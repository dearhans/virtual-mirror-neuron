"""model/compositional.py — 组合性先验孪生（攻克 ood_action 组合泛化）。

设计要点（呼应项目硬约束「机制而非记忆」「OOD」「不确定度」）：
  - 组合性先验：共享组件编码器 phi(p)（动作轴已剥离）；双扰动响应 =
        z = phi(p_a) + phi(p_b) - phi(null)
    即用「两条已见单扰动效应相加」来外推组合 —— 不是记忆组合，而是机制组合。
    动作轴 act 不进入 φ：单/双语义由「pb 是否为零」隐含，pa/pb 同处 Dp 维坐标 → φ 天然对称，
    消除 act=double 对已知基因的纯外推失真（P1 架构修正）。
  - 因果头：训练目标归一化为 z = y / g，把乘法调质机制变成可学先验；
    预测再乘回 g → 对新增益具备外推能力。
  - 课程学习：stage1 仅用单扰动组件拟合 phi（先学机制），
    stage2 用 warm_start 续训含双扰动组件（再学组合）。
  - 不确定度：集成(epistemic) + 新颖度(noOD 检测) 合成。

仅依赖 numpy / scikit-learn。
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors


class CompositionalTwin:
    """组合性多尺度孪生：fit(X, y, z_comp) 后 predict(X) -> (mean, std, nd_rel)。

    X 列布局：[p_a(0..Dp-1), p_b(Dp..2Dp-1), act(2Dp..2Dp+A_dim-1), g(2Dp+A_dim)]。
    """

    def __init__(
        self,
        Dp: int = 8,
        A_dim: int = 2,
        hidden: tuple = (32, 32),
        n_ensemble: int = 5,
        novelty_k: float = 0.8,
        k_neighbors: int = 5,
        random_state: int = 0,
        curriculum: bool = True,
    ):
        self.Dp = Dp
        self.A_dim = A_dim
        self.hidden = hidden
        self.n_ensemble = n_ensemble
        self.novelty_k = novelty_k
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.curriculum = curriculum
        self._fitted = False

    def _slice(self, X):
        pa = X[:, : self.Dp]
        pb = X[:, self.Dp: 2 * self.Dp]
        act = X[:, 2 * self.Dp: 2 * self.Dp + self.A_dim]
        g = X[:, 2 * self.Dp + self.A_dim: 2 * self.Dp + self.A_dim + 1].ravel()
        return pa, pb, act, g

    def fit(self, X: np.ndarray, y: np.ndarray, z_comp: np.ndarray = None):
        rng = np.random.RandomState(self.random_state)
        pa, pb, act, g = self._slice(X)

        # 因果头：增益归一化，把乘法调质机制变成可学先验。
        # 标量响应(1D)：g 保持 (n,) 直接除；向量响应(2D, 如 Norman K=256)：g 扩为 (n,1) 按列广播。
        z = (y / g.reshape(-1, 1)) if y.ndim == 2 else (y / g)

        # 组件特征：每个扰动分量 = 基因掩码 p（动作轴已从 φ 剥离）。
        # 单/双语义由「pb 是否为零」隐含（加法组合结构已编码），无需 act 输入：
        #   - 消除 act=double 对已知基因纯外推失真（P1 架构修正）；
        #   - pa/pb 同处 Dp 维坐标空间 → 共享 φ 天然对称，修复坐标不对称。
        fa = pa
        fb = pb
        null = np.zeros((len(X), self.Dp))

        # 单组件监督目标：
        #   - 若提供 z_comp（合成生成器已知机制）：直接用 per-component 贡献（ta, tb 已正确分解）。
        #   - 否则（真实数据，无机制先验）：加法分解无法监督——真实双扰动是 epistatic（非加和），
        #     无法把整体响应 z 拆成 ta/tb。故仅在「单扰动行」(pb=0) 上监督 φ 学单基因效应，
        #     双扰动由结构加法 φ(pa)+φ(pb)-2φ(0) 在预测时外推。这避免把双扰动整体响应错误灌入
        #     φ(pa) 造成「单/双目标冲突」而学崩（此前 id RMSE 因此从 0.57 退到 1.15）。
        #     因 φ 现只吃 Dp 维基因掩码，pa/pb 同坐标空间，φ(pb) 在预测时自动可用。
        if z_comp is not None:
            ta = np.asarray(z_comp)[:, 0]
            tb = np.asarray(z_comp)[:, 1]
            comp_X = np.vstack([fa, fb, null])
            comp_y = np.concatenate([ta, tb, np.zeros_like(ta)], axis=0)
            single_X = np.vstack([fa, null])
            single_y = np.concatenate([ta, np.zeros_like(ta)], axis=0)
        else:
            single_mask = (pb.sum(axis=1) == 0)
            fa_s = pa[single_mask]
            z_s = z[single_mask]
            null_s = np.zeros((int(single_mask.sum()), self.Dp))
            # 真实数据无 z_comp：null 监督为目标基线 b（而非 0），使 φ(null)=基线，
            # 加法公式 φ(p_a)+φ(p_b)−φ(null) 才等于「基线 + 两单扰动效应」（正确加性，不重复计基线）。
            # 合成数据走上方 z_comp 分支，其 null 本就为 0，保持 0。
            baseline = z_s.mean(axis=0)
            null_y = np.broadcast_to(baseline, z_s.shape).copy()
            single_X = np.vstack([fa_s, null_s])
            single_y = np.concatenate([z_s, null_y], axis=0)
            comp_X = single_X
            comp_y = single_y

        self.ensembles_ = []
        for _ in range(self.n_ensemble):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden,
                max_iter=800,
                tol=1e-4,
                warm_start=self.curriculum,
                random_state=int(rng.randint(1, 1_000_000)),
                early_stopping=False,
            )
            if self.curriculum:
                # stage1：仅单扰动组件（fa, ta）+ null（教 phi 学到单扰动机制）
                mlp.fit(single_X, single_y)
                # stage2：warm_start 续训（真实数据下与 stage1 同分布；合成数据下补双扰动分量）
                mlp.max_iter = 800
                mlp.fit(comp_X, comp_y)
            else:
                mlp.fit(comp_X, comp_y)
            self.ensembles_.append(mlp)

        # 新颖度模型：在 (p_a, p_b, act, g) 全特征空间上做 kNN
        self.train_feats_ = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_.fit(self.train_feats_)
        self.med_dist_ = float(np.median(self._knn_dist(self.train_feats_)))

        # P3-1：记录 ID epistemic (σ_epi) 的 95 分位数，predict 时封顶 OOD 爆炸
        self.sigma_id_95_ = float(self._sigma_epi_quantile(X[: min(len(X), 2000)]))

        self._fitted = True
        return self

    def _knn_dist(self, F: np.ndarray) -> np.ndarray:
        dists, _ = self.nn_.kneighbors(F)
        return dists.mean(axis=1)

    def _phi(self, feats: np.ndarray) -> tuple:
        preds = np.stack([m.predict(feats) for m in self.ensembles_], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def _sigma_epi_quantile(self, X, q=0.95):
        """P3-1：ID epistemic (σ_epi) 的 q 分位数，用于 predict 时封顶 OOD 爆炸。"""
        pa, pb, act, g = self._slice(X)
        g_col = g.reshape(-1, 1)
        fa, fb = pa, pb
        null = np.zeros((len(X), self.Dp))
        za_m, za_s = self._phi(fa)
        zb_m, zb_s = self._phi(fb)
        zn_m, zn_s = self._phi(null)
        z_mean = za_m + zb_m - zn_m
        z_std_epi = np.sqrt(za_s ** 2 + zb_s ** 2 + 2 * zn_s ** 2)
        if z_mean.ndim == 1:
            y_std_epi = z_std_epi * np.abs(g)
        else:
            y_std_epi = z_std_epi * np.abs(g_col)
        return float(np.quantile(np.abs(y_std_epi), q))

    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("CompositionalTwin.predict 前必须先 fit。")
        pa, pb, act, g = self._slice(X)
        g_col = g.reshape(-1, 1)  # (n,1)
        # 组件特征：仅基因掩码 p（动作轴不进入 φ，见 fit 注释）
        fa = pa
        fb = pb
        null = np.zeros((len(X), self.Dp))

        za_m, za_s = self._phi(fa)
        zb_m, zb_s = self._phi(fb)
        zn_m, zn_s = self._phi(null)

        # 加法组合：z = φ(p_a) + φ(p_b) − φ(null)（docstring 公式，null 仅减一次）。
        # 真实数据下 φ(null) 被监督为基线 b，故该式 = 基线 + effect_a + effect_b（正确加性，不重复计基线）。
        # 先前误写成 (φ(pa)−φ(null))+(φ(pb)−φ(null)) 会重复减基线 → 双扰动过预测约 2×。
        z_mean = za_m + zb_m - zn_m
        # 集成的 epistemic 不确定度：两独立组件 + null 在方差上相加
        z_std_epi = np.sqrt(za_s ** 2 + zb_s ** 2 + 2 * zn_s ** 2)

        # 标量响应(1D) 用 1D g；向量响应(2D) 用 (n,1) g 按列广播
        if z_mean.ndim == 1:
            y_mean = z_mean * g
            y_std_epi = z_std_epi * np.abs(g)
            F = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        else:
            y_mean = z_mean * g_col
            y_std_epi = z_std_epi * np.abs(g_col)
            F = np.hstack([pa, pb, act, g_col])
        nd = self._knn_dist(F)
        nd_relative = nd / (self.med_dist_ + 1e-9)
        # P3-1 σ 解耦 + 有界结构门控（ACC-CRL g(u) 风格）：
        # σ_pred = σ_epi * gate(nd)，gate = 1 + k·sigmoid(c·(nd-1)) ∈ [1, 1+k] 封顶。
        # 既让 OOD 不确定度随新颖度适度增大，又避免无界撑宽导致 coverage 锁 1.0（0.2125 饱和）。
        # 这是从「上半场加和」到「门控」的 TRIZ 条件分离破法。
        gate = 1.0 + self.novelty_k / (1.0 + np.exp(-2.0 * (nd_relative - 1.0)))
        gate_f = gate if y_std_epi.ndim == 1 else gate.reshape(-1, 1)
        # P3-1 封顶：OOD (held-out 基因对) 上 epistemic 方差无界爆炸 → 区间锁 1.0 饱和。
        # 用 ID epistemic 95 分位数封顶 σ_epi（允许 k 倍 ID 级波动），使区间有界且合理。
        cap = 10.0 * getattr(self, "sigma_id_95_", 1.0)
        sigma_pred = np.minimum(y_std_epi, cap) * gate_f
        return y_mean, sigma_pred, nd_relative

    def __call__(self, X: np.ndarray) -> np.ndarray:
        mean, _, _ = self.predict(X)
        return mean


class CompositionalInteractionTwin:
    """P2：加法组合先验(φ) + 可学习基因对交互项（拟合真实 epistasis）。

    设计（相对 P1 的升级）：
      z = φ(p_a) + φ(p_b) − φ(null)            [加法基，与 P1 完全相同：φ 仅吃基因掩码、单扰动监督]
          + ψ( h(p_a, p_b) )                    [基因对交互修正，仅在双扰动上监督]

    关键性质（为何能攻克 P1 失败）：
      - φ 冻结（同 P1，只在单扰动行学单基因效应）—— 单扰动行为完全继承 P1 的优良拟合。
      - ψ 是**对称**交互头：h = concat(e_a+e_b, e_a⊙e_b)（对换 a/b 不变）→ 小 MLP → K 维 epistasis 修正。
      - ψ **只依赖基因效应**(e_a=φ(p_a)−φ(null), e_b=φ(p_b)−φ(null))，**不依赖基因对身份**，
        故能外推到 held-out *基因对*(ood_action) —— 这正是 P1 加法先验做不到的。
      - 单扰动：p_b=0 → e_b=0 → 交互项仅对双扰动激活，故单扰动 z=加法基（保持 P1 单扰动拟合）。
      - 双扰动：ψ 由**训练双扰动**拟合 → 学到真实 epistasis（拮抗/协同）；
        对 held-out 对的泛化能力 = P2 是否成立的实证检验。
    仅依赖 numpy / scikit-learn（与 P1 一致，无新依赖）。
    """

    def __init__(
        self,
        Dp: int = 8,
        A_dim: int = 2,
        hidden: tuple = (32, 32),
        n_ensemble: int = 5,
        novelty_k: float = 0.8,
        k_neighbors: int = 5,
        random_state: int = 0,
        curriculum: bool = True,
        interaction_hidden: tuple = (64,),
        interaction_reg: float = 1e-1,
    ):
        self.Dp = Dp
        self.A_dim = A_dim
        self.hidden = hidden
        self.n_ensemble = n_ensemble
        self.novelty_k = novelty_k
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.curriculum = curriculum
        self.interaction_hidden = interaction_hidden
        self.interaction_reg = interaction_reg
        self._fitted = False

    def _slice(self, X):
        pa = X[:, : self.Dp]
        pb = X[:, self.Dp: 2 * self.Dp]
        act = X[:, 2 * self.Dp: 2 * self.Dp + self.A_dim]
        g = X[:, 2 * self.Dp + self.A_dim: 2 * self.Dp + self.A_dim + 1].ravel()
        return pa, pb, act, g

    def fit(self, X: np.ndarray, y: np.ndarray, z_comp: np.ndarray = None):
        rng = np.random.RandomState(self.random_state)
        pa, pb, act, g = self._slice(X)
        z = (y / g.reshape(-1, 1)) if y.ndim == 2 else (y / g)

        # ---- Stage A：同 P1 拟合 φ（仅单扰动行监督，动作轴已剥离）----
        fa = pa
        fb = pb
        null = np.zeros((len(X), self.Dp))
        if z_comp is not None:
            ta = np.asarray(z_comp)[:, 0]
            tb = np.asarray(z_comp)[:, 1]
            comp_X = np.vstack([fa, fb, null])
            comp_y = np.concatenate([ta, tb, np.zeros_like(ta)], axis=0)
            single_X = np.vstack([fa, null])
            single_y = np.concatenate([ta, np.zeros_like(ta)], axis=0)
        else:
            single_mask = (pb.sum(axis=1) == 0)
            fa_s = pa[single_mask]
            z_s = z[single_mask]
            null_s = np.zeros((int(single_mask.sum()), self.Dp))
            baseline = z_s.mean(axis=0)
            null_y = np.broadcast_to(baseline, z_s.shape).copy()
            single_X = np.vstack([fa_s, null_s])
            single_y = np.concatenate([z_s, null_y], axis=0)
            comp_X = single_X
            comp_y = single_y

        self.ensembles_ = []
        for _ in range(self.n_ensemble):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden,
                max_iter=800,
                tol=1e-4,
                warm_start=self.curriculum,
                random_state=int(rng.randint(1, 1_000_000)),
                early_stopping=False,
            )
            if self.curriculum:
                mlp.fit(single_X, single_y)
                mlp.max_iter = 800
                mlp.fit(comp_X, comp_y)
            else:
                mlp.fit(comp_X, comp_y)
            self.ensembles_.append(mlp)

        # ---- Stage B：拟合交互头 ψ（仅双扰动行，残差 = 真实 epistasis）----
        za_m, _ = self._phi(fa)
        zb_m, _ = self._phi(fb)
        zn_m, _ = self._phi(null)
        z_base = za_m + zb_m - zn_m
        ea = za_m - zn_m          # 单基因 a 的效应（已减基线）
        eb = zb_m - zn_m          # 单基因 b 的效应
        # 兼容 1D（合成标量响应）与 2D（Norman 向量响应）：列化后再水平拼接
        ea = ea.reshape(-1, 1) if ea.ndim == 1 else ea
        eb = eb.reshape(-1, 1) if eb.ndim == 1 else eb
        is_double = (pb.sum(axis=1) > 0)

        if int(is_double.sum()) > 0:
            # 对称交互特征：h = [e_a+e_b, e_a⊙e_b]（对换 a/b 不变）
            h = np.hstack([ea + eb, ea * eb])
            r = z - z_base                       # 残差 = 真实 epistasis（加法基未能解释）
            h_d = h[is_double]
            r_d = r[is_double]
            self.psi_ = MLPRegressor(
                hidden_layer_sizes=tuple(self.interaction_hidden),
                alpha=float(self.interaction_reg),
                max_iter=800,
                tol=1e-4,
                random_state=int(self.random_state),
                early_stopping=False,
            )
            self.psi_.fit(h_d, r_d)
            pred_r = self.psi_.predict(h_d)
            self.psi_sigma_ = float(np.sqrt(np.mean((pred_r - r_d) ** 2)))
        else:
            self.psi_ = None
            self.psi_sigma_ = 0.0

        # 新颖度模型（同 P1）
        self.train_feats_ = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_.fit(self.train_feats_)
        self.med_dist_ = float(np.median(self._knn_dist(self.train_feats_)))

        # P3-1：记录 ID epistemic (σ_epi) 的 95 分位数，predict 时封顶 OOD 爆炸
        self.sigma_id_95_ = float(self._sigma_epi_quantile(X[: min(len(X), 2000)]))

        self._fitted = True
        return self

    def _knn_dist(self, F: np.ndarray) -> np.ndarray:
        dists, _ = self.nn_.kneighbors(F)
        return dists.mean(axis=1)

    def _phi(self, feats: np.ndarray) -> tuple:
        preds = np.stack([m.predict(feats) for m in self.ensembles_], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def _sigma_epi_quantile(self, X, q=0.95):
        """P3-1：ID epistemic (σ_epi) 的 q 分位数，用于 predict 时封顶 OOD 爆炸。"""
        pa, pb, act, g = self._slice(X)
        g_col = g.reshape(-1, 1)
        fa, fb = pa, pb
        null = np.zeros((len(X), self.Dp))
        za_m, za_s = self._phi(fa)
        zb_m, zb_s = self._phi(fb)
        zn_m, zn_s = self._phi(null)
        z_mean = za_m + zb_m - zn_m
        z_std_epi = np.sqrt(za_s ** 2 + zb_s ** 2 + 2 * zn_s ** 2)
        if z_mean.ndim == 1:
            y_std_epi = z_std_epi * np.abs(g)
        else:
            y_std_epi = z_std_epi * np.abs(g_col)
        return float(np.quantile(np.abs(y_std_epi), q))

    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("CompositionalInteractionTwin.predict 前必须先 fit。")
        pa, pb, act, g = self._slice(X)
        g_col = g.reshape(-1, 1)
        fa = pa
        fb = pb
        null = np.zeros((len(X), self.Dp))

        za_m, za_s = self._phi(fa)
        zb_m, zb_s = self._phi(fb)
        zn_m, zn_s = self._phi(null)

        # 加法基（与 P1 同）
        z_base = za_m + zb_m - zn_m
        z_base_std = np.sqrt(za_s ** 2 + zb_s ** 2 + 2 * zn_s ** 2)

        # 交互项（对称特征 → ψ → 加法修正；仅双扰动激活）
        ea = za_m - zn_m
        eb = zb_m - zn_m
        # 兼容 1D（合成标量响应）与 2D（Norman 向量响应）：列化后再水平拼接
        ea = ea.reshape(-1, 1) if ea.ndim == 1 else ea
        eb = eb.reshape(-1, 1) if eb.ndim == 1 else eb
        is_double = (pb.sum(axis=1) > 0)
        z_int = np.zeros_like(z_base)
        if self.psi_ is not None and int(is_double.sum()) > 0:
            h = np.hstack([ea + eb, ea * eb])
            psi = self.psi_.predict(h)
            if z_base.ndim == 2:
                z_int = np.where(is_double[:, None], psi, 0.0)
            else:
                z_int = np.where(is_double, psi, 0.0)
        z_mean = z_base + z_int

        if z_mean.ndim == 1:
            y_mean = z_mean * g
            y_std_epi = z_base_std * np.abs(g)
            F = np.hstack([pa, pb, act, g.reshape(-1, 1)])
            psi_std_arr = (is_double * self.psi_sigma_).astype(float)
        else:
            y_mean = z_mean * g_col
            y_std_epi = z_base_std * np.abs(g_col)
            F = np.hstack([pa, pb, act, g_col])
            psi_std_arr = (is_double[:, None] * self.psi_sigma_).astype(float)

        # 交互头残差作为双扰动行的额外（同方差）不确定度
        y_std_epi = np.sqrt(y_std_epi ** 2 + psi_std_arr ** 2)

        nd = self._knn_dist(F)
        nd_relative = nd / (self.med_dist_ + 1e-9)
        # P3-1 σ 解耦 + 有界结构门控（ACC-CRL g(u) 风格）：σ_pred = σ_epi * gate(nd) 封顶。
        # 修复 ood_agent/ood_action 上 epistemic 方差结构性膨胀导致的饱和。
        gate = 1.0 + self.novelty_k / (1.0 + np.exp(-2.0 * (nd_relative - 1.0)))
        gate_f = gate if y_std_epi.ndim == 1 else gate.reshape(-1, 1)
        # P3-1 封顶：OOD (held-out 基因对) 上 epistemic 方差无界爆炸 → 区间锁 1.0 饱和。
        # 用 ID epistemic 95 分位数封顶 σ_epi（允许 k 倍 ID 级波动），使区间有界且合理。
        cap = 10.0 * getattr(self, "sigma_id_95_", 1.0)
        sigma_pred = np.minimum(y_std_epi, cap) * gate_f
        return y_mean, sigma_pred, nd_relative

    def __call__(self, X: np.ndarray) -> np.ndarray:
        mean, _, _ = self.predict(X)
        return mean
