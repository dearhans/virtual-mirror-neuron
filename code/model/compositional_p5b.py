"""model/compositional_p5b.py — P5-B · 密度感知多样性注入（结构性估计器方向，W33 §3.10）。

机制（TRIZ #1 空间分割 + #15 动态化；Einstein 修对称性非补输出）：
  当前 `CompositionalTwin` 的 n_ensemble 个 MLP 在**完全相同的数据**上独立训练 → 收敛到同一函数
  → OOD 方差塌缩 → σ 欠缩放（PC-3 根因）。P5-B 在**构造期**注入多样性：
    训练每个成员 m 时，按「局部训练支撑稀疏度」w(x) 加权样本，并叠加**逐成员平滑区域偏好**
    u_m(x)（低维随机投影 + 随机相位），使不同成员在不同稀疏区域被差异化强调 → 在稀疏/OOD 区
    天然分散、在密集 ID 区仍一致。σ(x) 对「到训练流形距离」τ(x) 的单调性（∂σ/∂τ>0）由构造满足。

实现要点：
  - w(x) = 连续 kNN 稀疏度（kNN 距离归一化），**非路 A 的二值 novelty**——重复细胞 w=0（密集），
    OOD 新组合 w→1（稀疏），中间连续。
  - 逐成员 u_m(x)=sigmoid(Z·a_m + b_m)，Z=特征随机投影；不同 (a_m,b_m) → 不同成员在不同区域被强调。
  - sample_weight = clip(1 + λ·w(x)·(2u_m(x)-1), 0.01, ∞)：稀疏区按成员偏好上/下采样 → 成员分歧。
  - predict 返回**原始集成 epistemic**（无 P3-1 门控/封顶），以便干净归因「多样性注入本身是否治本」。

仅依赖 numpy / scikit-learn（与父类一致）。
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors

from model.compositional import CompositionalTwin


class CompositionalTwinP5B(CompositionalTwin):
    """P5-B 密度感知多样性注入孪生（继承 CompositionalTwin 的 φ/因果头/课程学习）。"""

    def __init__(self, density_lambda: float = 0.1, density_proj_dim: int = 3,
                 random_state: int = 0, **kw):
        super().__init__(random_state=random_state, **kw)
        self.density_lambda = density_lambda
        self.density_proj_dim = density_proj_dim

    # --------------------------------------------------------------- 稀疏度权重
    def _sparsity_weight(self, F: np.ndarray) -> np.ndarray:
        """连续 kNN 稀疏度：0=密集（重复细胞/训练核心），1=稀疏（OOD-like 新组合）。"""
        d = self._knn_dist(F)  # 各点到 k 近邻的平均距离
        lo, hi = float(d.min()), float(d.max())
        return (d - lo) / max(hi - lo, 1e-9)

    # --------------------------------------------------------------- 训练（注入多样性）
    def fit(self, X: np.ndarray, y: np.ndarray, z_comp: np.ndarray = None):
        rng = np.random.RandomState(self.random_state)
        pa, pb, act, g = self._slice(X)

        z = (y / g.reshape(-1, 1)) if y.ndim == 2 else (y / g)
        fa, fb = pa, pb
        null = np.zeros((len(X), self.Dp))

        # —— 复用父类监督目标构造（单扰动行监督 φ；双扰动由结构加法外推）——
        if z_comp is not None:
            ta = np.asarray(z_comp)[:, 0]
            tb = np.asarray(z_comp)[:, 1]
            comp_X = np.vstack([fa, fb, null])
            comp_y = np.concatenate([ta, tb, np.zeros_like(ta)], axis=0)
            single_X = np.vstack([fa, null])
            single_y = np.concatenate([ta, np.zeros_like(ta)], axis=0)
            comp_idx = np.concatenate([np.arange(len(X)), np.arange(len(X)), np.arange(len(X))])
            single_idx = np.concatenate([np.arange(len(X)), np.arange(len(X))])
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
            sm_idx = np.flatnonzero(single_mask)
            comp_idx = np.concatenate([sm_idx, sm_idx])
            single_idx = np.concatenate([sm_idx, sm_idx])

        # —— kNN 密度（全训练特征空间），先于训练以获得 w(x) ——
        self.train_feats_ = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_.fit(self.train_feats_)
        self.med_dist_ = float(np.median(self._knn_dist(self.train_feats_)))
        w_train = self._sparsity_weight(self.train_feats_)  # (n_train,) 连续稀疏度

        # —— 逐成员平滑区域偏好（低维随机投影 + 随机相位）——
        rnd = np.random.RandomState(self.random_state + 7)
        proj = rnd.randn(self.train_feats_.shape[1], self.density_proj_dim)
        Z = self.train_feats_ @ proj  # (n_train, proj_dim)

        def member_sample_weight(idxs, a_vec, b_sc):
            wv = w_train[idxs]
            uv = 1.0 / (1.0 + np.exp(-(Z[idxs] @ a_vec + b_sc)))
            return np.clip(1.0 + self.density_lambda * wv * (2.0 * uv - 1.0), 0.01, None)

        self.ensembles_ = []
        self._member_meta = []
        for m_i in range(self.n_ensemble):
            a_vec = rnd.randn(self.density_proj_dim)
            b_sc = rnd.randn()
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden,
                max_iter=800,
                tol=1e-4,
                warm_start=self.curriculum,
                random_state=int(rng.randint(1, 1_000_000)),
                early_stopping=False,
            )
            if self.curriculum:
                mlp.fit(single_X, single_y, sample_weight=member_sample_weight(single_idx, a_vec, b_sc))
                mlp.max_iter = 800
                mlp.fit(comp_X, comp_y, sample_weight=member_sample_weight(comp_idx, a_vec, b_sc))
            else:
                mlp.fit(comp_X, comp_y, sample_weight=member_sample_weight(comp_idx, a_vec, b_sc))
            self.ensembles_.append(mlp)
            self._member_meta.append({"a_vec": a_vec.tolist(), "b_sc": float(b_sc)})

        # P3-1 同款 ID epistemic 95 分位记录（仅用于 cap 参考，predict 实际不封顶）
        self.sigma_id_95_ = float(self._sigma_epi_quantile(X[: min(len(X), 2000)]))
        self._fitted = True
        return self

    # --------------------------------------------------------------- predict（原始集成 epistemic）
    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("CompositionalTwinP5B.predict 前必须先 fit。")
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
            y_mean = z_mean * g
            y_std_epi = z_std_epi * np.abs(g)
            F = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        else:
            y_mean = z_mean * g_col
            y_std_epi = z_std_epi * np.abs(g_col)
            F = np.hstack([pa, pb, act, g_col])

        nd = self._knn_dist(F)
        nd_relative = nd / (self.med_dist_ + 1e-9)
        # 原始集成 epistemic，无 P3-1 门控/封顶——干净归因多样性注入本身
        return y_mean, y_std_epi, nd_relative
