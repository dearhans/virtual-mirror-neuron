"""code/model/compositional_p5c_gmm.py — P5-C：GMM 生成式密度门控 epistemic std。

P5 三机制面板最后一格（W33 §3.10 / §7）
------------------------------------------------
    A = proper-Bayesian / SGLD（内禀后验多样性，破共识塌缩）—— 精炼阴性（点估计代价 + ECE 不越 0.08）
    B = 密度感知注入（sample_weight + NCL）—— 双阴性（共识塌缩）
    C = 生成式 epistemic（GMM 密度门控）—— 本文件，未测

P5-C 机制（与路 A 同族、换平滑参数化密度）
--------------------------------------------
路 A 用 kNN 距离 nd(x) 作新颖度，在 ID 校准集上估「欠缩放比 r(nd)」→ 测试时按自身 nd 撑大 std。
P5-C 把同样门控逻辑搬到**平滑参数化密度**：用 sklearn GaussianMixture 在训练特征上拟合训练流形
密度 q(x)，以「密度新颖度」u(x)=log q_med − log q(x) 取代 kNN 距离——
    u 大 → 低密度（OOD）→ r(u) 大 → σ 撑大；u≈0（ID 主体）→ r≈1 → σ 保持诚实 ID 值。

直觉上 GMM 平滑密度比原始 kNN 距离在离散扰动空间外推更优雅（密度是连续可微函数，可外推到
训练点之间），故作为「三机制中密度门控的精致版」补完面板；但墙在机制层——
epistemic 欠缩放是深度集成 posterior 的性质，不是输入协变量密度的函数（路 A 同墙），
故 P5-C 大概率撞路 A 同墙。协议要求三机制全测，故廉价（无 torch/PyMC 工具链升级）补完以正当 kill-switch。

仅依赖 numpy / scikit-learn（与 model/ 一致）。fit 与父类完全相同（同监督口径 → 点估计增量
干净归因到 σ 合成改造），仅 override predict 的 σ 合成（密度门控）；另加 fit_density_gate 在
ID 校准集上估欠缩放比→密度门控倍率（与路 A fit_novelty_gate 同辙）。
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from .compositional import CompositionalTwin


class CompositionalTwinP5C_GMM(CompositionalTwin):
    """P5-C：GMM 密度门控 epistemic std（生成式 epistemic 三机制之一）。

    继承 CompositionalTwin 的 φ / 因果头 / 训练口径（标准 MLPRegressor 集成，无 SGLD），
    仅 override predict 的 σ 合成：σ_pred = σ_epi · g(u)，g 为 ID 校准拟合的
    「欠缩放比→密度新颖度 u」门控倍率。fit 标准集成后额外拟合 GMM 训练特征密度，
    fit_density_gate 在 ID 校准集上估门控。
    """

    def __init__(self, gmm_n_components: int = 8, gmm_cov_type: str = "diag",
                 gmm_reg_covar: float = 1e-6, gmm_random_state: int = 0,
                 gate_n_bins: int = 8, gate_cap_ratio: float = 50.0,
                 density_standardize: bool = True, random_state: int = 0, **kw):
        super().__init__(random_state=random_state, **kw)
        self.gmm_n_components = int(gmm_n_components)
        self.gmm_cov_type = gmm_cov_type
        self.gmm_reg_covar = float(gmm_reg_covar)
        self.gmm_random_state = int(gmm_random_state)
        self.gate_n_bins = int(gate_n_bins)
        self.gate_cap_ratio = float(gate_cap_ratio)
        self.density_standardize = bool(density_standardize)
        self._gmm_ok = False
        self._gate_fitted = False

    # --------------------------------------------------------------- 密度门控依赖
    def _build_feats(self, X: np.ndarray) -> np.ndarray:
        pa, pb, act, g = self._slice(X)
        return np.hstack([pa, pb, act, g.reshape(-1, 1)])

    def _std_feats(self, F: np.ndarray) -> np.ndarray:
        if not self.density_standardize:
            return F
        std = np.clip(self._gmm_std_, 1e-9, None)
        return (F - self._gmm_mean_) / std

    def _density_novelty(self, X: np.ndarray) -> np.ndarray:
        """密度新颖度 u(x) = log q_med − log q(x)（越大越 OOD / 低密度）。"""
        if not self._gmm_ok:
            return np.zeros(len(X))
        F = self._build_feats(X)
        Fs = self._std_feats(F)
        log_q = self.gmm_.score_samples(Fs)
        return (self._logq_med_ - log_q).reshape(-1)

    # --------------------------------------------------------------- 原始 epistemic（复用 路 A 口径）
    def _raw_epistemic(self, X: np.ndarray):
        """无门控/无封顶的原始集成 epistemic（供门控拟合与 predict 复用）。

        与路 A _raw_epistemic 同口径：σ_epi = sqrt(za_s²+zb_s²+2·zn_s²)·|g|，不施加 P3-1 有界门控与封顶。
        """
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
        nd_rel = nd / (self.med_dist_ + 1e-9)
        return y_mean, y_std_epi, nd_rel

    # --------------------------------------------------------------- fit：标准集成 + GMM 密度
    def fit(self, X: np.ndarray, y: np.ndarray, z_comp: np.ndarray = None):
        super().fit(X, y, z_comp=z_comp)  # 标准 MLPRegressor 集成（与 P1 同监督口径 → 点估计增量可归因）
        # —— GMM 拟合训练特征密度 q(x) ——
        F = self.train_feats_  # 父类 fit 已建（hstack[pa,pb,act,g]）
        if self.density_standardize:
            self._gmm_mean_ = F.mean(0)
            self._gmm_std_ = F.std(0)
        try:
            Fs = self._std_feats(F)
            self.gmm_ = GaussianMixture(
                n_components=self.gmm_n_components, covariance_type=self.gmm_cov_type,
                reg_covar=self.gmm_reg_covar, random_state=self.gmm_random_state,
                max_iter=200, n_init=1,
            ).fit(Fs)
            self._logq_med_ = float(np.median(self.gmm_.score_samples(Fs)))
            self._gmm_ok = True
        except Exception as e:  # noqa: BLE001
            self._gmm_ok = False
            self.gmm_ = None
            self._logq_med_ = 0.0
            print(f"[p5c] GMM fit failed ({e!r}); density gate disabled (multiplier≡1).", flush=True)
        return self

    # --------------------------------------------------------------- 门控拟合（ID 校准集估欠缩放比）
    def fit_density_gate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """在 ID 校准集上估欠缩放比 → 密度门控倍率（与路 A fit_novelty_gate 同辙）。

        r(u) = median|resid|/σ_epi 作为「密度新颖度 u」的函数（u 越大越 OOD），
        单调非减（cumulative-max）；参考点 u_ref=0（训练密度中位数）→ 倍率≈1。
        OOD（u 超出校准范围）按 cap 外推防饱和（PC-3 宽↔窄矛盾）。
        """
        if not self._gmm_ok:
            self._gate_u = np.array([0.0])
            self._gate_ratio = np.array([1.0])
            self._gate_r_ref = 1.0
            self._gate_fitted = True
            return self

        mu_c, std_c, _ = self._raw_epistemic(X_cal)
        score = np.abs(y_cal - mu_c) / np.clip(std_c, 1e-9, None)
        score1d = np.median(score, axis=1) if score.ndim == 2 else score

        u = self._density_novelty(X_cal)
        self._u_ref = 0.0  # 训练密度中位数即参考（校准同分布，u 中心≈0）

        edges = np.quantile(u, np.linspace(0, 1, self.gate_n_bins + 1))
        edges[0] = float(u.min()) - 1e-9
        edges[-1] = float(u.max()) + 1e-9
        centers, ratios = [], []
        for b in range(self.gate_n_bins):
            msk = (u >= edges[b]) & (u < edges[b + 1])
            if int(msk.sum()) >= 5:
                centers.append(float(np.median(u[msk])))
                ratios.append(float(np.median(score1d[msk])))
        centers = np.array(centers)
        ratios = np.array(ratios)

        # 按 u 排序并强制单调非减（消除小箱噪声，符合「越 OOD 越欠缩放」机制先验）
        order = np.argsort(centers)
        centers = centers[order]
        ratios = np.maximum.accumulate(ratios[order])

        r_ref = float(np.interp(self._u_ref, centers, ratios))
        self._gate_u = centers
        self._gate_ratio = ratios / max(r_ref, 1e-9)  # 相对参考点倍率（参考点=1）
        self._gate_r_ref = r_ref
        self._gate_fitted = True
        return self

    def _density_multiplier(self, u: np.ndarray) -> np.ndarray:
        """测试点密度新颖度 → std 撑大倍率。u 低于校准范围→最低箱倍率(≈1)；超出→按 cap 外推。"""
        if not self._gate_fitted:
            return np.ones_like(u)
        mult = np.interp(u, self._gate_u, self._gate_ratio,
                         left=float(self._gate_ratio[0]), right=self.gate_cap_ratio)
        return np.clip(mult, float(self._gate_ratio[0]), self.gate_cap_ratio)

    # --------------------------------------------------------------- predict
    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("CompositionalTwinP5C_GMM.predict 前必须先 fit。")
        y_mean, y_std_epi, nd_rel = self._raw_epistemic(X)
        u = self._density_novelty(X)
        mult = self._density_multiplier(u)
        mult_f = mult if y_std_epi.ndim == 1 else mult.reshape(-1, 1)
        sigma_pred = y_std_epi * mult_f
        return y_mean, sigma_pred, u
