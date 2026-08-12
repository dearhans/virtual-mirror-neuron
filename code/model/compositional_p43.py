"""model/compositional_p43.py — P4-3：ê_u 的经验贝叶斯逆方差收缩。

背景（W34 / 2026-08-09 留下的最高优先缺陷）
-----------------------------------------------
P4-2 的 `SoloEffectImputer` 把「仅在合作中出现过的基因」的单独效应插补为

        ê_u = mean_{(a,u) ∈ train doubles} [ z_au − b̂ − ê_a ]        （**等权平均**）

W34 实测：坍缩被修好（var_ratio 1.07e-26 → 0.503），但精度没变好
（ood_agent RMSE 0.5854 → 0.5887，pbRMSE 0.1927 → 0.1965）→ **插补噪声 ≥ 信号**。

诊断：被插补基因的共扰动条数跨度极大（62 ~ 2971，约 47×）。等权平均对每个基因内部
是无偏的，但**跨基因**看，低证据基因的 ê_u 方差远大于高证据基因；预测时把它们
同等信任地写进 φ，等于把一堆高方差估计原样注入模型 → 总 MSE 被少数噪声基因拖垮。

破法（TRIZ「预先反作用」：在噪声进入模型前先按证据量把它压回先验）
-----------------------------------------------
标准经验贝叶斯 / James–Stein 收缩，**不引入任何新的可学习参数**：

    对基因 u（n_u 条共扰动，逐输出维 k）
        raw_u    = mean_i r_i                       （r_i = z_au − b̂ − ê_a）
        s²_u     = 无偏的组内方差
        SE²_u    = s²_u / n_u                       （raw_u 的抽样方差）
    跨基因的矩估计先验
        μ₀       = 以 n_u 加权的 raw_u 均值
        V        = mean_u[(raw_u − μ₀)²]            （观测到的总离散度）
        τ²       = max(V − mean_u[SE²_u], ε)        （扣掉抽样噪声 = 真实基因间离散度）
    收缩
        λ_u      = τ² / (τ² + SE²_u)     ∈ (0, 1]
        ê_u      = μ₀ + λ_u · (raw_u − μ₀)
        SE²_post = λ_u · SE²_u                      （后验方差，恒 ≤ 原 SE²）

n_u 大 → SE² 小 → λ→1 → 几乎不动（高证据基因保真）；
n_u 小 → SE² 大 → λ→0 → 拉回 μ₀（低证据基因不再乱喷）。
这正是「逆方差加权」在层级模型下的正确形态。

诚实边界
-----------------------------------------------
- 收缩只降**估计方差**，不纠**加法先验的偏差**：若 z_au − b̂ − ê_a 的残差里
  混着真实 epistasis，那是系统性偏差，任何加权/收缩方案都无法分离。
  → 因此本方法若无效，正确结论是「噪声不在权重上，而在结构上」，应转 P4-4（结构分离），
    **不得**改判据或加网络层。
- μ₀ 取被插补基因群体的加权均值（同群体 EB）。另一可选参照是「已见基因效应的均值」，
  属不同先验假设；本实现固定前者并如实记录，避免事后挑参照。
- λ_u 逐输出维计算（K 维各自收缩），因不同基因程序在不同基因维上的离散度差异很大。

仅依赖 numpy。
"""
from __future__ import annotations

import numpy as np

from .compositional_p42 import SoloEffectImputer


class ShrunkSoloEffectImputer(SoloEffectImputer):
    """P4-2 插补器 + 经验贝叶斯逆方差收缩（P4-3）。

    继承 `SoloEffectImputer` 的 `predict` / `summary`，只替换 `fit` 末端的
    `solo_effect_` 与 `impute_se_`，因此两臂在 predict 路径上**完全同构**，
    差异可干净归因到收缩本身。
    """

    def __init__(self, min_count: int = 1, eps_tau: float = 1e-12):
        super().__init__(min_count=min_count)
        self.eps_tau = float(eps_tau)

    def fit(self, twin, X_fit: np.ndarray, y_fit: np.ndarray) -> "ShrunkSoloEffectImputer":
        # 先跑父类，拿到 seen_mask_ / b_hat_ / e_all_ / counts_ / has_solo_ / solo_effect_(等权)
        super().fit(twin, X_fit, y_fit)

        Dp = twin.Dp
        pa = X_fit[:, :Dp]
        pb = X_fit[:, Dp:2 * Dp]
        g = X_fit[:, 2 * Dp + twin.A_dim]
        z = (y_fit / g.reshape(-1, 1)) if y_fit.ndim == 2 else (y_fit / g)

        two_d = self.e_all_.ndim == 2
        K = self.e_all_.shape[1] if two_d else 1
        shape = (Dp, K) if two_d else (Dp,)

        sums = np.zeros(shape)
        sq = np.zeros(shape)
        cnts = np.zeros(Dp)

        dbl = np.flatnonzero(pb.sum(axis=1) > 0)
        if len(dbl):
            ca = pa[dbl].argmax(axis=1)
            cb = pb[dbl].argmax(axis=1)
            a_un = ~self.seen_mask_[ca]
            b_un = ~self.seen_mask_[cb]
            one = a_un ^ b_un
            if one.any():
                u = np.where(a_un[one], ca[one], cb[one])
                o = np.where(a_un[one], cb[one], ca[one])
                resid = z[dbl[one]] - self.b_hat_ - self.e_all_[o]
                np.add.at(sums, u, resid)
                np.add.at(sq, u, resid ** 2)
                np.add.at(cnts, u, 1.0)

        has = self.has_solo_
        idx = np.flatnonzero(has)
        self.shrinkage_applied_ = False
        self.shrinkage_report_ = {
            "n_genes_shrunk": 0,
            "reason_if_skipped": "no imputed genes",
        }
        if idx.size == 0:
            return self

        n = cnts[idx]                                   # (m,)
        n_col = n.reshape(-1, *([1] * (len(shape) - 1)))
        raw = sums[idx] / n_col                          # (m,K) 等权点估计（= 父类 solo_effect_）

        # 组内无偏方差 → raw 的抽样方差 SE²；n=1 的基因用池化组内方差兜底
        var_biased = np.maximum(sq[idx] / n_col - raw ** 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_unb = var_biased * (n_col / np.maximum(n_col - 1.0, 1e-9))
        multi = n > 1
        if multi.any():
            pooled = var_unb[multi].mean(axis=0)
        else:                                            # 全是 n=1：无法估噪声，放弃收缩
            self.shrinkage_report_["reason_if_skipped"] = "all imputed genes have n_u == 1"
            return self
        var_unb[~multi] = pooled
        se2 = var_unb / n_col                            # (m,K)

        # 跨基因矩估计先验（μ₀ 以证据量加权，更稳）
        w = n_col / n_col.sum()
        mu0 = (w * raw).sum(axis=0)                      # (K,)
        V = ((raw - mu0) ** 2).mean(axis=0)              # (K,)
        tau2 = np.maximum(V - se2.mean(axis=0), self.eps_tau)

        lam = tau2 / (tau2 + se2)                        # (m,K) ∈(0,1]
        shrunk = mu0 + lam * (raw - mu0)

        solo = np.zeros(shape)
        solo[idx] = shrunk
        self.solo_effect_raw_equal_weight_ = sums / np.maximum(cnts.reshape(-1, *([1] * (len(shape) - 1))), 1.0)
        self.solo_effect_ = solo

        # 后验方差 λ·SE²（恒 ≤ 原 SE²）→ 逐基因标量化，接口与父类一致
        se_post = np.zeros(Dp)
        post_var = lam * se2
        se_post[idx] = np.sqrt(post_var.mean(axis=1) if post_var.ndim == 2 else post_var)
        self.impute_se_ = se_post

        lam_g = lam.mean(axis=1) if lam.ndim == 2 else lam
        self.lambda_per_gene_ = {int(i): float(v) for i, v in zip(idx, lam_g)}
        self.shrinkage_applied_ = True
        self.shrinkage_report_ = {
            "n_genes_shrunk": int(idx.size),
            "codouble_count_min": int(n.min()),
            "codouble_count_max": int(n.max()),
            "codouble_count_median": float(np.median(n)),
            "codouble_count_ratio_max_over_min": float(n.max() / max(n.min(), 1)),
            "lambda_min": float(lam_g.min()),
            "lambda_max": float(lam_g.max()),
            "lambda_mean": float(lam_g.mean()),
            "lambda_median": float(np.median(lam_g)),
            "tau2_mean": float(np.mean(tau2)),
            "se2_mean": float(np.mean(se2)),
            "mean_abs_shift_vs_equal_weight": float(np.abs(shrunk - raw).mean()),
            "mean_abs_raw": float(np.abs(raw).mean()),
            "impute_se_mean_before": float(np.sqrt(
                (se2.mean(axis=1) if se2.ndim == 2 else se2)).mean()),
            "impute_se_mean_after": float(se_post[idx].mean()),
        }
        return self

    def summary(self) -> dict:
        out = super().summary()
        out["shrinkage"] = getattr(self, "shrinkage_report_", {})
        out["shrinkage_applied"] = bool(getattr(self, "shrinkage_applied_", False))
        return out
