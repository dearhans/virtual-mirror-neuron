"""model/compositional_p42.py — P4-2：共扰动→单独效应插补（Solo-Effect Imputation）。

背景（2026-08-07 W33 坍缩检测发现的活跃缺陷）
-----------------------------------------------
`CompositionalTwin`（P1）在 `ood_agent` 子集上**完全坍缩为常数预测器**
（跨扰动组 var_ratio=1.07e-26，20 组只输出 1 种预测）。

根因（本模块的核心诊断，非"再加一层网络"）
-----------------------------------------------
1. `ood_agent` = 「held-out 基因的**单**扰动」。
2. 但 `norman_adapter` 的切分规则里，**含 held-out 基因的双扰动被划进 train**
   （"含未见基因的双扰动 → 训练额外信号"）→ 该基因**在训练集中出现过**，
   只是从未以「单独扰动」的形式出现。
3. 而 P1 的 φ **只在单扰动行上监督**（`single_mask = pb.sum(1)==0`），
   双扰动行完全不进 φ 的训练集 → held-out 基因对应的 one-hot 坐标在 φ 的
   训练输入里恒为 0 → `φ(p_unseen) ≡ φ(0) = baseline` → 预测常数。

即：**信号存在于数据中，只是被 P1 的监督口径丢弃了**。这不是容量问题，是口径问题。

破法（TRIZ 自服务 / 预先作用：用系统自身的加法先验制造缺失坐标的监督）
-----------------------------------------------
在 P1 的加法结构 `z = φ(p_a) + φ(p_b) − φ(null)` 下，对一条训练双扰动 (a=已见, u=未见)：

        z_au = φ(a) + φ(u) − b        (b := φ(null) = 基线)
   ⟹    φ(u) = z_au − φ(a) + b
   ⟹    e_u := φ(u) − b = z_au − b − e_a ,   e_a := φ(a) − b

于是「只在合作中出现过的基因」的**单独效应** e_u 可由其所有训练共扰动求均值插补：

        ê_u = mean_{(a,u) ∈ train doubles} [ z_au − b̂ − ê_a ]

预测时对含 u 的行把 `φ(p_u)` 覆写为 `b̂ + ê_u`。

镜像神经元语义
-----------------------------------------------
这正是「自我/他者同源」的可计算版本：**推断一个只在合作场景中被观察过的主体，
单独行动时的效应**。共享潜空间（加法基 φ）提供了把"合作观察"投影回"单独执行"的通道。

不确定度（遵守项目硬约束）
-----------------------------------------------
插补不是免费的：为每个被插补基因记录残差离散度，作为**插补 epistemic 分量**
按方差相加进 σ_pred；共扰动样本越少 / 越不一致，区间越宽。

诚实边界
-----------------------------------------------
- ê_u 在真实 epistasis 下有偏（用加法基去解释非加和响应）。偏差量级由项目已量化的
  「非加和残差小」结论界定，但对强上位性基因对会失效——这是本方法的已知适用边界。
- 若某 held-out 基因在训练集中**连共扰动都没有**（cnt=0），本方法不插补，
  退回 P1 行为（仍坍缩）。这类基因是**真正不可识别**的，应如实报告。

仅依赖 numpy（不引入新依赖）。
"""
from __future__ import annotations

import numpy as np


class SoloEffectImputer:
    """包裹一个已 fit 的 CompositionalTwin / CompositionalInteractionTwin，
    用训练集双扰动为「仅在合作中出现过的基因」插补单独效应。

    用法::

        imp = SoloEffectImputer().fit(twin, X_fit, y_fit)
        mu, sigma, nd = imp.predict(twin, X_test)

    `X_fit / y_fit` 必须是**该 twin 实际拟合所用的行**（若走 conformal，
    则为 fidx 子集），否则插补会用到 twin 没见过的数据，破坏可比性。
    """

    def __init__(self, min_count: int = 1):
        self.min_count = int(min_count)
        self._fitted = False

    # --------------------------------------------------------------- fit
    def fit(self, twin, X_fit: np.ndarray, y_fit: np.ndarray) -> "SoloEffectImputer":
        Dp = twin.Dp
        pa = X_fit[:, :Dp]
        pb = X_fit[:, Dp:2 * Dp]
        g = X_fit[:, 2 * Dp + twin.A_dim]
        z = (y_fit / g.reshape(-1, 1)) if y_fit.ndim == 2 else (y_fit / g)

        # 1) φ 直接监督过的基因集合 = 训练单扰动行里出现过的基因
        single = pb.sum(axis=1) == 0
        seen_cols = np.flatnonzero(pa[single].sum(axis=0) > 0)
        seen_mask = np.zeros(Dp, dtype=bool)
        seen_mask[seen_cols] = True
        self.seen_mask_ = seen_mask

        # 2) 基线 b̂ 与所有基因的 one-hot 效应 ê（批量一次算完，≤Dp 行）
        b_hat = twin._phi(np.zeros((1, Dp)))[0][0]        # (K,) 或标量
        phi_all = twin._phi(np.eye(Dp))[0]                # (Dp, K) 或 (Dp,)
        self.b_hat_ = np.asarray(b_hat)
        self.e_all_ = np.asarray(phi_all) - self.b_hat_   # 仅对 seen 基因可信

        K = self.e_all_.shape[1] if self.e_all_.ndim == 2 else 1
        sums = np.zeros((Dp, K)) if self.e_all_.ndim == 2 else np.zeros(Dp)
        sq = np.zeros_like(sums)
        cnts = np.zeros(Dp)

        # 3) 训练双扰动中「恰好一个基因未被单扰动监督过」的行 → 插补该基因
        dbl = np.flatnonzero(pb.sum(axis=1) > 0)
        if len(dbl):
            ca = pa[dbl].argmax(axis=1)
            cb = pb[dbl].argmax(axis=1)
            a_un = ~seen_mask[ca]
            b_un = ~seen_mask[cb]
            one = a_un ^ b_un                              # 恰好一个未见
            if one.any():
                u = np.where(a_un[one], ca[one], cb[one])  # 待插补基因
                o = np.where(a_un[one], cb[one], ca[one])  # 已见搭档
                resid = z[dbl[one]] - self.b_hat_ - self.e_all_[o]
                np.add.at(sums, u, resid)
                np.add.at(sq, u, resid ** 2)
                np.add.at(cnts, u, 1.0)

        self.counts_ = cnts
        has = cnts >= self.min_count
        self.has_solo_ = has
        solo = np.zeros_like(sums)
        solo[has] = sums[has] / cnts[has].reshape(-1, *([1] * (sums.ndim - 1)))
        self.solo_effect_ = solo

        # 插补 epistemic：残差标准误（样本越少/越不一致 → 区间越宽）
        se = np.zeros(Dp)
        if has.any():
            c = cnts[has].reshape(-1, *([1] * (sums.ndim - 1)))
            var = np.maximum(sq[has] / c - (sums[has] / c) ** 2, 0.0)
            var_scalar = var.mean(axis=1) if var.ndim == 2 else var
            se[has] = np.sqrt(var_scalar / np.maximum(cnts[has], 1.0))
        self.impute_se_ = se

        self.n_imputed_ = int(has.sum())
        self.n_unseen_total_ = int((~seen_mask).sum())
        self._fitted = True
        return self

    # ----------------------------------------------------------- predict
    def predict(self, twin, X: np.ndarray):
        """返回 (mu, sigma, nd_relative)，接口与 twin.predict 对齐。"""
        if not self._fitted:
            raise RuntimeError("SoloEffectImputer.predict 前必须先 fit。")
        Dp = twin.Dp
        A_dim = twin.A_dim
        pa = X[:, :Dp]
        pb = X[:, Dp:2 * Dp]
        act = X[:, 2 * Dp:2 * Dp + A_dim]
        g = X[:, 2 * Dp + A_dim:2 * Dp + A_dim + 1].ravel()
        g_col = g.reshape(-1, 1)
        null = np.zeros((len(X), Dp))

        za_m, za_s = twin._phi(pa)
        zb_m, zb_s = twin._phi(pb)
        zn_m, zn_s = twin._phi(null)

        # ---- 核心：把被插补基因的 φ 覆写为 b̂ + ê_u ----
        extra_var = np.zeros(len(X))
        for comp, p, z_m in (("a", pa, za_m), ("b", pb, zb_m)):
            active = p.sum(axis=1) > 0
            idx = p.argmax(axis=1)
            hit = active & self.has_solo_[idx]
            if hit.any():
                repl = self.b_hat_ + self.solo_effect_[idx[hit]]
                z_m[hit] = repl
                extra_var[hit] += self.impute_se_[idx[hit]] ** 2

        z_mean = za_m + zb_m - zn_m
        z_std_epi = np.sqrt(za_s ** 2 + zb_s ** 2 + 2 * zn_s ** 2)

        if z_mean.ndim == 1:
            y_mean = z_mean * g
            y_std_epi = z_std_epi * np.abs(g)
            extra = np.sqrt(extra_var) * np.abs(g)
            F = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        else:
            y_mean = z_mean * g_col
            y_std_epi = z_std_epi * np.abs(g_col)
            extra = (np.sqrt(extra_var).reshape(-1, 1)) * np.abs(g_col)
            F = np.hstack([pa, pb, act, g_col])

        # 插补不确定度按方差相加（硬约束：不确定度必须随证据量变化）
        y_std_epi = np.sqrt(y_std_epi ** 2 + extra ** 2)

        nd = twin._knn_dist(F)
        nd_relative = nd / (twin.med_dist_ + 1e-9)
        # 沿用 P3-1 的 σ 解耦：有界门控 + ID 95 分位封顶（不改变既有语义）
        gate = 1.0 + twin.novelty_k / (1.0 + np.exp(-2.0 * (nd_relative - 1.0)))
        gate_f = gate if y_std_epi.ndim == 1 else gate.reshape(-1, 1)
        cap = 10.0 * getattr(twin, "sigma_id_95_", 1.0)
        sigma_pred = np.minimum(y_std_epi, cap) * gate_f
        return y_mean, sigma_pred, nd_relative

    # ------------------------------------------------------------ report
    def summary(self) -> dict:
        return {
            "n_genes_total": int(len(self.seen_mask_)),
            "n_genes_with_single_supervision": int(self.seen_mask_.sum()),
            "n_genes_without_single_supervision": self.n_unseen_total_,
            "n_genes_imputed_from_codoubles": self.n_imputed_,
            "n_genes_unidentifiable": int(self.n_unseen_total_ - self.n_imputed_),
            "codouble_counts_per_imputed_gene": {
                int(i): int(self.counts_[i]) for i in np.flatnonzero(self.has_solo_)
            },
        }
