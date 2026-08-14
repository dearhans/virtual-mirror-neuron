"""model/compositional_p4a.py — 路 A：novelty-gated epistemic std（攻克 PC-3 epistemic 欠缩放根因）。

背景（W33 §6 PC-3 / §7 路 A · P0）
-------------------------------
P4-1（localized conformal）已证伪「校准层重标定」方向：ID 校准集新颖度不覆盖 OOD，
分 subset q 退化回全局 41.26，ood_action 仍饱和（cov@0.95=0.999, ECE=0.206）。
根因是组合孪生 epistemic std **全子集一致欠缩放 ~41×**——深度集成在 OOD 上 confidently-wrong。
PC-3 因此重定位为 **epistemic 估计器缺陷**，唯一未排除的解法 = 路 A：让 std 自身随新颖度撑大。

P4-1 与路 A 的本质区别（避免重蹈覆辙）
---------------------------------------
P4-1 失败在「用 ID 校准行估 OOD 的 q」——ID 校准新颖度范围覆盖不了 OOD，
argmin 把全部校准行归到 nd 最低的 id 箱 → 四子集 q 全相等（结构性退化）。
路 A **不碰校准层 q**，而是**直接改 std 估计器本体**：在 predict 时按测试点「自身」
的新颖度 nd_relative（kNN 距离）撑大 epistemic std。nd_relative 在 ID 与 OOD 测试点都能
即时算到（不需要 ID 校准去「覆盖」OOD）——这正是二者机制层面的分水岭。

破法（TRIZ 条件分离：把「ID 内诚实 std」与「OOD 放大 std」按新颖度切分）
------------------------------------------------------------
1. 在 20% ID 校准集上估「欠缩放比 r(nd) = median|resid|/σ_epi 作为 nd 的函数」——
   即深度集成在每个新颖度层级上把不确定度低估了多少倍。
2. 定义门控倍率 g(nd) = r(nd) / r(nd_ref)，参考新颖度 nd_ref 取校准中位数（使 ID 主体 g≈1）。
3. predict 时 σ_pred = σ_epi · g(nd)：由「欠缩放比」直接补回缺失的 epistemic 量。
   → |resid|/σ_pred 在 ID 与 OOD 近似同分布，单标量 conformal q（在校准上用 gated std
     估计）在 OOD 也成立。这从根因上修 std，而非重标定 q。
4. OOD（nd 超出校准范围）按 g 外推，cap 在 gate_cap_ratio 防饱和（守住 PC-3 宽↔窄矛盾）。

仅依赖 numpy / scikit-learn（与既有 model/ 一致）。
"""
from __future__ import annotations

import numpy as np

from .compositional import CompositionalTwin


class NoveltyGatedCompositionalTwin(CompositionalTwin):
    """路 A：novelty-gated epistemic std（替换 P3-1 有界 sigmoid 门控 + 封顶）。

    继承 CompositionalTwin 的 _phi / 因果头 / 新颖度 kNN；仅 override predict 的
    σ 合成方式（用校准拟合的「欠缩放比→新颖度」门控倍率，而非固定的 [1,1+k] 有界门）。
    fit 与父类完全相同（同监督口径 → 点估计增量可干净归因到 σ 合成改造）。
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
        gate_kind: str = "empirical",
        gate_n_bins: int = 8,
        gate_cap_ratio: float = 50.0,
    ):
        super().__init__(
            Dp=Dp, A_dim=A_dim, hidden=hidden, n_ensemble=n_ensemble,
            novelty_k=novelty_k, k_neighbors=k_neighbors,
            random_state=random_state, curriculum=curriculum,
        )
        self.gate_kind = gate_kind
        self.gate_n_bins = int(gate_n_bins)
        self.gate_cap_ratio = float(gate_cap_ratio)
        self._gate_fitted = False

    # --------------------------------------------------------------- 新颖度（排除自匹配）
    def _knn_dist(self, F: np.ndarray) -> np.ndarray:
        """kNN 距离作为新颖度，**排除自匹配**。

        父类的 _knn_dist 直接取 k 近邻均值，但查询点若位于索引中（校准/训练点），
        第 0 近邻即自身（距离 0）→ med_dist 被压到 ~0 → nd_relative 在「近训练点」塌为 0、
        在「OOD」爆为 1e9，门控退化为二值开关。路 A 依赖平滑新颖度梯度，故此处取
        k_neighbors+1 近邻并丢弃第 0 列（自匹配），以「到其余训练点」的均值度量新颖度。
        """
        kk = self.k_neighbors + 1
        dists, _ = self.nn_.kneighbors(F, n_neighbors=kk)
        return dists[:, 1:].mean(axis=1)

    # --------------------------------------------------------------- 原始 epistemic
    def _raw_epistemic(self, X: np.ndarray):
        """无门控/无封顶的原始集成 epistemic（供门控拟合与 predict 复用）。

        与父类 predict 的 σ 合成前状态一致：σ_epi = sqrt(za_s²+zb_s²+2·zn_s²)·|g|，
        不施加 P3-1 的有界 sigmoid 门控与 10× ID 封顶。
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

    # --------------------------------------------------------------- 门控拟合
    def fit_novelty_gate(self, X_cal: np.ndarray, y_cal: np.ndarray, kind: str = None):
        """在 ID 校准集上估欠缩放比 → 门控倍率。

        kind="empirical"（novelty）：r(nd)=median|resid|/σ_epi 作为 kNN 新颖度 nd 的函数，
            单调非减，参考点取校准中位数新颖度。→ 测试时按自身新颖度撑大 std。
        kind="heteroscedastic"（σ_epi）：欠缩放比 U=|resid|/σ_epi 作为模型自身 epistemic
            std（σ_epi）的函数，log U ~ β·log(σ_ref/σ_epi) → 门控 g=(σ_ref/σ_epi)^β。
            诊断显示 U 与 σ_epi 呈负 Spearman 相关（-0.4），即「越自信越错」——
            此分支直接利用模型自身 epistemic 作为门控回归量（异方差重标定）。
        """
        kind = kind or self.gate_kind
        mu_c, std_c, nd_c = self._raw_epistemic(X_cal)
        score = np.abs(y_cal - mu_c) / np.clip(std_c, 1e-9, None)
        score1d = np.median(score, axis=1) if score.ndim == 2 else score
        sig_epi = np.median(std_c, axis=1) if std_c.ndim == 2 else std_c

        self._nd_ref = float(np.median(nd_c))

        if kind == "heteroscedastic":
            sel = sig_epi > 1e-9
            S = np.log(np.clip(sig_epi[sel], 1e-9, None))
            U = np.log(np.clip(score1d[sel], 1e-9, None))
            beta = float(np.polyfit(S, U, 1)[0])  # logU 对 logS 的斜率
            self._het_beta = beta
            self._het_sref = float(np.median(sig_epi[sel]))
            self._gate_kind = "heteroscedastic"
            self._gate_fitted = True
            return self

        # --- empirical（novelty）分支 ---
        nd_sorted = np.sort(nd_c)
        edges = np.quantile(nd_c, np.linspace(0, 1, self.gate_n_bins + 1))
        edges[0] = float(nd_sorted[0]) - 1e-9
        edges[-1] = float(nd_sorted[-1]) + 1e-9

        centers, ratios = [], []
        for b in range(self.gate_n_bins):
            msk = (nd_c >= edges[b]) & (nd_c < edges[b + 1])
            if int(msk.sum()) >= 5:
                centers.append(float(np.median(nd_c[msk])))
                ratios.append(float(np.median(score1d[msk])))
        centers = np.array(centers)
        ratios = np.array(ratios)

        # 按新颖度排序并保证单调非减（消除小箱噪声，符合「越 OOD 越欠缩放」机制先验）
        order = np.argsort(centers)
        centers = centers[order]
        ratios = np.maximum.accumulate(ratios[order])

        r_ref = float(np.interp(self._nd_ref, centers, ratios))
        self._gate_nd = centers
        self._gate_ratio = ratios / max(r_ref, 1e-9)  # 相对参考点的倍率（参考点=1）
        self._gate_r_ref = r_ref
        self._gate_kind = "empirical"
        self._gate_fitted = True
        return self

    def _novelty_multiplier(self, nd_rel: np.ndarray) -> np.ndarray:
        """测试点新颖度 → std 撑大倍率（empirical 分支）。

        nd 低于校准范围 → 取最低箱倍率（≈1）；nd 超出校准范围（OOD）→ 按 cap 外推防饱和。
        """
        if not self._gate_fitted or self._gate_kind != "empirical":
            return np.ones_like(nd_rel)
        mult = np.interp(
            nd_rel, self._gate_nd, self._gate_ratio,
            left=float(self._gate_ratio[0]), right=self.gate_cap_ratio,
        )
        return np.clip(mult, float(self._gate_ratio[0]), self.gate_cap_ratio)

    def _het_multiplier(self, sig_epi: np.ndarray) -> np.ndarray:
        """模型自身 epistemic std → 门控倍率（heteroscedastic 分支）。

        g=(σ_ref/σ_epi)^β：β>0 时「越自信（σ_epi 小）越撑大」，
        抵消 U∝σ_epi^{-β} 的欠缩放，使 |resid|/(σ·g) 在 ID 近似均匀。
        cap 防饱和（PC-3 宽↔窄矛盾）。
        """
        if not self._gate_fitted or self._gate_kind != "heteroscedastic":
            return np.ones_like(sig_epi)
        g = (self._het_sref / np.clip(sig_epi, 1e-9, None)) ** self._het_beta
        return np.clip(g, 1.0 / self.gate_cap_ratio, self.gate_cap_ratio)

    # --------------------------------------------------------------- predict
    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("NoveltyGatedCompositionalTwin.predict 前必须先 fit。")
        y_mean, y_std_epi, nd_rel = self._raw_epistemic(X)
        if self._gate_kind == "heteroscedastic":
            sig_epi_s = np.median(y_std_epi, axis=1) if y_std_epi.ndim == 2 else y_std_epi
            mult = self._het_multiplier(sig_epi_s)
        else:
            mult = self._novelty_multiplier(nd_rel)
        mult_f = mult if y_std_epi.ndim == 1 else mult.reshape(-1, 1)
        sigma_pred = y_std_epi * mult_f
        return y_mean, sigma_pred, nd_rel
