"""code/goai_compositional_twin.py — CompositionalTwin 赛题版（残差分解架构）

架构（对标教程 §3.2「残差分解」+ 本项目实证的 φ+ψ 交互头）：
  ŷ(s,c,ctx) = μ_global + μ_strain(s) + μ_compound(c) + ψ(s,c,ctx)

  其中：
    μ_global  : 全局蛋白均值 Δ（每蛋白，train 集估计，NaN-safe）
    μ_strain  : 菌株效应 — 已知菌株的每蛋白均值 Δ 超全局（未见菌株→0）
    μ_compound: 化合物效应 — 已知化合物的每蛋白均值 Δ 超全局（未见化合物→0）
    ψ         : 交互余量 = Δ − (μ_global + μ_strain + μ_compound)，
                 由 Ridge/MLP 从交互特征（strain×compound×context）预测。

  OOD 泛化：
    - 已知菌株+已知化合物：ψ 在残差上捕获 epistasis/交联，分离加和→提高可解释性。
    - 未见菌株/化合物：加和项归零 → 纯依靠 ψ 推断交互余量。
      若交互特征含可泛化信号（菌株基因组嵌入 + 化合物 SMILES 描述符），
      ψ 可推广至 unseen → 得分 OOD 三轴。
      目前 MVP 使用 one-hot（未见=全零）→ 交互头退化为全局偏差预测。

评估：对整体 Δ 预测（μ_global+μ_strain+μ_compound+ψ）用 goai_metrics.evaluate()。
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge


class GoaiCompositionalTwin:
    def __init__(self, interaction_model="ridge", alpha=1.0, random_state=0):
        self.interaction_model = interaction_model
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, Y_delta, strain_idx, comp_idx, X_ctxt, train_mask):
        """
        Y_delta   : (N, Kp) Δ 矩阵（含 NaN → 均值估计时 NaN-safe）
        strain_idx: (N,) 菌株 integer index
        comp_idx  : (N,) 化合物 integer index
        X_ctxt    : (N, Dctx) 上下文特征（已编码为 one-hot 或数值）
        train_mask: (N,) bool, True=train
        """
        N, Kp = Y_delta.shape
        Y_tr = Y_delta[train_mask]
        s_tr = strain_idx[train_mask]
        c_tr = comp_idx[train_mask]

        # 1. 全局均值（每蛋白，NaN-safe）
        self.mu_global = np.zeros(Kp, dtype=np.float64)
        for j in range(Kp):
            col = Y_tr[:, j]
            f = np.isfinite(col)
            self.mu_global[j] = col[f].mean() if f.sum() > 0 else 0.0

        # 2. 菌株效应（每蛋白，每个菌株，NaN-safe）
        n_strains = strain_idx.max() + 1
        self.mu_strain = np.zeros((n_strains, Kp), dtype=np.float64)
        for s in range(n_strains):
            ms = (s_tr == s)
            if ms.sum() < 2:
                continue
            Ys = Y_tr[ms]
            for j in range(Kp):
                col = Ys[:, j]
                f = np.isfinite(col)
                if f.sum() > 0:
                    self.mu_strain[s, j] = col[f].mean() - self.mu_global[j]

        # 3. 化合物效应
        n_comps = comp_idx.max() + 1
        self.mu_compound = np.zeros((n_comps, Kp), dtype=np.float64)
        for c in range(n_comps):
            mc = (c_tr == c)
            if mc.sum() < 2:
                continue
            Yc = Y_tr[mc]
            for j in range(Kp):
                col = Yc[:, j]
                f = np.isfinite(col)
                if f.sum() > 0:
                    self.mu_compound[c, j] = col[f].mean() - self.mu_global[j]

        # 4. 交互余量 = Δ - 加和项（全部样本）
        additive = (self.mu_global[None, :]
                     + self.mu_strain[strain_idx, :]
                     + self.mu_compound[comp_idx, :])
        Y_delta_filled = np.where(np.isfinite(Y_delta), Y_delta, additive)
        residual = Y_delta_filled - additive      # (N, Kp)

        # 5. 交互特征：strain one-hot + compound one-hot + 上下文 + cross
        S_onehot = np.eye(n_strains)[strain_idx]
        C_onehot = np.eye(n_comps)[comp_idx]
        # cross terms: strain×compound element-wise products (sparse)
        # 只对已知 pairs 有信号；未知 pair (zero row) → cross=0
        cross = np.zeros((N, n_strains * n_comps), dtype=np.float32)
        for i in range(N):
            cross[i, strain_idx[i] * n_comps + comp_idx[i]] = 1.0

        X_int = np.hstack([S_onehot.astype(np.float32),
                           C_onehot.astype(np.float32),
                           X_ctxt.astype(np.float32),
                           cross]).astype(np.float64)

        # 6. 学习 ψ：Ridge on interaction residual
        self.psi_model = Ridge(alpha=self.alpha, fit_intercept=True,
                               random_state=self.random_state)
        self.psi_model.fit(X_int[train_mask], residual[train_mask])
        return self

    def predict_delta(self, strain_idx, comp_idx, X_ctxt):
        """返回完整 Δ 预测（加和项 + ψ）。"""
        additive = (self.mu_global[None, :]
                     + self.mu_strain[strain_idx, :]
                     + self.mu_compound[comp_idx, :])

        n_strains = self.mu_strain.shape[0]
        n_comps = self.mu_compound.shape[0]
        S_onehot = np.eye(n_strains)[strain_idx]
        C_onehot = np.eye(n_comps)[comp_idx]
        cross = np.zeros((len(strain_idx), n_strains * n_comps), dtype=np.float32)
        for i in range(len(strain_idx)):
            cross[i, strain_idx[i] * n_comps + comp_idx[i]] = 1.0
        X_int = np.hstack([S_onehot.astype(np.float32),
                           C_onehot.astype(np.float32),
                           X_ctxt.astype(np.float32),
                           cross]).astype(np.float64)

        psi = self.psi_model.predict(X_int)
        return additive + psi
