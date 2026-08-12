"""model/compositional_p44.py — P4-4：φ = 线性头 ⊕ 机制残差头（空间分离 PC-1）。

背景（W33 §6 PC-1：项目核心物理矛盾）
-----------------------------------------------
同一 φ 既要记住 55464 条训练行的分布，又要对 held-out 基因/基因对外推。
P1 把全部信号压给单一加法基 MLP → 该 MLP 在训练基因上过拟合，加法组合到
held-out 对(ood_action)时不外推；而线性基线(纯加法)反而更好（W33 §1：线性 0.6266
优于孪生 0.6319/0.6355）。

破法（TRIZ 空间分离：把"记分布"与"外推"拆到两条通路）
-----------------------------------------------
把 φ 拆成：
    φ(p) = W_lin · p + b0   ⊕   r(p)          （r = 残差头 MLP 集成）
  - 线性头 W_lin·p + b0：在单扰动行上拟合（与 P1 同监督口径），但**架构是线性**，
    故加法组合 z = φ(pa)+φ(pb)−φ(null) 在 held-out 对上天然外推
    （线性件 = 远端外推件，外部判据 A「架构谦逊」）；
  - 残差头 r(p)：MLP 集成，仅在单扰动行上拟合 z − (W_lin·p + b0)，
    只补线性解释不了的非线性机制（单基因非线性效应）；
    held-out 基因上 r≈0 → 外推交给线性头（oved-out 基因的共扰动接管逻辑属 P4-2，本模块不混入）。

与 P4-2 / P4-3 的关系（记账纪律：分两次跑、分别记账）
-----------------------------------------------
- P4-2（共扰动插补）解 ood_agent 坍缩（PC-2）；P4-3（收缩）是其消融（已判阴性 P43_ACCEPTED=false）；
- P4-4 是**独立的结构分离**，只改 φ 架构、不引入插补/收缩；本类的 supervision 与 P1
  完全相同（仅单扰动行），故 P4-4 的增量可干净归因于「线性头 ⊕ 残差头」本身。
- 判据（W33 §7 #3，事前锁定）：组合孪生在 ood_action 上**首次与线性 CI 重叠或更优**。

仅依赖 numpy / scikit-learn（与既有 model/ 一致）。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors

from .compositional import CompositionalTwin


class DualPathCompositionalTwin(CompositionalTwin):
    """P4-4：双通路 φ（线性头 + 机制残差头）。

    继承 CompositionalTwin 的 predict / 因果头 / 新颖度 / P3-1 门控（仅 _phi 被替换）。
    线性头在单扰动行上以 LinearRegression 拟合；残差头在单扰动行上以 MLP 集成拟合
    z − 线性头(p)，只学线性解释不了的非线性。
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
        super().__init__(
            Dp=Dp, A_dim=A_dim, hidden=hidden, n_ensemble=n_ensemble,
            novelty_k=novelty_k, k_neighbors=k_neighbors,
            random_state=random_state, curriculum=curriculum,
        )

    # --------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray, z_comp: np.ndarray = None):
        rng = np.random.RandomState(self.random_state)
        pa, pb, act, g = self._slice(X)
        # 因果头：增益归一化（与 P1 同），z 空间做线性/残差分解
        z = (y / g.reshape(-1, 1)) if y.ndim == 2 else (y / g)

        # ---- 线性头：单扰动行（与 P1 同监督口径），架构线性 → 加法组合外推 ----
        single_mask = (pb.sum(axis=1) == 0)
        fa_s = pa[single_mask]
        z_s = z[single_mask]
        self.linear_ = LinearRegression(fit_intercept=True)
        self.linear_.fit(fa_s, z_s)
        # 截距 b0 ≈ 单扰动 z 的均值 = 基线；φ(null)=b0+残差(null)≈基线

        # ---- 残差头：单扰动行，目标 = z − 线性头(p) ----
        null_s = np.zeros((int(single_mask.sum()), self.Dp))
        baseline = z_s.mean(axis=0)
        null_y = np.broadcast_to(baseline, z_s.shape).copy()
        lin_pred_s = np.asarray(self.linear_.predict(fa_s))
        res_y_single = z_s - lin_pred_s
        lin_pred_null = np.asarray(self.linear_.predict(null_s))
        res_y_null = null_y - lin_pred_null          # ≈ 0（基线已被截距吸收）
        single_X = np.vstack([fa_s, null_s])
        comp_y = np.concatenate([res_y_single, res_y_null], axis=0)

        self.res_ensembles_ = []
        for _ in range(self.n_ensemble):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden, max_iter=800, tol=1e-4,
                warm_start=self.curriculum,
                random_state=int(rng.randint(1, 1_000_000)),
                early_stopping=False,
            )
            # 残差头只在单扰动行学非线性机制（与 P1 的 MLP 同分布监督）
            mlp.fit(single_X, comp_y)
            self.res_ensembles_.append(mlp)

        # ---- 新颖度 + P3-1 封顶（沿用父类机制）----
        self.train_feats_ = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_.fit(self.train_feats_)
        self.med_dist_ = float(np.median(self._knn_dist(self.train_feats_)))
        self.sigma_id_95_ = float(self._sigma_epi_quantile(X[: min(len(X), 2000)]))
        self._fitted = True
        return self

    # --------------------------------------------------------------- φ
    def _phi(self, feats: np.ndarray) -> tuple:
        """双通路 φ：线性头（确定性）+ 残差头集成（epistemic 来自残差集成方差）。

        返回 (mean, std)。线性头贡献点估计但不贡献 epistemic（固定线性映射）；
        held-out 基因上残差≈0 → 外推交给线性头。
        """
        lin = np.asarray(self.linear_.predict(feats))
        preds = np.stack([m.predict(feats) for m in self.res_ensembles_], axis=0)
        res_mean = preds.mean(axis=0)
        res_std = preds.std(axis=0)
        mean = lin + res_mean
        return mean, res_std
