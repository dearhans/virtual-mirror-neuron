"""model/twin.py — 多尺度虚拟镜像神经元孪生（替换 MLP 占位）。

设计（呼应七层框架的「建模层」与「因果层」，以及项目硬约束「别迷信黑箱」）：
多尺度结构显式拆解，而非端到端黑箱：
  - 尺度 A · 离子通道/细胞 (cellular)：非线性机制 phi(p)，由集成 MLP 学到。
  - 尺度 B · 突触/动作 (synaptic)：动作对称性 —— self/other 共享同一 phi，
        仅增益不同（mirror_k）。权重在 self/other 间「绑定」→ 镜像性涌现。
  - 尺度 C · 环路/神经调质 (circuit)：神经调质增益 g 作为乘法因子作用于响应。
因果头：训练时把目标归一化为 z = y / g，使「乘法调质机制」成为归纳偏置；
        预测时再乘回 g → 对未见/新调质具备外推能力（而非记忆）。
不确定度：集成(epistemic) + 新颖度(novelty, OOD 检测) 合成 →
        使 OOD 不再过度自信（修正此前同方差不确定度在 OOD 上覆盖度塌陷的问题）。
仅依赖 numpy / scikit-learn。
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors


class VirtualTwin:
    """多尺度虚拟孪生：fit(X, y) 后 predict(X) -> (mean, std)。

    X 列布局与 data/loaders.py 对齐：[p(0..Dp-1), act(Dp..Dp+A_dim-1), g(Dp+A_dim)]。
    支持标量或向量响应 y（向量时预测基因表达谱，不确定度按每样本聚合为标量）。
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
    ):
        self.Dp = Dp
        self.A_dim = A_dim
        self.hidden = hidden
        self.n_ensemble = n_ensemble
        self.novelty_k = novelty_k
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._fitted = False

    def _slice(self, X):
        p = X[:, : self.Dp]
        act = X[:, self.Dp : self.Dp + self.A_dim]
        g = X[:, self.Dp + self.A_dim : self.Dp + self.A_dim + 1].ravel()
        return p, act, g

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        p, act, g = self._slice(X)

        # 因果头：增益归一化，把乘法调质机制变成可学先验
        # 向量响应 y(n, K) 时，g(n,) 须按列广播 -> g(n,1)
        g_col = g.reshape(-1, 1) if y.ndim == 2 else g
        z = y / g_col

        feats = np.hstack([p, act])  # 机制输入（不含 agent：agent 是混杂因子）
        self.ensembles_ = []
        for _ in range(self.n_ensemble):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden,
                max_iter=800,
                tol=1e-4,
                random_state=int(rng.randint(1, 1_000_000)),
                early_stopping=False,
            )
            mlp.fit(feats, z)
            self.ensembles_.append(mlp)

        # 新颖度模型：在 (p, act, g) 全特征空间上做 kNN
        self.train_feats_ = np.hstack([p, act, g.reshape(-1, 1)])
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_.fit(self.train_feats_)
        self.med_dist_ = float(np.median(self._knn_dist(self.train_feats_)))

        self._fitted = True
        return self

    def _knn_dist(self, F: np.ndarray) -> np.ndarray:
        dists, _ = self.nn_.kneighbors(F)
        return dists.mean(axis=1)

    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("VirtualTwin.predict 前必须先 fit。")
        p, act, g = self._slice(X)
        g_col = g.reshape(-1, 1)
        F = np.hstack([p, act, g_col])

        feats = np.hstack([p, act])
        z_preds = np.stack([m.predict(feats) for m in self.ensembles_], axis=0)
        z_mean = z_preds.mean(axis=0)
        z_std = z_preds.std(axis=0)

        y_mean = z_mean * g_col
        # 集成的 epistemic 不确定度随 g 缩放
        y_std_epi = z_std * np.abs(g_col)

        # P3-1 σ 解耦：区间宽度只含集成 epistemic，novelty 独立返回（决策阶段使用）
        nd = self._knn_dist(F)
        nd_relative = nd / (self.med_dist_ + 1e-9)
        if y_mean.ndim == 1:
            y_std = y_std_epi
        else:
            y_std = np.sqrt((y_std_epi ** 2).mean(axis=1))
        return y_mean, y_std, nd_relative

    # 便捷接口：与 baseline 签名一致（仅返回 mean）
    def __call__(self, X: np.ndarray) -> np.ndarray:
        mean, _, _ = self.predict(X)
        return mean


class PredictiveUnit:
    """单神经元预测编码单元（概念模块）。

    预测编码视角：神经元持续预测下一时刻输入，仅对「预测误差」敏感。
    这里是轻量实现，用于环路级建模的 building block；本基准由 VirtualTwin 统一调用。
    """

    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self.prediction = 0.0

    def step(self, observation: float) -> float:
        error = observation - self.prediction
        self.prediction += self.lr * error
        return error


class MLPTwin:
    """朴素 MLP 占位（端到端黑箱基线，用于对照「带生物学先验」的 VirtualTwin）。"""

    def __init__(self, Dp: int = 8, hidden: tuple = (64, 64), random_state: int = 0):
        self.Dp = Dp
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden, max_iter=800, tol=1e-4,
            random_state=random_state, early_stopping=False,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        resid = y - self.model.predict(X)
        self.resid_std_ = float(np.std(resid)) + 1e-9
        self._fitted = True
        return self

    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("MLPTwin.predict 前必须先 fit。")
        mean = self.model.predict(X)
        # 同方差不确定度（已知会在 OOD 上过度自信，作为对照）
        std = np.full_like(mean, self.resid_std_)
        return mean, std

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
