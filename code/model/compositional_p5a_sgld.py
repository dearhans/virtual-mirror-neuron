"""code/model/compositional_p5a_sgld.py — P5-A proper-Bayesian 深度集成（SGLD）。

P5-B（sample_weight / NCL）双阴性，根因=**MSE 共识塌缩**：NCL 去相关项 ∝(f_m−f̄) 随成员收敛→0，
无法从塌缩集成自举多样性。P5-A 换一条**不依赖 (f_m−f̄) 自举**的机制：

    SGLD：θ_{t+1} = θ_t − α_t·∇L(θ_t) + N(0, 2·α_t·T·I)

每个成员是一条**独立 SGLD 链**（各自噪声序列）→ 采样**不同 posterior 模式** → 集成在 OOD
（低数据区 posterior 天然宽）正确分散 → epistemic σ 由构造随到训练流形距离单调增长（∂σ/∂τ>0）。
这是 P5 三机制中唯一有原理希望者（内禀 posterior 多样性），且不受共识塌缩约束。

仅依赖 numpy（与项目一致，无 torch）。复用 P5-B-NCL 的 `_NumpyMLP` 框架与 `_phi/predict` 口径，
将 SGD 更新替换为 SGLD 步（梯度 + 高斯噪声 ∝ √(2·α·T)）。
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from model.compositional_p5b_ncl import _NumpyMLP, _relu_d
from model.compositional import CompositionalTwin


class _SGLD_MLP(_NumpyMLP):
    """在 `_NumpyMLP` 反向传播基础上，把 SGD 更新替换为 SGLD 步（梯度 + 高斯噪声）。"""

    def sgld_step(self, dz, lr, temperature, rng, wdecay=0.0, clip_grad=None):
        # 梯度裁剪（防高 T 下瞬态大梯度触发 overflow）
        if clip_grad is not None:
            dz = np.clip(dz, -clip_grad, clip_grad)
        # 同 backward 算 MSE 梯度（∂L/∂W），但不立即更新；wdecay 提供恢复力（约束随机游走）
        dW3 = self.h2.T @ dz / len(self.X) + wdecay * self.W3
        db3 = dz.mean(0)
        dh2 = (dz @ self.W3.T) * _relu_d(self.h2)
        dW2 = self.h1.T @ dh2 / len(self.X) + wdecay * self.W2
        db2 = dh2.mean(0)
        dh1 = (dh2 @ self.W2.T) * _relu_d(self.h1)
        dW1 = self.X.T @ dh1 / len(self.X) + wdecay * self.W1
        db1 = dh1.mean(0)
        # SGLD 噪声：N(0, 2·lr·T) 每参数（minibatch 归一下噪声相对梯度放大 → 更易产生多样性）
        noise_scale = np.sqrt(2.0 * lr * max(temperature, 1e-12))
        self.W3 += -lr * dW3 + noise_scale * rng.standard_normal(self.W3.shape)
        self.b3 += -lr * db3 + noise_scale * rng.standard_normal(self.b3.shape)
        self.W2 += -lr * dW2 + noise_scale * rng.standard_normal(self.W2.shape)
        self.b2 += -lr * db2 + noise_scale * rng.standard_normal(self.b2.shape)
        self.W1 += -lr * dW1 + noise_scale * rng.standard_normal(self.W1.shape)
        self.b1 += -lr * db1 + noise_scale * rng.standard_normal(self.b1.shape)


class CompositionalTwinP5A_SGLD(CompositionalTwin):
    """P5-A SGLD 孪生（继承 CompositionalTwin 的 φ/因果头/课程学习口径；predict 返回原始集成 epistemic）。"""

    def __init__(self, sgld_temperature: float = 0.1, sgld_lr: float = 0.01,
                 sgld_epochs: int = 120, sgld_batch: int = 256, sgld_decay: int = 50,
                 sgld_wdecay: float = 0.0, clip_grad: float = None,
                 random_state: int = 0, **kw):
        super().__init__(random_state=random_state, **kw)
        self.sgld_temperature = sgld_temperature
        self.sgld_lr = sgld_lr
        self.sgld_epochs = sgld_epochs
        self.sgld_batch = sgld_batch
        self.sgld_decay = sgld_decay
        self.sgld_wdecay = sgld_wdecay
        self.clip_grad = clip_grad

    # --------------------------------------------------------------- 训练（SGLD 联合）
    def fit(self, X: np.ndarray, y: np.ndarray, z_comp: np.ndarray = None):
        pa, pb, act, g = self._slice(X)
        z = (y / g.reshape(-1, 1)) if y.ndim == 2 else (y / g)
        fa, fb = pa, pb
        null = np.zeros((len(X), self.Dp))

        # —— 复用父类监督目标（与 P5-B-NCL 同口径）——
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

        # —— 训练特征（供 predict 的 kNN 距离/相对距离）——
        self.train_feats_ = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_.fit(self.train_feats_)
        self.med_dist_ = float(np.median(self._knn_dist(self.train_feats_)))

        Xtr, Ytr = single_X, single_y
        din, dout = Xtr.shape[1], Ytr.shape[1]
        rng = np.random.default_rng(self.random_state)
        members = [
            _SGLD_MLP(din, self.hidden, dout,
                      rng=np.random.default_rng(int(rng.integers(1, 1_000_000))))
            for _ in range(self.n_ensemble)
        ]

        # —— SGLD：每成员独立链（各自噪声），lr 衰减；无密度权重（P5-A 多样性内禀）——
        n, B = len(Xtr), self.sgld_batch
        idx = np.arange(n)
        for ep in range(self.sgld_epochs):
            rng.shuffle(idx)
            lr_t = self.sgld_lr / (1.0 + ep / max(self.sgld_decay, 1)) ** 0.5
            for s in range(0, n, B):
                bi = idx[s:s + B]
                Xb, Yb = Xtr[bi], Ytr[bi]
                for m in members:
                    Fm = m.forward(Xb)
                    m.sgld_step(2.0 * (Fm - Yb) / len(Xb), lr_t, self.sgld_temperature, rng,
                                wdecay=self.sgld_wdecay, clip_grad=self.clip_grad)

        self.members_ = members
        self.sigma_id_95_ = float(self._sigma_epi_quantile(X[: min(len(X), 2000)]))
        self._fitted = True
        return self

    # --------------------------------------------------------------- φ / predict（同 P5-B-NCL 口径）
    def _phi(self, feats: np.ndarray):
        preds = np.stack([m.forward(feats) for m in self.members_], axis=0)  # [M,B,dout]
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("CompositionalTwinP5A_SGLD.predict 前必须先 fit。")
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
        return y_mean, y_std_epi, nd_relative
