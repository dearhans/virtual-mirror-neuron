"""code/model/compositional_p5b_ncl.py — P5-B·proper NCL（负相关性学习）实现。

P5-B 的 sklearn `sample_weight` 代理无法注入成员负相关（见 `experiments/20260814-P5B-closure.md`）：
重加权只能扰动「哪些点被拟合得更好」，表达不了 decorrelation 损失。本文件用 **numpy-only custom
训练循环** 实现真正的 negative correlation learning（Liu & Yao 1999 / Brown 2005）：

    L_m = MSE(f_m, y) + λ · mean_i[ w_i · (f_m(x_i) − f̄(x_i)) · (f̄(x_i) − y_i) ]
    f̄ = mean_m f_m ；  w_i = 局部训练支撑稀疏度（连续 kNN，高=稀疏/OOD-like）

第二项在**稀疏区强制成员负相关** → 集成在 OOD 天然分散 → ∂σ/∂τ>0 由构造满足（TRIZ #1+#15）。
稀疏度 w 连续（非路 A 二值 novelty），且损失层面直接耦合成员 → 克服 P5-B 的「主体主导收敛」瓶颈。

仅依赖 numpy（与项目一致，无 torch 新依赖）。继承 CompositionalTwin 的 φ 结构/因果头/predict 口径。
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from model.compositional import CompositionalTwin


def _relu(x):
    return np.maximum(0.0, x)


def _relu_d(x):
    return (x > 0.0).astype(np.float64)


class _NumpyMLP:
    """2 隐藏层 MLP（numpy-only），前向 + 手动反向传播。"""

    def __init__(self, din, hidden=(64, 32), dout=256, rng=None, scale=1e-2):
        rng = rng or np.random.default_rng(0)
        h1, h2 = hidden
        self.W1 = rng.standard_normal((din, h1)) * scale
        self.b1 = np.zeros(h1)
        self.W2 = rng.standard_normal((h1, h2)) * scale
        self.b2 = np.zeros(h2)
        self.W3 = rng.standard_normal((h2, dout)) * scale
        self.b3 = np.zeros(dout)

    def forward(self, X):
        self.X = X
        self.h1 = _relu(X @ self.W1 + self.b1)
        self.h2 = _relu(self.h1 @ self.W2 + self.b2)
        self.z = self.h2 @ self.W3 + self.b3
        return self.z

    def backward(self, dz, lr):
        # dz: [B, dout] = ∂L/∂z
        dW3 = self.h2.T @ dz / len(self.X)
        db3 = dz.mean(0)
        dh2 = (dz @ self.W3.T) * _relu_d(self.h2)
        dW2 = self.h1.T @ dh2 / len(self.X)
        db2 = dh2.mean(0)
        dh1 = (dh2 @ self.W2.T) * _relu_d(self.h1)
        dW1 = self.X.T @ dh1 / len(self.X)
        db1 = dh1.mean(0)
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


class CompositionalTwinP5B_NCL(CompositionalTwin):
    """P5-B proper-NCL 孪生（继承 CompositionalTwin 的 φ/因果头/课程学习口径）。"""

    def __init__(self, ncl_lambda: float = 0.3, ncl_lr: float = 0.01,
                 ncl_epochs: int = 120, ncl_batch: int = 512, random_state: int = 0, **kw):
        super().__init__(random_state=random_state, **kw)
        self.ncl_lambda = ncl_lambda
        self.ncl_lr = ncl_lr
        self.ncl_epochs = ncl_epochs
        self.ncl_batch = ncl_batch

    # --------------------------------------------------------------- 稀疏度权重
    def _sparsity_weight(self, F: np.ndarray) -> np.ndarray:
        d = self._knn_dist(F)
        lo, hi = float(d.min()), float(d.max())
        return (d - lo) / max(hi - lo, 1e-9)

    # --------------------------------------------------------------- 训练（NCL 联合）
    def fit(self, X: np.ndarray, y: np.ndarray, z_comp: np.ndarray = None):
        pa, pb, act, g = self._slice(X)
        z = (y / g.reshape(-1, 1)) if y.ndim == 2 else (y / g)
        fa, fb = pa, pb
        null = np.zeros((len(X), self.Dp))

        # —— 复用父类监督目标（真实数据：单扰动行监督 φ；双扰动由结构加法外推）——
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

        # —— kNN 密度 → 连续稀疏度 w（按原始训练行索引对齐）——
        self.train_feats_ = np.hstack([pa, pb, act, g.reshape(-1, 1)])
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_.fit(self.train_feats_)
        self.med_dist_ = float(np.median(self._knn_dist(self.train_feats_)))
        w_all = self._sparsity_weight(self.train_feats_)

        Xtr, Ytr = single_X, single_y
        if z_comp is None:
            sm = np.flatnonzero(single_mask)
            w_tr = np.concatenate([w_all[sm], w_all[sm]])  # fa_s 行 + null_s 行（同索引）
        else:
            w_tr = np.concatenate([w_all, w_all])

        din, dout = Xtr.shape[1], Ytr.shape[1]
        rng = np.random.default_rng(self.random_state)
        members = [
            _NumpyMLP(din, self.hidden, dout,
                      rng=np.random.default_rng(int(rng.integers(1, 1_000_000))))
            for _ in range(self.n_ensemble)
        ]

        # —— NCL 联合训练：每 mini-batch 前向所有成员，按 NCL 梯度更新各自参数 ——
        # 关键修复（W36 首次 NCL 跑批 λ 零效应 bug）：正确 NCL 交叉项梯度必须含**成员特异**的
        #   (f_m − f̄) 因子，否则把所有成员推向同一方向、多样性与 λ 无关（等价于 P5-B sample_weight 的破代理）。
        #   P_m = λ·mean_i[ w_i·(f_m − f̄)·(f̄ − y) ]
        #   ∂P_m/∂f_m^(i) = λ·w_i·[ (1−1/M)·(f̄^(i) − y^(i)) + (1/M)·(f_m^(i) − f̄^(i)) ]
        # 第二项 (f_m − f̄) 使每个成员的梯度互异 → 强制负相关 → σ 在 OOD 正确转移（TRIZ #1/#15）。
        n, B = len(Xtr), self.ncl_batch
        M = len(members)
        idx = np.arange(n)
        for ep in range(self.ncl_epochs):
            rng.shuffle(idx)
            for s in range(0, n, B):
                bi = idx[s:s + B]
                Xb, Yb, wb = Xtr[bi], Ytr[bi], w_tr[bi]
                Fs = [m.forward(Xb) for m in members]          # list[[B,dout]]
                Fbar = np.mean(np.stack(Fs, 0), axis=0)        # [B,dout]
                for m, Fm in zip(members, Fs):
                    mse_grad = 2.0 * (Fm - Yb) / len(Xb)
                    ncl_factor = (1.0 - 1.0 / M) * (Fbar - Yb) + (1.0 / M) * (Fm - Fbar)
                    ncl_grad = (self.ncl_lambda / len(Xb)) * (wb[:, None] * ncl_factor)
                    m.backward(mse_grad + ncl_grad, self.ncl_lr)

        self.members_ = members
        self.sigma_id_95_ = float(self._sigma_epi_quantile(X[: min(len(X), 2000)]))
        self._fitted = True
        return self

    # --------------------------------------------------------------- φ / predict
    def _phi(self, feats: np.ndarray):
        preds = np.stack([m.forward(feats) for m in self.members_], axis=0)  # [M,B,dout]
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("CompositionalTwinP5B_NCL.predict 前必须先 fit。")
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
        # 原始集成 epistemic（NCL 注入的多样性直接体现在 σ 中），无 P3-1 门控/封顶
        return y_mean, y_std_epi, nd_relative
