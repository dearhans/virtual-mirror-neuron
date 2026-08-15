#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-C 冒烟：用合成数据验证 GMM 密度门控机制不退化（不加载 Norman 全量数据）。

验证点：
    1. CompositionalTwinP5C_GMM 可 fit（标准集成）+ fit_density_gate（ID 校准）不报错；
    2. GMM 密度拟合成功（_gmm_ok=True），门控倍率单调非减、封顶生效；
    3. 密度新颖度 u 在 OOD（远离训练流形）显著 > ID，门控倍率随之 >1；
    4. predict 返回 (y_mean, sigma_pred, u)，sigma_pred 在 OOD 被撑大。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))

from model.compositional_p5c_gmm import CompositionalTwinP5C_GMM  # noqa: E402

rng = np.random.default_rng(0)
Dp, Adim = 8, 2
n = 600


def make_Xy(n, shift=0.0):
    pa = (rng.random((n, Dp)) < 0.3).astype(float) + shift
    pb = np.zeros((n, Dp))
    # 一半为双扰动（随机非零 pb）
    dbl = rng.random(n) < 0.5
    pb[dbl] = (rng.random((int(dbl.sum()), Dp)) < 0.3).astype(float)
    act = np.zeros((n, Adim))
    act[:, 0] = (pb.sum(1) == 0).astype(float)
    act[:, 1] = (pb.sum(1) > 0).astype(float)
    g = rng.uniform(0.5, 2.0, (n, 1))
    X = np.hstack([pa, pb, act, g])
    y = (pa * np.linspace(0.1, 1.0, Dp)).sum(1, keepdims=True) * g + rng.normal(0, 0.1, (n, 1))
    return X, y.ravel()


X, y = make_Xy(n)
# ID 校准集取前 20%，训练（fit 主体）取后 80%
cidx = np.arange(int(n * 0.2))
fidx = np.arange(int(n * 0.2), n)
# OOD 预测集：特征整体平移（远离训练流形）
X_ood, y_ood = make_Xy(120, shift=10.0)

m = CompositionalTwinP5C_GMM(Dp=Dp, A_dim=Adim, hidden=(16, 16), n_ensemble=3,
                             gmm_n_components=6, gate_cap_ratio=50.0, random_state=0)
m.fit(X[fidx], y[fidx])
m.fit_density_gate(X[cidx], y[cidx])

mu_tr, sd_tr, u_tr = m.predict(X[fidx])
mu_ood, sd_ood, u_ood = m.predict(X_ood)

print("== P5-C smoke ==")
print(f"gmm_ok={m._gmm_ok}  gate_fitted={m._gate_fitted}  logq_med={getattr(m,'_logq_med_',None):.3f}")
print(f"gate_u_centers={np.round(m._gate_u,3)}")
print(f"gate_ratio    ={np.round(m._gate_ratio,3)}  (monotone non-decreasing? {bool(np.all(np.diff(m._gate_ratio)>=0))})")
print(f"u id  median={np.median(u_tr):.3f} p95={np.quantile(u_tr,0.95):.3f}")
print(f"u ood median={np.median(u_ood):.3f} p95={np.quantile(u_ood,0.95):.3f}")
gm_tr = m._density_multiplier(u_tr)
gm_ood = m._density_multiplier(u_ood)
print(f"gate_mult id  median={np.median(gm_tr):.3f} (in [{gm_tr.min():.3f},{gm_tr.max():.3f}])")
print(f"gate_mult ood median={np.median(gm_ood):.3f} (in [{gm_ood.min():.3f},{gm_ood.max():.3f}])")
print(f"sigma_pred id  median={np.median(sd_tr):.4f}")
print(f"sigma_pred ood median={np.median(sd_ood):.4f}  (OOD inflated? {bool(np.median(sd_ood)>np.median(sd_tr))})")

ok = (m._gmm_ok and m._gate_fitted and np.all(np.diff(m._gate_ratio) >= 0)
      and np.median(u_ood) > np.median(u_tr) and np.median(gm_ood) > 1.0
      and np.median(sd_ood) > np.median(sd_tr))
print("SMOKE_P5C_OK" if ok else "SMOKE_P5C_FAIL")
sys.exit(0 if ok else 1)
