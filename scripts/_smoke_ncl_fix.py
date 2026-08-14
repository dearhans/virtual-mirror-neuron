#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke test：验证 NCL 梯度修复后 (a) 可跑 (b) λ 真正改变成员多样性。

若 λ=0.1 与 λ=1.0 的成员间标准差近似相同 → 修复仍失败，需进一步排查。
"""
from __future__ import annotations
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
import yaml  # noqa: E402
from benchmark_ood import load_or_generate, norman_to_compositional_X  # noqa: E402
from model.compositional_p5b_ncl import CompositionalTwinP5B_NCL  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"), encoding="utf-8"))
m = cfg["model"]
d = load_or_generate(cfg, ROOT)
X_raw, y, split = d["X"], d["y"], d["split"]
Xc = norman_to_compositional_X(X_raw)
tr = split == "train"
comp_Dp = (Xc.shape[1] - 1 - 2) // 2
Xtr, ytr = Xc[tr], y[tr]
rng = np.random.default_rng(0); idx = np.arange(len(Xtr)); rng.shuffle(idx)
sub = idx[:3000]  # 小子集加速

def member_diversity(lam):
    t0 = time.time()
    model = CompositionalTwinP5B_NCL(
        Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]), n_ensemble=m["n_ensemble"],
        novelty_k=m["novelty_k"], random_state=m["random_state"],
        curriculum=True, ncl_lambda=lam, ncl_lr=0.01, ncl_epochs=60, ncl_batch=256)
    model.fit(Xtr[sub], ytr[sub], z_comp=None)
    # 用 _phi 取成员级预测（跨成员），输入为 pa（Dp 维）
    pa, pb, act, g_ = model._slice(Xtr[sub])
    preds = np.stack([mem.forward(pa) for mem in model.members_], axis=0)  # [M, N, Dp]
    # 稀疏度 w（与训练一致口径）
    w = model._sparsity_weight(model.train_feats_)
    # 全局成员标准差
    std_all = float(preds.std(axis=0).mean())
    # 高稀疏尾部（top 10% w）成员标准差 —— NCL 应在稀疏区注入多样性
    top = w >= np.quantile(w, 0.90)
    std_tail = float(preds[:, top, :].std(axis=0).mean()) if top.any() else 0.0
    corr = float(np.corrcoef(preds[0, :, 0], preds[1, :, 0])[0, 1])
    print(f"[smoke] λ={lam}: fit {time.time()-t0:.1f}s  std_all={std_all:.4f}  std_tail(top10% w)={std_tail:.4f}  corr={corr:+.4f}", flush=True)
    return std_all, std_tail, corr

s01, t01, c01 = member_diversity(0.1)
s10, t10, c10 = member_diversity(1.0)
if abs(t10 - t01) < 1e-9 and abs(s10 - s01) < 1e-9:
    print("[smoke] FAIL: λ 未改变成员多样性（修复仍无效）", flush=True); sys.exit(2)
print(f"[smoke] OK: λ=1.0 vs 0.1 → std_all {s10/s01:.2f}×, std_tail {t10/max(t01,1e-12):.2f}×, 修复生效", flush=True)

