#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NCL 机制诊断：隔离「稀疏度 w」与「λ 量级」两因素，判断 NCL 是否真能注入多样性。

场景：
  A. 默认稀疏 w（P5-B 设计），λ=0.1 vs 1.0   —— 复现 smoke 结果
  B. 强制 w=1（uniform NCL，去掉稀疏瓶颈），λ=0.1 vs 1.0
  C. 默认 w，λ=1.0 vs 10                       —— 看高阶 λ 是否能撬动共识塌缩
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
d = load_or_generate(cfg, ROOT); X_raw, y, split = d["X"], d["y"], d["split"]
Xc = norman_to_compositional_X(X_raw); tr = split == "train"
comp_Dp = (Xc.shape[1] - 1 - 2) // 2
Xtr, ytr = Xc[tr], y[tr]
rng = np.random.default_rng(0); idx = np.arange(len(Xtr)); rng.shuffle(idx)
sub = idx[:3000]


class UniformW(CompositionalTwinP5B_NCL):
    """强制稀疏度 w=1（uniform NCL）。"""
    def _sparsity_weight(self, F):
        return np.ones(len(F))


def run(model_cls, lam, epochs=60):
    t0 = time.time()
    model = model_cls(Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]), n_ensemble=m["n_ensemble"],
                      novelty_k=m["novelty_k"], random_state=m["random_state"],
                      curriculum=True, ncl_lambda=lam, ncl_lr=0.01, ncl_epochs=epochs, ncl_batch=256)
    model.fit(Xtr[sub], ytr[sub], z_comp=None)
    pa, pb, act, g_ = model._slice(Xtr[sub])
    preds = np.stack([mem.forward(pa) for mem in model.members_], axis=0)
    w = model._sparsity_weight(model.train_feats_)
    top = w >= np.quantile(w, 0.90)
    std_tail = float(preds[:, top, :].std(axis=0).mean())
    corr = float(np.corrcoef(preds[0, :, 0], preds[1, :, 0])[0, 1])
    print(f"   {model_cls.__name__:10s} λ={lam:>5}: fit {time.time()-t0:.1f}s  std_tail={std_tail:.4f}  corr={corr:+.4f}", flush=True)
    return std_tail, corr


print("[diag] A: 默认稀疏 w", flush=True)
a1 = run(CompositionalTwinP5B_NCL, 0.1)
a2 = run(CompositionalTwinP5B_NCL, 1.0)
print("[diag] B: uniform w=1", flush=True)
b1 = run(UniformW, 0.1)
b2 = run(UniformW, 1.0)
print("[diag] C: 默认 w, λ=10", flush=True)
c1 = run(CompositionalTwinP5B_NCL, 1.0)
c2 = run(CompositionalTwinP5B_NCL, 10.0)

print("\n[diag] 结论判定：", flush=True)
print(f"   A: λ0.1 vs 1.0  std_tail {a1[0]:.4f} vs {a2[0]:.4f}  (差 {abs(a2[0]-a1[0]):.2e})", flush=True)
print(f"   B: λ0.1 vs 1.0  std_tail {b1[0]:.4f} vs {b2[0]:.4f}  (差 {abs(b2[0]-b1[0]):.2e})", flush=True)
print(f"   C: λ1.0 vs 10   std_tail {c1[0]:.4f} vs {c2[0]:.4f}  (差 {abs(c2[0]-c1[0]):.2e})", flush=True)
multiplicative = (abs(a2[0]-a1[0])>1e-4) or (abs(b2[0]-b1[0])>1e-4) or (abs(c2[0]-c1[0])>1e-4)
if multiplicative:
    print("[diag] NCL 机制存活：λ 能改变多样性 → 跑完整 sweep 验证。", flush=True)
else:
    print("[diag] NCL 机制失效：即便 uniform w / 10×λ 也撬不动共识塌缩 → P5-B 概念在本数据/工具链下不可行。", flush=True)
