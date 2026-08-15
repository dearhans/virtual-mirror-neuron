#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-A 精炼冒烟：在最高 T=0.5（首跑批曾 overflow 发散）+ wdecay+梯度裁剪 下确认稳定且不塌缩。

验证两件事：
  1. T=0.5 不再 NaN / overflow（wdecay 提供恢复力，clip 防瞬态）；
  2. 成员仍保持持久多样性（post-fit member-std>0，corr 不→1），SGLD 机制存活。
"""
from __future__ import annotations
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
import yaml  # noqa: E402
from benchmark_ood import load_or_generate, norman_to_compositional_X  # noqa: E402
from model.compositional_p5a_sgld import CompositionalTwinP5A_SGLD  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"), encoding="utf-8"))
m = cfg["model"]
d = load_or_generate(cfg, ROOT)
X_raw, y, split = d["X"], d["y"], d["split"]
Xc = norman_to_compositional_X(X_raw)
tr = split == "train"
rng = np.random.default_rng(int(m.get("random_state", 0)))
idx = np.arange(len(Xc[tr])); rng.shuffle(idx)
nsub = 3000
sub = idx[:nsub]

comp_Dp = (Xc.shape[1] - 1 - 2) // 2
T = 0.5
WDECAY = 1e-3
CLIP = 1.0
t0 = time.time()
model = CompositionalTwinP5A_SGLD(
    Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]), n_ensemble=m["n_ensemble"],
    novelty_k=m["novelty_k"], random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
    sgld_temperature=T, sgld_lr=0.01, sgld_epochs=40, sgld_batch=256, sgld_decay=50,
    sgld_wdecay=WDECAY, clip_grad=CLIP,
)
model.fit(Xc[tr][sub], y[tr][sub], z_comp=None)
pa, pb, act, g_ = model._slice(Xc[tr][sub])
preds = np.stack([mem.forward(pa) for mem in model.members_], axis=0)  # [M,B,dout]
a = preds[:, :, 0]
nan_frac = float(np.isnan(preds).mean())
corr = float(np.corrcoef(a[0], a[1])[0, 1])
std_across_members = float(preds.std(axis=0).mean())
# 权重范数（确认未发散）
w_norms = [float(np.sqrt(sum((getattr(mem, w).astype(float)**2).sum() for w in ("W1","W2","W3")))) for mem in model.members_]
print(f"[smoke-refine] T={T} wdecay={WDECAY} clip={CLIP}: fit {time.time()-t0:.1f}s", flush=True)
print(f"  NaN_frac(preds)={nan_frac:.4f}  member-std={std_across_members:.4f}  corr(m0,m1)={corr:+.4f}", flush=True)
print(f"  weight-norms(members)={[round(x,1) for x in w_norms]}", flush=True)
if nan_frac > 0 or not np.isfinite(std_across_members) or std_across_members < 1e-3:
    print("[smoke-refine] FAIL: T=0.5 仍发散/塌缩", flush=True); sys.exit(2)
print(f"[smoke-refine] OK: T=0.5 稳定(sigma 内禀多样性存活), 可进入完整精炼跑批", flush=True)
