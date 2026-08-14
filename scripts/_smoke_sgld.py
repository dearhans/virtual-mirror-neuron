#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SGLD 冒烟测试：验证 (a) 可跑 (b) 成员多样性持久（不似 NCL 共识塌缩）(c) T 控制多样性。
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
d = load_or_generate(cfg, ROOT); X_raw, y, split = d["X"], d["y"], d["split"]
Xc = norman_to_compositional_X(X_raw); tr = split == "train"
comp_Dp = (Xc.shape[1] - 1 - 2) // 2
Xtr, ytr = Xc[tr], y[tr]
rng = np.random.default_rng(0); idx = np.arange(len(Xtr)); rng.shuffle(idx)
sub = idx[:8000]  # 中子集加速


def member_diversity(T, epochs=40):
    t0 = time.time()
    model = CompositionalTwinP5A_SGLD(
        Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]), n_ensemble=m["n_ensemble"],
        novelty_k=m["novelty_k"], random_state=m["random_state"],
        curriculum=True, sgld_temperature=T, sgld_lr=0.01, sgld_epochs=epochs, sgld_batch=256, sgld_decay=50)
    model.fit(Xtr[sub], ytr[sub], z_comp=None)
    # 端到端 predict 校验
    mu, sd, nd = model.predict(Xc[sub[:500]])
    assert mu.shape == sd.shape and sd.ndim == 2, f"predict 形状异常 {mu.shape}/{sd.shape}"
    # 成员多样性：跨成员 std（pa 输入）
    pa, pb, act, g_ = model._slice(Xc[sub])
    preds = np.stack([mem.forward(pa) for mem in model.members_], axis=0)
    std_all = float(preds.std(axis=0).mean())
    corr = float(np.corrcoef(preds[0, :, 0], preds[1, :, 0])[0, 1])
    print(f"[smoke] T={T}: fit {time.time()-t0:.1f}s  std_all={std_all:.4f}  corr(m0,m1)={corr:+.4f}  predict_ok", flush=True)
    return std_all, corr


s001, c001 = member_diversity(0.01)
s100, c100 = member_diversity(1.0)
if s100 <= s001 * 1.5:
    print(f"[smoke] FAIL: T 未显著改变成员多样性（{s001:.4f} vs {s100:.4f}），SGLD 可能仍塌缩", flush=True); sys.exit(2)
if s001 < 1e-4:
    print(f"[smoke] FAIL: T=0.01 成员多样性≈0（共识塌缩），需提高最低 T", flush=True); sys.exit(2)
print(f"[smoke] OK: T=0.01→1.0 成员多样性 {s001:.4f}→{s100:.4f}（{s100/s001:.1f}×），SGLD 持久多样性成立", flush=True)
