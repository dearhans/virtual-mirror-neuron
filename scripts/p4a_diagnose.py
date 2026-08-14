#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路 A 诊断：epistemic 欠缩放比是否与「测试时可获得的信号」相关？

若欠缩放（|resid|/σ_epi）与任何测试时可得特征（集成 std、修正后 kNN 新颖度、
与线性基线分歧、增益 g）都无关，则 novelty-gated std 在原理上不可行——
因为门控倍率必须能从测试时特征预测，而欠缩放若不依赖这些特征，就无信号可门控。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code")); sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
import yaml
from benchmark_ood import load_or_generate, norman_to_compositional_X, LinearBaseline
from model.compositional import CompositionalTwin
from scipy.stats import spearmanr

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"), encoding="utf-8"))
m = cfg["model"]
t0 = time.time()
d = load_or_generate(cfg, ROOT)
X_raw, y, split = d["X"], d["y"], d["split"]
Xc = norman_to_compositional_X(X_raw)
tr = split == "train"
Xtr, ytr = Xc[tr], y[tr]
rng = np.random.default_rng(int(m.get("random_state", 0)))
idx = np.arange(len(Xtr)); rng.shuffle(idx)
ncal = max(1, int(len(idx) * 0.2)); cidx, fidx = idx[:ncal], idx[ncal:]
comp_Dp = (Xc.shape[1] - 1 - 2) // 2

twin = CompositionalTwin(Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]), n_ensemble=m["n_ensemble"],
                         novelty_k=m["novelty_k"], random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)))
twin.fit(Xtr[fidx], ytr[fidx], z_comp=None)
lin = LinearBaseline().fit(Xtr, ytr)
print(f"[diag] fit {time.time()-t0:.1f}s", flush=True)

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
feat_names = ["sigma_epi", "novelty_nd", "disagree_linear", "gain_g"]
rows = []
for s in SUBSETS:
    msk = split == s
    if not msk.any(): continue
    Xs, ys = Xc[msk], y[msk]
    mu, sd, nd = twin.predict(Xs)   # 父类 gated std/sigma; 用 raw_epistemic 取 σ_epi
    mu_raw, sd_epi, nd_raw = twin._raw_epistemic(Xs) if hasattr(twin, "_raw_epistemic") else (mu, sd, nd)
    # 逐 (样本,基因) 欠缩放比
    score = np.abs(ys - mu_raw) / np.clip(sd_epi, 1e-9, None)
    us = np.median(score, axis=1)              # 每样本欠缩放（中位数跨基因）
    sig = np.median(sd_epi, axis=1) if sd_epi.ndim == 2 else sd_epi
    nov = nd_raw
    dis = np.abs(mu_raw.mean(1) - np.asarray(lin.predict(Xs)).mean(1)) if mu_raw.ndim == 2 else np.abs(mu_raw - np.asarray(lin.predict(Xs)))
    gv = Xs[:, -1]
    feats = {"sigma_epi": sig, "novelty_nd": nov, "disagree_linear": dis, "gain_g": gv}
    cors = {}
    for fn in feat_names:
        f = np.asarray(feats[fn], dtype=float)
        if np.std(f) < 1e-12:
            cors[fn] = float("nan")   # 常量特征（如二元新颖度）无相关性
        else:
            r, p = spearmanr(us, f)
            cors[fn] = float(r)
    rows.append((s, us, cors))
    print(f"[diag] {s:12s} n={msk.sum():6d}  under-scale median={np.median(us):.2f} p95={np.quantile(us,0.95):.2f}"
          + "  | spearman(under-scale vs): " + ", ".join(f"{fn}={cors[fn]:+.3f}" for fn in feat_names), flush=True)

# 跨子集池化（看整体是否任何信号有用）
all_us = np.concatenate([r[1] for r in rows])
print("\n=== 跨子集池化（受子集混杂，仅参考）===")
print("under-scale整体 median=%.2f p95=%.2f" % (np.median(all_us), np.quantile(all_us, 0.95)))
print(f"[diag] done {time.time()-t0:.1f}s", flush=True)
