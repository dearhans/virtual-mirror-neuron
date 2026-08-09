#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定向核验：孪生在 ood_agent 上是否塌成常数预测器（macro-per-gene PCC = nan 的根因）。

判据：跨扰动组的预测方差 / 真实方差。比值 ~0 即为「预测均值」坍缩。
只拟合 compositional_twin + linear（对照），约 50s。
"""
import os, sys, json, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))
import yaml
from benchmark_ood import (load_or_generate, norman_to_compositional_X,
                           CompositionalTwinPredictor, LinearBaseline, MeanBaseline)

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"), encoding="utf-8"))
d = load_or_generate(cfg, ROOT)
X, y, split = d["X"], d["y"], d["split"]
Praw = d["X"][:, :107]
X = norman_to_compositional_X(X)
tr = split == "train"
t0 = time.time()
ct = CompositionalTwinPredictor(cfg, conformal=bool(cfg["model"].get("conformal", False))).fit(
    X[tr], y[tr], split=split[tr], z_comp=None)
ln = LinearBaseline().fit(X[tr], y[tr]); mn = MeanBaseline().fit(X[tr], y[tr])
print(f"fit done {time.time()-t0:.1f}s", flush=True)

def gk(P):
    return np.asarray(["ctrl" if (r > .5).sum() == 0 else "+".join(map(str, sorted(np.flatnonzero(r > .5).tolist()))) for r in P])

out = {}
for s in ["id", "ood_agent", "ood_action", "ood_neuro"]:
    m = split == s
    keys = gk(Praw[m]); uk = np.unique(keys)
    pbt = np.stack([y[m][keys == k].mean(0) for k in uk])
    row = {"n_groups": int(len(uk)), "true_across_group_var": float(pbt.var(0).mean())}
    for nm, p in [("compositional_twin", ct), ("linear", ln), ("mean", mn)]:
        yp = np.asarray(p.predict(X[m]))
        pbp = np.stack([yp[keys == k].mean(0) for k in uk])
        v = float(pbp.var(0).mean())
        row[nm] = {"pred_across_group_var": v,
                   "var_ratio_vs_true": v / max(row["true_across_group_var"], 1e-30),
                   "n_distinct_pred_rows": int(len(np.unique(np.round(pbp, 6), axis=0)))}
    out[s] = row
    print(f"== {s} groups={row['n_groups']} true_var={row['true_across_group_var']:.6f}")
    for nm in ["compositional_twin", "linear", "mean"]:
        r = row[nm]
        print(f"   {nm:20s} pred_var={r['pred_across_group_var']:.3e} ratio={r['var_ratio_vs_true']:.4f} distinct_rows={r['n_distinct_pred_rows']}/{row['n_groups']}")
json.dump(out, open(os.path.join(ROOT, "experiments/20260807-collapse-check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("written -> experiments/20260807-collapse-check.json")
