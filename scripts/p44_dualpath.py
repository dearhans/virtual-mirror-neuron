#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-4 新证据槽：φ = 线性头 ⊕ 机制残差头（双通路，空间分离 PC-1）。

两臂受控对照（共享同一数据/切分/种子，零混淆）
------------------------------------------------------
    臂 A（P1 对照）   ：canonical `compositional_twin` —— 必须**逐位复现**当周基准
    臂 D（P4-4）      ：DualPathCompositionalTwin（线性头 + 残差头）—— 本周被检验对象
    linear / mean     ：简单基线，用于判据对比

任一复现核验失败即视为管线污染，**不出结论**（GATE_REPRO=False）。

判据（W33 §7 #3，事前锁定，不得事后修改）
------------------------------------------------------
    ood_action 上 P4-4 的 RMSE 与线性基线 **CI 重叠或更优**（点估计 ≤ 线性）
    → P4-4 成立（组合孪生首次在 ood_action 与线性并驾或超越）；
    否则记为阴性结果并重新审视 PC-1 破法。

用法：
    python scripts/p44_dualpath.py --ref experiments/20260809-benchmark.json \
        --out experiments/20260812-p44-dualpath.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))

import yaml  # noqa: E402
from benchmark_ood import (  # noqa: E402
    load_or_generate, norman_to_compositional_X, bootstrap_ci, calibration, rmse,
    LinearBaseline, MeanBaseline,
)
from model.compositional import CompositionalTwin  # noqa: E402
from model.compositional_p44 import DualPathCompositionalTwin  # noqa: E402

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
SATURATION_ECE = 0.2125

# --- 事前锁定的判据（W33 §7 #3） ---
CRITERION = {
    "subset": "ood_action",
    "compare_to": "linear",
    "rule": "ci_overlap_or_point_better",
}

# 线性基线在 ood_action 的当周锚点（W33 §1，用于诚实标注"首次"语境）
W33_LINEAR_OOD_ACTION = {"rmse": 0.6266, "ci_low": 0.6218, "ci_high": 0.6310}
W33_TWIN_OOD_ACTION = {  # 基础组合孪生（P1），用于对照"是否首次"
    "rmse": 0.6355, "ci_low": 0.6301, "ci_high": 0.6400,
}
ANCHOR_TOL = {"rmse": 5e-3}


def group_keys(P: np.ndarray) -> np.ndarray:
    return np.asarray([
        "ctrl" if (r > .5).sum() == 0 else "+".join(map(str, sorted(np.flatnonzero(r > .5).tolist())))
        for r in P
    ])


def variance_guard(keys: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    uk = np.unique(keys)
    pbt = np.stack([y_true[keys == k].mean(0) for k in uk])
    pbp = np.stack([y_pred[keys == k].mean(0) for k in uk])
    tv = float(pbt.var(0).mean())
    pv = float(pbp.var(0).mean())
    return {
        "n_groups": int(len(uk)),
        "true_across_group_var": tv,
        "pred_across_group_var": pv,
        "var_ratio_vs_true": pv / max(tv, 1e-30),
        "n_distinct_pred_rows": int(len(np.unique(np.round(pbp, 6), axis=0))),
        "collapsed": bool(pv / max(tv, 1e-30) < 1e-3),
        "pb_rmse": float(np.sqrt(np.mean((pbt - pbp) ** 2))),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True, help="当周基准 JSON（臂 A 复现核验 + 基线取数）")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    m, ev = cfg["model"], cfg["eval"]
    levels = list(ev["calibration_levels"])
    n_boot = int(ev["bootstrap_samples"])
    ci_alpha = float(ev["ci_alpha"])

    t0 = time.time()
    d = load_or_generate(cfg, ROOT)
    X_raw, y, split = d["X"], d["y"], d["split"]
    Praw = X_raw[:, :107]
    Xc = norman_to_compositional_X(X_raw)
    tr = split == "train"
    print(f"[p44] data ready {time.time()-t0:.1f}s X={Xc.shape}", flush=True)

    Xtr, ytr = Xc[tr], y[tr]
    rng = np.random.default_rng(int(m.get("random_state", 0)))
    idx = np.arange(len(Xtr))
    rng.shuffle(idx)
    ncal = max(1, int(len(idx) * float(m.get("conformal_calib_frac", 0.2))))
    cidx, fidx = idx[:ncal], idx[ncal:]

    comp_Dp = (Xc.shape[1] - 1 - 2) // 2

    twin = CompositionalTwin(
        Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]),
        n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
        random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
    )
    t1 = time.time()
    twin.fit(Xtr[fidx], ytr[fidx], z_comp=None)
    print(f"[p44] P1 twin fit {time.time()-t1:.1f}s", flush=True)

    p44 = DualPathCompositionalTwin(
        Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]),
        n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
        random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
    )
    t2 = time.time()
    p44.fit(Xtr[fidx], ytr[fidx], z_comp=None)
    print(f"[p44] P4-4 dual-path twin fit {time.time()-t2:.1f}s", flush=True)

    alpha = float(m.get("conformal_alpha", 0.9))

    def conf_q(mu_c, sd_c):
        return float(np.quantile(np.abs(ytr[cidx] - mu_c) / np.clip(sd_c, 1e-9, None), alpha))

    muA_c, sdA_c, _ = twin.predict(Xtr[cidx])
    qA = conf_q(muA_c, sdA_c)
    muD_c, sdD_c, _ = p44.predict(Xtr[cidx])
    qD = conf_q(muD_c, sdD_c)
    print(f"[p44] conformal q: A(P1)={qA:.4f} D(P4-4)={qD:.4f}", flush=True)

    lin = LinearBaseline().fit(Xtr, ytr)
    mean_bl = MeanBaseline().fit(Xtr, ytr)
    ref = json.load(open(a.ref, encoding="utf-8")) if os.path.exists(a.ref) else {}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.relpath(a.config, ROOT).replace("\\", "/"),
        "ref": os.path.relpath(a.ref, ROOT).replace("\\", "/"),
        "criterion_prelocked": CRITERION,
        "w33_anchors": {
            "linear_ood_action": W33_LINEAR_OOD_ACTION,
            "twin_P1_ood_action": W33_TWIN_OOD_ACTION,
        },
        "conformal_q": {"armA_P1": qA, "armD_P44": qD},
        "subsets": {},
    }

    for s in SUBSETS:
        msk = split == s
        if not msk.any():
            continue
        Xs, ys = Xc[msk], y[msk]
        keys = group_keys(Praw[msk])
        row = {"n": int(msk.sum())}

        muA, sdA, _ = twin.predict(Xs)
        muD, sdD, _ = p44.predict(Xs)
        preds = {
            "armA_P1_compositional_twin": (muA, qA * sdA),
            "armD_P44_dual_path": (muD, qD * sdD),
            "linear": (np.asarray(lin.predict(Xs)), None),
            "mean": (np.asarray(mean_bl.predict(Xs)), None),
        }
        for name, (mu, sd) in preds.items():
            mu = np.asarray(mu)
            mval, lo, hi = bootstrap_ci(ys, mu, n_boot, ci_alpha, seed=0)
            entry = {"rmse": rmse(ys, mu), "rmse_boot_mean": mval, "ci_low": lo, "ci_high": hi}
            entry.update(variance_guard(keys, ys, mu))
            if sd is not None:
                cal = calibration(ys, mu, sd, levels)
                cov = cal["coverage"]
                entry["coverage"] = {str(k): v for k, v in cov.items()}
                entry["ece"] = cal["ece"]
                entry["calibration_curve"] = cal["calibration_curve"]
                entry["saturated"] = bool(
                    abs(cal["ece"] - SATURATION_ECE) < 1e-3 and all(v > 0.999 for v in cov.values()))
            row[name] = entry
        out["subsets"][s] = row
        print(f"[p44] == {s} n={row['n']}", flush=True)
        for name in preds:
            e = row[name]
            print("     %-32s rmse=%.4f [%.4f,%.4f] pbRMSE=%.4f var_ratio=%.3e %s"
                  % (name, e["rmse"], e["ci_low"], e["ci_high"], e["pb_rmse"], e["var_ratio_vs_true"],
                     ("ECE=%.4f%s" % (e["ece"], " SAT" if e.get("saturated") else "")) if "ece" in e else ""),
                  flush=True)

    # ---------- 复现核验（污染即中止出结论） ----------
    repro = {"armA_vs_canonical": {"performed": False}}
    if ref.get("results"):
        deltas, ok = {}, True
        for s in SUBSETS:
            r = ref["results"].get(s, {}).get("compositional_twin", {})
            if "rmse" in r and s in out["subsets"]:
                dd = float(out["subsets"][s]["armA_P1_compositional_twin"]["rmse_boot_mean"]) - float(r["rmse"])
                deltas[s] = dd
                ok = ok and abs(dd) < 1e-9
        repro["armA_vs_canonical"] = {"performed": True, "match": bool(ok),
                                      "compare_key": "rmse_boot_mean_vs_canonical_rmse",
                                      "tol": 1e-9, "deltas": deltas}
    gate = bool(repro["armA_vs_canonical"].get("match"))
    out["reproduction_check"] = repro
    out["GATE_REPRO"] = gate

    # ---------- 事前判据裁决 ----------
    verdict = {"evaluated": False}
    if gate and CRITERION["subset"] in out["subsets"]:
        c = out["subsets"][CRITERION["subset"]]["armD_P44_dual_path"]
        lin_e = out["subsets"][CRITERION["subset"]]["linear"]
        a1_e = out["subsets"][CRITERION["subset"]]["armA_P1_compositional_twin"]
        ci_overlap = not (c["ci_high"] < lin_e["ci_low"] or lin_e["ci_high"] < c["ci_low"])
        point_better = c["rmse"] <= lin_e["rmse"]
        verdict = {
            "evaluated": True,
            "subset": CRITERION["subset"],
            "p44_rmse": c["rmse"], "p44_ci": [c["ci_low"], c["ci_high"]],
            "linear_rmse": lin_e["rmse"], "linear_ci": [lin_e["ci_low"], lin_e["ci_high"]],
            "p1_twin_rmse": a1_e["rmse"], "p1_twin_ci": [a1_e["ci_low"], a1_e["ci_high"]],
            "ci_overlap_with_linear": bool(ci_overlap),
            "point_better_than_linear": bool(point_better),
            "p44_vs_p1_twin_delta": c["rmse"] - a1_e["rmse"],
            "P44_ACCEPTED": bool(ci_overlap or point_better),
        }
    out["verdict_p44"] = verdict

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[p44] GATE_REPRO={gate}  verdict={json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[p44] written -> {a.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
