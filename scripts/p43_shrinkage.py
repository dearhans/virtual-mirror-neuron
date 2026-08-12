#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-3 新证据槽：ê_u 经验贝叶斯逆方差收缩（对 P4-2 等权平均的直接消融）。

三臂受控消融（共享同一个已 fit 的 φ，零混淆）
------------------------------------------------------
    臂 A（P1 对照）   ：canonical `compositional_twin` —— 必须**逐位复现**当周基准
    臂 B（P4-2）      ：SoloEffectImputer（等权平均）—— 必须复现 W34 记录值
    臂 C（P4-3）      ：ShrunkSoloEffectImputer（EB 逆方差收缩）—— 本周被检验对象

任一复现核验失败即视为管线污染，**不出结论**（GATE_REPRO=False）。

判据（W34 定下，事前锁定，不得事后修改）
------------------------------------------------------
    ood_agent 臂 C：RMSE ≤ 0.5854（即回到/超过 P1 水平） 且 var_ratio > 0.1（坍缩不复发）
    两条同时满足 → P4-3 成立；否则记为阴性结果并转 P4-4（结构分离）。

用法：
    python scripts/p43_shrinkage.py --ref experiments/20260810-benchmark.json \
        --out experiments/20260810-p43-shrinkage.json
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
from model.compositional_p42 import SoloEffectImputer  # noqa: E402
from model.compositional_p43 import ShrunkSoloEffectImputer  # noqa: E402

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
SATURATION_ECE = 0.2125

# --- 事前锁定的判据与历史锚点（W34 / experiments/20260809-p42-solo-imputation.json） ---
CRITERION = {
    "subset": "ood_agent",
    "rmse_max": 0.5854,      # P1 点估计水平
    "var_ratio_min": 0.10,   # 坍缩不得复发
}
W34_ANCHORS = {"armB_rmse": 0.5887, "armB_var_ratio": 0.503}
ANCHOR_TOL = {"rmse": 5e-3, "var_ratio": 5e-2}


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
    print(f"[p43] data ready {time.time()-t0:.1f}s X={Xc.shape}", flush=True)

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
    print(f"[p43] twin fit {time.time()-t1:.1f}s", flush=True)

    impB = SoloEffectImputer(min_count=1).fit(twin, Xtr[fidx], ytr[fidx])
    impC = ShrunkSoloEffectImputer(min_count=1).fit(twin, Xtr[fidx], ytr[fidx])
    print(f"[p43] imputers fit. B_imputed={impB.summary()['n_genes_imputed_from_codoubles']} "
          f"C_shrunk={impC.summary()['shrinkage'].get('n_genes_shrunk')}", flush=True)
    print(f"[p43] shrinkage report: {json.dumps(impC.summary()['shrinkage'], ensure_ascii=False)}", flush=True)

    alpha = float(m.get("conformal_alpha", 0.9))

    def conf_q(mu_c, sd_c):
        return float(np.quantile(np.abs(ytr[cidx] - mu_c) / np.clip(sd_c, 1e-9, None), alpha))

    muA_c, sdA_c, _ = twin.predict(Xtr[cidx])
    qA = conf_q(muA_c, sdA_c)
    muB_c, sdB_c, _ = impB.predict(twin, Xtr[cidx])
    qB = conf_q(muB_c, sdB_c)
    muC_c, sdC_c, _ = impC.predict(twin, Xtr[cidx])
    qC = conf_q(muC_c, sdC_c)
    print(f"[p43] conformal q: A={qA:.4f} B={qB:.4f} C={qC:.4f}", flush=True)

    lin = LinearBaseline().fit(Xtr, ytr)
    mean_bl = MeanBaseline().fit(Xtr, ytr)
    ref = json.load(open(a.ref, encoding="utf-8")) if os.path.exists(a.ref) else {}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.relpath(a.config, ROOT).replace("\\", "/"),
        "ref": os.path.relpath(a.ref, ROOT).replace("\\", "/"),
        "criterion_prelocked": CRITERION,
        "w34_anchors": W34_ANCHORS,
        "conformal_q": {"armA_P1": qA, "armB_P42": qB, "armC_P43": qC},
        "imputer_summary_B": impB.summary(),
        "imputer_summary_C": impC.summary(),
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
        muB, sdB, _ = impB.predict(twin, Xs)
        muC, sdC, _ = impC.predict(twin, Xs)
        preds = {
            "armA_P1_compositional_twin": (muA, qA * sdA),
            "armB_P42_equal_weight": (muB, qB * sdB),
            "armC_P43_shrinkage": (muC, qC * sdC),
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
        print(f"[p43] == {s} n={row['n']}", flush=True)
        for name in preds:
            e = row[name]
            print("     %-30s rmse=%.4f [%.4f,%.4f] pbRMSE=%.4f var_ratio=%.3e %s"
                  % (name, e["rmse"], e["ci_low"], e["ci_high"], e["pb_rmse"], e["var_ratio_vs_true"],
                     ("ECE=%.4f%s" % (e["ece"], " SAT" if e.get("saturated") else "")) if "ece" in e else ""),
                  flush=True)

    # ---------- 复现核验（污染即中止出结论） ----------
    repro = {"armA_vs_canonical": {"performed": False}, "armB_vs_w34_anchor": {"performed": False}}
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
    if "ood_agent" in out["subsets"]:
        b = out["subsets"]["ood_agent"]["armB_P42_equal_weight"]
        dr = abs(b["rmse"] - W34_ANCHORS["armB_rmse"])
        dv = abs(b["var_ratio_vs_true"] - W34_ANCHORS["armB_var_ratio"])
        repro["armB_vs_w34_anchor"] = {
            "performed": True, "rmse_now": b["rmse"], "var_ratio_now": b["var_ratio_vs_true"],
            "abs_delta_rmse": dr, "abs_delta_var_ratio": dv, "tol": ANCHOR_TOL,
            "match": bool(dr < ANCHOR_TOL["rmse"] and dv < ANCHOR_TOL["var_ratio"]),
        }
    gate = bool(repro["armA_vs_canonical"].get("match") and repro["armB_vs_w34_anchor"].get("match"))
    out["reproduction_check"] = repro
    out["GATE_REPRO"] = gate

    # ---------- 事前判据裁决 ----------
    verdict = {"evaluated": False}
    if gate and CRITERION["subset"] in out["subsets"]:
        c = out["subsets"][CRITERION["subset"]]["armC_P43_shrinkage"]
        b = out["subsets"][CRITERION["subset"]]["armB_P42_equal_weight"]
        a1 = out["subsets"][CRITERION["subset"]]["armA_P1_compositional_twin"]
        pass_rmse = c["rmse"] <= CRITERION["rmse_max"]
        pass_var = c["var_ratio_vs_true"] > CRITERION["var_ratio_min"]
        ci_overlap = not (c["ci_high"] < b["ci_low"] or b["ci_high"] < c["ci_low"])
        verdict = {
            "evaluated": True,
            "armC_rmse": c["rmse"], "armC_ci": [c["ci_low"], c["ci_high"]],
            "armC_var_ratio": c["var_ratio_vs_true"], "armC_pb_rmse": c["pb_rmse"],
            "armB_rmse": b["rmse"], "armB_pb_rmse": b["pb_rmse"],
            "armA_rmse": a1["rmse"], "armA_pb_rmse": a1["pb_rmse"],
            "pass_rmse_le_0.5854": bool(pass_rmse),
            "pass_var_ratio_gt_0.10": bool(pass_var),
            "P43_ACCEPTED": bool(pass_rmse and pass_var),
            "armC_vs_armB_ci_overlap": bool(ci_overlap),
            "delta_rmse_C_minus_B": c["rmse"] - b["rmse"],
            "delta_pb_rmse_C_minus_B": c["pb_rmse"] - b["pb_rmse"],
        }
    out["verdict_p43"] = verdict

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[p43] GATE_REPRO={gate}  verdict={json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[p43] written -> {a.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
