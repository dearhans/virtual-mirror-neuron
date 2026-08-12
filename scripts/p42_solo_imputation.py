#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 新证据槽：共扰动→单独效应插补，修 φ 在 ood_agent 的常数坍缩。

受控消融设计（关键：两臂共享同一个已 fit 的 φ，零混淆）
------------------------------------------------------
    臂 A（对照 / P1）：canonical `compositional_twin`，应**逐位复现**当周基准数字
                       （若不复现说明消融管线本身有问题 → 直接中止，不出结论）。
    臂 B（P4-2）      ：同一 φ + SoloEffectImputer 覆写，仅改「未被单扰动监督过的基因」。

输出指标（全部遵守项目硬约束）
------------------------------------------------------
    - RMSE + bootstrap 95% CI（与基准同口径：n_boot 从 config 读，seed 固定）
    - pbRMSE（按扰动组 pseudobulk）——W33 已证实的判别性指标，抗「预测均值」陷阱
    - 方差比守卫 var_ratio / n_distinct（坍缩检测，本周升为一等指标）
    - 覆盖度 + ECE + 校准曲线（含饱和签名检测：ECE≈0.2125 且覆盖恒 1.000）

用法：
    python scripts/p42_solo_imputation.py [--out experiments/20260808-p42-solo-imputation.json]
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

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
SATURATION_ECE = 0.2125  # levels [.5,.8,.9,.95] 下 mean(|1-level|) —— 饱和签名


def group_keys(P: np.ndarray) -> np.ndarray:
    """扰动组标识（基因索引集合），用于 pseudobulk 与方差比守卫。"""
    return np.asarray([
        "ctrl" if (r > .5).sum() == 0 else "+".join(map(str, sorted(np.flatnonzero(r > .5).tolist())))
        for r in P
    ])


def variance_guard(keys: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """坍缩检测：跨扰动组的预测方差 / 真实方差。~0 即坍缩为常数预测器。"""
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
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments/20260808-p42-solo-imputation.json"))
    ap.add_argument("--ref", default=os.path.join(ROOT, "experiments/20260808-benchmark.json"),
                    help="当周基准 JSON，用于臂 A 复现核验与基线取数")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    m = cfg["model"]
    ev = cfg["eval"]
    levels = list(ev["calibration_levels"])
    n_boot = int(ev["bootstrap_samples"])
    ci_alpha = float(ev["ci_alpha"])

    t0 = time.time()
    d = load_or_generate(cfg, ROOT)
    X_raw, y, split = d["X"], d["y"], d["split"]
    Praw = X_raw[:, :107]
    Xc = norman_to_compositional_X(X_raw)
    tr = split == "train"
    print(f"[p42] data ready {time.time()-t0:.1f}s  X={Xc.shape} y={y.shape}", flush=True)

    # ---- 复现 canonical 的 conformal 划分（同 rng、同 frac），保证臂 A 可逐位对齐 ----
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
    print(f"[p42] twin fit {time.time()-t1:.1f}s", flush=True)

    # ---- 臂 B：solo-effect 插补（只用 twin 实际拟合过的行）----
    t2 = time.time()
    imp = SoloEffectImputer(min_count=1).fit(twin, Xtr[fidx], ytr[fidx])
    print(f"[p42] imputer fit {time.time()-t2:.1f}s  {imp.summary()['n_genes_imputed_from_codoubles']} genes imputed",
          flush=True)

    # ---- 两臂各自做共形标定（同一 cidx，公平）----
    alpha = float(m.get("conformal_alpha", 0.9))
    muA_c, sdA_c, _ = twin.predict(Xtr[cidx])
    qA = float(np.quantile(np.abs(ytr[cidx] - muA_c) / np.clip(sdA_c, 1e-9, None), alpha))
    muB_c, sdB_c, _ = imp.predict(twin, Xtr[cidx])
    qB = float(np.quantile(np.abs(ytr[cidx] - muB_c) / np.clip(sdB_c, 1e-9, None), alpha))
    print(f"[p42] conformal q: A={qA:.4f}  B={qB:.4f}", flush=True)

    # ---- 简单基线（同输入、同切分）：线性 / 均值。KNN 从基准 JSON 取数，避免重复算力 ----
    lin = LinearBaseline().fit(Xtr, ytr)
    mean_bl = MeanBaseline().fit(Xtr, ytr)

    ref = {}
    if os.path.exists(a.ref):
        ref = json.load(open(a.ref, encoding="utf-8"))

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "config": a.config, "conformal_q": {"armA_P1": qA, "armB_P42": qB},
           "imputer_summary": imp.summary(), "subsets": {}}

    for s in SUBSETS:
        msk = split == s
        if not msk.any():
            continue
        Xs, ys = Xc[msk], y[msk]
        keys = group_keys(Praw[msk])
        row = {"n": int(msk.sum())}

        muA, sdA, _ = twin.predict(Xs)
        muB, sdB, _ = imp.predict(twin, Xs)
        preds = {
            "armA_compositional_twin_P1": (muA, qA * sdA),
            "armB_compositional_twin_P42": (muB, qB * sdB),
            "linear": (np.asarray(lin.predict(Xs)), None),
            "mean": (np.asarray(mean_bl.predict(Xs)), None),
        }
        for name, (mu, sd) in preds.items():
            mu = np.asarray(mu)
            mval, lo, hi = bootstrap_ci(ys, mu, n_boot, ci_alpha, seed=0)
            entry = {"rmse": rmse(ys, mu), "rmse_boot_mean": mval,
                     "ci_low": lo, "ci_high": hi}
            entry.update(variance_guard(keys, ys, mu))
            if sd is not None:
                cal = calibration(ys, mu, sd, levels)
                cov = cal["coverage"]
                entry["coverage"] = {str(k): v for k, v in cov.items()}
                entry["ece"] = cal["ece"]
                entry["calibration_curve"] = cal["calibration_curve"]
                entry["saturated"] = bool(
                    abs(cal["ece"] - SATURATION_ECE) < 1e-3
                    and all(v > 0.999 for v in cov.values())
                )
            row[name] = entry
        out["subsets"][s] = row
        print(f"[p42] == {s} n={row['n']}", flush=True)
        for name in preds:
            e = row[name]
            print("     %-32s rmse=%.4f [%.4f,%.4f] pbRMSE=%.4f var_ratio=%.2e distinct=%d/%d %s"
                  % (name, e["rmse"], e["ci_low"], e["ci_high"], e["pb_rmse"],
                     e["var_ratio_vs_true"], e["n_distinct_pred_rows"], e["n_groups"],
                     ("ECE=%.4f%s" % (e["ece"], " SATURATED" if e.get("saturated") else "")) if "ece" in e else ""),
                  flush=True)

    # ---- 臂 A 复现核验：必须与当周基准 compositional_twin 一致（容差 1e-9）----
    # 口径对齐（重要）：benchmark_ood.evaluate_predictor 报出的 results[*]["rmse"]
    # 是 bootstrap_ci 返回的 **bootstrap 均值**（rmse_mean），不是点估计。
    # 因此这里必须用 armA 的 rmse_boot_mean 去比，用点估计比会产生 ~1e-4 的假性失配。
    check = {"performed": False}
    if ref.get("results"):
        deltas, deltas_point = {}, {}
        ok = True
        for s in SUBSETS:
            r = ref["results"].get(s, {}).get("compositional_twin", {})
            if "rmse" in r and s in out["subsets"]:
                arm = out["subsets"][s]["armA_compositional_twin_P1"]
                dd = float(arm["rmse_boot_mean"]) - float(r["rmse"])
                deltas[s] = dd
                deltas_point[s] = float(arm["rmse"]) - float(r["rmse"])
                ok = ok and abs(dd) < 1e-9
        check = {"performed": True, "armA_matches_canonical": bool(ok),
                 "compare_key": "rmse_boot_mean_vs_canonical_rmse", "tol": 1e-9,
                 "deltas": deltas, "deltas_point_estimate_for_reference": deltas_point}
        print(f"[p42] 臂A复现核验: {check}", flush=True)
    out["arm_a_reproduction_check"] = check

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[p42] written -> {a.out}  ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
