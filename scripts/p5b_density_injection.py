#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-B 新证据槽：密度感知多样性注入（攻克 PC-3 epistemic 欠缩放根因，W33 §3.10）。

两臂受控对照（共享同一数据/切分/种子，零混淆）
------------------------------------------------------
    臂 A（P1 对照）   ：canonical `compositional_twin` —— 必须**逐位复现**当周基准（GATE_REPRO）
    臂 R（P5-B）      ：CompositionalTwinP5B（训练期按 kNN 连续稀疏度 w(x) 注入逐成员区域偏好多样性）
    linear / mean     ：简单基线，用于判据上下文对比

判据（W33 §3.10，与路 A 同，公平比较）
------------------------------------------------------
    ood_action 上「P5-B」的 覆盖@0.95 ∈ [0.93, 0.97] **且** ECE < 0.08
    → P5-B 成立（构造期多样性注入使 σ 在 OOD 正确转移，区间回到名义覆盖、消除饱和）
    且 id/ood_agent/ood_neuro 相对 P1 不退化超阈值（覆盖@0.95 不跌破 0.85 且 ECE 不爆）
    否则记为阴性结果，激活 kill-switch（三 spike 全失败则停 P5、接受 R4）。

机制区分：路 A 改 σ *估计器输出*（测试时乘门控 g）；P5-B 改 σ *构造*（训练期让集成成员在稀疏区
天然分散）。前者已证伪（新颖度二值 / σ_epi 弱负相关）；后者是 TRIZ #1+#15 落地——σ 成为训练密度
的连续函数，由构造满足 ∂σ/∂τ>0。

用法：
    python scripts/p5b_density_injection.py --ref experiments/20260810-benchmark.json \
        --out experiments/20260814-p5b-density-injection.json
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
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))

import yaml  # noqa: E402
from benchmark_ood import (  # noqa: E402
    load_or_generate, norman_to_compositional_X, bootstrap_ci, calibration, rmse,
    LinearBaseline, MeanBaseline,
)
from model.compositional import CompositionalTwin  # noqa: E402
from model.compositional_p5b import CompositionalTwinP5B  # noqa: E402

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
SATURATION_ECE = 0.2125
LAMBDAS = [0.01, 0.05, 0.1]

CRITERION = {
    "subset": "ood_action",
    "cov_level": 0.95,
    "cov_band": [0.93, 0.97],
    "ece_max": 0.08,
    "conformal_alpha": 0.95,
    # 退化护栏：其他子集覆盖@0.95 不得跌破、ECE 不得爆
    "degrade_cov_min": 0.85,
    "degrade_ece_max": 0.25,
}


def group_keys(P: np.ndarray) -> np.ndarray:
    return np.asarray([
        "ctrl" if (r > .5).sum() == 0 else "+".join(map(str, sorted(np.flatnonzero(r > .5).tolist())))
        for r in P
    ])


def variance_guard(keys, y_true, y_pred):
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


def _pergene_scores(y, mu, std):
    std_col = std.reshape(-1, 1) if std.ndim == 1 else std
    return np.abs(y - mu) / np.clip(std_col, 1e-9, None)


def evaluate_subset(name, Xs, ys, keys, models, q_map, levels, n_boot, ci_alpha, lin, mean_bl):
    """对单子集评估多模型（臂 A / 各 λ 的 P5-B / linear / mean）。"""
    row = {"n": int(len(Xs))}
    preds = {}
    for mname, (model, q) in models.items():
        mu, sd, nd = model.predict(Xs)
        preds[mname] = (mu, sd, q)
    # linear / mean 基线
    preds["linear"] = (np.asarray(lin.predict(Xs)), None, None)
    preds["mean"] = (np.asarray(mean_bl.predict(Xs)), None, None)

    for mname, (mu, sd, q) in preds.items():
        mu = np.asarray(mu)
        mval, lo, hi = bootstrap_ci(ys, mu, n_boot, ci_alpha, seed=0)
        entry = {"rmse": rmse(ys, mu), "rmse_boot_mean": mval, "ci_low": lo, "ci_high": hi}
        entry.update(variance_guard(keys, ys, mu))
        if sd is not None and q is not None:
            cal = calibration(ys, mu, sd, levels)
            cov = cal["coverage"]
            entry["coverage"] = {str(k): v for k, v in cov.items()}
            entry["ece"] = cal["ece"]
            entry["calibration_curve"] = cal["calibration_curve"]
            entry["saturated"] = bool(
                abs(cal["ece"] - SATURATION_ECE) < 1e-3 and all(v > 0.999 for v in cov.values()))
        row[mname] = entry
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True)
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
    print(f"[p5b] data ready {time.time()-t0:.1f}s X={Xc.shape}", flush=True)

    Xtr, ytr = Xc[tr], y[tr]
    rng = np.random.default_rng(int(m.get("random_state", 0)))
    idx = np.arange(len(Xtr))
    rng.shuffle(idx)
    ncal = max(1, int(len(idx) * float(m.get("conformal_calib_frac", 0.2))))
    cidx, fidx = idx[:ncal], idx[ncal:]
    comp_Dp = (Xc.shape[1] - 1 - 2) // 2
    alpha = CRITERION["conformal_alpha"]

    # ---- 臂 A：P1 canonical 组合孪生 ----
    twin = CompositionalTwin(
        Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]),
        n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
        random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
    )
    t1 = time.time()
    twin.fit(Xtr[fidx], ytr[fidx], z_comp=None)
    print(f"[p5b] P1 twin fit {time.time()-t1:.1f}s", flush=True)

    # P1 校准 q + 原始集成 epistemic（用于多样性注入诊断）
    muA_c, sdA_c, _ = twin.predict(Xtr[cidx])
    qA = float(np.quantile(_pergene_scores(ytr[cidx], muA_c, sdA_c).ravel(), alpha))

    lin = LinearBaseline().fit(Xtr, ytr)
    mean_bl = MeanBaseline().fit(Xtr, ytr)
    ref = json.load(open(a.ref, encoding="utf-8")) if os.path.exists(a.ref) else {}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.relpath(a.config, ROOT).replace("\\", "/"),
        "ref": os.path.relpath(a.ref, ROOT).replace("\\", "/"),
        "criterion_prelocked": CRITERION,
        "lambdas": LAMBDAS,
        "conformal_q": {"armA_P1": qA},
        "lambda_sweeps": {},
        "reproduction_check": {"armA_vs_canonical": {"performed": False}},
        "GATE_REPRO": False,
        "verdict_p5b": {"evaluated": False},
    }

    best = None  # 记录满足判据的最优 λ
    for lam in LAMBDAS:
        p5b = CompositionalTwinP5B(
            Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]),
            n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
            random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
            density_lambda=lam, density_proj_dim=3,
        )
        t2 = time.time()
        p5b.fit(Xtr[fidx], ytr[fidx], z_comp=None)
        print(f"[p5b] P5-B(λ={lam}) fit {time.time()-t2:.1f}s", flush=True)

        muR_c, sdR_c, _ = p5b.predict(Xtr[cidx])
        qR = float(np.quantile(_pergene_scores(ytr[cidx], muR_c, sdR_c).ravel(), alpha))
        out["conformal_q"][f"armR_P5B_lambda{lam}"] = qR

        models = {
            "armA_P1_compositional_twin": (twin, qA),
            f"armR_P5B_lambda{lam}": (p5b, qR),
        }
        sweep = {"density_lambda": lam, "q_at_level": qR, "subsets": {}}
        for s in SUBSETS:
            msk = split == s
            if not msk.any():
                continue
            Xs, ys = Xc[msk], y[msk]
            keys = group_keys(Praw[msk])
            row = evaluate_subset(s, Xs, ys, keys, models, None, levels, n_boot, ci_alpha, lin, mean_bl)

            # —— 多样性注入诊断：P5-B 原始 σ_epi vs P1 原始 σ_epi 的欠缩放比 ——
            muR_raw, sdR_raw, ndR_raw = p5b.predict(Xs)
            muA_raw, sdA_raw, ndA_raw = twin.predict(Xs)
            scR = _pergene_scores(ys, muR_raw, sdR_raw)
            scA = _pergene_scores(ys, muA_raw, sdA_raw)
            row["diversity_injection_diagnostic"] = {
                "sigma_epi_median_P5B": float(np.median(sdR_raw)),
                "sigma_epi_median_P1": float(np.median(sdA_raw)),
                "sigma_epi_ratio_P5B_over_P1": float(np.median(sdR_raw) / max(np.median(sdA_raw), 1e-9)),
                "underscale_median_P5B": float(np.median(scR)),
                "underscale_median_P1": float(np.median(scA)),
                "underscale_p95_P5B": float(np.quantile(scR, 0.95)),
                "underscale_p95_P1": float(np.quantile(scA, 0.95)),
            }
            sweep["subsets"][s] = row
            print(f"[p5b] == λ={lam} {s} n={row['n']}", flush=True)
            for nm in ["armA_P1_compositional_twin", f"armR_P5B_lambda{lam}"]:
                e = row[nm]
                print("     %-32s rmse=%.4f [%.4f,%.4f] pbRMSE=%.4f var_ratio=%.3e %s"
                      % (nm, e["rmse"], e["ci_low"], e["ci_high"], e["pb_rmse"], e["var_ratio_vs_true"],
                         ("ECE=%.4f%s" % (e["ece"], " SAT" if e.get("saturated") else "")) if "ece" in e else ""),
                      flush=True)
            print("     diversity: σ_epi_ratio(P5B/P1)=%.3f underscale_med P5B=%.3f P1=%.3f p95 P5B=%.3f P1=%.3f"
                  % (row["diversity_injection_diagnostic"]["sigma_epi_ratio_P5B_over_P1"],
                     row["diversity_injection_diagnostic"]["underscale_median_P5B"],
                     row["diversity_injection_diagnostic"]["underscale_median_P1"],
                     row["diversity_injection_diagnostic"]["underscale_p95_P5B"],
                     row["diversity_injection_diagnostic"]["underscale_p95_P1"]), flush=True)

        out["lambda_sweeps"][str(lam)] = sweep

        # 单 λ 判据检查
        if CRITERION["subset"] in sweep["subsets"]:
            e = sweep["subsets"][CRITERION["subset"]][f"armR_P5B_lambda{lam}"]
            a1 = sweep["subsets"][CRITERION["subset"]]["armA_P1_compositional_twin"]
            cov_l = e["coverage"].get(str(CRITERION["cov_level"]), float("nan"))
            cov_a = a1["coverage"].get(str(CRITERION["cov_level"]), float("nan"))
            lo_b, hi_b = CRITERION["cov_band"]
            cov_in = bool(lo_b <= cov_l <= hi_b)
            ece_ok = bool(e["ece"] < CRITERION["ece_max"])
            sat_fixed = bool(not e.get("saturated"))
            # 退化护栏
            degraded = False
            for s in SUBSETS:
                if s == CRITERION["subset"]:
                    continue
                se = sweep["subsets"][s][f"armR_P5B_lambda{lam}"]
                sa = sweep["subsets"][s]["armA_P1_compositional_twin"]
                sc = se["coverage"].get(str(CRITERION["cov_level"]), 1.0)
                if sc < CRITERION["degrade_cov_min"]:
                    degraded = True
                if se["ece"] > CRITERION["degrade_ece_max"]:
                    degraded = True
                if sa["ece"] > 0 and se["ece"] > sa["ece"] * 2.0:
                    degraded = True
            lam_ok = bool(cov_in and ece_ok and sat_fixed and not degraded)
            if lam_ok and best is None:
                best = {"lambda": lam, "cov": cov_l, "ece": e["ece"], "armA_cov": cov_a, "armA_ece": a1["ece"]}

    # ---------- 复现核验（污染即中止） ----------
    if ref.get("results"):
        deltas, ok = {}, True
        for s in SUBSETS:
            r = ref["results"].get(s, {}).get("compositional_twin", {})
            if "rmse" in r and s in out["lambda_sweeps"][str(LAMBDAS[0])]["subsets"]:
                dd = float(out["lambda_sweeps"][str(LAMBDAS[0])]["subsets"][s]["armA_P1_compositional_twin"]["rmse_boot_mean"]) - float(r["rmse"])
                deltas[s] = dd
                ok = ok and abs(dd) < 1e-9
        out["reproduction_check"]["armA_vs_canonical"] = {
            "performed": True, "match": bool(ok),
            "compare_key": "rmse_boot_mean_vs_canonical_rmse", "tol": 1e-9, "deltas": deltas}
    out["GATE_REPRO"] = bool(out["reproduction_check"]["armA_vs_canonical"].get("match"))

    # ---------- 事前判据裁决 ----------
    out["verdict_p5b"] = {
        "evaluated": True,
        "criterion_subset": CRITERION["subset"],
        "gate_repro": out["GATE_REPRO"],
        "best_lambda": (best["lambda"] if best else None),
        "best_lambda_cov_at_level": (best["cov"] if best else None),
        "best_lambda_ece": (best["ece"] if best else None),
        "P5B_ACCEPTED": bool(out["GATE_REPRO"] and best is not None),
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[p5b] GATE_REPRO={out['GATE_REPRO']}  verdict={json.dumps(out['verdict_p5b'], ensure_ascii=False)}", flush=True)
    print(f"[p5b] written -> {a.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
