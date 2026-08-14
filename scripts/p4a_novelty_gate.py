#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路 A 新证据槽：novelty-gated epistemic std（攻克 PC-3 epistemic 欠缩放根因，W33 §7 P0）。

两臂受控对照（共享同一数据/切分/种子，零混淆）
------------------------------------------------------
    臂 A（P1 对照）   ：canonical `compositional_twin` —— 必须**逐位复现**当周基准（GATE_REPRO）
    臂 R（路 A）      ：NoveltyGatedCompositionalTwin（校准拟合欠缩放比→新颖度门控倍率）
    linear / mean     ：简单基线，用于判据上下文对比

判据（W33 §7 路 A，事前锁定，不得事后修改）
------------------------------------------------------
    ood_action 上「路 A」的 覆盖@0.95 ∈ [0.93, 0.97] **且** ECE < 0.08
    → 路 A 成立（epistemic std 被新颖度撑大，区间回到名义覆盖、消除饱和、且不因过宽锁 1.0）
    否则记为阴性结果，重新审视 PC-3 是否确为 epistemic 根因。

机制区分（对 P4-1 阴性诚实）：路 A 改 std 估计器本体（predict 时按测试点自身新颖度撑大），
不重标定校准层 q；新颖度是 kNN 距离，ID/OOD 测试点皆可得，故不依赖「ID 校准覆盖 OOD」——
这正是 P4-1（分 subset q 退化回全局 41.26）结构性失效所缺失的前提。

用法：
    python scripts/p4a_novelty_gate.py --ref experiments/20260810-benchmark.json \
        --out experiments/20260814-p4a-novelty-gate.json
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
# 自包含依赖：sklearn 等置于项目本地 .pylibs（托管 Python 隔离环境）
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))

import yaml  # noqa: E402
from benchmark_ood import (  # noqa: E402
    load_or_generate, norman_to_compositional_X, bootstrap_ci, calibration, rmse,
    LinearBaseline, MeanBaseline,
)
from model.compositional import CompositionalTwin  # noqa: E402
from model.compositional_p4a import NoveltyGatedCompositionalTwin  # noqa: E402

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
SATURATION_ECE = 0.2125

# --- 事前锁定的判据（W33 §7 路 A，目标带直接对齐 ood_action coverage@0.95） ---
CRITERION = {
    "subset": "ood_action",
    "cov_level": 0.95,
    "cov_band": [0.93, 0.97],
    "ece_max": 0.08,
    # conformal q 估计层级：与判据 cov@0.95 对齐（P4-1 同口径用 0.95）
    "conformal_alpha": 0.95,
}
# 路 A 门控建模选择（校准拟合，非测试拟合；详见结案文档诚实边界）
GATE_CAP_RATIO = 50.0


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
    """逐 (样本,基因) 绝对残差分数（与 calibration() 口径一致）。"""
    std_col = std.reshape(-1, 1) if std.ndim == 1 else std
    return np.abs(y - mu) / np.clip(std_col, 1e-9, None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True, help="当周基准 JSON（臂 A 复现核验 + 基线取数）")
    ap.add_argument("--gate-kind", default="empirical",
                    choices=["empirical", "heteroscedastic"],
                    help="路 A 门控机制：empirical=novelty 门控；heteroscedastic=模型自身 "
                         "epistemic std 异方差重标定（诊断显示欠缩放与 σ_epi 负相关）")
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
    print(f"[p4a] data ready {time.time()-t0:.1f}s X={Xc.shape}", flush=True)

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
    print(f"[p4a] P1 twin fit {time.time()-t1:.1f}s", flush=True)

    # ---- 臂 R：路 A novelty-gated 孪生 ----
    p4a = NoveltyGatedCompositionalTwin(
        Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]),
        n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
        random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
        gate_cap_ratio=GATE_CAP_RATIO,
    )
    t2 = time.time()
    p4a.fit(Xtr[fidx], ytr[fidx], z_comp=None)
    p4a.fit_novelty_gate(Xtr[cidx], ytr[cidx], kind=a.gate_kind)
    _gk = a.gate_kind
    _gparams = (f"beta={p4a._het_beta:.3f} sref={p4a._het_sref:.4f}" if _gk == "heteroscedastic"
                else f"nd_ref={p4a._nd_ref:.3f} r_ref={getattr(p4a, '_gate_r_ref', float('nan')):.3f}")
    print(f"[p4a] route-A twin fit+gate {time.time()-t2:.1f}s ({_gk}: {_gparams} cap={GATE_CAP_RATIO})", flush=True)

    # ---- 校准 q@0.95（两臂，均在 ID 校准集上用各自 std 估计）----
    muA_c, sdA_c, _ = twin.predict(Xtr[cidx])
    qA = float(np.quantile(_pergene_scores(ytr[cidx], muA_c, sdA_c).ravel(), alpha))
    muR_c, sdR_c, _ = p4a.predict(Xtr[cidx])
    qR = float(np.quantile(_pergene_scores(ytr[cidx], muR_c, sdR_c).ravel(), alpha))
    print(f"[p4a] conformal q@0.95: A(P1,P3-1门控)={qA:.4f}  R(路A,novelty门控)={qR:.4f}", flush=True)

    lin = LinearBaseline().fit(Xtr, ytr)
    mean_bl = MeanBaseline().fit(Xtr, ytr)
    ref = json.load(open(a.ref, encoding="utf-8")) if os.path.exists(a.ref) else {}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.relpath(a.config, ROOT).replace("\\", "/"),
        "ref": os.path.relpath(a.ref, ROOT).replace("\\", "/"),
        "criterion_prelocked": CRITERION,
        "gate_kind": a.gate_kind,
        "gate_cap_ratio": GATE_CAP_RATIO,
        "conformal_q": {"armA_P1_P3gate": qA, "armR_routeA": qR},
        "gate_map": {
            "gate_kind": a.gate_kind,
            "nd_ref": p4a._nd_ref,
            "r_ref": getattr(p4a, "_gate_r_ref", None),
            "nd_centers": getattr(p4a, "_gate_nd", np.array([])).tolist() if hasattr(p4a, "_gate_nd") else [],
            "ratio": getattr(p4a, "_gate_ratio", np.array([])).tolist() if hasattr(p4a, "_gate_ratio") else [],
            "het_beta": getattr(p4a, "_het_beta", None),
            "het_sref": getattr(p4a, "_het_sref", None),
            "cap_ratio": GATE_CAP_RATIO,
        },
        "subsets": {},
    }

    for s in SUBSETS:
        msk = split == s
        if not msk.any():
            continue
        Xs, ys = Xc[msk], y[msk]
        keys = group_keys(Praw[msk])
        row = {"n": int(msk.sum())}

        muA, sdA, ndA = twin.predict(Xs)
        muR, sdR, ndR = p4a.predict(Xs)

        # 诊断：门控前（原始集成 epistemic）欠缩放比 = median|resid|/σ_epi
        muR_raw, sdR_raw, ndR_raw = p4a._raw_epistemic(Xs)
        raw_score = _pergene_scores(ys, muR_raw, sdR_raw)
        raw_underscale = {
            "median_abs_resid_over_epi": float(np.median(raw_score)),
            "p95_abs_resid_over_epi": float(np.quantile(raw_score, 0.95)),
            "nd_relative_median": float(np.median(ndR_raw)),
            "nd_relative_max": float(np.max(ndR_raw)),
            "gate_multiplier_median": float(np.median(p4a._novelty_multiplier(ndR_raw))),
        }

        preds = {
            "armA_P1_compositional_twin": (muA, qA * sdA),
            "armR_routeA": (muR, qR * sdR),
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
        row["routeA_underscale_diagnostic"] = raw_underscale
        out["subsets"][s] = row
        print(f"[p4a] == {s} n={row['n']}", flush=True)
        for name in preds:
            e = row[name]
            print("     %-30s rmse=%.4f [%.4f,%.4f] pbRMSE=%.4f var_ratio=%.3e %s"
                  % (name, e["rmse"], e["ci_low"], e["ci_high"], e["pb_rmse"], e["var_ratio_vs_true"],
                     ("ECE=%.4f%s" % (e["ece"], " SAT" if e.get("saturated") else "")) if "ece" in e else ""),
                  flush=True)
        print("     routeA underscale: median|r|/epi=%.3f p95=%.3f nd_med=%.3f nd_max=%.3f gate_med=%.3f"
              % (raw_underscale["median_abs_resid_over_epi"], raw_underscale["p95_abs_resid_over_epi"],
                 raw_underscale["nd_relative_median"], raw_underscale["nd_relative_max"],
                 raw_underscale["gate_multiplier_median"]), flush=True)

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
        e = out["subsets"][CRITERION["subset"]]["armR_routeA"]
        a1 = out["subsets"][CRITERION["subset"]]["armA_P1_compositional_twin"]
        cov_l = e["coverage"].get(str(CRITERION["cov_level"]), float("nan"))
        cov_a = a1["coverage"].get(str(CRITERION["cov_level"]), float("nan"))
        lo_b, hi_b = CRITERION["cov_band"]
        cov_in_band = bool(lo_b <= cov_l <= hi_b)
        ece_ok = bool(e["ece"] < CRITERION["ece_max"])
        saturated_fixed = bool(not e.get("saturated"))
        verdict = {
            "evaluated": True,
            "subset": CRITERION["subset"],
            "routeA_cov_at_level": cov_l,
            "routeA_ece": e["ece"],
            "armA_baseline_cov_at_level": cov_a,
            "armA_baseline_ece": a1["ece"],
            "cov_in_band": cov_in_band,
            "ece_below_max": ece_ok,
            "saturation_fixed": saturated_fixed,
            "P4A_ACCEPTED": bool(cov_in_band and ece_ok and saturated_fixed),
        }
    out["verdict_p4a"] = verdict

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[p4a] GATE_REPRO={gate}  verdict={json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[p4a] written -> {a.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
