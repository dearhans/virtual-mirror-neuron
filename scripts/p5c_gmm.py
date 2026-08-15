#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-C 新证据槽：GMM 生成式密度门控 epistemic std（补完 P5 三机制面板，W33 §3.10 / §7）。

两臂受控对照（共享同一数据/切分/种子，零混淆）
------------------------------------------------------
    臂 A（P1 对照）   ：canonical `compositional_twin` —— 必须**逐位复现**当周基准（GATE_REPRO）
    臂 R（P5-C GMM）  ：CompositionalTwinP5C_GMM（标准集成 + GMM 密度门控倍率）
    linear / mean     ：简单基线，用于判据上下文对比

判据（W33 §3.10，与路 A / P5-B / P5-A 同，事前锁定，公平比较）
---------------------------------------------------------------
    ood_action 上「P5-C GMM」的 覆盖@0.95 ∈ [0.93,0.97] **且** ECE < 0.08
    → P5-C 成立（GMM 密度门控使 σ 在 OOD 正确转移，区间回名义覆盖、消除饱和）
    且 id/ood_agent/ood_neuro 相对 P1 不退化超阈值（覆盖@0.95 不跌破 0.85 且 ECE 不爆 0.25、不 2× 基线）
    否则记为阴性结果；三机制（A 精炼阴性 / B 双阴性 / C 待测）全失败 → 正当触发 kill-switch 接受 R4。

机制区分：P5-C 与路 A 同族（测试时协变量密度门控 σ），但用 sklearn GaussianMixture 的**平滑参数化
密度** q(x) 取代 kNN 距离 nd(x)。诚实预测：撞路 A 同墙（epistemic 欠缩放是 posterior 性质、非协变量
密度函数，conformal 重标定压平门控）；但协议要求测完三机制方停 P5。

用法：
    python scripts/p5c_gmm.py --ref experiments/20260810-benchmark.json \
        --out experiments/20260814-p5c-gmm.json
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
from model.compositional_p5c_gmm import CompositionalTwinP5C_GMM  # noqa: E402

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
SATURATION_ECE = 0.2125

# --- 事前锁定的判据（W33 §3.10，目标带直接对齐 ood_action coverage@0.95） ---
CRITERION = {
    "subset": "ood_action",
    "cov_level": 0.95,
    "cov_band": [0.93, 0.97],
    "ece_max": 0.08,
    "conformal_alpha": 0.95,
    "degrade_cov_min": 0.85,
    "degrade_ece_max": 0.25,
}
GATE_CAP_RATIO = 50.0
GMM_N_COMPONENTS = 8
GMM_COV_TYPE = "diag"


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
    ap.add_argument("--gmm-n-components", type=int, default=GMM_N_COMPONENTS,
                    help="GMM 成分数（训练流形密度拟合）")
    ap.add_argument("--gmm-cov-type", default=GMM_COV_TYPE,
                    choices=["full", "tied", "diag", "spherical"],
                    help="GMM 协方差类型；diag 最稳（19 维特征防奇异）")
    ap.add_argument("--gate-cap-ratio", type=float, default=GATE_CAP_RATIO,
                    help="密度门控倍率封顶（防饱和，PC-3 宽↔窄矛盾）")
    ap.add_argument("--no-standardize", action="store_true",
                    help="GMM 拟合前不对训练特征做列 z-score 标准化")
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
    print(f"[p5c] data ready {time.time()-t0:.1f}s X={Xc.shape}", flush=True)

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
    print(f"[p5c] P1 twin fit {time.time()-t1:.1f}s", flush=True)

    # ---- 臂 R：P5-C GMM 密度门控孪生 ----
    p5c = CompositionalTwinP5C_GMM(
        Dp=comp_Dp, A_dim=2, hidden=tuple(m["hidden"]),
        n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
        random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
        gmm_n_components=a.gmm_n_components, gmm_cov_type=a.gmm_cov_type,
        gate_cap_ratio=a.gate_cap_ratio, density_standardize=(not a.no_standardize),
    )
    t2 = time.time()
    p5c.fit(Xtr[fidx], ytr[fidx], z_comp=None)
    p5c.fit_density_gate(Xtr[cidx], ytr[cidx])
    gmm_info = {
        "gmm_n_components": a.gmm_n_components, "gmm_cov_type": a.gmm_cov_type,
        "gmm_ok": bool(p5c._gmm_ok), "gate_fitted": bool(p5c._gate_fitted),
        "logq_med": (float(p5c._logq_med_) if p5c._gmm_ok else None),
        "u_ref": float(getattr(p5c, "_u_ref", 0.0)),
        "gate_u_centers": getattr(p5c, "_gate_u", np.array([])).tolist(),
        "gate_ratio": getattr(p5c, "_gate_ratio", np.array([])).tolist(),
        "gate_r_ref": float(getattr(p5c, "_gate_r_ref", 1.0)),
        "cap_ratio": a.gate_cap_ratio,
        "standardize": (not a.no_standardize),
    }
    print(f"[p5c] P5-C twin fit+gate {time.time()-t2:.1f}s "
          f"(gmm_ok={gmm_info['gmm_ok']} cap={a.gate_cap_ratio})", flush=True)

    # ---- 校准 q@0.95（两臂，均在 ID 校准集上用各自 std 估计）----
    muA_c, sdA_c, _ = twin.predict(Xtr[cidx])
    qA = float(np.quantile(_pergene_scores(ytr[cidx], muA_c, sdA_c).ravel(), alpha))
    muR_c, sdR_c, _ = p5c.predict(Xtr[cidx])
    qR = float(np.quantile(_pergene_scores(ytr[cidx], muR_c, sdR_c).ravel(), alpha))
    print(f"[p5c] conformal q@0.95: A(P1)={qA:.4f}  R(P5-C,GMM门控)={qR:.4f}", flush=True)

    lin = LinearBaseline().fit(Xtr, ytr)
    mean_bl = MeanBaseline().fit(Xtr, ytr)
    ref = json.load(open(a.ref, encoding="utf-8")) if os.path.exists(a.ref) else {}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.relpath(a.config, ROOT).replace("\\", "/"),
        "ref": os.path.relpath(a.ref, ROOT).replace("\\", "/"),
        "criterion_prelocked": CRITERION,
        "gate_cap_ratio": a.gate_cap_ratio,
        "gmm_params": {"n_components": a.gmm_n_components, "cov_type": a.gmm_cov_type,
                       "standardize": (not a.no_standardize)},
        "conformal_q": {"armA_P1": qA, "armR_P5C_GMM": qR},
        "gmm_gate_map": gmm_info,
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
        muR, sdR, uR = p5c.predict(Xs)

        # 诊断：门控前（原始集成 epistemic）欠缩放比 = median|resid|/σ_epi
        muR_raw, sdR_raw, uR_raw = p5c._raw_epistemic(Xs)
        raw_score = _pergene_scores(ys, muR_raw, sdR_raw)
        raw_underscale = {
            "median_abs_resid_over_epi": float(np.median(raw_score)),
            "p95_abs_resid_over_epi": float(np.quantile(raw_score, 0.95)),
            "density_novelty_median": float(np.median(uR_raw)),
            "density_novelty_p95": float(np.quantile(uR_raw, 0.95)),
            "gate_multiplier_median": float(np.median(p5c._density_multiplier(uR_raw))),
            "gate_multiplier_p95": float(np.quantile(p5c._density_multiplier(uR_raw), 0.95)),
        }

        preds = {
            "armA_P1_compositional_twin": (muA, qA * sdA),
            "armR_P5C_GMM": (muR, qR * sdR),
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
        row["p5c_underscale_diagnostic"] = raw_underscale
        out["subsets"][s] = row
        print(f"[p5c] == {s} n={row['n']}", flush=True)
        for name in preds:
            e = row[name]
            print("     %-30s rmse=%.4f [%.4f,%.4f] pbRMSE=%.4f var_ratio=%.3e %s"
                  % (name, e["rmse"], e["ci_low"], e["ci_high"], e["pb_rmse"], e["var_ratio_vs_true"],
                     ("ECE=%.4f%s" % (e["ece"], " SAT" if e.get("saturated") else "")) if "ece" in e else ""),
                  flush=True)
        print("     p5c underscale: median|r|/epi=%.3f p95=%.3f u_med=%.3f u_p95=%.3f gate_med=%.3f gate_p95=%.3f"
              % (raw_underscale["median_abs_resid_over_epi"], raw_underscale["p95_abs_resid_over_epi"],
                 raw_underscale["density_novelty_median"], raw_underscale["density_novelty_p95"],
                 raw_underscale["gate_multiplier_median"], raw_underscale["gate_multiplier_p95"]), flush=True)

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

    # ---------- 事前判据裁决（含跨子集退化检查，P5 面板口径） ----------
    verdict = {"evaluated": False}
    if gate and CRITERION["subset"] in out["subsets"]:
        e = out["subsets"][CRITERION["subset"]]["armR_P5C_GMM"]
        a1 = out["subsets"][CRITERION["subset"]]["armA_P1_compositional_twin"]
        cov_l = e["coverage"].get(str(CRITERION["cov_level"]), float("nan"))
        cov_a = a1["coverage"].get(str(CRITERION["cov_level"]), float("nan"))
        lo_b, hi_b = CRITERION["cov_band"]
        cov_in_band = bool(lo_b <= cov_l <= hi_b)
        ece_ok = bool(e["ece"] < CRITERION["ece_max"])
        saturated_fixed = bool(not e.get("saturated"))
        degraded = False
        for s in SUBSETS:
            if s == CRITERION["subset"]:
                continue
            se = out["subsets"][s]["armR_P5C_GMM"]
            sa = out["subsets"][s]["armA_P1_compositional_twin"]
            sc = se["coverage"].get(str(CRITERION["cov_level"]), 1.0)
            if sc < CRITERION["degrade_cov_min"]:
                degraded = True
            if se["ece"] > CRITERION["degrade_ece_max"]:
                degraded = True
            if sa["ece"] > 0 and se["ece"] > sa["ece"] * 2.0:
                degraded = True
        verdict = {
            "evaluated": True,
            "subset": CRITERION["subset"],
            "p5c_cov_at_level": cov_l,
            "p5c_ece": e["ece"],
            "armA_baseline_cov_at_level": cov_a,
            "armA_baseline_ece": a1["ece"],
            "cov_in_band": cov_in_band,
            "ece_below_max": ece_ok,
            "saturation_fixed": saturated_fixed,
            "no_other_subset_degrade": (not degraded),
            "P5C_GMM_ACCEPTED": bool(cov_in_band and ece_ok and saturated_fixed and not degraded),
        }
    out["verdict_p5c"] = verdict

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[p5c] GATE_REPRO={gate}  verdict={json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[p5c] written -> {a.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
