#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-1 新证据槽：localized（分 subset）conformal 校准 —— 解 PC-3（ood_action 区间饱和）。

缺陷（来自 20260812 基准，compositional_twin 已走全局共形 q）：
    ood_action 上 coverage@0.95 = 0.998、ECE = 0.1965  —— 区间严重过宽、恒过覆盖（饱和签名）。
    根因：单一全局 q 在「混合训练校准集（以 ID 为主、std 未被新颖度门控撑大）」上估计，
          应用到 ood_action（std 被 P3-1 门控/封顶放大）时 q·std 过大 → 过覆盖。
          文献监测(2026-08-12)已指明：单标量 `_q` 应退役，按 ood_* 子集分别报覆盖/ECE。

破法（TRIZ 空间/局部分离）：
    把全局 q 换成 **per-subset localized q** —— 每个测试子集用自己的校准行估计分位数：
    校准行按「新颖度 nd_relative」匹配到最近测试子集的新颖度中位数，子集 s 的 q_s
    仅由匹配到 s 的校准行估计。ood_action（高新颖度）因此得到「为自己的放大 std 定标」
    的、更紧的 q，区间回到名义覆盖，ECE 下降。

本脚本自包含、只读式验证（与 p43/p44 同口径）：
    - 用与 canonical 完全相同的 P1 组合孪生（fit 于 80% train，复现闸门对齐）；
    - 在 20% train 校准集上算全局 q（基线）与分 subset localized q；
    - 在 4 个子集成对算（global  vs  localized）的 RMSE[CI] / cov@0.95 / ECE / var_ratio；
    - 事前锁定判据：ood_action 上 localized 的 cov@0.95 ∈ [0.93,0.97] 且 ECE < 0.08。

纯校准层改动：不改 φ 结构、不引入 P4-3 收缩，归因干净。

用法：
    python scripts/p41_localized_conformal.py --ref experiments/20260812-benchmark.json \
        --out experiments/20260812-p41-localized-conformal.json
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

SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
SATURATION_ECE = 0.2125

# --- 事前锁定判据（W33 §7 #1） ---
CRITERION = {
    "subset": "ood_action",
    "cov_level": 0.95,
    "cov_band": [0.93, 0.97],
    "ece_max": 0.08,
}

# P4-1 目标名义覆盖级（与判据 cov@0.95 对齐）：校准 q 在 alpha=0.95 上估计
LOCALIZED_ALPHA = 0.95
# 同时保留 canonical 全局 q 在 0.9 上的值，用于与 20260812 基准逐位对照
GLOBAL_ALPHA_REF = 0.9


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
        "collapsed": bool(pv / max(tv, 1e-30) < 1e-3),
        "pb_rmse": float(np.sqrt(np.mean((pbt - pbp) ** 2))),
    }


def _pergene_scores(y, mu, std):
    """逐 (样本,基因) 标准化残差分数（与 calibration() 口径一致，避免 P3-1b 的 √K 偏置）。"""
    std_col = std.reshape(-1, 1) if std.ndim == 1 else std
    return np.abs(y - mu) / np.clip(std_col, 1e-9, None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/benchmark_ood_norman_canonical.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True, help="当周基准 JSON（复现核验 + 基线取数）")
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
    print(f"[p41] data ready {time.time()-t0:.1f}s X={Xc.shape}", flush=True)

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
    print(f"[p41] P1 twin fit {time.time()-t1:.1f}s", flush=True)

    # —— 校准集上的分数 + 新颖度 ——
    mu_c, std_c, nd_c = twin.predict(Xtr[cidx])
    scores_c = _pergene_scores(ytr[cidx], mu_c, std_c).ravel()  # (n*K,)

    # 全局 q（基线）：canonical 用 0.9；为对齐判据 cov@0.95 额外算 0.95
    q_global_09 = float(np.quantile(scores_c, GLOBAL_ALPHA_REF))
    q_global_95 = float(np.quantile(scores_c, LOCALIZED_ALPHA))

    # —— localized q：每个测试子集用「匹配到自己新颖度中位数的校准行」估计 ——
    # 先算各测试子集的新颖度中位数（用全量测试行，避免小校准集方差）
    nd_by_subset = {}
    for s in SUBSETS:
        msk = split == s
        if msk.any():
            _, _, nd_s = twin.predict(Xc[msk])
            nd_by_subset[s] = nd_s
    med_arr = np.array([float(np.median(nd_by_subset[s])) for s in SUBSETS])
    # 每条校准行分配到最近子集中位数（逐样本）
    assign = np.argmin(np.abs(nd_c[:, None] - med_arr[None, :]), axis=1)
    # scores_c 是逐 (样本,基因) 展平 (n_calib*K,)，每条基因继承其样本的子集归属
    assign_exp = np.repeat(assign, ytr[cidx].shape[1])
    q_local = {}
    for ji, s in enumerate(SUBSETS):
        mask = assign_exp == ji
        q_local[s] = float(np.quantile(scores_c[mask], LOCALIZED_ALPHA)) if mask.sum() > 0 else q_global_95
    print(f"[p41] q_global@0.9={q_global_09:.4f} q_global@0.95={q_global_95:.4f}", flush=True)
    print(f"[p41] q_local=" + ", ".join(f"{s}={q_local[s]:.4f}" for s in SUBSETS), flush=True)

    lin = LinearBaseline().fit(Xtr, ytr)
    mean_bl = MeanBaseline().fit(Xtr, ytr)
    ref = json.load(open(a.ref, encoding="utf-8")) if os.path.exists(a.ref) else {}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.relpath(a.config, ROOT).replace("\\", "/"),
        "ref": os.path.relpath(a.ref, ROOT).replace("\\", "/"),
        "criterion_prelocked": CRITERION,
        "conformal_q": {
            "global_0.9": q_global_09, "global_0.95": q_global_95,
            "localized_0.95": q_local,
        },
        "subsets": {},
    }

    cov_lv = CRITERION["cov_level"]
    for s in SUBSETS:
        msk = split == s
        if not msk.any():
            continue
        Xs, ys = Xc[msk], y[msk]
        keys = group_keys(Praw[msk])
        mu, std, nd = twin.predict(Xs)
        sd_global = q_global_95 * std
        sd_local = q_local[s] * std
        row = {"n": int(msk.sum())}

        # 基线（全局 q）臂
        mval_g, lo_g, hi_g = bootstrap_ci(ys, mu, n_boot, ci_alpha, seed=0)
        eg = {"rmse": rmse(ys, mu), "rmse_boot_mean": mval_g, "ci_low": lo_g, "ci_high": hi_g}
        eg.update(variance_guard(keys, ys, mu))
        cal_g = calibration(ys, mu, sd_global, levels)
        eg["coverage"] = {str(k): v for k, v in cal_g["coverage"].items()}
        eg["ece"] = cal_g["ece"]
        eg["saturated"] = bool(abs(cal_g["ece"] - SATURATION_ECE) < 1e-3 and all(v > 0.999 for v in cal_g["coverage"].values()))
        row["global_q_baseline"] = eg

        # P4-1（localized q）臂
        mval_l, lo_l, hi_l = bootstrap_ci(ys, mu, n_boot, ci_alpha, seed=0)
        el = {"rmse": rmse(ys, mu), "rmse_boot_mean": mval_l, "ci_low": lo_l, "ci_high": hi_l}
        el.update(variance_guard(keys, ys, mu))
        cal_l = calibration(ys, mu, sd_local, levels)
        el["coverage"] = {str(k): v for k, v in cal_l["coverage"].items()}
        el["ece"] = cal_l["ece"]
        el["saturated"] = bool(abs(cal_l["ece"] - SATURATION_ECE) < 1e-3 and all(v > 0.999 for v in cal_l["coverage"].values()))
        row["p41_localized"] = el

        # 简单基线（点估计对照）
        for nm, bl in (("linear", lin), ("mean", mean_bl)):
            bmu = np.asarray(bl.predict(Xs))
            bm, blo, bhi = bootstrap_ci(ys, bmu, n_boot, ci_alpha, seed=0)
            row[nm] = {"rmse": rmse(ys, bmu), "rmse_boot_mean": bm, "ci_low": blo, "ci_high": bhi}
            row[nm].update(variance_guard(keys, ys, bmu))

        out["subsets"][s] = row
        print(f"[p41] == {s} n={row['n']}", flush=True)
        print("     global  ECE=%.4f cov@0.95=%.3f" % (eg["ece"], eg["coverage"].get("0.95", float("nan"))), flush=True)
        print("     local   ECE=%.4f cov@0.95=%.3f" % (el["ece"], el["coverage"].get("0.95", float("nan"))), flush=True)

    # ---------- 复现核验（污染即中止出结论） ----------
    repro = {"arm_global_vs_canonical": {"performed": False}}
    if ref.get("results"):
        deltas, ok = {}, True
        for s in SUBSETS:
            r = ref["results"].get(s, {}).get("compositional_twin", {})
            if "rmse" in r and s in out["subsets"]:
                dd = float(out["subsets"][s]["global_q_baseline"]["rmse_boot_mean"]) - float(r["rmse"])
                deltas[s] = dd
                ok = ok and abs(dd) < 1e-9
        repro["arm_global_vs_canonical"] = {"performed": True, "match": bool(ok),
                                             "compare_key": "rmse_boot_mean_vs_canonical_rmse",
                                             "tol": 1e-9, "deltas": deltas}
    gate = bool(repro["arm_global_vs_canonical"].get("match"))
    out["reproduction_check"] = repro
    out["GATE_REPRO"] = gate

    # ---------- 事前判据裁决 ----------
    verdict = {"evaluated": False}
    if gate and CRITERION["subset"] in out["subsets"]:
        e = out["subsets"][CRITERION["subset"]]["p41_localized"]
        g = out["subsets"][CRITERION["subset"]]["global_q_baseline"]
        cov_l = e["coverage"].get(str(cov_lv), float("nan"))
        cov_g = g["coverage"].get(str(cov_lv), float("nan"))
        lo_b, hi_b = CRITERION["cov_band"]
        cov_in_band = bool(lo_b <= cov_l <= hi_b)
        ece_ok = bool(e["ece"] < CRITERION["ece_max"])
        verdict = {
            "evaluated": True,
            "subset": CRITERION["subset"],
            "localized_cov_at_level": cov_l,
            "localized_ece": e["ece"],
            "global_baseline_cov_at_level": cov_g,
            "global_baseline_ece": g["ece"],
            "cov_in_band": cov_in_band,
            "ece_below_max": ece_ok,
            "P41_ACCEPTED": bool(cov_in_band and ece_ok),
        }
    out["verdict_p41"] = verdict

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[p41] GATE_REPRO={gate}  verdict={json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[p41] written -> {a.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
