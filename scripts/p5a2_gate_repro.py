#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-A2 复赛 P0 · 任务 #4 GATE_REPRO 固化 —— 正式回归守卫流水线（GOAI 真实数据）

对齐初赛「连续零增量」冻结纪律：把当前运行的**四栏指标**与已冻结基线对比，
Δ 全部 ≤ 容差 → 守卫 PASS（ZERO-DELTA，可复现）；任一超阈 → REGRESSION 告警。

四栏（报告 §五#4 口径）：
  L1 覆盖（校准栏） : localized conformal cov@0.95，中心=P1，宽=fglex σ（分层，见 #3）
  L2 ECE（校准栏）  : canonical 多档 |cov(L)-L| 均值
  L3 单调性（H1）   : corr(σ_tot 均值, τ)，τ = 样本到训练流形 NN 距离（全量 id 口径）
  L4 RMSE（点预测栏）: P1 = GoaiCompositionalTwin → goai_metrics.evaluate() 加权总分

冻结基线（已发布 artifacts，同口径可比）：
  - 覆盖/ECE     : experiments/20260819-p5a2-layered.json        （localized_conformal）
  - 单调性       : experiments/20260819-p5a2-fglex-full.json      （localized mono；subsample 口径，参考）
  - RMSE 加权分   : experiments/20260819-p5a2-precision-vs-p1.json （P1 = 0.4271）

容差（零增量纪律）：
  cov@0.95 ±0.01 / ECE ±0.01 / mono ±0.05 / weighted ±0.02

运行（全量 id 训练，约 4-5 分钟）：
  .venv_p5a2/Scripts/python.exe scripts/p5a2_gate_repro.py

产出：experiments/20260821-p5a2-gate-repro.json
诚实边界：mono 冻结基线为 fglex-full 的 subsample 口径，本次守卫为全量 id 口径，
Δ 差异标注口径、不判 REGRESSION；同时把本次全量 mono 存为新冻结值供后续守卫。
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, "code")
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from data.goai_loader import build_dataset
from data.goai_compound_features import make_functionalgroup_block
from model.compositional_p5a2_pergene import PerGeneBayesTwin
from goai_compositional_twin import GoaiCompositionalTwin
from goai_metrics import evaluate, MODULE_WEIGHTS

CACHE = "data/processed/goai_cache.npz"
LAYERED = "experiments/20260819-p5a2-layered.json"
FGLEX_FULL = "experiments/20260819-p5a2-fglex-full.json"
PRECISION = "experiments/20260819-p5a2-precision-vs-p1.json"
OUT = "experiments/20260821-p5a2-gate-repro.json"

BAND = (0.93, 0.97)
ECE_BAR = 0.08
CANON = [0.5, 0.8, 0.9, 0.95]
SUBSETS = ["id", "ood_action", "ood_agent", "ood_s3", "ood_time"]
OOD = ["ood_action", "ood_agent", "ood_s3", "ood_time"]
TOL = {"cov095": 0.01, "ece": 0.01, "mono": 0.05, "weighted": 0.02}


def build_fglex_X(d):
    X = d["X"]
    names = list(d["compound_names"])
    Dc = len(names)
    rownz = X[:, :Dc].argmax(1)
    per_sample = [names[i] for i in rownz]
    fg, _ = make_functionalgroup_block(per_sample)          # (N, 14)
    X_rest = X[:, Dc:]
    return np.ascontiguousarray(np.hstack([fg, X_rest]).astype(np.float64)), fg.shape[1]


def impute_id_median(Y, id_mask):
    gm = np.nanmedian(Y[id_mask], axis=0)
    gm = np.where(np.isnan(gm), 0.0, gm)
    Ys = Y.copy()
    m = np.isnan(Ys)
    Ys[m] = gm[np.nonzero(m)[1]]
    return Ys


def zc(level):
    return float(norm.ppf(0.5 + level / 2.0))


def coverage_at(y, mu, s, level):
    q = zc(level)
    return float(((y >= mu - q * s) & (y <= mu + q * s)).mean())


def ece_canon(y, mu, s):
    cov = [coverage_at(y, mu, s, L) for L in CANON]
    return float(np.mean([abs(c - L) for c, L in zip(cov, CANON)]))


def _load_json(p):
    """健壮读取实验 JSON（部分历史产物为 GBK 编码，兼容 utf-8/gbk/latin-1）。"""
    last = None
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            last = e
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-genes", type=int, default=5243)
    ap.add_argument("--n-iter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    d = build_dataset(cache_path=CACHE)
    X = d["X"]; Y_delta = d["Y_delta"]; Y_abs = d["Y_abs"]; Y_ctrl = d["Y_ctrl"]
    split = d["split"].astype(str)
    Kp = Y_delta.shape[1]
    id_mask = split == "id"
    print(f"[load] N={len(X)} Kp={Kp} id={int(id_mask.sum())}", flush=True)

    # ---- fglex 特征 ----
    Xc, Dfg = build_fglex_X(d)

    # ---- 基因选择 + NaN 填补（id 中位数，与 layered 一致）----
    cand = np.arange(Kp)
    if args.n_genes < Kp:
        cand = np.argsort(np.isnan(Y_delta).mean(0))[:args.n_genes]
    Ys = Y_delta[:, cand].copy()
    keep = ~np.isnan(Ys).all(0)
    cand = cand[keep]; Ys = Ys[:, keep]
    Ys = impute_id_median(Ys, id_mask)
    Xtr = Xc[id_mask]; Ytr = Ys[id_mask]
    noise_var = float(np.mean((Ytr - Xtr @ np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]).var(0)))
    print(f"[prep] genes={len(cand)} train_rows={Xtr.shape[0]} noise_var={noise_var:.4f}", flush=True)

    # ---- fit fglex per-gene 贝叶斯孪生（校准栏）----
    twin = PerGeneBayesTwin(noise_var=noise_var, prior_lambda=1.0, seed=args.seed)
    twin.fit_torch_sgld(Xtr, Ytr, n_chains=3, n_iter=args.n_iter,
                        burnin=150, thin=10, lr=5e-4, T=0.1, clip_grad=1.0)
    mu_sel, sig_epi = twin.predict(Xc)                      # (N, n_genes)
    sig_tot_sel = np.sqrt(sig_epi ** 2 + noise_var)
    Ys_sel = Ys
    print(f"[twin] done {time.time()-t0:.0f}s", flush=True)

    # ---- fit P1（点预测栏）----
    Dc = len(d["compound_names"]); Ds = len(d["strain_names"])
    Dctx = X.shape[1] - Dc - Ds - 3
    Ctx = X[:, Dc:Dc + Dctx]
    S = X[:, Dc + Dctx:Dc + Dctx + Ds]
    comp_idx = X[:, :Dc].argmax(1)
    strain_idx = S.argmax(1)
    p1 = GoaiCompositionalTwin().fit(Y_delta, strain_idx, comp_idx, Ctx, id_mask)
    pred_delta_p1 = p1.predict_delta(strain_idx, comp_idx, Ctx)    # (N, Kp) 完整
    pred_delta_p1_sel = pred_delta_p1[:, cand]
    print(f"[p1] fit done", flush=True)

    # ================= L1/L2: localized conformal（中心=P1，宽=fglex σ） =================
    localized = {}
    for s in SUBSETS:
        m = split == s
        yv = Ys_sel[m]; muP = pred_delta_p1_sel[m]; st = sig_tot_sel[m]
        n = len(yv); j = rng.permutation(n); h = max(int(n * 0.5), 1)
        yc, muC, sc = yv[j[:h]], muP[j[:h]], st[j[:h]]
        yt, muT, stT = yv[j[h:]], muP[j[h:]], st[j[h:]]
        s_s = float(np.sqrt(np.mean(((yc - muC) / np.clip(sc, 1e-9, None)) ** 2)))
        sig_cal = stT * s_s
        cov = {str(L): round(coverage_at(yt.ravel(), muT.ravel(), sig_cal.ravel(), L), 4)
               for L in CANON}
        ece = ece_canon(yt.ravel(), muT.ravel(), sig_cal.ravel())
        localized[s] = {
            "coverage_by_level": cov, "coverage@0.95": cov["0.95"],
            "ece_multilevel_canon": round(ece, 4), "s_local": round(s_s, 4),
            "in_band_095": bool(BAND[0] <= cov["0.95"] <= BAND[1]),
            "passes_ece_bar": bool(ece < ECE_BAR),
        }
        print(f"  [L1/L2 {s}] cov@0.95={cov['0.95']} ece={ece:.4f} s_local={s_s:.4f}", flush=True)

    # ================= L3: 单调性 σ vs τ（全量 id 口径） =================
    nn = NearestNeighbors(n_neighbors=5).fit(Xtr)
    mono = {}
    for s in SUBSETS:
        m = split == s
        t = nn.kneighbors(Xc[m])[0].mean(1)
        sig_mean = sig_tot_sel[m].mean(1)
        mono[s] = round(float(np.corrcoef(sig_mean, t)[0, 1]), 4) if np.std(sig_mean) > 0 else None
        print(f"  [L3 {s}] mono(σ vs τ)={mono[s]}", flush=True)

    # ================= L4: RMSE 点预测栏（P1 → evaluate） =================
    pred_abs = Y_ctrl + pred_delta_p1
    ev = evaluate(d, pred_abs, pred_delta_p1, model_tag="P1_gate")
    weighted = float(ev["weighted_score"])
    per_fc = ev.get("per_subset_fc_pcc", {})
    print(f"  [L4] P1 weighted_score={weighted:.4f}", flush=True)

    # ================= GATE_REPRO: 与冻结基线对比 =================
    layered = _load_json(LAYERED)["localized_conformal"]
    fglex_full = _load_json(FGLEX_FULL)["results"]["fglex"]["localized"]
    precision = _load_json(PRECISION)["comparison"]["compositional_twin"]

    guard = {"tolerances": TOL, "vs_layered": {}, "vs_fglexfull_mono": {}, "vs_precision": {}}
    all_pass = True
    for s in SUBSETS:
        f = layered[s]
        dc = abs(localized[s]["coverage@0.95"] - f["coverage@0.95"])
        de = abs(localized[s]["ece_multilevel_canon"] - f["ece_multilevel_canon"])
        pc = dc <= TOL["cov095"] and de <= TOL["ece"]
        all_pass &= pc
        guard["vs_layered"][s] = {
            "freeze_cov095": f["coverage@0.95"], "current_cov095": localized[s]["coverage@0.95"],
            "d_cov095": round(dc, 4),
            "freeze_ece": f["ece_multilevel_canon"], "current_ece": localized[s]["ece_multilevel_canon"],
            "d_ece": round(de, 4), "pass": bool(pc),
        }
    for s in OOD:
        fm = fglex_full[s]["mono_corr_sigma_tau"]
        cm = mono[s]
        dm = abs(cm - fm) if cm is not None else float("inf")
        guard["vs_fglexfull_mono"][s] = {
            "freeze_mono(subsample4000)": fm, "current_mono(fullid)": cm, "d_mono": round(dm, 4),
            "pass(参考)": bool(dm <= TOL["mono"]),
            "note": "口径差异：冻结=fglex-full subsample(4000训练/抽样评测)，当前=全量 id(5169)；Δ 仅参考不判 REGRESSION",
        }
    dw = abs(weighted - float(precision["weighted_score"]))
    wp = dw <= TOL["weighted"]
    all_pass &= wp
    guard["vs_precision"] = {
        "freeze_P1_weighted": float(precision["weighted_score"]), "current_P1_weighted": weighted,
        "d_weighted": round(dw, 4), "pass": bool(wp),
    }
    guard["verdict"] = "ZERO-DELTA_PASS" if all_pass else "REGRESSION"
    guard["note"] = ("同口径栏（覆盖/ECE/加权分）全部 Δ≤容差 → 守卫 PASS，可复现；"
                     "mono 因冻结基线为 subsample 口径，仅作参考并已存全量口径冻结值。")

    out = {
        "stage": "GATE_REPRO（任务#4 固化）",
        "config": {"n_genes": len(cand), "train_rows": int(id_mask.sum()),
                   "D_fglex": Dfg, "noise_var": noise_var, "n_iter": args.n_iter,
                   "seed": args.seed, "T": 0.1,
                   "note": "四栏：覆盖/ECE（localized 校准栏, 中心P1） + 单调性（σvsτ） + RMSE（P1 weighted）"},
        "current_four_columns": {
            "L1_coverage_localized": {s: localized[s]["coverage@0.95"] for s in SUBSETS},
            "L2_ece_localized": {s: localized[s]["ece_multilevel_canon"] for s in SUBSETS},
            "L3_mono_sigma_tau": mono,
            "L4_P1_weighted_score": weighted,
            "P1_per_subset_fc_pcc": per_fc,
        },
        "localized_detail": localized,
        "mono_detail": mono,
        "guard": guard,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*64}\n  GATE_REPRO 守卫判定: {guard['verdict']}\n{'='*64}")
    for s in SUBSETS:
        g = guard["vs_layered"][s]
        print(f"  [{s}] d_cov095={g['d_cov095']:.4f} d_ece={g['d_ece']:.4f} pass={g['pass']}")
    print(f"  weighted: d={guard['vs_precision']['d_weighted']:.4f} pass={guard['vs_precision']['pass']}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
