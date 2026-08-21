#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-A2 复赛 P0 · Stage B-full — 全量 5243 基因 per-gene 贝叶斯线性孪生 + localized conformal

把 pilot/calib 的结论扩展到全量蛋白，确认可推广并产出真实基准数字。
架构/校准同 p5a2_real_calib.py，区别：
    - 使用全部 5243 基因（W(D×5243) 向量化 SGLD；_sgld_predict 增量 Welford，不爆内存）
    - NaN 用 id_train 每基因中位数填补（不泄漏测试信息）
    - 评估 raw / global(ID s) / localized(每子集 s_s) 三臂，报告 ood_action 预锁判据

用法（torch/PyMC venv）：
    .venv_p5a2/Scripts/python.exe scripts/p5a2_real_full.py \
        --out experiments/20260816-p5a2-real-full.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from model.compositional_p5a2_pergene import PerGeneBayesTwin  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

CANON_LEVELS = [0.5, 0.8, 0.9, 0.95]
BAND = (0.93, 0.97)
ECE_BAR = 0.08
SUBSETS = ["id", "ood_action", "ood_agent", "ood_s3", "ood_time"]


def _z(level):
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2.0))


def coverage_at(y, mu, sigma, level):
    q = _z(level)
    return float(((y >= mu - q * sigma) & (y <= mu + q * sigma)).mean())


def ece_multilevel(y, mu, sigma, levels=CANON_LEVELS):
    return float(np.mean([abs(coverage_at(y, mu, sigma, L) - L) for L in levels]))


def metrics(X, Y, mu, sig, Xref, noise_var, s):
    sig_tot = np.sqrt(sig ** 2 + noise_var) * s
    yf, muf, sf = Y.ravel(), mu.ravel(), sig_tot.ravel()
    cov = {str(L): round(coverage_at(yf, muf, sf, L), 4) for L in CANON_LEVELS}
    ec = ece_multilevel(yf, muf, sf)
    t = NearestNeighbors(n_neighbors=5).fit(Xref).kneighbors(X)[0].mean(1)
    sig_mean = (np.sqrt(sig ** 2 + noise_var)).mean(1)
    mono = float(np.corrcoef(sig_mean, t)[0, 1]) if np.std(sig_mean) > 0 else float("nan")
    return {"coverage_by_level": cov, "coverage@0.95": cov["0.95"],
            "ece_multilevel_canon": round(ec, 4),
            "in_band_095": bool(BAND[0] <= cov["0.95"] <= BAND[1]),
            "passes_ece_bar": bool(ec < ECE_BAR),
            "mono_corr_sigma_tau": round(mono, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed/goai_cache.npz")
    ap.add_argument("--out", default="experiments/20260816-p5a2-real-full.json")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    cache = np.load(args.cache, allow_pickle=True)
    X = cache["X"]
    Y = cache["Y_delta"]                                   # mmap (N,5243)
    split = cache["split"].astype(str)
    N, D = X.shape
    Kp = Y.shape[1]
    rng = np.random.default_rng(args.seed)

    Yf = np.asarray(Y, dtype=np.float64)                    # (N,5243) copy
    id_m = split == "id"
    iid = rng.permutation(int(id_m.sum()))
    ntr = min(args.n_train, len(iid))
    Xtr = np.asarray(X[id_m], dtype=np.float64)[iid[:ntr]]
    # NaN 用 id_train 每基因中位数填补（不泄漏测试）
    gene_med = np.nanmedian(Yf[id_m][iid[:ntr]], axis=0)
    gene_med = np.where(np.isnan(gene_med), 0.0, gene_med)   # 全 nan 基因兜底 0
    mask = np.isnan(Yf)
    Yf[mask] = gene_med[np.nonzero(mask)[1]]                 # 按列填该基因中位数
    Ys = np.ascontiguousarray(Yf)
    assert not np.isnan(Ys).any(), "Ys NaN after impute"
    Ytr = Ys[id_m][iid[:ntr]]                                # 填补后重新抽取（避免 NaN 级联）
    print(f"[full] X={X.shape} Kp={Kp} n_train={ntr} impute-gene-median(id_train)", flush=True)

    # 每子集 50/50 cal/test
    cal_idx, test_idx = {}, {}
    for s in SUBSETS:
        m = split == s
        Xs = np.asarray(X[m], dtype=np.float64); Ys_ = Ys[m]
        ii = rng.permutation(len(Xs))
        h = len(Xs) // 2
        cal_idx[s] = (Xs[ii[:h]], Ys_[ii[:h]])
        test_idx[s] = (Xs[ii[h:]], Ys_[ii[h:]])

    W_ols, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    noise_var = float(np.mean((Ytr - Xtr @ W_ols).var(0)))
    print(f"[full] noise_var(OLS)={noise_var:.4f}  fit on {ntr} id", flush=True)

    twin = PerGeneBayesTwin(noise_var=noise_var, prior_lambda=1.0, seed=args.seed)
    twin.fit_torch_sgld(Xtr, Ytr, n_chains=3, n_iter=600, burnin=150, thin=10,
                        lr=5e-4, T=0.1, clip_grad=1.0, wdecay=1e-3)
    print(f"[full] fit done {time.time()-t0:.1f}s", flush=True)

    Xc_id, Yc_id = cal_idx["id"]
    muc, sigc = twin.predict(Xc_id)
    s_global = float(np.sqrt(np.mean(((Yc_id - muc).ravel()
                                      / np.clip(np.sqrt(sigc ** 2 + noise_var).ravel(), 1e-9, None)) ** 2)))
    s_local = {}
    for s in SUBSETS:
        Xcs, Ycs = cal_idx[s]
        mus, sigs = twin.predict(Xcs)
        st = np.sqrt(sigs ** 2 + noise_var)
        s_local[s] = float(np.sqrt(np.mean(((Ycs - mus).ravel()
                                            / np.clip(st.ravel(), 1e-9, None)) ** 2)))
    print(f"[full] s_global(ID)={s_global:.4f}  s_local=" +
          ", ".join(f"{s}={s_local[s]:.4f}" for s in SUBSETS), flush=True)

    out = {"stage": "B-full-real-goai-pergene-5243",
           "config": vars(args), "Kp": Kp, "n_train": ntr,
           "noise_var_ols": round(noise_var, 4),
           "s_global_id": round(s_global, 4), "s_local": {k: round(v, 4) for k, v in s_local.items()},
           "canonical_levels": CANON_LEVELS, "prelocked_band_095": list(BAND), "ece_bar": ECE_BAR,
           "arms": {}}
    for arm, smap in (("raw", {s: 1.0 for s in SUBSETS}),
                      ("global", {s: s_global for s in SUBSETS}),
                      ("localized", s_local)):
        arm_res = {}
        for s in SUBSETS:
            Xte, Yte = test_idx[s]
            mu, sig = twin.predict(Xte)
            arm_res[s] = metrics(Xte, Yte, mu, sig, Xtr, noise_var, smap[s])
        out["arms"][arm] = arm_res

    oa = {arm: out["arms"][arm]["ood_action"] for arm in out["arms"]}
    out["verdict_ood_action"] = {
        a: {"cov095": oa[a]["coverage@0.95"], "ece": oa[a]["ece_multilevel_canon"],
            "in_band": oa[a]["in_band_095"], "ece_ok": oa[a]["passes_ece_bar"]} for a in oa}
    out["localized_passes_prelocked"] = bool(
        oa["localized"]["in_band_095"] and oa["localized"]["passes_ece_bar"])

    for arm in out["arms"]:
        print(f"--- {arm} ---", flush=True)
        for s in SUBSETS:
            r = out["arms"][arm][s]
            print(f"  [{s:10s}] cov95={r['coverage@0.95']:.3f} ece={r['ece_multilevel_canon']:.3f} "
                  f"band={r['in_band_095']} mono={r['mono_corr_sigma_tau']:+.3f}", flush=True)
    print("\n=== ood_action verdict (raw/global/localized) ===")
    print(json.dumps(out["verdict_ood_action"], ensure_ascii=False, indent=1))
    print(f"localized_passes_prelocked={out['localized_passes_prelocked']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
