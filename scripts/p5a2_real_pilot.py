#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-A2 复赛 P0 · Stage B-pilot — 真实 GOAI 数据 per-gene 贝叶斯线性孪生（小子集）

目的（de-risk 全量 99k×217×5243 前）：
    1) per-gene 贝叶斯线性结构在真实 GOAI 数据上能跑（架构可行性）；
    2) proper-Bayesian σ 量级在真实数据上是否合理（OOD 有真实 epistemic，不像合成 toy 过尺度）；
    3) ID 校准的全局重标定 s=rms(r_id/σ_id) 能否把 OOD 覆盖压回预锁带 [0.93,0.97]；
    4) H1：σ 随流形距离 τ 单调（corr>0）在真实数据上保持。

数据：data/processed/goai_cache.npz（mmap，避免 2GB Y_delta 全载入）
    X (N,D) = 化合物one-hot+上下文+菌株one-hot+agent+g；Y_delta (N,5243) 蛋白 log2 Δ
    split = id/ood_action/ood_agent/ood_s3/ood_time（官方 OOD 划分）

做法：
    - 随机取 --n-genes 个蛋白 + 子样本样本（id 三切分 fit/cal/eval；ood 仅 eval）
    - OLS 残差方差估 noise_var（data-driven）
    - PerGeneBayesTwin(torch SGLD) 在 id_train 拟 5243→子集 个贝叶斯线性回归
    - honest 全局 s 仅用 id_cal；σ_cal=σ_tot·s
    - 每子集：(sample,gene) 展平算 canonical 4 档覆盖 + multilevel ECE + 带检查；H1 用基因均值 σ vs τ

判据（预锁）：ood_action cov@0.95∈[0.93,0.97] 且 ECE<0.08；H1 corr>0。

用法（torch/PyMC venv）：
    .venv_p5a2/Scripts/python.exe scripts/p5a2_real_pilot.py \
        --out experiments/20260816-p5a2-real-pilot.json
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
OOD_SUBSETS = ["ood_action", "ood_agent", "ood_s3", "ood_time"]


def _z(level):
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2.0))


def coverage_at(y, mu, sigma, level):
    q = _z(level)
    return float(((y >= mu - q * sigma) & (y <= mu + q * sigma)).mean())


def ece_multilevel(y, mu, sigma, levels=CANON_LEVELS):
    return float(np.mean([abs(coverage_at(y, mu, sigma, L) - L) for L in levels]))


def evaluate(twin, X, Y, Xref, noise_var, s=1.0):
    mu, sig = twin.predict(X)                      # (N,K)
    sig_tot = np.sqrt(sig ** 2 + noise_var)
    sig_cal = sig_tot * s
    # canonical: per (sample,gene) raveled
    yf, muf, sf = Y.ravel(), mu.ravel(), sig_cal.ravel()
    cov = {str(L): round(coverage_at(yf, muf, sf, L), 4) for L in CANON_LEVELS}
    ec = ece_multilevel(yf, muf, sf)
    # H1: gene-mean sigma vs tau(manifold distance to Xref)
    t = NearestNeighbors(n_neighbors=5).fit(Xref).kneighbors(X)[0].mean(1)
    sig_mean = sig_tot.mean(1)
    mono = float(np.corrcoef(sig_mean, t)[0, 1]) if np.std(sig_mean) > 0 else float("nan")
    return {
        "n": int(X.shape[0]),
        "coverage_by_level": cov,
        "coverage@0.95": cov["0.95"],
        "ece_multilevel_canon": round(ec, 4),
        "in_band_095": bool(BAND[0] <= cov["0.95"] <= BAND[1]),
        "passes_ece_bar": bool(ec < ECE_BAR),
        "mono_corr_sigma_tau": round(mono, 4),
        "sigma_epi_mean": round(float(sig.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed/goai_cache.npz")
    ap.add_argument("--out", default="experiments/20260816-p5a2-real-pilot.json")
    ap.add_argument("--n-genes", type=int, default=80)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-cal", type=int, default=1000)
    ap.add_argument("--n-ood", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    cache = np.load(args.cache, allow_pickle=True)
    X = cache["X"]                                          # mmap (N,D)
    Y = cache["Y_delta"]                                    # mmap (N,5243)
    split = cache["split"].astype(str) if cache["split"].dtype.kind == "O" else cache["split"].astype(str)
    N, D = X.shape
    Kp = Y.shape[1]
    print(f"[pilot] cache X={X.shape} Y={Y.shape} D={D} Kp={Kp}", flush=True)
    for s in ["id"] + OOD_SUBSETS:
        print(f"    {s}: {(split==s).sum()}", flush=True)

    rng = np.random.default_rng(args.seed)
    # 真实蛋白丰度含 NaN（非阳性/未检出 -> log2(nan)）。先取候选基因、按 NaN 率排序挑低缺失的，
    # 残余 NaN 用列中位数填补，保证拟合/校准数值稳定。
    n_cand = max(args.n_genes * 4, 200)
    cand = rng.choice(Kp, size=min(n_cand, Kp), replace=False)
    Yc = np.asarray(Y[:, cand], dtype=np.float64)           # (N, n_cand) copy
    nan_frac = np.isnan(Yc).mean(0)
    chosen = np.argsort(nan_frac)[:args.n_genes]
    Ys = Yc[:, chosen].copy()                               # (N, n_genes)
    for j in range(Ys.shape[1]):                            # 中位数填补候选基因里残余 NaN
        col = Ys[:, j]
        m = np.isnan(col)
        if m.any():
            col[m] = np.nanmedian(col)
    Ys = np.ascontiguousarray(Ys)
    assert not np.isnan(Ys).any(), "Ys still contains NaN after impute"

    id_m = split == "id"
    Xid = np.asarray(X[id_m], dtype=np.float64)
    Yid = Ys[id_m]
    iid = rng.permutation(len(Xid))
    ntr, ncal = min(args.n_train, len(Xid)), min(args.n_cal, len(Xid) - args.n_train)
    i_tr, i_cal, i_ev = iid[:ntr], iid[ntr:ntr + ncal], iid[ntr + ncal:]
    Xtr, Ytr = Xid[i_tr], Yid[i_tr]
    Xcal, Ycal = Xid[i_cal], Yid[i_cal]
    Xid_ev, Yid_ev = Xid[i_ev], Yid[i_ev]

    # OOD eval（子样本）
    ood = {}
    for s in OOD_SUBSETS:
        m = split == s
        Xs = np.asarray(X[m], dtype=np.float64)
        Ys_ = Ys[m]
        if len(Xs) > args.n_ood:
            j = rng.choice(len(Xs), args.n_ood, replace=False)
            Xs, Ys_ = Xs[j], Ys_[j]
        ood[s] = (Xs, Ys_)

    # data-driven noise_var (OLS residual var, mean over genes)
    W_ols, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    resid = Ytr - Xtr @ W_ols
    noise_var = float(np.mean(resid.var(0)))
    print(f"[pilot] noise_var(OLS)={noise_var:.4f}  n_train={len(Xtr)} n_cal={len(Xcal)} "
          f"id_eval={len(Xid_ev)}", flush=True)

    twin = PerGeneBayesTwin(noise_var=noise_var, prior_lambda=1.0, seed=args.seed)
    twin.fit_torch_sgld(Xtr, Ytr, n_chains=3, n_iter=600, burnin=150, thin=10,
                        lr=5e-4, T=0.1, clip_grad=1.0, wdecay=1e-3)
    print(f"[pilot] fit done {time.time()-t0:.1f}s", flush=True)

    mu_cal, sig_cal = twin.predict(Xcal)
    sig_cal_tot = np.sqrt(sig_cal ** 2 + noise_var)
    s_global = float(np.sqrt(np.mean(((Ycal - mu_cal).ravel()
                                      / np.clip(sig_cal_tot.ravel(), 1e-9, None)) ** 2)))
    print(f"[pilot] s_global(ID)={s_global:.4f}", flush=True)

    out = {
        "stage": "B-pilot-real-goai-pergene",
        "config": vars(args),
        "data": {"N": N, "D": D, "Kp": Kp, "n_genes": int(Ys.shape[1]),
                 "n_train": len(Xtr), "n_cal": len(Xcal), "id_eval": len(Xid_ev)},
        "noise_var_ols": round(noise_var, 4),
        "s_global_id": round(s_global, 4),
        "canonical_levels": CANON_LEVELS, "prelocked_band_095": list(BAND), "ece_bar": ECE_BAR,
        "subsets_before": {}, "subsets_after": {}, "verdict": {},
    }
    all_names = ["id_eval"] + OOD_SUBSETS
    data_map = {"id_eval": (Xid_ev, Yid_ev)}
    data_map.update(ood)
    for name in all_names:
        Xs, Ys_ = data_map[name]
        b = evaluate(twin, Xs, Ys_, Xtr, noise_var, s=1.0)
        a = evaluate(twin, Xs, Ys_, Xtr, noise_var, s=s_global)
        out["subsets_before"][name] = b
        out["subsets_after"][name] = a
        print(f"[{name:10s}] cov95 {b['coverage@0.95']:.3f}→{a['coverage@0.95']:.3f}  "
              f"ece {b['ece_multilevel_canon']:.3f}→{a['ece_multilevel_canon']:.3f}  "
              f"band={a['in_band_095']} mono={a['mono_corr_sigma_tau']:+.3f}", flush=True)

    oa = out["subsets_after"]["ood_action"]
    out["verdict"] = {
        "ood_action_cov095_after": oa["coverage@0.95"],
        "ood_action_ece_after": oa["ece_multilevel_canon"],
        "in_band": oa["in_band_095"],
        "ece_ok": oa["passes_ece_bar"],
        "h1_monotonicity_preserved": bool(oa["mono_corr_sigma_tau"] > 0),
        "P5A2_REAL_PILOT_ACCEPTED": bool(oa["in_band_095"] and oa["passes_ece_bar"]),
        "per_ood_band": {s: out["subsets_after"][s]["in_band_095"] for s in OOD_SUBSETS},
    }
    print("\n=== VERDICT (real GOAI pilot) ===")
    print(json.dumps(out["verdict"], ensure_ascii=False, indent=1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
