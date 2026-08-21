#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-A2 复赛 P0 · H1 救援 — 连续化合物表征替 one-hot（离线代理对比）

目的：检验 *编码假设* —— H1(σ 随流形距离 τ 单调) 在真实数据失败是否源于
    化合物/菌株 **one-hot 编码**（未见轴 epistemic=0），而非 proper-Bayesian 机制本身。

两种离线连续化合物表征（沙箱 PubChem / SMILES 均不可达，无 RDKit）：
  - namehash : 化合物名字符 n-gram 哈希袋（make_namehash_block）—— 已证 H1 救援
  - fglex    : 化合物名法规派生**官能团指示向量**（make_functionalgroup_block）
               —— 化学含义更强（每维可解释为 halogen/hydroxy/amine/...），任务#2 的
                  「PubChem 增强」离线替代；真 PubChem 2D/SMILES 可达时换之（TODO）

跑法：
    .venv_p5a2/Scripts/python.exe scripts/p5a2_real_contcomp.py --comp-mode both \
        --out experiments/20260819-p5a2-contcomp.json
    .venv_p5a2/Scripts/python.exe scripts/p5a2_real_contcomp.py --comp-mode fglex \
        --n-genes 5243 --out experiments/20260819-p5a2-fglex-full.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from model.compositional_p5a2_pergene import PerGeneBayesTwin        # noqa: E402
from data.goai_compound_features import (                             # noqa: E402
    make_namehash_block, make_functionalgroup_block,
)
from sklearn.neighbors import NearestNeighbors                       # noqa: E402

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
    mu, sig = twin.predict(X)
    sig_tot = np.sqrt(sig ** 2 + noise_var)
    sig_cal = sig_tot * s
    yf, muf, sf = Y.ravel(), mu.ravel(), sig_cal.ravel()
    cov = {str(L): round(coverage_at(yf, muf, sf, L), 4) for L in CANON_LEVELS}
    ec = ece_multilevel(yf, muf, sf)
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


def build_X_cont(per_sample, X, Dc, mode, args):
    if mode == "namehash":
        desc, _ = make_namehash_block(per_sample, dim=args.desc_dim)
    elif mode == "fglex":
        desc, _ = make_functionalgroup_block(per_sample)
    else:
        raise ValueError(mode)
    return np.ascontiguousarray(np.hstack([desc, X[:, Dc:]]).astype(np.float64))


def run_mode(mode, per_sample, X, Ys, split, Dc, args, rng, t0):
    X_cont = build_X_cont(per_sample, X, Dc, mode, args)
    print(f"[{mode}] X_cont={X_cont.shape}", flush=True)

    id_m = split == "id"
    Xid, Yid = X_cont[id_m], Ys[id_m]
    iid = rng.permutation(len(Xid))
    ntr = min(args.n_train, len(Xid))
    ncal = min(args.n_cal, len(Xid) - ntr)
    i_tr, i_cal, i_ev = iid[:ntr], iid[ntr:ntr + ncal], iid[ntr + ncal:]
    Xtr, Ytr = Xid[i_tr], Yid[i_tr]
    Xcal, Ycal = Xid[i_cal], Yid[i_cal]
    Xid_ev, Yid_ev = Xid[i_ev], Yid[i_ev]

    ood = {}
    for s in OOD_SUBSETS:
        m = split == s
        Xs, Ys_ = X_cont[m], Ys[m]
        if len(Xs) > args.n_ood:
            j = rng.choice(len(Xs), args.n_ood, replace=False)
            Xs, Ys_ = Xs[j], Ys_[j]
        ood[s] = (Xs, Ys_)

    W_ols, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    resid = Ytr - Xtr @ W_ols
    noise_var = float(np.mean(resid.var(0)))
    print(f"[{mode}] noise_var(OLS)={noise_var:.4f} n_train={len(Xtr)}", flush=True)

    twin = PerGeneBayesTwin(noise_var=noise_var, prior_lambda=1.0, seed=args.seed)
    twin.fit_torch_sgld(Xtr, Ytr, n_chains=3, n_iter=600, burnin=150, thin=10,
                        lr=5e-4, T=0.1, clip_grad=1.0, wdecay=1e-3)
    print(f"[{mode}] fit done {time.time()-t0:.1f}s", flush=True)

    mu_cal, sig_cal = twin.predict(Xcal)
    sig_cal_tot = np.sqrt(sig_cal ** 2 + noise_var)
    s_global = float(np.sqrt(np.mean(((Ycal - mu_cal).ravel()
                                      / np.clip(sig_cal_tot.ravel(), 1e-9, None)) ** 2)))

    out = {
        "encoding": mode,
        "desc_dim": X_cont.shape[1] - (X.shape[1] - Dc),
        "noise_var_ols": round(noise_var, 4),
        "s_global_id": round(s_global, 4),
        "raw": {}, "global": {}, "localized": {},
    }
    for name in ["id_eval"] + OOD_SUBSETS:
        Xs, Ys_ = (Xid_ev, Yid_ev) if name == "id_eval" else ood[name]
        out["raw"][name] = evaluate(twin, Xs, Ys_, Xtr, noise_var, s=1.0)
        out["global"][name] = evaluate(twin, Xs, Ys_, Xtr, noise_var, s=s_global)
    for name in ["id_eval"] + OOD_SUBSETS:
        Xs, Ys_ = (Xid_ev, Yid_ev) if name == "id_eval" else ood[name]
        j = rng.permutation(len(Xs))
        h = len(j) // 2
        Xc, Yc_ = Xs[j[:h]], Ys_[j[:h]]
        Xt, Yt_ = Xs[j[h:]], Ys_[j[h:]]
        muc, sigc = twin.predict(Xc)
        s_s = float(np.sqrt(np.mean(((Yc_ - muc).ravel()
                                     / np.clip(np.sqrt(sigc ** 2 + noise_var).ravel(),
                                               1e-9, None)) ** 2)))
        out["localized"][name] = evaluate(twin, Xt, Yt_, Xtr, noise_var, s=s_s)
        out["localized"][name]["s_local"] = round(s_s, 4)

    loc_pass = all(out["localized"][n]["in_band_095"] and out["localized"][n]["passes_ece_bar"]
                   for n in ["id_eval"] + OOD_SUBSETS)
    out["verdict"] = {
        "ood_action_mono_raw": out["raw"]["ood_action"]["mono_corr_sigma_tau"],
        "ood_action_mono_localized": out["localized"]["ood_action"]["mono_corr_sigma_tau"],
        "h1_rescued_ood_action": bool(out["localized"]["ood_action"]["mono_corr_sigma_tau"] > 0),
        "localized_passes_prelocked": bool(loc_pass),
        "ood_action_cov095_localized": out["localized"]["ood_action"]["coverage@0.95"],
        "ood_action_ece_localized": out["localized"]["ood_action"]["ece_multilevel_canon"],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed/goai_cache.npz")
    ap.add_argument("--out", default="experiments/20260819-p5a2-contcomp.json")
    ap.add_argument("--comp-mode", choices=["namehash", "fglex", "both"],
                    default="namehash")
    ap.add_argument("--n-genes", type=int, default=80)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-cal", type=int, default=1000)
    ap.add_argument("--n-ood", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--desc-dim", type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()

    cache = np.load(args.cache, allow_pickle=True)
    X = cache["X"]
    Y = cache["Y_delta"]
    split = cache["split"].astype(str)
    Kp = Y.shape[1]
    cnames = list(cache["compound_names"])
    Dc = len(cnames)
    print(f"[contcomp] cache X={X.shape} Y={Y.shape} Dc={Dc}", flush=True)

    P = np.asarray(X[:, :Dc], dtype=np.float64)
    assert (P.sum(1) == 1).all(), "compound one-hot 应每行恰一 1"
    per_sample = [cnames[i] for i in P.argmax(1)]

    # 共享基因选择 + NaN 处理（与编码无关）
    rng = np.random.default_rng(args.seed)
    n_cand = max(args.n_genes * 4, 200)
    cand = rng.choice(Kp, size=min(n_cand, Kp), replace=False)
    Yc = np.asarray(Y[:, cand], dtype=np.float64)
    chosen = np.argsort(np.isnan(Yc).mean(0))[:args.n_genes]
    Ys = Yc[:, chosen].copy()
    keep = ~np.isnan(Ys).all(0)
    Ys = Ys[:, keep]
    for j in range(Ys.shape[1]):
        col = Ys[:, j]
        m = np.isnan(col)
        if m.any():
            col[m] = np.nanmedian(col)
    Ys = np.ascontiguousarray(Ys)
    assert not np.isnan(Ys).any()

    modes = ["namehash", "fglex"] if args.comp_mode == "both" else [args.comp_mode]
    results = {}
    for mode in modes:
        print(f"\n########## MODE={mode} ##########", flush=True)
        results[mode] = run_mode(mode, per_sample, X, Ys, split, Dc, args, rng, t0)

    # 打印对照
    print("\n=== H1 / 校准 对照（localized 臂，按编码）===")
    print(f"{'subset':12s} " + " ".join(
        f"{m:>22s}" for m in modes))
    for metric in ["mono_corr_sigma_tau", "coverage@0.95", "ece_multilevel_canon"]:
        print(f"-- {metric} --")
        for name in ["id_eval"] + OOD_SUBSETS:
            row = f"{name:12s} "
            for m in modes:
                v = results[m]["localized"][name][metric]
                row += f"{v:22.4f} "
            print(row)
    for m in modes:
        print(f"\n=== VERDICT ({m}) ===")
        print(json.dumps(results[m]["verdict"], ensure_ascii=False, indent=1))

    out = {
        "stage": "B-contcomp-real-goai-compare",
        "config": vars(args),
        "modes": modes,
        "canonical_levels": CANON_LEVELS, "prelocked_band_095": list(BAND), "ece_bar": ECE_BAR,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
