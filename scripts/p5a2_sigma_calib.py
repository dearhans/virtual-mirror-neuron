#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-A2 复赛 P0 · σ 尺度标定验证（合成数据，honest ID-calibrated recalibration）

动机（复赛 P0 任务#1）：
    P5-A2 proper-Bayesian 在合成数据上验证 H1(σ 随流形距离 τ 单调)成立，但
    σ 过尺度 → ood_action 覆盖@0.95≈1.000（饱和签名），越出预锁带 [0.93,0.97]，
    canonical 4 档 ECE≈0.21（饱和地板），判据不通过。
    本脚本回答：用**诚实 ID 校准集**拟合的乘性重标定 s=rms(r_id/σ_id)、σ_cal=σ·s，
    能否把覆盖压回 [0.93,0.97] 且保留 H1？并诊断过尺度是否**异方差**
    （ood 比 id 更过尺度 → 全局标量 s 不够，需 τ-条件重标定）。

设计（与 multiseed_recalibrate.py 同口径，但套到 P5-A2 sigma + 预锁判据）：
    ID 三切分：fit(70%) / cal(15%) / eval-id(15%)；ood 三层仅 eval。
    仅用 cal(ID) 估计 s_global（测试时不许见 ood 校准行，honest）。
    诊断：另报各 eval 子集经验 s*（仅在测试集算，用于诊断，不用于 honest 评估）。

判据（预锁，对齐 p5a2_readjudicate / p41）：
    ood_action 上 cov@0.95 ∈ [0.93,0.97] 且 canonical 4 档 ECE < 0.08。
    H1 = corr(σ_epi_mean, τ) > 0（重标定是乘性标量，秩不变 → H1 必保留；验证之）。

用法（torch/PyMC venv）：
    .venv_p5a2/Scripts/python.exe scripts/p5a2_sigma_calib.py \
        --out experiments/20260816-p5a2-sigma-calib.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from model.compositional_p5a2_bayes import ProperBayesTwin  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

CANON_LEVELS = [0.5, 0.8, 0.9, 0.95]
BAND = (0.93, 0.97)
ECE_BAR = 0.08
SUBSETS = ["id_eval", "ood_action", "ood_agent", "ood_neuro"]


# ---------------------------------------------------------------- synthetic data (byte-id to spike)
def make_Xy(n_id=2000, n_ood=500, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    pa, pb = dim // 2, dim // 4
    act = max(1, dim // 6)

    def block(n, center):
        X = rng.normal(0, 1, size=(n, dim))
        X[:, :pa] += center
        return X

    Xid = block(n_id, 0.0)
    Xact = block(n_ood, 6.0)
    Xage = block(n_ood, -6.0)
    Xneu = block(n_ood, 1.5)

    def f(X):
        a = X[:, :pa].sum(1)
        b = X[:, pa:pa + pb].prod(1) if pb > 0 else np.zeros(X.shape[0])
        c = np.tanh(X[:, pa + pb:pa + pb + act].sum(1)) if act > 0 else np.zeros(X.shape[0])
        return np.stack([np.sin(a), np.cos(0.5 * a), b / (1 + b ** 2), c], axis=1)

    Yid = f(Xid) + rng.normal(0, 0.5, size=(n_id, 4))
    Yact = f(Xact) + rng.normal(0, 0.5, size=(n_ood, 4))
    Yage = f(Xage) + rng.normal(0, 0.5, size=(n_ood, 4))
    Yneu = f(Xneu) + rng.normal(0, 0.5, size=(n_ood, 4))
    return (Xid, Yid, Xact, Yact, Xage, Yage, Xneu, Yneu)


def tau(X, Xref, k=5):
    nn = NearestNeighbors(n_neighbors=k).fit(Xref)
    d, _ = nn.kneighbors(X)
    return d.mean(1)


def _z(level):
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2.0))


def coverage_at(y, mu, sigma, level):
    q = _z(level)
    return float(((y >= mu - q * sigma) & (y <= mu + q * sigma)).mean())


def ece_multilevel(y, mu, sigma, levels=CANON_LEVELS):
    return float(np.mean([abs(coverage_at(y, mu, sigma, L) - L) for L in levels]))


def evaluate(twin, X, Y, Xref, s=1.0):
    mu, sig = twin.predict(X)
    sig_tot = np.sqrt(sig ** 2 + twin.noise_var)
    sig_cal = sig_tot * s
    cov = {str(L): round(coverage_at(Y, mu, sig_cal, L), 4) for L in CANON_LEVELS}
    ec = ece_multilevel(Y, mu, sig_cal)
    t = tau(X, Xref)
    mono = float(np.corrcoef(sig.mean(1), t)[0, 1])
    # 经验 s*（诊断用，不用于 honest 评估）
    s_star = float(np.sqrt(np.mean(((Y - mu).ravel() / np.clip(sig_tot.ravel(), 1e-9, None)) ** 2)))
    return {
        "coverage_by_level": cov,
        "coverage@0.95": cov["0.95"],
        "ece_multilevel_canon": round(ec, 4),
        "in_band_095": bool(BAND[0] <= cov["0.95"] <= BAND[1]),
        "passes_ece_bar": bool(ec < ECE_BAR),
        "mono_corr_sigma_tau": round(mono, 4),
        "sigma_epi_mean": round(float(sig.mean()), 4),
        "s_star_diagnostic": round(s_star, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/20260816-p5a2-sigma-calib.json")
    ap.add_argument("--n-id", type=int, default=2000)
    ap.add_argument("--n-ood", type=int, default=500)
    ap.add_argument("--hidden", default="32,16")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))
    t0 = time.time()

    (Xid, Yid, Xact, Yact, Xage, Yage, Xneu, Yneu) = make_Xy(args.n_id, args.n_ood)
    # ID 三切分：fit 70 / cal 15 / eval 15
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Xid))
    n = len(Xid)
    n_fit, n_cal = int(n * 0.70), int(n * 0.15)
    i_fit, i_cal, i_eval = idx[:n_fit], idx[n_fit:n_fit + n_cal], idx[n_fit + n_cal:]
    Xfit, Yfit = Xid[i_fit], Yid[i_fit]
    Xcal, Ycal = Xid[i_cal], Yid[i_cal]
    Xid_eval, Yid_eval = Xid[i_eval], Yid[i_eval]

    # 测试集
    Xtest = np.vstack([Xid_eval, Xact, Xage, Xneu])
    Ytest = np.vstack([Yid_eval, Yact, Yage, Yneu])
    strata = (["id_eval"] * len(Xid_eval) + ["ood_action"] * len(Xact)
              + ["ood_agent"] * len(Xage) + ["ood_neuro"] * len(Xneu))

    tw = ProperBayesTwin(hidden=hidden, noise_var=0.25, prior_lambda=1.0, seed=args.seed)
    tw.fit_torch_sgld(Xfit, Yfit, n_chains=3, n_iter=800, burnin=300, thin=4,
                      lr=1e-3, T=0.1, clip_grad=1.0, wdecay=1e-3)
    print(f"[calib] fit done {time.time()-t0:.1f}s", flush=True)

    # honest 全局 s：仅用 ID cal 行
    mu_cal, sig_cal_raw = tw.predict(Xcal)
    sig_cal_tot = np.sqrt(sig_cal_raw ** 2 + tw.noise_var)
    s_global = float(np.sqrt(np.mean(((Ycal - mu_cal).ravel()
                                      / np.clip(sig_cal_tot.ravel(), 1e-9, None)) ** 2)))
    print(f"[calib] s_global(ID)={s_global:.4f}", flush=True)

    out = {
        "stage": "A-synthetic-sigma-calib",
        "config": vars(args),
        "canonical_levels": CANON_LEVELS,
        "prelocked_band_095": list(BAND),
        "ece_bar": ECE_BAR,
        "s_global_id": round(s_global, 4),
        "subsets_before": {}, "subsets_after": {},
        "verdict": {},
    }

    for name in SUBSETS:
        m = np.array(strata) == name
        Xs, Ys = Xtest[m], Ytest[m]
        out["subsets_before"][name] = evaluate(tw, Xs, Ys, Xfit, s=1.0)
        out["subsets_after"][name] = evaluate(tw, Xs, Ys, Xfit, s=s_global)
        b, a = out["subsets_before"][name], out["subsets_after"][name]
        print(f"[{name:10s}] cov95 {b['coverage@0.95']:.3f}→{a['coverage@0.95']:.3f}  "
              f"ece {b['ece_multilevel_canon']:.3f}→{a['ece_multilevel_canon']:.3f}  "
              f"band={a['in_band_095']} mono={a['mono_corr_sigma_tau']:+.3f}  "
              f"s*={a['s_star_diagnostic']:.3f}", flush=True)

    # 判据（ood_action，honest 全局 s）
    oa = out["subsets_after"]["ood_action"]
    cov_l = oa["coverage@0.95"]
    out["verdict"] = {
        "ood_action_cov095_after": cov_l,
        "ood_action_ece_after": oa["ece_multilevel_canon"],
        "in_band": oa["in_band_095"],
        "ece_ok": oa["passes_ece_bar"],
        "h1_monotonicity_preserved": bool(oa["mono_corr_sigma_tau"] > 0),
        "P5A2_CALIBRATED_ACCEPTED": bool(oa["in_band_095"] and oa["passes_ece_bar"]),
        # 异方差诊断：ood 经验 s* 相对 id_eval s* 的倍数（>1 表示 ood 比 id 更过尺度）
        "ood_over_scaling_factor": round(oa["s_star_diagnostic"]
                                         / max(out["subsets_after"]["id_eval"]["s_star_diagnostic"], 1e-9), 3),
        "interpretation": (
            "全局 ID 标量 s 已把 ood_action 压回带内 → 过尺度近似同方差，单一乘性重标定足够"
            if oa["in_band_095"]
            else "ood_action 仍越带 → 过尺度异方差(ood 比 id 更过尺度)，需 τ-条件重标定（Stage B）"),
    }
    print("\n=== VERDICT ===")
    print(json.dumps(out["verdict"], ensure_ascii=False, indent=1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
