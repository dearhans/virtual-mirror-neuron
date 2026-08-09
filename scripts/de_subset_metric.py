#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新证据槽（W33）：ood_action 判别性指标复核
==========================================

背景（2026-08-07 文献监测 + P2 复核待办 (a)）：
    P2 结论「ood_action 所有模型 ~0.63、无判别力」是在**全基因、单细胞级 RMSE**下得到的。
    文献（Response Magnitude, arXiv 2608.00152；Empirical Comparison of Virtual Cell Models,
    rs-10434123）指出：全基因 RMSE 被「预测均值」策略最大化，深度模型会坍缩到均值，
    真实增益只在 DE 子集 / 表征层 / 组合 context 上显现。

    → 假设 H_metric：ood_action 的「无判别力」是**指标伪影**，不是机制结论。

本脚本的证伪设计（只用缓存 + 廉价基线，不重训孪生）：
    探针 = 「均值基线」（退化策略）vs「线性基线」（当前最强外推器）。
    若两者在单细胞全基因 RMSE 上几乎并列，却在**伪bulk / DE 子集 / ΔPCC** 上显著拉开，
    则证明指标本身缺乏判别力 → H_metric 成立，ood_action 判据须升级后重跑全部预测器。
    若两者在所有指标上都并列，则 H_metric 被证伪，「信号量小」的原结论增强。

指标：
    1. cell_rmse_all      单细胞 × 全 256 基因 RMSE（现行主判据）
    2. cell_rmse_de       单细胞 × top-K DE 基因 RMSE
    3. pseudobulk_rmse    按扰动身份聚合后 RMSE（去单细胞噪声）
    4. delta_pcc          Δ 谱（减训练集全局均值）的 macro-per-perturbation Pearson
    5. macro_gene_pcc     macro-per-gene Pearson（跨扰动组）
    6. e_distance         能量距离（预测分布 vs 真实分布，子采样）

DE 基因定义：按**伪bulk 真值谱跨扰动组的方差**排序取 top-K（真正在响应的基因）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def log(msg):
    print(f"[de_metric {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_key(P: np.ndarray) -> np.ndarray:
    """扰动身份 = P 的非零列索引组合（单/双扰动都适用）。返回字符串键数组。"""
    keys = []
    for row in P:
        idx = np.nonzero(row)[0]
        keys.append("+".join(str(int(i)) for i in sorted(idx.tolist())))
    return np.asarray(keys)


def pseudobulk(y: np.ndarray, keys: np.ndarray):
    uk = np.unique(keys)
    out = np.zeros((len(uk), y.shape[1]), dtype=np.float64)
    for i, k in enumerate(uk):
        out[i] = y[keys == k].mean(0)
    return uk, out


def macro_pcc(a: np.ndarray, b: np.ndarray, axis: int) -> float:
    """沿 axis 逐行/列算 Pearson 后取均值（忽略零方差项）。"""
    if axis == 1:
        A, B = a, b
    else:
        A, B = a.T, b.T
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    na = np.sqrt((A ** 2).sum(1))
    nb = np.sqrt((B ** 2).sum(1))
    ok = (na > 1e-12) & (nb > 1e-12)
    if ok.sum() == 0:
        return float("nan")
    r = (A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])
    return float(np.mean(r))


def energy_distance(Xp: np.ndarray, Xt: np.ndarray, rng, cap=400) -> float:
    """能量距离 E = 2*E|p-t| - E|p-p'| - E|t-t'|（欧氏，子采样到 cap）。"""
    def sub(M):
        if len(M) > cap:
            return M[rng.choice(len(M), cap, replace=False)]
        return M
    P, T = sub(Xp), sub(Xt)

    def md(A, B):
        # 分块算平均成对距离，控内存
        s, n = 0.0, 0
        for i in range(0, len(A), 128):
            d = np.linalg.norm(A[i:i + 128, None, :] - B[None, :, :], axis=2)
            s += d.sum(); n += d.size
        return s / max(n, 1)
    return float(2 * md(P, T) - md(P, P) - md(T, T))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "processed", "norman_cache.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "20260807-de-subset-metric.json"))
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200, help="自助法次数（重采样扰动组）")
    ap.add_argument("--with-mlp", action="store_true", help="额外跑 MLP 探针（慢，默认关）")
    a = ap.parse_args(argv)

    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(a.seed)

    log(f"载入缓存 {a.cache}")
    d = np.load(a.cache, allow_pickle=True)
    X, y, sp = d["X"], d["y"], d["split"]
    P = X[:, :107]
    tr = sp == "train"
    log(f"train={tr.sum()} 全量={len(sp)}")

    log("拟合线性探针（Ridge, alpha=1.0）…")
    t0 = time.time()
    lin = Ridge(alpha=1.0).fit(X[tr], y[tr])
    log(f"  完成 {time.time()-t0:.1f}s")
    mu_train = y[tr].mean(0)          # 均值基线（退化策略）
    glob_ref = mu_train.copy()        # Δ 谱参照

    preds = {"mean": None, "linear": None}
    mlp = None
    if a.with_mlp:
        from sklearn.neural_network import MLPRegressor
        log("拟合 MLP 探针（64,32；early_stopping）…")
        t0 = time.time()
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=a.seed,
                           early_stopping=True, n_iter_no_change=5, max_iter=60).fit(X[tr], y[tr])
        log(f"  完成 {time.time()-t0:.1f}s")
        preds["mlp"] = None

    out = {"topk": a.topk, "seed": a.seed, "subsets": {}}
    for subset in ["id", "ood_agent", "ood_action", "ood_neuro"]:
        m = sp == subset
        Xs, ys, Ps = X[m], y[m], P[m]
        keys = group_key(Ps)
        uk, pb_true = pseudobulk(ys, keys)
        # DE 基因 = 伪bulk 真值谱跨扰动组方差最大的 top-K
        gene_var = pb_true.var(0)
        de_idx = np.argsort(gene_var)[::-1][: a.topk]
        de_share = float(gene_var[de_idx].sum() / max(gene_var.sum(), 1e-12))

        model_pred = {
            "mean": np.tile(mu_train, (len(ys), 1)),
            "linear": lin.predict(Xs),
        }
        if mlp is not None:
            model_pred["mlp"] = mlp.predict(Xs)

        rec = {
            "n_cells": int(m.sum()), "n_pert_groups": int(len(uk)),
            "de_gene_idx": de_idx.tolist(),
            "de_variance_share": de_share,
            "models": {},
        }
        for name, yp in model_pred.items():
            _, pb_pred = pseudobulk(yp, keys)
            r = {
                "cell_rmse_all": float(np.sqrt(((yp - ys) ** 2).mean())),
                "cell_rmse_de": float(np.sqrt(((yp[:, de_idx] - ys[:, de_idx]) ** 2).mean())),
                "pseudobulk_rmse": float(np.sqrt(((pb_pred - pb_true) ** 2).mean())),
                "pseudobulk_rmse_de": float(np.sqrt(((pb_pred[:, de_idx] - pb_true[:, de_idx]) ** 2).mean())),
                "delta_pcc": macro_pcc(pb_pred - glob_ref, pb_true - glob_ref, axis=1),
                "macro_gene_pcc": macro_pcc(pb_pred, pb_true, axis=0),
                "e_distance": energy_distance(yp, ys, rng),
            }
            # 不确定度（硬约束）：自助法重采样**扰动组**（正确统计单元），B 次
            ng = pb_true.shape[0]
            bs = {"pseudobulk_rmse": [], "pseudobulk_rmse_de": [], "delta_pcc": []}
            brng = np.random.default_rng(a.seed + 1)
            for _ in range(a.bootstrap):
                ii = brng.integers(0, ng, ng)
                bp, bt = pb_pred[ii], pb_true[ii]
                bs["pseudobulk_rmse"].append(np.sqrt(((bp - bt) ** 2).mean()))
                bs["pseudobulk_rmse_de"].append(np.sqrt(((bp[:, de_idx] - bt[:, de_idx]) ** 2).mean()))
                bs["delta_pcc"].append(macro_pcc(bp - glob_ref, bt - glob_ref, axis=1))
            for k, v in bs.items():
                v = np.asarray(v, dtype=float)
                r[k + "_lo"] = float(np.nanpercentile(v, 2.5))
                r[k + "_hi"] = float(np.nanpercentile(v, 97.5))
            rec["models"][name] = r
            log(f"  {subset:11s} {name:7s} cellRMSE={r['cell_rmse_all']:.4f} "
                f"pbRMSE={r['pseudobulk_rmse']:.4f}[{r['pseudobulk_rmse_lo']:.4f},{r['pseudobulk_rmse_hi']:.4f}] "
                f"ΔPCC={r['delta_pcc']:.4f}[{r['delta_pcc_lo']:.4f},{r['delta_pcc_hi']:.4f}] "
                f"macroGenePCC={r['macro_gene_pcc']:.4f} Edist={r['e_distance']:.4f}")
        out["subsets"][subset] = rec

    # 判别力比值：线性 vs 均值 的相对改善，逐指标
    disc = {}
    for subset, rec in out["subsets"].items():
        mm, ml = rec["models"]["mean"], rec["models"]["linear"]
        disc[subset] = {
            "cell_rmse_all_gain_pct": 100 * (mm["cell_rmse_all"] - ml["cell_rmse_all"]) / mm["cell_rmse_all"],
            "cell_rmse_de_gain_pct": 100 * (mm["cell_rmse_de"] - ml["cell_rmse_de"]) / mm["cell_rmse_de"],
            "pseudobulk_rmse_gain_pct": 100 * (mm["pseudobulk_rmse"] - ml["pseudobulk_rmse"]) / mm["pseudobulk_rmse"],
            "pseudobulk_rmse_de_gain_pct": 100 * (mm["pseudobulk_rmse_de"] - ml["pseudobulk_rmse_de"]) / mm["pseudobulk_rmse_de"],
            "delta_pcc_abs_gain": ml["delta_pcc"] - mm["delta_pcc"],
            "e_distance_gain_pct": 100 * (mm["e_distance"] - ml["e_distance"]) / max(abs(mm["e_distance"]), 1e-12),
        }
    out["discriminability_linear_vs_mean"] = disc

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"写出 {a.out}")

    print("\n=== 判别力对照（线性 vs 均值，正值=线性更好）===")
    print("%-11s %14s %14s %14s %14s %12s" % ("subset", "cellRMSE%", "cellRMSE_DE%", "pbRMSE%", "pbRMSE_DE%", "ΔPCC(abs)"))
    for s, v in disc.items():
        print("%-11s %13.2f%% %13.2f%% %13.2f%% %13.2f%% %12.4f" % (
            s, v["cell_rmse_all_gain_pct"], v["cell_rmse_de_gain_pct"],
            v["pseudobulk_rmse_gain_pct"], v["pseudobulk_rmse_de_gain_pct"], v["delta_pcc_abs_gain"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
