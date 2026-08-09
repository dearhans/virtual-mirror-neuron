#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P3-2 · 判别性指标重裁（re-adjudication）
=======================================

背景（为什么需要这个脚本）
------------------------
W32 之前所有 OOD 判据都用「单细胞 × 全基因 RMSE」。2026-08-07 的
`scripts/de_subset_metric.py` 证明：该指标在 ood_action 上把
线性 vs 均值 的真实差距从 50.7%（伪bulk）压到 6.1%（单细胞全基因），
动态范围被单细胞泊松噪声压缩约 8.4 倍 —— 于是「所有模型都挤在 0.63」
被误读成「无机制判别力」。

本脚本用**同一批 benchmark 预测器**（不是重新实现的简化版）在判别性
指标上重跑，回答唯一的问题：

    换一把有刻度的尺子后，孪生在 ood_action 上是否仍然打不过简单基线？

与 de_subset_metric.py 的区别
----------------------------
- de_subset_metric.py：只跑 mean/linear，用途是**验证指标本身有动态范围**。
- 本脚本：跑 benchmark_ood.py 的全部 7 个预测器，用途是**重判模型**。

效率
----
benchmark_ood.py 的 evaluate_predictor 每个 (预测器 × 子集) 都重新 fit
（28 次拟合）。本脚本每个预测器只 fit 一次，再对 4 个子集分别 predict，
拟合次数 28 → 7。

指标
----
- pseudobulk_rmse       : 按扰动组聚合后的 RMSE（消掉单细胞噪声）
- pseudobulk_rmse_de    : 仅 top-K DE 基因上的伪bulk RMSE（信号最集中处）
- delta_pcc             : 扰动效应向量 (pb − 全局对照) 的 macro PCC（逐扰动组求相关再平均）
- macro_gene_pcc        : 逐基因跨扰动组的 PCC 再平均（对抗「预测均值」陷阱）
- e_distance            : 能量距离（分布层面，非仅一阶矩）
- cell_rmse_all         : 旧主判据，保留用于换算「压缩倍率」

不确定度
--------
所有伪bulk 指标做自助法 95% CI，**重采样单位 = 扰动组**（正确统计单元；
按细胞重采样会低估不确定度，因为同组细胞不独立）。

复现
----
    PYTHONPATH=<repo>/.pylibs python scripts/de_readjudicate.py \
        --config configs/benchmark_ood_norman_canonical.yaml --bootstrap 200 --seed 0
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

_T0 = time.time()


def log(msg):
    print(f"[{time.time()-_T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 指标工具
def group_key(P: np.ndarray) -> np.ndarray:
    """扰动身份 = 独热/多热扰动矩阵的行签名。对照细胞(全 0)自成一组。"""
    keys = []
    for row in P:
        idx = np.flatnonzero(row > 0.5)
        keys.append("ctrl" if idx.size == 0 else "+".join(map(str, sorted(idx.tolist()))))
    return np.asarray(keys)


def pseudobulk(y: np.ndarray, keys: np.ndarray):
    uk = np.unique(keys)
    return uk, np.stack([y[keys == k].mean(0) for k in uk])


def macro_pcc(a: np.ndarray, b: np.ndarray, axis: int) -> float:
    """axis=1: 逐行(逐扰动组)相关后平均；axis=0: 逐列(逐基因)相关后平均。"""
    A, B = (a, b) if axis == 1 else (a.T, b.T)
    out = []
    for u, v in zip(A, B):
        su, sv = u.std(), v.std()
        if su < 1e-12 or sv < 1e-12:
            continue
        out.append(float(np.corrcoef(u, v)[0, 1]))
    return float(np.mean(out)) if out else float("nan")


def energy_distance(Xp: np.ndarray, Xt: np.ndarray, rng, cap=400) -> float:
    def sub(M):
        if len(M) <= cap:
            return M
        return M[rng.choice(len(M), cap, replace=False)]

    A, B = sub(Xp), sub(Xt)

    def md(U, V):
        d = 0.0
        for i in range(0, len(U), 100):
            d += np.linalg.norm(U[i:i + 100, None, :] - V[None, :, :], axis=2).sum()
        return d / (len(U) * len(V))

    return float(2 * md(A, B) - md(A, A) - md(B, B))


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark_ood_norman_canonical.yaml")
    ap.add_argument("--topk", type=int, default=20, help="DE 基因数（按跨扰动组方差选）")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/20260807-de-readjudication.json")
    a = ap.parse_args(argv)

    import yaml
    from benchmark_ood import (  # type: ignore
        load_or_generate, norman_to_compositional_X,
        CompositionalTwinPredictor, CompositionalInteractionPredictor,
        VirtualTwinPredictor, MLPTwinPredictor,
        LinearBaseline, KNNBaseline, MeanBaseline,
    )

    cfg = yaml.safe_load(open(os.path.join(ROOT, a.config), encoding="utf-8"))
    log(f"config={a.config} source={cfg['data'].get('source')}")

    data = load_or_generate(cfg, ROOT)
    X, y, split = data["X"], data["y"], data["split"]
    z_comp = data.get("z_comp")
    comp = bool(cfg["model"].get("compositional"))
    if comp and cfg["data"].get("source") == "norman":
        X = norman_to_compositional_X(X)
    log(f"X={X.shape} y={y.shape} compositional_layout={comp}")

    tr = split == "train"
    Xtr, ytr = X[tr], y[tr]
    ztr = z_comp[tr] if z_comp is not None else None
    str_split = split[tr]

    predictors = [
        CompositionalTwinPredictor(cfg, conformal=bool(cfg["model"].get("conformal", False))),
        CompositionalInteractionPredictor(cfg, conformal=bool(cfg["model"].get("conformal", False))),
        VirtualTwinPredictor(cfg),
        MLPTwinPredictor(cfg),
        LinearBaseline(),
        KNNBaseline(k=cfg["baselines"]["knn_k"]),
        MeanBaseline(),
    ]

    # ---- 每个预测器只拟合一次 ----
    fitted = {}
    for p in predictors:
        t = time.time()
        p.fit(Xtr, ytr, split=str_split, z_comp=ztr)
        fitted[p.name] = p
        log(f"  fit {p.name:32s} {time.time()-t:6.1f}s")

    # 扰动身份取自「组合布局前」的原始多热 P（前 107 列）
    P_raw = data["X"][:, :107]
    keys_all = group_key(P_raw)
    rng = np.random.default_rng(a.seed)

    # 全局对照参照：训练集对照细胞均值（用于算扰动效应向量 Δ）
    ctrl_tr = (keys_all == "ctrl") & tr
    glob_ref = y[ctrl_tr].mean(0) if ctrl_tr.any() else ytr.mean(0)
    log(f"control cells in train: {int(ctrl_tr.sum())}")

    out = {"topk": a.topk, "seed": a.seed, "bootstrap": a.bootstrap,
           "config": a.config, "subsets": {}}

    for subset in ["id", "ood_agent", "ood_action", "ood_neuro"]:
        m = split == subset
        if not m.any():
            continue
        ys, keys = y[m], keys_all[m]
        uk, pb_true = pseudobulk(ys, keys)
        # DE 基因 = 伪bulk 上跨扰动组方差最大的 top-K（真实响应定义，与模型无关）
        de_idx = np.argsort(pb_true.var(0))[::-1][:a.topk]
        de_share = float(pb_true.var(0)[de_idx].sum() / pb_true.var(0).sum())
        rec = {"n_cells": int(m.sum()), "n_pert_groups": int(len(uk)),
               "de_variance_share": de_share, "models": {}}
        log(f"== {subset}: cells={int(m.sum())} groups={len(uk)} DEtop{a.topk}var={de_share:.3f}")

        Xs = X[m]
        for name, p in fitted.items():
            yp = np.asarray(p.predict(Xs))
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
            # 自助法 CI：重采样单位 = 扰动组
            if a.bootstrap > 0 and len(uk) > 2:
                bs = {"pseudobulk_rmse": [], "pseudobulk_rmse_de": [], "delta_pcc": []}
                ng = len(uk)
                for _ in range(a.bootstrap):
                    ii = rng.integers(0, ng, ng)
                    bs["pseudobulk_rmse"].append(np.sqrt(((pb_pred[ii] - pb_true[ii]) ** 2).mean()))
                    bs["pseudobulk_rmse_de"].append(
                        np.sqrt(((pb_pred[np.ix_(ii, de_idx)] - pb_true[np.ix_(ii, de_idx)]) ** 2).mean()))
                    bs["delta_pcc"].append(
                        macro_pcc(pb_pred[ii] - glob_ref, pb_true[ii] - glob_ref, axis=1))
                for k, v in bs.items():
                    v = np.asarray(v, dtype=float)
                    r[k + "_lo"] = float(np.nanpercentile(v, 2.5))
                    r[k + "_hi"] = float(np.nanpercentile(v, 97.5))
            rec["models"][name] = r
            log(f"   {name:32s} pbRMSE={r['pseudobulk_rmse']:.4f} "
                f"pbRMSE_DE={r['pseudobulk_rmse_de']:.4f} dPCC={r['delta_pcc']:.4f} "
                f"mGenePCC={r['macro_gene_pcc']:.4f} cellRMSE={r['cell_rmse_all']:.4f}")
        out["subsets"][subset] = rec

    # ---- 重裁：孪生 vs 最佳简单基线（每个指标独立判） ----
    SIMPLE = ["mean", "linear", "knn"]
    TWINS = ["compositional_twin", "compositional_interaction_twin", "virtual_twin"]
    verdict = {}
    for subset, rec in out["subsets"].items():
        M = rec["models"]
        vs = {}
        for metric, lower_better in [("cell_rmse_all", True), ("pseudobulk_rmse", True),
                                     ("pseudobulk_rmse_de", True), ("delta_pcc", False),
                                     ("macro_gene_pcc", False), ("e_distance", True)]:
            base_vals = [M[b][metric] for b in SIMPLE if b in M]
            best_base = min(base_vals) if lower_better else max(base_vals)
            best_twin_name, best_twin = None, None
            for t in TWINS:
                if t not in M:
                    continue
                v = M[t][metric]
                if best_twin is None or (v < best_twin if lower_better else v > best_twin):
                    best_twin, best_twin_name = v, t
            # CI 重叠判显著性（仅对有 CI 的指标）
            sig = None
            lo_k, hi_k = metric + "_lo", metric + "_hi"
            if best_twin_name and lo_k in M[best_twin_name]:
                bb_name = min(SIMPLE, key=lambda b: M[b][metric]) if lower_better \
                    else max(SIMPLE, key=lambda b: M[b][metric])
                t_lo, t_hi = M[best_twin_name][lo_k], M[best_twin_name][hi_k]
                b_lo, b_hi = M[bb_name][lo_k], M[bb_name][hi_k]
                sig = bool(t_hi < b_lo or b_hi < t_lo)
            vs[metric] = {
                "best_twin": best_twin_name, "twin_value": best_twin,
                "best_baseline_value": best_base,
                "twin_wins": bool((best_twin < best_base) if lower_better else (best_twin > best_base)),
                "ci_disjoint": sig,
                "rel_gain_pct": float(100 * (best_base - best_twin) / abs(best_base)) if lower_better
                else float(100 * (best_twin - best_base) / (abs(best_base) + 1e-9)),
            }
        verdict[subset] = vs
    out["readjudication"] = verdict

    # ---- 指标压缩倍率：判别力(线性 vs 均值) 在旧尺子 vs 新尺子 ----
    comp_ratio = {}
    for subset, rec in out["subsets"].items():
        M = rec["models"]
        if "mean" not in M or "linear" not in M:
            continue
        old = 100 * (M["mean"]["cell_rmse_all"] - M["linear"]["cell_rmse_all"]) / M["mean"]["cell_rmse_all"]
        new = 100 * (M["mean"]["pseudobulk_rmse"] - M["linear"]["pseudobulk_rmse"]) / M["mean"]["pseudobulk_rmse"]
        new_de = 100 * (M["mean"]["pseudobulk_rmse_de"] - M["linear"]["pseudobulk_rmse_de"]) / M["mean"]["pseudobulk_rmse_de"]
        comp_ratio[subset] = {"old_cell_rmse_gain_pct": old, "new_pb_gain_pct": new,
                              "new_pb_de_gain_pct": new_de,
                              "compression_factor": float(new / old) if old > 1e-9 else None}
    out["metric_compression"] = comp_ratio

    op = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"written -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
