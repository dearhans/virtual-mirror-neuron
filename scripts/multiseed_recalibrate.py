#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiseed_recalibrate.py — 多种子稳健性 + 不确定度重标定（不改动 benchmark_ood.py / norman_adapter.py）

背景：原每周 OOD 基准 seed 固定（norman seed=0）且 load_norman 命中固定缓存时无视 seed，
导致多次运行 bit-identical、零新信息。本脚本：

(1) 多种子稳健性：对 5 个独立种子，各用【独立 cache_path】强制重解析 Norman 数据
    （不同 seed → 不同 held-out 基因/基因对/批次切分 + 不同模型初始化），
    调用 bm.run() 取得与 canonical 完全一致的 RMSE / bootstrap-CI / 「疑似仅记忆」判定。

(2) 不确定度重标定（约束③：先校准再比较）：
    run() 只返回聚合 coverage/ECE，不暴露逐样本 (mu, σ, y)。
    故本脚本在每种子下重载数据（命中该种子缓存，快），refit 含 σ 的 4 个模型，
    取逐样本残差 r = y - mu 与 σ，仅在 ID 子集拟合单一标量
        s = rms(r_id / σ_id)
    再 σ_cal = σ * s（**乘性缩放**，修正「当除数」的方向错误），
    用 bm.calibration 重算各子集 coverage / ECE。

产出：
    experiments/multiseed/aggregate.json          跨种子聚合（RMSE / flags / recalibration 前→后）
    experiments/multiseed/seed{SEED}/benchmark.{md,json}  每种子 canonical 报告（溯源）

用法：
    python scripts/multiseed_recalibrate.py --seeds 0,7,13,29,51
    python scripts/multiseed_recalibrate.py --seeds 0 --pilot     # 单种子试跑+计时
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(ROOT, "code") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "code"))

import benchmark_ood as bm  # noqa: E402
from benchmark_ood import (  # noqa: E402
    CompositionalTwinPredictor,
    CompositionalInteractionPredictor,
    VirtualTwinPredictor,
    MLPTwinPredictor,
)

SEED_DEFAULT = "0,7,13,29,51"
# 含 σ 的模型（recalibration 目标）。baselines 的 σ 由残差 std 给出，同理由 s 缩放，亦纳入。
RECAL_MODELS = ["compositional_twin", "compositional_interaction_twin", "virtual_twin", "mlp"]


def build_cfg(seed: int) -> dict:
    """加载 canonical config，按种子覆盖切分 seed / 模型 init / 独立缓存 / 隔离输出。"""
    cfg = bm.load_config(os.path.join(ROOT, "configs", "benchmark_ood_norman_canonical.yaml"))
    cfg["data"]["norman"]["seed"] = int(seed)
    cfg["data"]["norman"]["raw_dir"] = os.path.join(ROOT, "data", "raw", "norman2019")
    # 关键：独立 cache_path → 强制该种子重解析（否则命中固定缓存、无视 seed、白跑）
    cfg["data"]["norman"]["cache_path"] = os.path.join(
        ROOT, "experiments", "multiseed", f"norman_cache_s{seed}.npz")
    cfg["model"]["random_state"] = int(seed)
    cfg["output"]["dir"] = os.path.join("experiments", "multiseed", f"seed{seed}")
    cfg["output"]["prefix"] = "benchmark"
    return cfg


def run_seed(seed: int):
    """返回 (report_dict, recal_dict)。report 来自 bm.run()（canonical 语义）；recal 为本脚本逐样本重标定。"""
    t0 = time.time()
    cfg = build_cfg(seed)
    # (1) canonical run —— RMSE / flags / 原始 calibration
    report = bm.run(cfg, ROOT)
    # 溯源：把每种子 canonical 报告落盘
    try:
        bm.write_report(report, ROOT)
    except Exception as e:  # pragma: no cover
        print(f"[seed {seed}] write_report 跳过: {e}")

    # (2) 重标定 —— 重载数据（命中本种子缓存，快），refit 含 σ 模型取逐样本数组
    data = bm.load_or_generate(cfg, ROOT)
    X, y, split = data["X"], data["y"], data["split"]
    source = cfg["data"].get("source")
    comp = bool(cfg["model"].get("compositional"))
    if comp and source == "norman":
        X = bm.norman_to_compositional_X(X)
    train_mask = split == "train"
    Xtr, ytr = X[train_mask], y[train_mask]
    ztr = data.get("z_comp")
    str_split = split[train_mask]
    present = report["subsets"]
    test_masks = {s: (split == s) for s in present}

    levels = cfg["eval"]["calibration_levels"]
    recal = {}
    for mdl_name in RECAL_MODELS:
        if mdl_name == "compositional_twin":
            pred = CompositionalTwinPredictor(cfg, conformal=bool(cfg["model"].get("conformal", False)))
        elif mdl_name == "compositional_interaction_twin":
            pred = CompositionalInteractionPredictor(cfg, conformal=bool(cfg["model"].get("conformal", False)))
        elif mdl_name == "virtual_twin":
            pred = VirtualTwinPredictor(cfg)
        else:
            pred = MLPTwinPredictor(cfg)
        pred.fit(Xtr, ytr, split=str_split, z_comp=ztr)

        per_subset = {}
        id_r = id_std = None
        for s in present:
            Xs, ys = X[test_masks[s]], y[test_masks[s]]
            mu, std = pred.predict_with_std(Xs, subset=s)
            mu = np.asarray(mu, dtype=float)
            std = np.asarray(std, dtype=float)
            if mu.ndim == 2 and std.ndim == 1:
                std = std.reshape(-1, 1)
            std1d = (std[:, 0] if std.ndim == 2 else std.ravel())
            r = ys - mu
            per_subset[s] = {"mu": mu, "std": std, "std1d": std1d, "r": r, "y": ys}
            if s == "id":
                id_r = r  # 2D (n,K)
                # std 形状 3 种：2D 多列(逐基因) / 2D 单列(逐样本) / 1D(逐样本)；
                # 统一广播成 (n,K) 以匹配 r（与 bm.calibration 内部广播一致）
                is_per_sample = std.ndim == 1 or (std.ndim == 2 and std.shape[1] == 1)
                id_std = np.broadcast_to(std.reshape(-1, 1), r.shape) if is_per_sample else std

        # 仅 ID 拟合标量 s（guard 除零）
        id_std_safe = np.clip(np.abs(id_std), 1e-9, None)
        s_val = float(np.sqrt(np.mean((id_r.ravel() / id_std_safe.ravel()) ** 2)))

        model_recal = {"s": s_val, "subsets": {}}
        for s in present:
            ps = per_subset[s]
            # 校准前（用原始 σ 重算，应与 run() 的 calibration 一致，作交叉校验）
            cal_before = bm.calibration(ps["y"], ps["mu"], ps["std"], levels)
            # 校准后：σ_cal = σ * s（乘性，修正除/乘方向）
            std_cal = ps["std"] * s_val
            cal_after = bm.calibration(ps["y"], ps["mu"], std_cal, levels)
            cov_before = {float(k): float(v) for k, v in cal_before["coverage"].items()}
            cov_after = {float(k): float(v) for k, v in cal_after["coverage"].items()}
            model_recal["subsets"][s] = {
                "ece_before": float(cal_before["ece"]),
                "ece_after": float(cal_after["ece"]),
                "cov_before": cov_before,
                "cov_after": cov_after,
            }
        recal[mdl_name] = model_recal

    elapsed = time.time() - t0
    print(f"[seed {seed}] 完成 用时 {elapsed/60:.1f}min  "
          f"n_train={report['n_train']} n_total={report['n_total']}  "
          f"ood_action_flag={'疑似仅记忆' if any('疑似' in f['verdict'] for f in report['flags']) else '无'}")
    return report, recal


def aggregate(seeds, reports, recals):
    agg = {"seeds": seeds, "source": "norman", "generated_at": _now()}
    # —— RMSE 聚合 ——
    subsets = reports[0]["subsets"]
    models = list(reports[0]["results"][subsets[0]].keys())
    rmse = {}
    for s in subsets:
        rmse[s] = {}
        for m in models:
            vals = [rep["results"][s][m]["rmse"] for rep in reports]
            rmse[s][m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": [round(float(v), 4) for v in vals],
            }
    agg["rmse"] = rmse
    agg["models"] = models
    agg["subsets"] = subsets

    # —— flags 聚合（ood_action 疑似仅记忆在多少种子触发）——
    flag_models = list(reports[0]["flags"][0]["twin"]) if reports[0]["flags"] else []
    ood_action_mem = sum(
        1 for rep in reports
        if any(f["subset"] == "ood_action" and "疑似" in f["verdict"] for f in rep["flags"])
    )
    agg["flags"] = {
        "ood_action_mem_seeds": ood_action_mem,
        "ood_action_mem_count": f"{ood_action_mem}/{len(seeds)}",
        "per_seed_detail": [
            {"seed": sd, "flags": [{"subset": f["subset"], "twin": f["twin"], "verdict": f["verdict"]}
                                   for f in rep["flags"]]}
            for sd, rep in zip(seeds, reports)
        ],
    }

    # —— recalibration 聚合 ——
    rec = {}
    for mdl in RECAL_MODELS:
        rec[mdl] = {"s": _meanstd([rc[mdl]["s"] for rc in recals])}
        rec[mdl]["subsets"] = {}
        for s in subsets:
            eb = [rc[mdl]["subsets"][s]["ece_before"] for rc in recals]
            ea = [rc[mdl]["subsets"][s]["ece_after"] for rc in recals]
            c95a = [rc[mdl]["subsets"][s]["cov_after"][0.95] for rc in recals]
            rec[mdl]["subsets"][s] = {
                "ece_before": _meanstd(eb),
                "ece_after": _meanstd(ea),
                "cov95_after": _meanstd(c95a),
            }
    agg["recalibration"] = rec
    return agg


def _meanstd(xs):
    xs = [float(v) for v in xs]
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "per_seed": [round(v, 4) for v in xs]}


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=SEED_DEFAULT, help="逗号分隔的种子列表")
    ap.add_argument("--pilot", action="store_true", help="只跑第一个种子（试跑+计时）")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    if args.pilot:
        seeds = seeds[:1]

    print(f"[multiseed] 种子={seeds}")
    reports, recals = [], []
    for sd in seeds:
        rep, rc = run_seed(sd)
        reports.append(rep)
        recals.append(rc)

    agg = aggregate(seeds, reports, recals)
    out_dir = os.path.join(ROOT, "experiments", "multiseed")
    os.makedirs(out_dir, exist_ok=True)
    agg_path = os.path.join(out_dir, f"aggregate_{'_'.join(map(str, seeds))}.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"[multiseed] 聚合落盘: {agg_path}")
    # 终端摘要
    print("\n=== RMSE 均值±std（跨种子）===")
    for s in agg["subsets"]:
        row = "  ".join(f"{m}={agg['rmse'][s][m]['mean']:.3f}±{agg['rmse'][s][m]['std']:.3f}"
                        for m in agg["models"])
        print(f"  {s:10s} {row}")
    print(f"\nood_action 疑似仅记忆: {agg['flags']['ood_action_mem_count']}")
    print("=== recalibration: ECE 前→后（均值±std）===")
    for mdl in RECAL_MODELS:
        line = []
        for s in agg["subsets"]:
            eb = agg["recalibration"][mdl]["subsets"][s]["ece_before"]
            ea = agg["recalibration"][mdl]["subsets"][s]["ece_after"]
            line.append(f"{s}:{eb['mean']:.3f}→{ea['mean']:.3f}")
        print(f"  {mdl:30s} " + "  ".join(line))


if __name__ == "__main__":
    main()
