#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实重标定驱动（seed=0）：读已有 canonical JSON 拿 RMSE/flags，复用缓存跑 recalibration，写 aggregate.json。
不改动 benchmark_ood.py / norman_adapter.py。目的：用真值验证/推翻 W32 narrative 的 R2（ECE 0.212→0.056）。
"""
import os, sys, json, time, datetime
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "code")):
    if p not in sys.path: sys.path.insert(0, p)
import benchmark_ood as bm
from benchmark_ood import (CompositionalTwinPredictor, CompositionalInteractionPredictor,
                           VirtualTwinPredictor, MLPTwinPredictor)

SEED = 0
RECAL_MODELS = [("compositional_twin", CompositionalTwinPredictor),
                ("compositional_interaction_twin", CompositionalInteractionPredictor),
                ("virtual_twin", VirtualTwinPredictor),
                ("mlp", MLPTwinPredictor)]

def main():
    cfg = bm.load_config(os.path.join(ROOT, "configs", "benchmark_ood_norman_canonical.yaml"))
    cfg["data"]["norman"]["seed"] = SEED
    cfg["data"]["norman"]["raw_dir"] = os.path.join(ROOT, "data", "raw", "norman2019")
    cfg["data"]["norman"]["cache_path"] = os.path.join(ROOT, "experiments", "multiseed", f"norman_cache_s{SEED}.npz")
    cfg["model"]["random_state"] = SEED
    levels = cfg["eval"]["calibration_levels"]

    # 1) 读已有 canonical JSON（真实 RMSE/flags）
    canon_path = os.path.join(ROOT, "experiments", "multiseed", f"seed{SEED}", "20260806-benchmark.json")
    with open(canon_path, encoding="utf-8") as f:
        report = json.load(f)
    print(f"[recal] 载入 canonical RMSE/flags: {canon_path}")

    # 2) 重标定：复用缓存重载数据，refit 4 模型取逐样本(mu,std)
    t0 = time.time()
    data = bm.load_or_generate(cfg, ROOT)
    X, y, split = data["X"], data["y"], data["split"]
    if bool(cfg["model"].get("compositional")) and cfg["data"].get("source") == "norman":
        X = bm.norman_to_compositional_X(X)
    train_mask = split == "train"
    Xtr, ytr = X[train_mask], y[train_mask]
    ztr = data.get("z_comp")
    str_split = split[train_mask]
    present = report["subsets"]
    test_masks = {s: (split == s) for s in present}

    recal = {}
    for mdl_name, Cls in RECAL_MODELS:
        # compositional 类接受 conformal；virtual_twin / mlp 不接受
        if mdl_name in ("compositional_twin", "compositional_interaction_twin"):
            pred = Cls(cfg, conformal=bool(cfg["model"].get("conformal", False)))
        else:
            pred = Cls(cfg)
        pred.fit(Xtr, ytr, split=str_split, z_comp=ztr)
        per_subset = {}
        id_r = id_std = None
        for s in present:
            Xs, ys = X[test_masks[s]], y[test_masks[s]]
            mu, std = pred.predict_with_std(Xs, subset=s)
            mu = np.asarray(mu, float); std = np.asarray(std, float)
            if mu.ndim == 2 and std.ndim == 1:
                std = std.reshape(-1, 1)
            r = ys - mu
            per_subset[s] = {"mu": mu, "std": std, "y": ys}
            if s == "id":
                id_r = r  # 2D (n,K)
                # std 可能是 2D(逐基因) 或 1D(逐样本)；统一广播成 (n,K) 以匹配 r，与 calibration 内部广播一致
                is_per_sample = std.ndim == 1 or (std.ndim == 2 and std.shape[1] == 1)
                id_std = np.broadcast_to(std.reshape(-1, 1), r.shape) if is_per_sample else std
        id_std_safe = np.clip(np.abs(id_std), 1e-9, None)
        s_val = float(np.sqrt(np.mean((id_r.ravel() / id_std_safe.ravel()) ** 2)))
        model_recal = {"s": s_val, "subsets": {}}
        for s in present:
            ps = per_subset[s]
            cal_before = bm.calibration(ps["y"], ps["mu"], ps["std"], levels)  # 传 2D std，内部 std[:,0]，与 run() 一致
            std_cal = ps["std"] * s_val
            cal_after = bm.calibration(ps["y"], ps["mu"], std_cal, levels)
            model_recal["subsets"][s] = {
                "ece_before": float(cal_before["ece"]),
                "ece_after": float(cal_after["ece"]),
                "cov_before_95": float(cal_before["coverage"][0.95]),
                "cov_after_95": float(cal_after["coverage"][0.95]),
            }
        recal[mdl_name] = model_recal
        print(f"  [{mdl_name}] s={s_val:.4f}  ood_action ECE {model_recal['subsets']['ood_action']['ece_before']:.4f}"
              f"→{model_recal['subsets']['ood_action']['ece_after']:.4f}")

    elapsed = time.time() - t0
    # 3) 聚合（seed=0 单种子）
    agg = {"seeds": [SEED], "source": "norman", "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "note": "seed=0 only; canonical RMSE/flags 来自 experiments/multiseed/seed0/20260806-benchmark.json; recalibration 为本脚本实算",
           "rmse": report["results"], "flags": report["flags"], "n_train": report["n_train"], "n_total": report["n_total"],
           "recalibration": recal}
    out_dir = os.path.join(ROOT, "experiments", "multiseed")
    agg_path = os.path.join(out_dir, "aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"[recal] 落盘 {agg_path} 用时 {elapsed/60:.1f}min")

if __name__ == "__main__":
    main()
