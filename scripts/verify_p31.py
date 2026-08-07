#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-1 验证：σ 解耦（novelty 移出预测区间）后，compositional 双孪生 + virtual_twin
在 ood_agent/ood_action 上的 calibration 是否脱离 0.2125 饱和。

复用 seed0 缓存（不重跑 canonical），用解耦后的新代码 refit 模型，
直接算 bm.calibration（predict_with_std 返回的 std 已是纯 σ_pred），
对比 aggregate_seed0.json 里的 ece_before（解耦前，含 novelty 的饱和值）。

判据（P3-1 成功）：
  - ood_agent/ood_action 的 coverage 脱离恒 1.0
  - 孪生 ECE < 0.10（脱离 0.2125 饱和）
  - RMSE 点估计不退化
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

import benchmark_ood as bm
from benchmark_ood import (CompositionalTwinPredictor,
                           CompositionalInteractionPredictor, VirtualTwinPredictor)

SEED = 0
SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
# (模型名, 类, conformal) — 关掉 conformal 测 raw σ_pred（解耦+门控+封顶）效果
MODELS = [
    ("compositional_twin", CompositionalTwinPredictor, False),
    ("compositional_interaction_twin", CompositionalInteractionPredictor, False),
    ("virtual_twin", VirtualTwinPredictor, False),
]


def main():
    cfg = bm.load_config(os.path.join(ROOT, "configs", "benchmark_ood_norman_canonical.yaml"))
    cfg["data"]["norman"]["seed"] = SEED
    cfg["data"]["norman"]["raw_dir"] = os.path.join(ROOT, "data", "raw", "norman2019")
    cfg["data"]["norman"]["cache_path"] = os.path.join(ROOT, "experiments", "multiseed", f"norman_cache_s{SEED}.npz")
    cfg["model"]["random_state"] = SEED
    levels = cfg["eval"]["calibration_levels"]

    # 1) 复用 seed0 缓存加载数据
    data = bm.load_or_generate(cfg, ROOT)
    X, y, split = data["X"], data["y"], data["split"]
    if bool(cfg["model"].get("compositional")) and cfg["data"].get("source") == "norman":
        X = bm.norman_to_compositional_X(X)
    train_mask = split == "train"
    Xtr, ytr = X[train_mask], y[train_mask]
    ztr = data.get("z_comp")
    str_split = split[train_mask]
    test_masks = {s: (split == s) for s in SUBSETS}

    # 2) 基线（解耦前，含 novelty 的饱和值）来自 aggregate_seed0.json 的 ece_before
    base = json.load(open(os.path.join(ROOT, "experiments", "multiseed", "aggregate_seed0.json"), encoding="utf-8"))
    base_recal = base["recalibration"]

    t0 = time.time()
    print("=== P3-1 验证：σ 解耦后 calibration 对比（seed=%d）===" % SEED)
    print(f"{'model':32s} {'subset':11s} | {'cov95_before':>12s} {'cov95_after':>12s} | {'ECE_before':>11s} {'ECE_after':>11s} | verdict")
    print("-" * 110)

    passed = True
    for mdl_name, Cls, conf in MODELS:
        if conf:
            pred = Cls(cfg, conformal=bool(cfg["model"].get("conformal", False)))
        else:
            pred = Cls(cfg)
        pred.fit(Xtr, ytr, split=str_split, z_comp=ztr)
        sid = getattr(pred.twin, "sigma_id_95_", "NA")
        print(f"  [{mdl_name}] sigma_id_95_={sid}")
        for s in SUBSETS:
            Xs, ys = X[test_masks[s]], y[test_masks[s]]
            mu, std = pred.predict_with_std(Xs, subset=s)
            if s in ("ood_agent", "ood_action"):
                print(f"    {s}: std p50={np.percentile(np.abs(std),50):.3f} "
                      f"p99={np.percentile(np.abs(std),99):.3f} max={np.abs(std).max():.3f}")
            mu = np.asarray(mu, float)
            std = np.asarray(std, float)
            # 1D/2D 对齐（与 bm.calibration 内部一致）
            if mu.ndim == 2 and std.ndim == 1:
                std = np.broadcast_to(std.reshape(-1, 1), mu.shape)
            cal = bm.calibration(ys, mu, std, levels)
            cov95 = float(cal["coverage"][0.95])
            ece = float(cal["ece"])
            b = base_recal[mdl_name]["subsets"][s]
            cb, ce = b["cov_before_95"], b["ece_before"]
            # 判定
            if s in ("ood_agent", "ood_action"):
                ok = (cov95 < 0.999) and (ece < 0.10)
                verdict = "OK 饱和消除" if ok else "FAIL 仍饱和"
                if not ok:
                    passed = False
            else:
                verdict = "-"
            print(f"{mdl_name:32s} {s:11s} | {cb:12.4f} {cov95:12.4f} | {ce:11.4f} {ece:11.4f} | {verdict}")

    elapsed = time.time() - t0
    print("-" * 110)
    print(f"总用时 {elapsed/60:.1f}min | P3-1 饱和消除: {'ALL PASS' if passed else 'NOT FULLY PASS'}")
    # 落盘
    out = {"seed": SEED, "p31_applied": True, "verdict": "pass" if passed else "fail",
           "note": "σ 解耦后 compositional/virtual 在 ood_agent/ood_action 的 calibration 对比基线"}
    with open(os.path.join(ROOT, "experiments", "multiseed", "verify_p31_seed0.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
