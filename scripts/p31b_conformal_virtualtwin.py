#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-1b · 修 virtual_twin 未走 conformal 路径的灾难性过度自信
================================================================

缺陷（来自 20260810 基准）：
    virtual_twin 在 ID / ood_agent / ood_action / ood_neuro 的 ECE 分别为
    0.722 / 0.733 / 0.706 / 0.72x，覆盖@0.9 仅 0.078 / 0.065 / 0.097 / ...
    —— 区间严重偏窄、灾难性过度自信。根因：canonical 管线里 `VirtualTwinPredictor`
    以 `VirtualTwinPredictor(cfg)` 实例化，**没有走 conformal 校准路径**，直接返回
    集成 epistemic std，而该 std 在 Norman 向量响应上系统性偏小。

破法（TRIZ 空间分离 / 流程分离）：
    把 virtual_twin 的不确定度拆成「集成 epistemic（预测用）」与「conformal 校准（区间用）」
    两条通道——与组合性孪生(P1/P2)完全对齐：在训练集拆校准子集，按归一化共形分数
    `|残差|/模型std` 取分位数 q，预测时区间半宽 = q × 模型 std。

本脚本自包含、只读式验证（类比 p42 消融）：
    - 用与 canonical 完全相同的 VirtualTwinPredictor + 组合性布局 X；
    - 在训练集 80% 上 fit，20% 上校准 q（与组合性孪生同协议）；
    - 在 4 个子集上分别用「raw std（before）」与「q×std（after）」算 ECE/覆盖度；
    - 同时把当周基准里的 virtual_twin 校准作为「生产基线」引用，证明 conformal 能把它拉回可接受区。

用法：
    python scripts/p31b_conformal_virtualtwin.py --date 20260812
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(ROOT, "code") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np  # noqa: E402

from benchmark_ood import (  # noqa: E402
    load_config,
    load_or_generate,
    norman_to_compositional_X,
    VirtualTwinPredictor,
    calibration,
)


def _cov_get(coverage, level):
    if level in coverage:
        return coverage[level]
    return coverage.get(str(level), float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-1b virtual_twin conformal calibration")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "benchmark_ood_norman_canonical.yaml"))
    ap.add_argument("--date", default="20260812")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    data = load_or_generate(cfg, a.root)
    X, y, split = data["X"], data["y"], data["split"]

    comp = bool(cfg["model"].get("compositional"))
    if comp:
        X = norman_to_compositional_X(X)

    subsets = ["id", "ood_agent", "ood_action", "ood_neuro"]
    test_masks = {s: (split == s) for s in subsets}
    train_mask = split == "train"
    Xtr, ytr = X[train_mask], y[train_mask]

    m = cfg["model"]
    rng = np.random.default_rng(int(m.get("random_state", 0)))
    idx = np.arange(len(Xtr))
    rng.shuffle(idx)
    ncal = max(1, int(len(idx) * float(m.get("conformal_calib_frac", 0.2))))
    cidx = idx[:ncal]
    fidx = idx[ncal:]

    # —— before/after 共用同一个在 fidx 上 fit 的 virtual_twin（隔离 conformal 效应）——
    # VirtualTwin.predict 返回每样本标量 std（n,），残差须先按基因维取范数再归一化。
    vt = VirtualTwinPredictor(cfg)
    vt.fit(Xtr[fidx], ytr[fidx])
    mu_cal, std_cal = vt.predict_with_std(Xtr[cidx])
    # 关键：score 必须与 calibration() 的覆盖度口径一致——逐 (样本,基因) 标准化残差，
    # 而非跨基因 L2 范数（后者会引入 √K 偏置，使 q 暴涨→区间饱和→ECE=0.212）。
    resid_std = np.abs(ytr[cidx] - mu_cal) / np.clip(std_cal.reshape(-1, 1), 1e-9, None)
    scores = resid_std.ravel()
    alpha = float(m.get("conformal_alpha", 0.9))
    q = float(np.quantile(scores, alpha)) if len(scores) else 1.645

    levels = cfg["eval"]["calibration_levels"]
    out_subsets = {}
    for s in subsets:
        Xte, yte = X[test_masks[s]], y[test_masks[s]]
        mu, std = vt.predict_with_std(Xte)
        cal_before = calibration(yte, mu, std, levels)
        cal_after = calibration(yte, mu, q * std, levels)
        out_subsets[s] = {
            "before": {
                "ece": cal_before["ece"],
                "coverage": {float(k): float(v) for k, v in cal_before["coverage"].items()},
            },
            "after": {
                "ece": cal_after["ece"],
                "coverage": {float(k): float(v) for k, v in cal_after["coverage"].items()},
            },
        }

    # —— 生产基线：当周基准里 virtual_twin 的校准（full fit，无 conformal）——
    bench_path = os.path.join(a.root, f"experiments/{a.date}-benchmark.json")
    prod = {}
    if os.path.exists(bench_path):
        bench = json.load(open(bench_path, encoding="utf-8"))
        for s in subsets:
            c = bench.get("calibration", {}).get(s, {}).get("virtual_twin")
            if c:
                prod[s] = {
                    "ece": c.get("ece"),
                    "coverage": {float(k): float(_cov_get(c["coverage"], k)) for k in levels},
                }

    payload = {
        "slot": "P3-1b",
        "title": "virtual_twin 走 conformal 路径修过度自信",
        "conformal_q": q,
        "alpha": alpha,
        "n_fit": int(len(fidx)),
        "n_calib": int(len(cidx)),
        "subsets": out_subsets,
        "production_baseline_virtual_twin": prod,
        "levels": [float(k) for k in levels],
    }

    out_path = os.path.join(a.root, f"experiments/{a.date}-p31b-conformal-virtualtwin.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # —— 控制台摘要 ——
    print(f"[p31b] conformal q={q:.4f} (alpha={alpha}, n_calib={len(cidx)})")
    print(f"{'subset':12s} {'prod_ECE':>9s} {'raw_ECE':>9s} {'conf_ECE':>9s} {'conf_cov@0.9':>12s}")
    for s in subsets:
        pe = prod.get(s, {}).get("ece", float("nan"))
        be = out_subsets[s]["before"]["ece"]
        ae = out_subsets[s]["after"]["ece"]
        ac = _cov_get(out_subsets[s]["after"]["coverage"], 0.9)
        print(f"{s:12s} {pe:9.3f} {be:9.3f} {ae:9.3f} {ac:12.3f}")
    print(f"[p31b] 写出: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
