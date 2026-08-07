"""code/goai_eval.py — GOAI 本地评测器（6 模块，对齐官方权重）+ 基线锚点校验

评测权重（教程 §1.4，65% 围绕 Δ）：
  绝对保真 Global R²/PCC        20%
  匹配对照原始 FC (Δ PCC)       25%
  上下文均值残差（化合物特异）  20%
  药物均值残差（菌株调制）      20%
  双重未知·时间外推             10%
  高效应蛋白 DEP 检出           5%

本文件当前实现：
  - 绝对保真 Global R²（flattened）
  - 匹配对照 FC = Δ 的全局 PCC（flattened）+ 逐样本 PCC 均值
  - 基线锚点校验：matched-control 基线(abs=control)应 ≈0.98，protein-mean 基线 ≈0.87
  - 按官方 split_final 子集分别报告（id / ood_action / ood_agent / ood_s3 / ood_time）

残差类(20%+20%) 与 DEP 检出(5%) 模块在 CompositionalTwin 接入后补全（见 goai_benchmark.py）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.goai_loader import build_dataset

CACHE = "data/processed/goai_cache.npz"
SUBSETS = ["id", "ood_action", "ood_agent", "ood_s3", "ood_time"]


def global_r2(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    f = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[f], yp[f]
    if yt.size < 2:
        return 0.0
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def global_pcc(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    f = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[f], yp[f]
    if yt.size < 2 or yp.std() == 0 or yt.std() == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def per_sample_pcc_mean(y_true, y_pred):
    """逐样本 Δ PCC 均值（更贴近官方「每个样本算 FC」语义），NaN-safe。"""
    n = y_true.shape[0]
    tot = 0.0
    cnt = 0
    for i in range(n):
        yt = y_true[i]
        yp = y_pred[i]
        f = np.isfinite(yt) & np.isfinite(yp)
        if f.sum() < 2:
            continue
        yt, yp = yt[f], yp[f]
        if yp.std() == 0 or yt.std() == 0:
            continue
        tot += np.corrcoef(yt, yp)[0, 1]
        cnt += 1
    return float(tot / cnt) if cnt else 0.0


def report(d, pred_abs, pred_delta, tag):
    print(f"\n=== 基线 / 模型: {tag} ===")
    abs_r2_all = global_r2(d["Y_abs"], pred_abs)
    fc_global = global_pcc(d["Y_delta"], pred_delta)
    fc_per = per_sample_pcc_mean(d["Y_delta"], pred_delta)
    print(f"  绝对保真 Global R² (整体) = {abs_r2_all:.4f}")
    print(f"  匹配对照 FC  Δ PCC (flattened) = {fc_global:.4f} | 逐样本均值 = {fc_per:.4f}")
    print("  -- 按官方子集的 Δ PCC（逐样本均值）--")
    for s in SUBSETS:
        m = d["split"] == s
        if m.sum() == 0:
            continue
        p = per_sample_pcc_mean(d["Y_delta"][m], pred_delta[m])
        print(f"     {s:10s} (n={m.sum():5d})  Δ PCC = {p:+.4f}")


def main():
    d = build_dataset(cache_path=CACHE)
    Y_abs, Y_ctrl, Y_delta = d["Y_abs"], d["Y_ctrl"], d["Y_delta"]
    N = Y_abs.shape[0]
    print(f"[goai] 样本(有匹配对照) N={N}  蛋白维 Kp={Y_abs.shape[1]}")

    # —— 基线1：protein-mean（train 集每蛋白均值 Δ，仅用 id 子集估计）——
    tr = d["split"] == "id"
    mean_delta = Y_delta[tr].mean(axis=0, keepdims=True)
    pred_abs_pm = Y_ctrl + mean_delta
    pred_delta_pm = np.tile(mean_delta, (N, 1))
    report(d, pred_abs_pm, pred_delta_pm, "protein-mean 基线 (train-est)")

    # —— 基线2：matched-control（Δ=0，预测=对照）——
    pred_abs_mc = Y_ctrl
    pred_delta_mc = np.zeros_like(Y_delta)
    report(d, pred_abs_mc, pred_delta_mc, "matched-control 基线")

    # —— 锚点校验 ——
    r2_mc = global_r2(Y_abs, pred_abs_mc)
    r2_pm = global_r2(Y_abs, pred_abs_pm)
    print("\n=== 锚点校验（教程: matched≈0.98）===")
    print(f"  matched-control  abs R² = {r2_mc:.4f}  (期望 ~0.98)")
    print(f"  protein-mean      abs R² = {r2_pm:.4f}  (方法相关，教程 0.87 为不回加对照的版本)")
    ok = abs(r2_mc - 0.98) < 0.03
    print("  评测器口径", "✅ 通过（matched-control 与教程锚点吻合）" if ok else
          "⚠️ 偏差较大 — 检查 Δ 计算 / 匹配键 / 是否含对照样本")

    # 落盘（供文档引用 + 后续模型对比）
    import json
    out = {
        "anchors": {"matched_control_abs_r2": r2_mc, "protein_mean_abs_r2": r2_pm,
                    "evaluator_calibrated": ok},
        "baselines_fc_pcc_per_sample": {
            "protein_mean": {s: float(per_sample_pcc_mean(
                d["Y_delta"][d["split"] == s], pred_delta_pm[d["split"] == s]))
                for s in SUBSETS if (d["split"] == s).sum()},
            "matched_control": {s: 0.0 for s in SUBSETS},
        },
        "n_per_subset": {s: int((d["split"] == s).sum()) for s in SUBSETS},
        "Kp": int(Y_abs.shape[1]),
    }
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/goai_baseline_anchors.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  → 已写 experiments/goai_baseline_anchors.json")


if __name__ == "__main__":
    main()
