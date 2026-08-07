"""code/goai_metrics.py — GOAI 6 模块本地评测器（对齐官方赛道三.pdf §评分模块）

官方权重（§评分模块，100%）：
  M1 绝对保真度            (20%)  适用全部划分
  M2 匹配对照原始 FC Δ PCC (25%)  所有 OOD 划分核心指标
  M3 上下文均值残差(新化合物特异响应) (20%) 适用 test_chem_only / S1
  M4 药物均值残差(新菌株背景调制)    (20%) 适用 test_strain_only / S2
  M5 双重未知·时间外推       (10%)  test_both(S3)+test_time
  M6 高效应蛋白与 DEP 检出    (5%)   全部划分补充指标

官方算法要点（已对齐）：
  M1 = 逐样本 R² 与逐蛋白 R² 分别聚合后平均（样本轴 / 蛋白轴），避免高丰度蛋白掩盖困难样本。
  M2 = Δ_pred = ŷ_treat − y_control；Δ_true = y_treat − y_control；PCC(Δ_pred, Δ_true)。
  M3 = 在 S1(ood_action) 子集，基线 μ_ctx = 同菌株(id训练集)药物 Δ_true 均值；
       PCC(Δ_pred − μ_ctx, Δ_true − μ_ctx)，剔除共享药物响应、聚焦化合物特异机制。
  M4 = 在 S2(ood_agent) 子集，基线 μ_drug = 同化合物(id训练集) Δ_true 均值；
       PCC(Δ_pred − μ_drug, Δ_true − μ_drug)，剔除共享平均机制、评估菌株调制。
  M6 = 对 |Δ_true| > 1 的蛋白（官方阈值），计算全局 PCC（高效应蛋白预测）。

所有指标 NaN-safe（成对删除）。返回字典 + 加权总分。
防泄漏纪律：所有基线统计量仅用训练数据（split=="id"）冻结。
"""
from __future__ import annotations
import numpy as np
from typing import Dict

SUBSETS = ["id", "ood_action", "ood_agent", "ood_s3", "ood_time"]
MODULE_WEIGHTS = {
    "abs_fidelity_r2": 0.20,
    "fc_delta_pcc": 0.25,
    "ctx_residual_pcc": 0.20,
    "drug_residual_pcc": 0.20,
    "double_unk_fc_pcc": 0.10,
    "dep_top_pcc": 0.05,
}
DEP_THRESHOLD = 1.0  # 官方：|Δ_true| > 1 视为高效应蛋白


def nan_global_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    f = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[f], yp[f]
    if yt.size < 2 or yp.std() == 0 or yt.std() == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def nan_global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    f = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[f], yp[f]
    if yt.size < 2:
        return 0.0
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def per_sample_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐样本 R²（每个样本的预测向量 vs 真值向量，基准=该样本蛋白均值），跨样本聚合均值。"""
    n = y_true.shape[0]
    tot, cnt = 0.0, 0
    for i in range(n):
        yt, yp = y_true[i], y_pred[i]
        f = np.isfinite(yt) & np.isfinite(yp)
        if f.sum() < 2:
            continue
        yt, yp = yt[f], yp[f]
        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        if ss_tot > 0:
            tot += 1.0 - ss_res / ss_tot
            cnt += 1
    return float(tot / cnt) if cnt else 0.0


def per_protein_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐蛋白 R²（每个蛋白跨样本的向量 vs 真值向量，基准=该蛋白样本均值），跨蛋白聚合均值。"""
    vals = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        f = np.isfinite(yt) & np.isfinite(yp)
        if f.sum() < 2:
            continue
        yt, yp = yt[f], yp[f]
        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        if ss_tot > 0:
            vals.append(1.0 - ss_res / ss_tot)
    return float(np.mean(vals)) if vals else 0.0


def per_sample_pcc_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐样本 PCC 均值（匹配官方「每个样本算 FC」语义）。"""
    n = y_true.shape[0]
    tot, cnt = 0.0, 0
    for i in range(n):
        yt, yp = y_true[i], y_pred[i]
        f = np.isfinite(yt) & np.isfinite(yp)
        if f.sum() < 2:
            continue
        yt, yp = yt[f], yp[f]
        if yp.std() == 0 or yt.std() == 0:
            continue
        tot += np.corrcoef(yt, yp)[0, 1]
        cnt += 1
    return float(tot / cnt) if cnt else 0.0


def _train_baseline_residual_pcc(
    y_true: np.ndarray, y_pred: np.ndarray,
    group_idx: np.ndarray, sub_mask: np.ndarray, train_mask: np.ndarray,
) -> float:
    """对 sub_mask 子集，基线 = 同 group(id 训练集) 的 Δ_true 均值；
    返回 PCC(Δ_pred − μ, Δ_true − μ)（展平所有蛋白）。"""
    if sub_mask.sum() == 0:
        return 0.0
    # 训练集每组基线（蛋白均值）
    base = {}
    for i in np.where(train_mask)[0]:
        g = group_idx[i]
        base.setdefault(g, []).append(y_true[i])
    mu = {g: np.nanmean(np.stack(v), axis=0) for g, v in base.items()}
    yt_sub, yp_sub = y_true[sub_mask], y_pred[sub_mask]
    gi_sub = group_idx[sub_mask]
    res_t, res_p = [], []
    for k in range(yt_sub.shape[0]):
        g = gi_sub[k]
        if g not in mu:
            continue
        m = mu[g]
        res_t.append(yt_sub[k] - m)
        res_p.append(yp_sub[k] - m)
    if not res_t:
        return 0.0
    return nan_global_pcc(np.stack(res_t), np.stack(res_p))


def evaluate(
    d: Dict,
    pred_abs: np.ndarray,
    pred_delta: np.ndarray,
    model_tag: str = "",
) -> Dict:
    """主评测函数。返回各模块分数 + 加权总分 + 按子集的逐项分数。"""
    Y_abs, Y_delta = d["Y_abs"], d["Y_delta"]
    split = d["split"]
    Dc = len(d["compound_names"])
    comp_idx = np.argmax(d["X"][:, :Dc], axis=1)
    strain_idx = d["meta"][:, 1]
    train_mask = split == "id"

    # ---- M1: 绝对保真度（逐样本 R² + 逐蛋白 R² 分别聚合）----
    ps_r2 = per_sample_r2(Y_abs, pred_abs)
    pp_r2 = per_protein_r2(Y_abs, pred_abs)
    abs_r2 = (ps_r2 + pp_r2) / 2.0
    abs_r2_per = {s: (per_sample_r2(Y_abs[split == s], pred_abs[split == s])
                      + per_protein_r2(Y_abs[split == s], pred_abs[split == s])) / 2.0
                  for s in SUBSETS if (split == s).sum() > 0}

    # ---- M2: 匹配对照 FC Δ PCC（全部划分）----
    fc_pcc = per_sample_pcc_mean(Y_delta, pred_delta)
    fc_pcc_per = {s: per_sample_pcc_mean(Y_delta[split == s], pred_delta[split == s])
                  for s in SUBSETS if (split == s).sum() > 0}

    # ---- M3: 上下文均值残差（S1 / ood_action，同菌株训练基线）----
    m3 = split == "ood_action"
    ctx_pcc = _train_baseline_residual_pcc(Y_delta, pred_delta, strain_idx, m3, train_mask)

    # ---- M4: 药物均值残差（S2 / ood_agent，同化合物训练基线）----
    m4 = split == "ood_agent"
    drug_pcc = _train_baseline_residual_pcc(Y_delta, pred_delta, comp_idx, m4, train_mask)

    # ---- M5: 双重未知·时间外推（ood_s3 + ood_time 的 FC PCC）----
    m5 = (split == "ood_s3") | (split == "ood_time")
    double_unk_pcc = per_sample_pcc_mean(Y_delta[m5], pred_delta[m5]) if m5.sum() else 0.0

    # ---- M6: 高效应蛋白与 DEP 检出（|Δ_true| > 1 蛋白的全局 PCC）----
    abs_delta_mean = np.nanmean(np.abs(Y_delta), axis=0)
    top_mask = abs_delta_mean > DEP_THRESHOLD
    if top_mask.sum() == 0:
        dep_pcc = 0.0
    else:
        dep_pcc = nan_global_pcc(Y_delta[:, top_mask], pred_delta[:, top_mask])

    # ---- 加权总分 ----
    score = (
        MODULE_WEIGHTS["abs_fidelity_r2"] * abs_r2
        + MODULE_WEIGHTS["fc_delta_pcc"] * fc_pcc
        + MODULE_WEIGHTS["ctx_residual_pcc"] * ctx_pcc
        + MODULE_WEIGHTS["drug_residual_pcc"] * drug_pcc
        + MODULE_WEIGHTS["double_unk_fc_pcc"] * double_unk_pcc
        + MODULE_WEIGHTS["dep_top_pcc"] * dep_pcc
    )

    result = {
        "model": model_tag,
        "weighted_score": float(score),
        "modules": {
            "abs_fidelity_r2": float(abs_r2),
            "fc_delta_pcc": float(fc_pcc),
            "ctx_residual_pcc": float(ctx_pcc),
            "drug_residual_pcc": float(drug_pcc),
            "double_unk_fc_pcc": float(double_unk_pcc),
            "dep_top_pcc": float(dep_pcc),
        },
        "per_subset_fc_pcc": {s: float(v) for s, v in fc_pcc_per.items()},
        "per_subset_abs_r2": {s: float(v) for s, v in abs_r2_per.items()},
        "n_per_subset": {s: int((split == s).sum()) for s in SUBSETS},
        "weights": MODULE_WEIGHTS,
    }
    return result


def print_report(result: Dict):
    m = result["modules"]
    print(f"\n{'='*60}")
    print(f"  模型: {result['model']}    加权总分 = {result['weighted_score']:.4f}")
    print(f"{'='*60}")
    print(f"  M1 绝对保真(逐样本+逐蛋白聚合) (20%) = {m['abs_fidelity_r2']:.4f}")
    print(f"  M2 匹配对照 FC Δ PCC         (25%) = {m['fc_delta_pcc']:.4f}")
    print(f"  M3 上下文均值残差(S1,菌株基线)(20%) = {m['ctx_residual_pcc']:.4f}")
    print(f"  M4 药物均值残差(S2,化合物基线)(20%) = {m['drug_residual_pcc']:.4f}")
    print(f"  M5 双重未知·时间外推         (10%) = {m['double_unk_fc_pcc']:.4f}")
    print(f"  M6 高效应蛋白 DEP(|Δ|>1 PCC) (5%) = {m['dep_top_pcc']:.4f}")
    print(f"  -- 按官方子集 FC PCC --")
    for s, v in result["per_subset_fc_pcc"].items():
        print(f"     {s:10s} (n={result['n_per_subset'][s]:5d})  Δ PCC = {v:+.4f}")
    print(f"  -- 按官方子集 Abs R²(聚合) --")
    for s, v in result["per_subset_abs_r2"].items():
        print(f"     {s:10s} (n={result['n_per_subset'][s]:5d})  R² = {v:.4f}")
