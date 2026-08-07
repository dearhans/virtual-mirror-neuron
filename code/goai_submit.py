"""code/goai_submit.py — GOAI test 预测 → submission prediction.csv

1. 用 train_val 全集拟合 CompositionalTwin
2. 加载 test metadata + proteome → 匹配 test 内对照
3. 构建 test 特征（未见过菌株/化合物归零，ψ 仅用于已知 pair）
4. 预测 Δ → 绝对丰度 = control_log2 + Δ → prediction.csv
"""
import os, sys, csv as _csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.goai_loader import build_dataset, RAW_DIR
from goai_compositional_twin import GoaiCompositionalTwin

OUT = "data/submissions/prediction.csv"
MATCH_KEYS = ["Strains", "Medium", "Temperature", "pert_time", "data_source"]
CONTROL_SET = {"water", "dmso"}


def _log2(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log2(np.where(x > 0, x, np.nan))


def main():
    # 1. 全量 train_val 拟合
    print("[1/5] 加载 train_val ...")
    d = build_dataset(cache_path="data/processed/goai_cache.npz")
    X, Y, Y_ctrl, split = d["X"], d["Y_delta"], d["Y_ctrl"], d["split"]
    Dc, Ds = len(d["compound_names"]), len(d["strain_names"])
    Dctx = X.shape[1] - Dc - Ds - 3
    train_mask = np.ones(len(Y), dtype=bool)  # 全量
    ct = GoaiCompositionalTwin(alpha=1.0, random_state=0)
    ct.fit(Y_delta=Y, strain_idx=d["meta"][:, 1],
           comp_idx=np.argmax(X[:, :Dc], axis=1),
           X_ctxt=X[:, Dc:Dc + Dctx], train_mask=train_mask)
    comp_names = list(d["compound_names"])
    strain_names = list(d["strain_names"])
    comp2i = {c: i for i, c in enumerate(comp_names)}
    strain2i = {s: i for i, s in enumerate(strain_names)}

    # 2. 加载 test
    print("[2/5] 加载 test ...")
    raw = RAW_DIR
    meta_t = pd.read_csv(os.path.join(raw, "WAYB_WAYC_metadata_test(1).csv"))
    prot_t = pd.read_csv(os.path.join(raw, "WAYB_WAYC_proteome_raw_test.csv")).set_index("sample_ID")
    meta_t = meta_t.set_index("sample_ID")
    prot_t = prot_t.loc[meta_t.index]
    prot_cols = list(prot_t.columns)
    Y_test_raw = prot_t.values.astype(np.float64)
    Nt = len(meta_t)
    print(f"  test: {Nt} samples, {len(prot_cols)} proteins")

    # 3. 检测 test 内对照
    print("[3/5] 匹配 test 对照 ...")
    is_ctrl = meta_t["perturbation_no_concentration"].str.strip().str.lower().isin(CONTROL_SET).values
    print(f"  对照: {is_ctrl.sum()}  处理: {(~is_ctrl).sum()}")
    ctrl_map = {}
    for i in np.where(is_ctrl)[0]:
        key = tuple(str(meta_t.iloc[i][k]) for k in MATCH_KEYS)
        ctrl_map.setdefault(key, []).append(i)

    # 全局备选对照：test 所有对照的 per-protein median log2
    ctrl_global_log2 = _log2(Y_test_raw[is_ctrl])
    ctrl_global_median = np.nanmedian(ctrl_global_log2, axis=0)
    ctrl_global_median = np.nan_to_num(ctrl_global_median, nan=0.0)

    treat_idx_full = np.where(~is_ctrl)[0]
    matched, matched_ctrl = [], []
    unmatched = []
    for i in treat_idx_full:
        key = tuple(str(meta_t.iloc[i][k]) for k in MATCH_KEYS)
        cands = ctrl_map.get(key, [])
        if cands:
            matched.append(i)
            matched_ctrl.append(np.nanmean(Y_test_raw[cands], axis=0))
        else:
            unmatched.append(i)
    matched = np.array(matched)
    matched_ctrl = np.array(matched_ctrl)
    unmatched = np.array(unmatched)
    print(f"  有匹配对照的处理: {len(matched)}  无匹配: {len(unmatched)}（用全局对照中位数）")

    # 合并：全部处理组
    all_treat = np.concatenate([matched, unmatched]) if len(unmatched) > 0 else matched
    # 对照值：匹配的用 matched_ctrl，未匹配的用全局中位数的 raw
    all_ctrl_log2 = np.zeros((len(all_treat), Y_test_raw.shape[1]), dtype=np.float64)
    for j, i in enumerate(matched):
        all_ctrl_log2[j] = _log2(matched_ctrl[j])
        all_ctrl_log2[j] = np.nan_to_num(all_ctrl_log2[j], nan=0.0)
    for j, i in enumerate(unmatched):
        all_ctrl_log2[len(matched) + j] = ctrl_global_median

    # 4. 构建特征 + 预测
    print("[4/5] 预测 ...")
    # strain/compound encoding on all_treat
    strain_idx = np.array([strain2i.get(str(meta_t.iloc[i]["Strains"]), -1) for i in all_treat])
    comp_idx = np.array([comp2i.get(str(meta_t.iloc[i]["perturbation_no_concentration"]), -1) for i in all_treat])
    ctx_sigs = ["||".join([str(meta_t.iloc[i][k]) for k in MATCH_KEYS])
                for i in all_treat]
    uniq_ctx = sorted(set(ctx_sigs))
    ctx2i = {c: k for k, c in enumerate(uniq_ctx)}
    Ctx_int = np.eye(len(uniq_ctx))[[ctx2i[c] for c in ctx_sigs]].astype(np.float32)
    Dctx_t = len(uniq_ctx)

    # ---- Predict additive ----
    Kp = ct.mu_global.shape[0]
    n_s, n_c = ct.mu_strain.shape[0], ct.mu_compound.shape[0]
    additive = np.tile(ct.mu_global[None, :], (len(all_treat), 1))
    for k, s in enumerate(strain_idx):
        if s >= 0:
            additive[k] += ct.mu_strain[s]
    for k, c in enumerate(comp_idx):
        if c >= 0:
            additive[k] += ct.mu_compound[c]

    # ---- Predict ψ for known pairs ----
    psi_pred = np.zeros_like(additive)
    known = (strain_idx >= 0) & (comp_idx >= 0)
    if known.sum() > 0:
        ki = np.where(known)[0]
        X_int = np.hstack([
            np.eye(n_s)[strain_idx[known]].astype(np.float32),
            np.eye(n_c)[comp_idx[known]].astype(np.float32),
            np.zeros((known.sum(), Dctx), dtype=np.float32),
            np.zeros((known.sum(), n_s * n_c), dtype=np.float32),
        ]).astype(np.float64)
        for i, ii in enumerate(ki):
            X_int[i, n_s + n_c + Dctx + strain_idx[ii] * n_c + comp_idx[ii]] = 1.0
        psi_pred[known] = ct.psi_model.predict(X_int)

    pred_delta = additive + psi_pred

    # 5. 绝对丰度
    print("[5/5] 写出 submission ...")
    pred_abs = all_ctrl_log2 + pred_delta

    sample_ids = list(meta_t.iloc[all_treat].index)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["sample_ID"] + prot_cols)
        for sid, row in zip(sample_ids, pred_abs):
            w.writerow([sid] + [f"{v:.6f}" for v in row])
    print(f"→ {OUT}  ({len(sample_ids)} samples × {len(prot_cols)+1} cols)")


if __name__ == "__main__":
    main()
