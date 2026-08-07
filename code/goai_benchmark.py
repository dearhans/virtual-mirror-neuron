"""code/goai_benchmark.py — GOAI 虚拟细胞扰动响应基准评测（官方 split_final）

模型：
  baseline_protein_mean — 每蛋白均值 Δ（train 集估计）
  baseline_matched_ctrl  — Δ=0（预测=对照）
  linear                — Ridge 回归 on X → Δ（多输出）
  mlp                   — MLP(256,128) on X → Δ

评测：goai_metrics.evaluate() → 6 模块加权分 + 逐子集 FC PCC。
输出：experiments/goai_benchmark.json + 控制台报告。
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.goai_loader import build_dataset
from goai_metrics import evaluate, print_report

CACHE = "data/processed/goai_cache.npz"
OUT = "experiments/goai_benchmark.json"


def get_baseline_preds(d, tag):
    """返回 (pred_abs, pred_delta, train_mask_for_estimation)"""
    tr = d["split"] == "id"
    Y_ctrl, Y_delta = d["Y_ctrl"], d["Y_delta"]
    if tag == "matched_control":
        return Y_ctrl, np.zeros_like(Y_delta)
    elif tag == "protein_mean":
        # 仅用 train 集计算每蛋白均值，NaN-safe
        mean_delta = np.full((1, Y_delta.shape[1]), np.nan)
        for j in range(Y_delta.shape[1]):
            col = Y_delta[tr, j]
            f = np.isfinite(col)
            mean_delta[0, j] = col[f].mean() if f.sum() > 0 else 0.0
        return Y_ctrl + mean_delta, np.tile(mean_delta, (len(Y_delta), 1))
    else:
        raise ValueError(f"unknown baseline tag {tag}")


def run_model(name, pred_abs, pred_delta, d, results):
    r = evaluate(d, pred_abs, pred_delta, model_tag=name)
    print_report(r)
    results.append({k: v for k, v in r.items() if k != "weights"})


def main():
    d = build_dataset(cache_path=CACHE)
    X, Y, Y_abs, Y_ctrl, split = d["X"], d["Y_delta"], d["Y_abs"], d["Y_ctrl"], d["split"]
    train = split == "id"
    Kp = Y.shape[1]
    results = []

    # ---- Baselines ----
    print("\n" + "=" * 60)
    print("  GOAI 虚拟细胞 6 模块基准评测（官方 split_final）")
    print("=" * 60)

    for tag in ["matched_control", "protein_mean"]:
        pa, pd = get_baseline_preds(d, tag)
        run_model(f"baseline_{tag}", pa, pd, d, results)

    # ---- 公用：NaN 蛋白用 train 均值填充（否则 sklearn 报错）；全缺列的蛋白填 0 ----
    Y_tr_raw = Y[train]
    Y_mean = np.full(Y.shape[1], 0.0)
    for j in range(Y.shape[1]):
        col = Y_tr_raw[:, j]
        f = np.isfinite(col)
        if f.sum() > 0:
            Y_mean[j] = col[f].mean()
    Y_imp = np.where(np.isfinite(Y), Y, Y_mean)   # fill NaN with train mean (or 0 for all-NaN)
    X_tr, Y_tr = X[train], Y_imp[train]
    X_val = X[~train]; Y_val = Y_imp[~train]
    train_n = train.sum()
    print(f"\n  训练 N={train_n}  验证 N={len(X_val)}")

    # ---- Ridge 线性模型 ----
    print("\n--- Ridge 回归 ---")
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X_tr, Y_tr)
    pred_delta_ridge = ridge.predict(X)
    pred_abs_ridge = Y_ctrl + pred_delta_ridge
    run_model("linear_ridge", pred_abs_ridge, pred_delta_ridge, d, results)

    # ---- MLP ----
    print("\n--- MLP (256,128) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr_s = X_scaled[train]
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=200,
                       early_stopping=True, validation_fraction=0.1,
                       random_state=0, verbose=False)
    t0 = time.time()
    mlp.fit(X_tr_s, Y_tr)
    dt = time.time() - t0
    print(f"  MLP fit: {dt:.1f}s, loss={mlp.loss_:.4f}, n_iter={mlp.n_iter_}")
    pred_delta_mlp = mlp.predict(X_scaled)
    pred_abs_mlp = Y_ctrl + pred_delta_mlp
    run_model("mlp_256_128", pred_abs_mlp, pred_delta_mlp, d, results)

    # ---- CompositionalTwin 赛题版（残差分解：μ + μ_strain + μ_compound + ψ）----
    from goai_compositional_twin import GoaiCompositionalTwin
    print("\n--- CompositionalTwin 赛题版 ---")
    Dc = len(d["compound_names"])
    Ds = len(d["strain_names"])
    Dctx = X.shape[1] - Dc - Ds - 3   # X = [P(Dc) | Ctx(Dctx) | S(Ds) | A(2) | g(1)]
    comp_idx = np.argmax(X[:, :Dc], axis=1)
    strain_idx = d["meta"][:, 1]
    X_ctxt = X[:, Dc:Dc + Dctx]        # 上下文特征

    ct = GoaiCompositionalTwin(interaction_model="ridge", alpha=1.0, random_state=0)
    ct.fit(Y_delta=Y, strain_idx=strain_idx, comp_idx=comp_idx,
           X_ctxt=X_ctxt, train_mask=train)
    pred_delta_ct = ct.predict_delta(strain_idx, comp_idx, X_ctxt)
    pred_abs_ct = Y_ctrl + pred_delta_ct
    run_model("compositional_twin", pred_abs_ct, pred_delta_ct, d, results)

    # ---- Comparison Table ----
    print("\n" + "=" * 60)
    print("  模型对比（加权总分 + 关键指标）")
    print("=" * 60)
    header = f"  {'Model':<22s} {'Weighted':>8s} {'Abs R²':>8s} {'FC PCC':>8s} {'CtxPCC':>8s} {'DrugPCC':>8s} {'DblUnk':>8s} {'DEP':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(results, key=lambda x: -x["weighted_score"]):
        m = r["modules"]
        print(f"  {r['model']:<22s} {r['weighted_score']:>8.4f} "
              f"{m['abs_fidelity_r2']:>8.4f} {m['fc_delta_pcc']:>8.4f} "
              f"{m['ctx_residual_pcc']:>8.4f} {m['drug_residual_pcc']:>8.4f} "
              f"{m['double_unk_fc_pcc']:>8.4f} {m['dep_top_pcc']:>8.4f}")

    # 写出 JSON
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"results": results, "weights": {k: v for k, v in results[0].get("weights", {}).items() if v} or list(results[0].get("modules",{}).keys())}, f, indent=2, ensure_ascii=False)
    print(f"\n  → 已写 {OUT}")


if __name__ == "__main__":
    main()
