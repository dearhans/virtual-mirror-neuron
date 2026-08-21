"""scripts/p5a2_precision_vs_p1.py — 决策测量：fglex per-gene 贝叶斯孪生 点预测精度 vs P1 基线(compositional_twin)

目的（用户授权「现在就跑」）
--------------------------
复赛 P0 路径1「从 fglex 孪生重生成 prediction.csv 能否抬升 45% 主榜精度」的决定性测量。
平台 45%(技术性能) = goai_metrics.evaluate() 的 6 模块加权分。最公平的对比 =
把 fglex 孪生接进**同一个 evaluate()**，直接比 goai_benchmark.json 里的
compositional_twin（P1 基线，同一 cache 同口径）。

公平性
------
- 两者皆在 split=="id" 上训练，在 ood_*(val_chem_only 等) 上评（evaluate 用 id 冻结基线，无泄漏）。
- fglex 用全量 id(5169)训练 + 全 5243 基因；点预测 = SGLD 后验均值（即会提交的量）。
- pred_delta 直接是模型输出；pred_abs = Y_ctrl + pred_delta（与基线同口径还原绝对丰度）。

输出
----
experiments/20260819-p5a2-precision-vs-p1.json + 控制台对比表（weighted + 6 模块 + 逐子集 FC PCC）。
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, "code")
from data.goai_loader import build_dataset
from data.goai_compound_features import make_functionalgroup_block
from goai_metrics import evaluate, MODULE_WEIGHTS, SUBSETS
from model.compositional_p5a2_pergene import PerGeneBayesTwin

CACHE = "data/processed/goai_cache.npz"
BENCH = "experiments/goai_benchmark.json"
OUT = "experiments/20260819-p5a2-precision-vs-p1.json"


def build_fglex_X(d):
    """替换化合物 one-hot 块为 fglex 官能团连续块，返回 X_cont + 列维度。"""
    X = d["X"]
    names = list(d["compound_names"])
    Dc = len(names)
    P = X[:, :Dc]
    rownz = P.argmax(1)
    per_sample = [names[i] for i in rownz]
    fg, _ = make_functionalgroup_block(per_sample)          # (N, 14)
    X_rest = X[:, Dc:]                                       # ctx + strain one-hot + A + g
    return np.ascontiguousarray(np.hstack([fg, X_rest]).astype(np.float64)), fg.shape[1]


def impute_id_median(Y, id_mask):
    """按 id 训练集每基因中位数填补 NaN（不泄漏 ood）。"""
    gm = np.nanmedian(Y[id_mask], axis=0)
    gm = np.where(np.isnan(gm), 0.0, gm)
    Ys = Y.copy()
    m = np.isnan(Ys)
    Ys[m] = gm[np.nonzero(m)[1]]
    return Ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-genes", type=int, default=5243)
    ap.add_argument("--n-iter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    d = build_dataset(cache_path=CACHE)
    X, Y_delta, Y_ctrl, split = d["X"], d["Y_delta"], d["Y_ctrl"], d["split"]
    Kp = Y_delta.shape[1]
    id_mask = split == "id"
    print(f"[load] N={len(X)} Kp={Kp}  id={int(id_mask.sum())}  "
          f"ood_action={int((split=='ood_action').sum())} ood_agent={int((split=='ood_agent').sum())} "
          f"ood_s3={int((split=='ood_s3').sum())} ood_time={int((split=='ood_time').sum())}")

    # ---- fglex 特征 ----
    Xc, Dfg = build_fglex_X(d)
    print(f"[feat] fglex X_cont shape={Xc.shape} (Dfg={Dfg})")

    # ---- 基因选择 + NaN 填补 ----
    if args.n_genes < Kp:
        nan_frac = np.isnan(Y_delta).mean(0)
        cand = np.argsort(nan_frac)[:args.n_genes]
        Ys = Y_delta[:, cand].copy()
        id_mask_g = id_mask
    else:
        cand = np.arange(Kp)
        Ys = Y_delta.copy()
    keep = ~np.isnan(Ys).all(0)                  # 丢全 NaN 基因
    cand = cand[keep]
    Ys = Ys[:, keep]
    Ys = impute_id_median(Ys, id_mask)
    Xtr = Xc[id_mask]; Ytr = Ys[id_mask]
    print(f"[prep] genes={len(cand)}  train_rows={Xtr.shape[0]}  D={Xtr.shape[1]}")

    # ---- 拟合 fglex per-gene 贝叶斯孪生（全量 id）----
    noise_var = float(np.mean((Ytr - Xtr @ np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]).var(0)))
    print(f"[fit] OLS noise_var={noise_var:.4f}")
    twin = PerGeneBayesTwin(noise_var=noise_var, prior_lambda=1.0, seed=args.seed)
    twin.fit_torch_sgld(Xtr, Ytr, n_chains=3, n_iter=args.n_iter,
                        burnin=150, thin=10, lr=5e-4, T=0.1, clip_grad=1.0)
    mu_all, _ = twin.predict(Xc)
    # mu_all 仅对选中基因；其余基因用 0（与选基因一致）
    pred_delta = np.zeros((len(Xc), Kp), dtype=np.float64)
    pred_delta[:, cand] = mu_all
    pred_abs = Y_ctrl + pred_delta
    print(f"[pred] delta range [{pred_delta.min():.3f},{pred_delta.max():.3f}]  "
          f"abs NaN={bool(np.isnan(pred_abs).any())}")

    # ---- 评测（同 evaluate，平台 6 模块加权分）----
    res = evaluate(d, pred_abs, pred_delta, model_tag="fglex_pergene_fullid")
    w = res["weighted_score"]; m = res["modules"]
    print(f"\n{'='*64}\n  fglex_pergene 加权总分 = {w:.4f}  ({time.time()-t0:.0f}s)\n{'='*64}")
    for k in MODULE_WEIGHTS:
        print(f"  {k:20s} (w={MODULE_WEIGHTS[k]:.2f}) = {m[k]:+.4f}")
    for s in SUBSETS:
        if s in res["per_subset_fc_pcc"]:
            print(f"    {s:10s} FC_PCC={res['per_subset_fc_pcc'][s]:+.4f}  "
                  f"AbsR2={res['per_subset_abs_r2'].get(s, float('nan')):+.4f}")

    # ---- 对比 P1 基线（goai_benchmark.json）----
    comparison = {"fglex_pergene_fullid": {"weighted_score": w, "modules": m,
                                            "per_subset_fc_pcc": res["per_subset_fc_pcc"],
                                            "per_subset_abs_r2": res["per_subset_abs_r2"]}}
    if os.path.isfile(BENCH):
        with open(BENCH) as f:
            bj = json.load(f)
        print(f"\n{'='*64}\n  模型对比（6 模块加权分 + 分项）\n{'='*64}")
        header = f"  {'Model':<24s} {'W':>7s} " + " ".join(f"{k[:6]:>7s}" for k in MODULE_WEIGHTS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        rows = []
        for r in bj.get("results", []):
            rows.append((r["model"], r["weighted_score"], r["modules"]))
        rows.append(("fglex_pergene_fullid", w, m))
        for name, sc, mm in sorted(rows, key=lambda x: -x[1]):
            line = f"  {name:<24s} {sc:>7.4f} " + " ".join(f"{mm[k]:>7.4f}" for k in MODULE_WEIGHTS)
            print(line)
            comparison[name] = {"weighted_score": sc, "modules": mm}
        print("  模块: abs=绝对保真 fc=匹配FCΔ ctx=上下文残差 drug=药物残差 "
              "dbl=双未知·时间 dep=高效应DEP")

    out = {
        "config": {"n_genes": len(cand), "train_rows": int(id_mask.sum()),
                   "D_fglex": Dfg, "n_iter": args.n_iter, "noise_var": noise_var,
                   "T": 0.1, "seed": args.seed,
                   "note": "fglex per-gene Bayesian twin, full id train, evaluate on cache ood_* (val)"},
        "comparison": comparison,
        "weights": MODULE_WEIGHTS,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  -> {args.out}")
    # 决策判读
    p1 = comparison.get("compositional_twin", {}).get("weighted_score")
    if p1 is not None:
        delta = w - p1
        verdict = "fglex 优于 P1 → 重生成 prediction.csv 可抬分" if delta > 0 else \
                  "fglex 不优于 P1 → 重生成会拉低主榜分，路径1 不成立"
        print(f"\n[决策] fglex={w:.4f} vs compositional_twin(P1)={p1:.4f}  Δ={delta:+.4f}  => {verdict}")


if __name__ == "__main__":
    main()
