"""code/goai_bootstrap.py — GOAI bootstrap CI (两栏: 记忆vs机制)

逐子集 bootstrap (N=200) 计算 FC PCC 95% CI，产 JSON + LaTeX 表。
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.goai_loader import build_dataset
from goai_metrics import evaluate
from goai_benchmark import get_baseline_preds
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from goai_compositional_twin import GoaiCompositionalTwin

CACHE = "data/processed/goai_cache.npz"
OUT = "experiments/goai_bootstrap_ci.json"
N_BOOT = 200
SEED = 42
SUBSETS = ["id", "ood_action", "ood_agent", "ood_s3", "ood_time"]


def bootstrap_ci(values):
    v = np.sort(values)
    lo = int(2.5 / 100 * len(v))
    hi = int(97.5 / 100 * len(v))
    return float(np.median(v)), float(v[lo]), float(v[hi])


def main():
    d = build_dataset(cache_path=CACHE)
    X, Y, Y_abs, Y_ctrl, split = d["X"], d["Y_delta"], d["Y_abs"], d["Y_ctrl"], d["split"]
    train = split == "id"; N = len(Y); Kp = Y.shape[1]
    Dc, Ds = len(d["compound_names"]), len(d["strain_names"])

    # ---- 拟合模型 ----
    models = {}
    for tag in ["matched_control", "protein_mean"]:
        models[tag] = get_baseline_preds(d, tag)
    Y_mean = np.zeros(Kp)
    for j in range(Kp):
        col = Y[train, j]; f = np.isfinite(col)
        if f.sum(): Y_mean[j] = col[f].mean()
    Y_imp = np.where(np.isfinite(Y), Y, Y_mean)
    ridge = Ridge(alpha=1.0, fit_intercept=True).fit(X[train], Y_imp[train])
    models["linear_ridge"] = (Y_ctrl + ridge.predict(X), ridge.predict(X))
    Dctx = X.shape[1] - Dc - Ds - 3
    ct = GoaiCompositionalTwin(alpha=1.0, random_state=0)
    ct.fit(Y_delta=Y, strain_idx=d["meta"][:, 1],
           comp_idx=np.argmax(X[:, :Dc], axis=1),
           X_ctxt=X[:, Dc:Dc + Dctx], train_mask=train)
    pd_ct = ct.predict_delta(d["meta"][:, 1], np.argmax(X[:, :Dc], axis=1), X[:, Dc:Dc + Dctx])
    models["compositional_twin"] = (Y_ctrl + pd_ct, pd_ct)

    # ---- MLP (256,128) — 黑箱对照（与 goai_benchmark.py 同构，补齐 P0 CI 缺口）----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=200,
                       early_stopping=True, validation_fraction=0.1,
                       random_state=0, verbose=False)
    mlp.fit(X_scaled[train], Y_imp[train])
    pred_delta_mlp = mlp.predict(X_scaled)
    models["mlp_256_128"] = (Y_ctrl + pred_delta_mlp, pred_delta_mlp)

    # ---- Bootstrap ----
    rng = np.random.default_rng(SEED)
    all_ci = {}
    for name, (pa, pd) in models.items():
        t0 = time.time()
        boot_fc = []
        boot_per = {s: [] for s in SUBSETS if (split == s).sum() > 0}
        for _ in range(N_BOOT):
            idx = rng.choice(N, size=N, replace=True)
            # fast: use Y/Y_ctrl/split indexed
            pa_b = pa[idx]; pd_b = pd[idx]
            Y_abs_b = Y_abs[idx]; Y_ctrl_b = Y_ctrl[idx]; Y_b = Y[idx]; sp_b = split[idx]
            # rebuild minimal d for evaluate
            d_boot = {"Y_abs": Y_abs_b, "Y_ctrl": Y_ctrl_b, "Y_delta": Y_b,
                      "X": X[idx], "meta": d["meta"][idx], "split": sp_b,
                      "compound_names": d["compound_names"], "strain_names": d["strain_names"]}
            r = evaluate(d_boot, pa_b, pd_b)
            boot_fc.append(r["modules"]["fc_delta_pcc"])
            for s in boot_per:
                if s in r["per_subset_fc_pcc"]:
                    boot_per[s].append(r["per_subset_fc_pcc"][s])
        m, l, h = bootstrap_ci(boot_fc)
        print(f"  {name}: FC PCC median={m:.4f} [{l:.4f},{h:.4f}] ({time.time()-t0:.1f}s)")
        all_ci[name] = {"fc_delta_pcc": {"median": m, "ci95_lo": l, "ci95_hi": h}}
        for s in boot_per:
            m, l, h = bootstrap_ci(boot_per[s])
            all_ci[name][f"fc_{s}"] = {"median": m, "ci95_lo": l, "ci95_hi": h}

    with open(OUT, "w") as f:
        json.dump({"bootstrap": all_ci, "n_boot": N_BOOT, "seed": SEED}, f, indent=2)
    print(f"-> {OUT}")

    # LaTeX table
    LABEL = {"id": "记忆(ID)", "ood_action": "OOD-化合物", "ood_agent": "OOD-菌株",
             "ood_s3": "OOD-双未知", "ood_time": "OOD-时间"}
    print("\n=== 两栏：记忆 vs 机制泛化（FC PCC [95% CI]）===")
    for name in ["compositional_twin", "mlp_256_128", "linear_ridge", "protein_mean"]:
        ci = all_ci.get(name, {})
        print(f"\n  {name}:")
        for s in SUBSETS:
            k = f"fc_{s}"
            if k in ci:
                m, l, h = ci[k]["median"], ci[k]["ci95_lo"], ci[k]["ci95_hi"]
                print(f"    {LABEL.get(s,s):15s}  {m:+.4f} [{l:+.4f}, {h:+.4f}]")


if __name__ == "__main__":
    main()
