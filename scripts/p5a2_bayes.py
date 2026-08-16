"""P5-A2 spike driver — proper-Bayesian salvage for PC-3 epistemic under-scaling.

Run inside the torch/PyMC venv:
    source .venv_p5a2/Scripts/activate
    python scripts/p5a2_bayes.py --method both --out experiments/20260815-p5a2-spike.json

Hypotheses tested (the PC-3 crux):
  H1  proper posterior sigma(x) is MONOTONE in tau(x)=dist-to-training-manifold  ->  corr(sigma, tau) > 0
  H2  id coverage@0.95 ~= 0.95 (well-calibrated in-domain)
  H3  ood coverage >= id coverage (intervals WIDEN out-of-domain, not collapse)
  H4  ECE acceptable (< 0.15 for a spike; < 0.08 is the panel bar)

If H1+H2+H3 hold, the proper-Bayesian path is the validated salvage for PC-3 (P5-A/B/C all failed).
"""
import argparse
import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: put ROOT/code (not ROOT) on path to avoid the stdlib `code` module collision in the venv.
sys.path.insert(0, os.path.join(ROOT, "code"))

from model.compositional_p5a2_bayes import ProperBayesTwin
from sklearn.neighbors import NearestNeighbors


def make_Xy(n_id=2000, n_ood=500, dim=12, seed=0):
    """Compositional-style data: id on a compact manifold; ood strata at increasing distance.

    y = f(x) + noise, f nonlinear. Epistemic should grow with distance to the id manifold.
    """
    rng = np.random.default_rng(seed)
    pa = dim // 2
    pb = dim // 4
    act = max(1, dim // 6)
    gc = dim - pa - pb - act  # g-context

    def block(n, center):
        X = rng.normal(0, 1, size=(n, dim))
        X[:, :pa] += center
        return X

    Xid = block(n_id, 0.0)
    Xact = block(n_ood, 6.0)    # ood_action: far
    Xage = block(n_ood, -6.0)   # ood_agent: far
    Xneu = block(n_ood, 1.5)    # ood_neuro: near

    def f(X):
        a = X[:, :pa].sum(1)
        b = X[:, pa:pa + pb].prod(1) if pb > 0 else np.zeros(X.shape[0])
        c = np.tanh(X[:, pa + pb:pa + pb + act].sum(1)) if act > 0 else np.zeros(X.shape[0])
        return np.stack([np.sin(a), np.cos(0.5 * a), b / (1 + b ** 2), c], axis=1)

    Yid = f(Xid) + rng.normal(0, 0.5, size=(n_id, 4))
    Yact = f(Xact) + rng.normal(0, 0.5, size=(n_ood, 4))
    Yage = f(Xage) + rng.normal(0, 0.5, size=(n_ood, 4))
    Yneu = f(Xneu) + rng.normal(0, 0.5, size=(n_ood, 4))
    return (Xid, Yid, Xact, Yact, Xage, Yage, Xneu, Yneu)


def tau(X, Xid, k=5):
    """distance-to-training-manifold proxy: mean kNN distance to id training points."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xid)
    d, _ = nn.kneighbors(X)
    return d.mean(1)


def coverage(y_true, y_mean, sigma, q=1.96):
    lo, hi = y_mean - q * sigma, y_mean + q * sigma
    inside = ((y_true >= lo) & (y_true <= hi)).mean()
    return float(inside)


def ece(y_true, y_mean, sigma, n_bins=10):
    # calibration error on the 0.95 nominal band (binned by sigma)
    z = np.abs(y_true - y_mean) / np.maximum(sigma, 1e-9)
    obs = (z <= 1.96).astype(float)
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    err = 0.0
    wsum = 0.0
    for i in range(n_bins):
        m = (sigma >= edges[i]) & (sigma <= edges[i + 1])
        if m.sum() > 0:
            w = m.mean()
            err += w * abs(obs[m].mean() - 0.95)
            wsum += w
    return float(err / wsum) if wsum > 0 else float("nan")


def evaluate(twin, Xtest, Ytest, Xid):
    y_mean, sigma_epi = twin.predict(Xtest)
    sigma_tot = np.sqrt(sigma_epi ** 2 + twin.noise_var)
    cov = coverage(Ytest, y_mean, sigma_tot)
    ec = ece(Ytest, y_mean, sigma_tot)
    t = tau(Xtest, Xid)
    # monotonicity: corr of epistemic std (mean over outputs) with tau
    mono = float(np.corrcoef(sigma_epi.mean(1), t)[0, 1])
    return {"coverage@0.95": round(cov, 4), "ece": round(ec, 4),
            "mono_corr_sigma_tau": round(mono, 4),
            "sigma_epi_mean": round(float(sigma_epi.mean()), 4),
            "sigma_epi_ood/id_ratio": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["torch", "pymc", "both"], default="both")
    ap.add_argument("--out", default="experiments/20260815-p5a2-spike.json")
    ap.add_argument("--n-id", type=int, default=2000)
    ap.add_argument("--n-ood", type=int, default=500)
    ap.add_argument("--hidden", default="32,16")
    ap.add_argument("--subset", type=int, default=600, help="pymc NUTS subset size")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--tune", type=int, default=200)
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))

    (Xid, Yid, Xact, Yact, Xage, Yage, Xneu, Yneu) = make_Xy(args.n_id, args.n_ood)
    Xtest = np.vstack([Xid, Xact, Xage, Xneu])
    Ytest = np.vstack([Yid, Yact, Yage, Yneu])
    strata = (["id"] * len(Xid) + ["ood_action"] * len(Xact)
              + ["ood_agent"] * len(Xage) + ["ood_neuro"] * len(Xneu))

    results = {"config": vars(args), "hidden": hidden, "noise_var": 0.25, "strata": {}}

    def run_one(method):
        tw = ProperBayesTwin(hidden=hidden, noise_var=0.25, prior_lambda=1.0, seed=1)
        if method == "torch":
            tw.fit_torch_sgld(Xid, Yid, n_chains=3, n_iter=800, burnin=300, thin=4,
                              lr=1e-3, T=0.1, clip_grad=1.0, wdecay=1e-3)
        else:
            idx = np.random.default_rng(7).choice(len(Xid), size=min(args.subset, len(Xid)), replace=False)
            tw.fit_pymc_nuts(Xid[idx], Yid[idx], draws=args.draws, tune=args.tune, chains=2)
        for name in ["id", "ood_action", "ood_agent", "ood_neuro"]:
            m = np.array(strata) == name
            ev = evaluate(tw, Xtest[m], Ytest[m], Xid)
            results["strata"][f"{method}:{name}"] = ev
            print(f"[{method}] {name:10s} cov={ev['coverage@0.95']:.3f} "
                  f"ece={ev['ece']:.3f} mono={ev['mono_corr_sigma_tau']:+.3f} "
                  f"sigma_epi={ev['sigma_epi_mean']:.3f}")
        return tw

    if args.method in ("torch", "both"):
        run_one("torch")
    if args.method in ("pymc", "both"):
        run_one("pymc")

    # verdict (spike bar): H1 monotone, H2 id cov~0.95, H3 ood>=id, H4 ece<0.15
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
