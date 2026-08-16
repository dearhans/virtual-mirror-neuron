"""P5-A2 acceptance-criterion AUDIT (metric calibration before comparison).

Motivation (W37 / 2026-08-16 weekly OOD benchmark, new-evidence slot):
  The 2026-08-15 P5-A2 synthetic spike was logged as "positive" partly on the basis of
  `ECE < 0.08`. But `scripts/p5a2_bayes.py::ece()` evaluates a SINGLE nominal level (0.95).
  Under complete interval saturation (empirical coverage == 1.0000) that estimator has a
  FLOOR of |1.0 - 0.95| = 0.0500, which sits BELOW the 0.08 acceptance bar.
  => a fully degenerate (over-wide, uninformative) interval PASSES the bar.

  Three strata in experiments/20260815-p5a2-spike.json show exactly cov=1.0000 & ece=0.0500
  (torch:ood_action, torch:ood_agent, pymc:ood_agent) -> saturation signature, not calibration.

This script re-adjudicates the SAME spike (same data seed, same model hyper-params) under the
canonical project calibration口径 (4 nominal levels [0.5,0.8,0.9,0.95]) and reports:

  * ece_single_095   -> reproduces the 2026-08-15 number  (GATE_REPRO)
  * ece_multilevel   -> canonical project metric; saturation signature = mean(|1-L|) = 0.2125
  * coverage_by_level-> the actual degeneracy evidence
  * band check       -> pre-locked P5 criterion: cov@0.95 in [0.93, 0.97]
  * metric floor analysis -> why the single-level bar was un-gated

Usage (inside the torch/PyMC venv):
    .venv_p5a2/Scripts/python.exe scripts/p5a2_readjudicate.py --method torch \
        --out experiments/20260816-p5a2-criterion-audit.json

Honesty: this AUDITS a criterion. It does not re-run the real-data benchmark and makes no
claim about the real Norman basis. Point predictions are untouched.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from model.compositional_p5a2_bayes import ProperBayesTwin  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

# canonical project calibration levels (configs/benchmark_ood_norman_canonical.yaml: eval.calibration_levels)
CANON_LEVELS = [0.5, 0.8, 0.9, 0.95]
# pre-locked P5 acceptance band (experiments/20260814-P5-structural-estimator-proposal.md)
BAND = (0.93, 0.97)
ECE_BAR = 0.08


# ---------------------------------------------------------------- data (byte-identical to spike)
def make_Xy(n_id=2000, n_ood=500, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    pa, pb = dim // 2, dim // 4
    act = max(1, dim // 6)

    def block(n, center):
        X = rng.normal(0, 1, size=(n, dim))
        X[:, :pa] += center
        return X

    Xid = block(n_id, 0.0)
    Xact = block(n_ood, 6.0)
    Xage = block(n_ood, -6.0)
    Xneu = block(n_ood, 1.5)

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
    nn = NearestNeighbors(n_neighbors=k).fit(Xid)
    d, _ = nn.kneighbors(X)
    return d.mean(1)


# ---------------------------------------------------------------- metrics
def _z_for_level(level):
    """two-sided gaussian quantile for a nominal central interval."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2.0))


def coverage_at(y_true, y_mean, sigma, level):
    q = _z_for_level(level)
    lo, hi = y_mean - q * sigma, y_mean + q * sigma
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def ece_single_095(y_true, y_mean, sigma, n_bins=10):
    """EXACT reproduction of scripts/p5a2_bayes.py::ece  (sigma-binned, single level 0.95)."""
    z = np.abs(y_true - y_mean) / np.maximum(sigma, 1e-9)
    obs = (z <= 1.96).astype(float)
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    err = wsum = 0.0
    for i in range(n_bins):
        m = (sigma >= edges[i]) & (sigma <= edges[i + 1])
        if m.sum() > 0:
            w = m.mean()
            err += w * abs(obs[m].mean() - 0.95)
            wsum += w
    return float(err / wsum) if wsum > 0 else float("nan")


def ece_multilevel(y_true, y_mean, sigma, levels=CANON_LEVELS):
    """canonical project ECE: mean_L |empirical_coverage(L) - L|."""
    return float(np.mean([abs(coverage_at(y_true, y_mean, sigma, L) - L) for L in levels]))


def saturation_signature(levels=CANON_LEVELS):
    """ECE value attained by a FULLY saturated (coverage==1 at every level) interval."""
    return float(np.mean([abs(1.0 - L) for L in levels]))


def metric_floor(levels):
    """floor of the ECE estimator under saturation, per level-set -> shows the un-gated bar."""
    return float(np.mean([abs(1.0 - L) for L in levels]))


def evaluate(twin, Xtest, Ytest, Xid):
    y_mean, sigma_epi = twin.predict(Xtest)
    sigma_tot = np.sqrt(sigma_epi ** 2 + twin.noise_var)
    cov_by_level = {str(L): round(coverage_at(Ytest, y_mean, sigma_tot, L), 4) for L in CANON_LEVELS}
    cov95 = cov_by_level["0.95"]
    e1 = ece_single_095(Ytest, y_mean, sigma_tot)
    em = ece_multilevel(Ytest, y_mean, sigma_tot)
    sat = saturation_signature()
    t = tau(Xtest, Xid)
    mono = float(np.corrcoef(sigma_epi.mean(1), t)[0, 1])
    # saturation flag: every level over-covers AND multilevel ECE sits on the signature
    all_over = all(v >= 0.995 for v in cov_by_level.values())
    return {
        "coverage_by_level": cov_by_level,
        "coverage@0.95": cov95,
        "ece_single_095_repro": round(e1, 4),
        "ece_multilevel_canon": round(em, 4),
        "saturation_signature": round(sat, 4),
        "saturated": bool(all_over and abs(em - sat) < 0.02),
        "in_band_095": bool(BAND[0] <= cov95 <= BAND[1]),
        "passes_ece_bar_single": bool(e1 < ECE_BAR),
        "passes_ece_bar_canon": bool(em < ECE_BAR),
        "mono_corr_sigma_tau": round(mono, 4),
        "sigma_epi_mean": round(float(sigma_epi.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["torch", "pymc", "both"], default="torch")
    ap.add_argument("--out", default="experiments/20260816-p5a2-criterion-audit.json")
    ap.add_argument("--n-id", type=int, default=2000)
    ap.add_argument("--n-ood", type=int, default=500)
    ap.add_argument("--hidden", default="32,16")
    ap.add_argument("--subset", type=int, default=600)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--tune", type=int, default=200)
    ap.add_argument("--prior-json", default="experiments/20260815-p5a2-spike.json",
                    help="2026-08-15 spike artifact used for the GATE_REPRO comparison")
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))

    (Xid, Yid, Xact, Yact, Xage, Yage, Xneu, Yneu) = make_Xy(args.n_id, args.n_ood)
    Xtest = np.vstack([Xid, Xact, Xage, Xneu])
    Ytest = np.vstack([Yid, Yact, Yage, Yneu])
    strata_lab = np.array(["id"] * len(Xid) + ["ood_action"] * len(Xact)
                          + ["ood_agent"] * len(Xage) + ["ood_neuro"] * len(Xneu))

    prior = {}
    if os.path.exists(args.prior_json):
        prior = json.load(open(args.prior_json)).get("strata", {})

    out = {
        "audit": "P5-A2 acceptance-criterion audit (metric calibration before comparison)",
        "date_utc": "2026-08-16",
        "config": vars(args),
        "canonical_levels": CANON_LEVELS,
        "prelocked_band_095": list(BAND),
        "ece_bar": ECE_BAR,
        "metric_floor_under_saturation": {
            "single_level_0.95": metric_floor([0.95]),
            "canonical_4_level": metric_floor(CANON_LEVELS),
        },
        "strata": {},
        "gate_repro": {},
    }
    # the核心 metrology point, computed not asserted
    out["criterion_defect"] = {
        "claim": "single-level(0.95) ECE floor under saturation is BELOW the 0.08 acceptance bar",
        "floor_single": metric_floor([0.95]),
        "bar": ECE_BAR,
        "floor_below_bar": bool(metric_floor([0.95]) < ECE_BAR),
        "floor_canon": metric_floor(CANON_LEVELS),
        "canon_floor_below_bar": bool(metric_floor(CANON_LEVELS) < ECE_BAR),
    }

    def run_one(method):
        tw = ProperBayesTwin(hidden=hidden, noise_var=0.25, prior_lambda=1.0, seed=1)
        if method == "torch":
            tw.fit_torch_sgld(Xid, Yid, n_chains=3, n_iter=800, burnin=300, thin=4,
                              lr=1e-3, T=0.1, clip_grad=1.0, wdecay=1e-3)
        else:
            idx = np.random.default_rng(7).choice(len(Xid), size=min(args.subset, len(Xid)),
                                                  replace=False)
            tw.fit_pymc_nuts(Xid[idx], Yid[idx], draws=args.draws, tune=args.tune, chains=2)
        for name in ["id", "ood_action", "ood_agent", "ood_neuro"]:
            m = strata_lab == name
            ev = evaluate(tw, Xtest[m], Ytest[m], Xid)
            key = f"{method}:{name}"
            out["strata"][key] = ev
            p = prior.get(key)
            if p is not None:
                out["gate_repro"][key] = {
                    "prior_cov095": p.get("coverage@0.95"),
                    "audit_cov095": ev["coverage@0.95"],
                    "d_cov095": round(abs(float(p.get("coverage@0.95", np.nan))
                                          - ev["coverage@0.95"]), 4),
                    "prior_ece": p.get("ece"),
                    "audit_ece_single": ev["ece_single_095_repro"],
                    "d_ece_single": round(abs(float(p.get("ece", np.nan))
                                              - ev["ece_single_095_repro"]), 4),
                }
            print(f"[{method}] {name:11s} cov95={ev['coverage@0.95']:.4f} "
                  f"ece_single={ev['ece_single_095_repro']:.4f} "
                  f"ece_canon={ev['ece_multilevel_canon']:.4f} "
                  f"sat={ev['saturated']} band={ev['in_band_095']} "
                  f"mono={ev['mono_corr_sigma_tau']:+.3f}")
        return tw

    if args.method in ("torch", "both"):
        run_one("torch")
    if args.method in ("pymc", "both"):
        run_one("pymc")

    # ---- verdict under the PRE-LOCKED criterion (band + ECE bar) on ood_action
    verdict = {}
    for method in (["torch", "pymc"] if args.method == "both" else [args.method]):
        k = f"{method}:ood_action"
        if k in out["strata"]:
            s = out["strata"][k]
            verdict[method] = {
                "ood_action_in_band": s["in_band_095"],
                "ood_action_ece_canon": s["ece_multilevel_canon"],
                "ood_action_ece_canon_pass": s["passes_ece_bar_canon"],
                "ACCEPTED_prelocked": bool(s["in_band_095"] and s["passes_ece_bar_canon"]),
                "ACCEPTED_as_logged_0815": bool(s["passes_ece_bar_single"]),
            }
    out["verdict"] = verdict
    # H1 is the claim NOT confounded by saturation -> keep it separable
    out["h1_monotonicity_unconfounded"] = {
        k: v["mono_corr_sigma_tau"] for k, v in out["strata"].items()
    }
    n_sat = sum(1 for v in out["strata"].values() if v["saturated"])
    out["summary"] = {
        "n_strata": len(out["strata"]),
        "n_saturated": n_sat,
        "P5A2_SYNTH_ACCEPTED_prelocked": bool(all(v.get("ACCEPTED_prelocked")
                                                  for v in verdict.values())) if verdict else None,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\n=== VERDICT (pre-locked criterion) ===")
    print(json.dumps(out["verdict"], indent=1))
    print(json.dumps(out["summary"], indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
