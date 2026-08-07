"""MISATO Hand-2 OOD benchmark — binding-domain migration of the Virtual Mirror Neuron
OOD evaluation framework.

Goal: replicate the omics "additive prior vs black-box on combinatorial OOD" experiment
in the small-molecule / protein binding domain.

Two target sources (switch with --qm-mode):
  - qm (default): QM.hdf5 -> ligand-level QM scalar (Ionization_Potential etc.).
      These are LIGAND molecular properties, NOT a protein-ligand binding energy.
      Used to prove the protocol migrates + characterize OOD degradation (target is
      protein-independent -> OOD does NOT degrade). See experiments/misato_ood_benchmark.md.
  - md: MD.hdf5 -> frames_interaction_energy (MM interaction energy, true protein-ligand target).
      This is the SCIENTIFICALLY interesting run: a real combinatorial OOD signal
      (held-out ligand x protein COMBINATIONS) where we test whether an additive prior
      (ligand_effect + protein_effect, =P1) generalizes as well as a black-box MLP.

Features (--feat):
  hash    : deterministic hash vectors of ligand/protein IDs  [weak baseline, self-contained]
  qmprops : ligand QM-property vector (7 scalars from QM.hdf5 mol_properties) + protein hash
            [local semantic upgrade, no SMILES needed]
  rdkit   : ligand Morgan 2048-bit fingerprint + 7 RDKit descriptors (needs ligand SMILES
            map: run code/fetch_misato_smiles.py on a machine with internet) + protein hash
            [true chemical semantics; strongest test of the additive-vs-blackbox question]

The deliverable is the OOD EVALUATION PROTOCOL + the cross-domain degradation
characterization; --feat upgrades how semantically meaningful the ligand axis is.

Usage:
  python benchmark_misato.py --inspect                  # print QM.hdf5 schema
  python benchmark_misato.py --run                      # QM OOD benchmark (hash feat)
  python benchmark_misato.py --inspect --qm-mode md     # print MD.hdf5 schema (binding energy)
  python benchmark_misato.py --run --qm-mode md --feat qmprops   # MD OOD, QM ligand props
  python benchmark_misato.py --run --qm-mode md --feat rdkit     # MD OOD, RDKit ligand fp
  python benchmark_misato.py --run --qm-mode md --prefer-key frames_bSASA   # force a frames_* key
"""
import argparse
import json
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ---- local libs (managed python + .pylibs) ----
H5 = None
def _ensure_h5():
    global H5
    if H5 is None:
        import h5py
        H5 = h5py
    return H5


# ----------------------------------------------------------------------------
# 1. Schema inspection (mode-agnostic)
# ----------------------------------------------------------------------------
def inspect(path, mode="qm"):
    h5 = _ensure_h5()
    f = h5.File(path, "r")
    keys = list(f.keys())
    print(f"[inspect] mode={mode}; top-level groups: {len(keys)}; sample: {keys[:3]}")
    for pdb in keys[:2]:
        g = f[pdb]
        print(f"[inspect] group '{pdb}' members:")
        def _walk(name, obj):
            if isinstance(obj, h5.Dataset):
                try:
                    shp = obj.shape
                except Exception:
                    shp = "?"
                print(f"    {name:50s} dataset shape={shp} dtype={obj.dtype}")
            else:
                print(f"    {name:50s} group")
        g.visititems(_walk)
    f.close()


# ----------------------------------------------------------------------------
# 2a. QM.hdf5 target extraction (schema-robust)
# ----------------------------------------------------------------------------
def extract_targets(qm_path, prefer_key=None):
    """Return (dict {pdb_id: scalar_target}, chosen_key) from QM.hdf5.

    Verified schema: each complex has a `mol_properties/` group with 7 ligand-level
    QM scalars (Electron_Affinity, Electronegativity, Hardness, Ionization_Potential,
    Koopman, molecular_weight, total_charge). These are LIGAND molecular properties,
    NOT a protein-ligand binding energy -> protein dimension is redundant for this target.
    """
    h5 = _ensure_h5()
    f = h5.File(qm_path, "r")
    order = ([prefer_key] if prefer_key else []) + [
        "Ionization_Potential", "Electron_Affinity", "Electronegativity",
        "Hardness", "Koopman", "molecular_weight", "total_charge",
    ]
    chosen = None
    for pdb in f.keys():
        g = f[pdb]
        mp = g.get("mol_properties") if "mol_properties" in g else None
        if mp is None:
            continue
        for ck in order:
            if ck in mp and isinstance(mp[ck], h5.Dataset) and mp[ck].shape == ():
                chosen = ck
                break
        if chosen:
            break
    if chosen is None:
        f.close()
        raise RuntimeError("no mol_properties scalar found in QM.hdf5")
    print(f"[extract-qm] chosen target key = '{chosen}'")
    out = {}
    for pdb in f.keys():
        g = f[pdb]
        if "mol_properties" not in g:
            continue
        mp = g["mol_properties"]
        if chosen in mp and mp[chosen].shape == ():
            v = float(mp[chosen][()])
            if np.isfinite(v):
                out[pdb] = v
    f.close()
    return out, chosen


# ----------------------------------------------------------------------------
# 2b. MD.hdf5 binding-energy target extraction (robust MM-GBSA scores probe)
# ----------------------------------------------------------------------------
def extract_targets_md(md_path, prefer_key=None):
    """Return (dict {pdb_id: scalar_target}, chosen_key) from MD.hdf5.

    Verified MISATO MD.hdf5 schema: each complex group contains per-frame scalar
    datasets `frames_interaction_energy` (MM interaction energy -> the binding-
    relevant quantity), `frames_bSASA`, `frames_distance`, `frames_rmsd_ligand`,
    each shape (100,). We pick a binding-energy-like component and aggregate the
    frames by mean to get one scalar per complex. The script probes a candidate
    list (and any --prefer-key) directly in the complex group, so it is robust to
    minor schema variants; it lists available `frames_*` keys on failure.
    """
    h5 = _ensure_h5()
    f = h5.File(md_path, "r")
    candidates = []
    if prefer_key:
        candidates.append(prefer_key)
    candidates += [
        "frames_interaction_energy", "frames_bSASA", "frames_distance",
        "frames_rmsd_ligand",
    ]
    chosen = None
    for pdb in f.keys():
        g = f[pdb]
        for ck in candidates:
            if ck in g and isinstance(g[ck], h5.Dataset) and len(g[ck].shape) >= 1:
                chosen = ck
                break
        if chosen:
            break
    if chosen is None:
        sample = None
        for pdb in f.keys():
            g = f[pdb]
            sample = [k for k in g.keys() if k.startswith("frames_")]
            if sample:
                break
        f.close()
        raise RuntimeError(
            f"no frames_* target found in MD.hdf5. sample frames keys={sample}. "
            f"Pass --prefer-key to force one. candidates tried={candidates}"
        )
    print(f"[extract-md] chosen target key = '{chosen}' (per-frame frames_* dataset)")
    out = {}
    for pdb in f.keys():
        g = f[pdb]
        if chosen not in g:
            continue
        arr = np.asarray(g[chosen]).reshape(-1)
        if arr.size == 0:
            continue
        v = float(np.mean(arr))
        if np.isfinite(v):
            out[pdb] = v
    f.close()
    return out, chosen


# ----------------------------------------------------------------------------
# 3. Weak featurization (deterministic hash vectors; placeholder for RDKit)
# ----------------------------------------------------------------------------
def _hash_vec(token, dim=32, seed=0):
    """Stable FINITE continuous vector for an ID string.

    We map the sha256 hex digest to uint16 integers, then to [-1,1]. This
    avoids interpreting raw hash bytes as float32 (which yields NaN/inf).
    """
    h = hashlib.sha256(f"{seed}:{token}".encode()).hexdigest()
    n = min(dim * 2, len(h))
    ints = np.frombuffer(bytes.fromhex(h[:n]), dtype=np.uint16)
    if ints.size < dim:
        ints = np.pad(ints, (0, dim - ints.size), mode="wrap")
    vals = (ints[:dim].astype(np.float32) / 65535.0) * 2.0 - 1.0
    return vals / (np.linalg.norm(vals) + 1e-8)


def build_features(pdb_ids, ligand_of, protein_of, dim=32):
    """X[i] = concat(hash_vec(ligand_i), hash_vec(protein_i)) -> 2*dim."""
    X = np.zeros((len(pdb_ids), 2 * dim), dtype=np.float32)
    for i, p in enumerate(pdb_ids):
        lig = ligand_of.get(p, "?")
        prot = protein_of.get(p, "?")
        X[i] = np.concatenate([_hash_vec("L:" + lig, dim), _hash_vec("P:" + prot, dim)])
    return X


# ----------------------------------------------------------------------------
# 3b. Semantic-ligand feature variants (RDKit / QM-property baselines)
# ----------------------------------------------------------------------------
_QMPROPS_KEYS = [
    "Ionization_Potential", "Electron_Affinity", "Electronegativity",
    "Hardness", "Koopman", "molecular_weight", "total_charge",
]


def load_ligand_qmprops(qm_path):
    """Per-complex 7-dim ligand QM-property vector from QM.hdf5 mol_properties.

    These are LIGAND molecular properties (real chemistry, no SMILES needed),
    a sandbox-local semantic feature upgrade over the ID hash.
    """
    h5 = _ensure_h5()
    f = h5.File(qm_path, "r")
    out = {}
    for pdb in f.keys():
        g = f[pdb]
        if "mol_properties" not in g:
            continue
        mp = g["mol_properties"]
        vec = []
        ok = True
        for k in _QMPROPS_KEYS:
            if k in mp and mp[k].shape == ():
                vec.append(float(mp[k][()]))
            else:
                ok = False
                break
        if ok and all(np.isfinite(v) for v in vec):
            out[pdb] = np.asarray(vec, dtype=np.float32)
    f.close()
    return out


def build_rdkit_lig_vecs(codes, smiles_map):
    """Per-ligand RDKit vector = Morgan 2048-bit fingerprint + 7 descriptors.

    Requires the ligand_code -> SMILES map (fetched on a machine with internet).
    Missing SMILES -> zero vector (model then falls back to protein hash only).
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Descriptors
    cache = {}

    def lig_vec(code):
        if code in cache:
            return cache[code]
        smi = smiles_map.get(code)
        vec = None
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048)
                bits = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8).astype(np.float32)
                desc = np.array([
                    Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                    Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                    Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                ], dtype=np.float32)
                vec = np.concatenate([bits, desc])
        if vec is None:
            vec = np.zeros(2048 + 7, dtype=np.float32)
        cache[code] = vec
        return vec

    return np.array([lig_vec(c) for c in codes], dtype=np.float32)


def _standardize_cols(mat, idxs, train_mask):
    sub = mat[np.ix_(train_mask, idxs)]
    mu = sub.mean(0)
    sd = sub.std(0)
    sd[sd < 1e-6] = 1.0
    mat[:, idxs] = (mat[:, idxs] - mu) / sd
    return mat


def build_X(pdb_ids, ligand_of, protein_of, feat, smiles_map, qmprops_map,
            dim=32, train_idx=None):
    """Dispatch feature builder by --feat.

      hash    : ligand_hash(32) + protein_hash(32)            [baseline]
      qmprops : ligand_QMprops(7) + protein_hash(32)          [local semantic]
      rdkit   : ligand_Morgan2048+desc(2055) + protein_hash(32) [RDKit semantic]
    """
    if feat == "hash":
        return build_features(pdb_ids, ligand_of, protein_of, dim=dim)
    prot = np.array([_hash_vec("P:" + protein_of.get(p, "?"), dim) for p in pdb_ids],
                    dtype=np.float32)
    if feat == "qmprops":
        lig = []
        for p in pdb_ids:
            lig.append(qmprops_map[p] if (qmprops_map is not None and p in qmprops_map)
                       else np.zeros(7, dtype=np.float32))
        lig = np.array(lig, dtype=np.float32)
        X = np.concatenate([lig, prot], axis=1)
        if train_idx is not None and len(train_idx) > 5:
            X = _standardize_cols(X, list(range(7)), train_idx)
        return X
    if feat == "rdkit":
        lig = build_rdkit_lig_vecs([ligand_of.get(p, "?") for p in pdb_ids], smiles_map)
        X = np.concatenate([lig, prot], axis=1)
        d0 = 2048
        if train_idx is not None and len(train_idx) > 5:
            X = _standardize_cols(X, list(range(d0, d0 + 7)), train_idx)
        return X
    raise ValueError(f"unknown feat={feat}")


# ----------------------------------------------------------------------------
# 4. Models + OOD evaluation (mirrors benchmark_ood.py)
# ----------------------------------------------------------------------------
def _fit_predict_ridge(Xtr, ytr, Xte):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    return m.predict(Xte)


def _fit_predict_mlp(Xtr, ytr, Xte):
    from sklearn.neural_network import MLPRegressor
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def _fit_predict_knn(Xtr, ytr, Xte):
    from sklearn.neighbors import KNeighborsRegressor
    m = KNeighborsRegressor(n_neighbors=5).fit(Xtr, ytr)
    return m.predict(Xte)


def _mean_predict(ytr, yte_shape):
    return np.full(yte_shape[0], float(np.mean(ytr)))


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _bootstrap_rmse(y, p, n_boot=200, seed=0):
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 5:
        return _rmse(y, p), _rmse(y, p)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bs.append(_rmse(y[idx], p[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return _rmse(y, p), (float(lo), float(hi))


def evaluate(X, y, splits, models):
    """splits: dict subset_name -> list of indices into X/y."""
    results = {}
    tr = splits["train"]
    Xtr, ytr = X[tr], y[tr]
    preds = {}
    for name, fn in models.items():
        if name == "mean":
            preds[name] = _mean_predict(ytr, y.shape)
        else:
            preds[name] = fn(Xtr, ytr, X)
    for sub, idx in splits.items():
        if len(idx) == 0:
            continue
        yi = y[idx]
        row = {}
        for name in models:
            pi = preds[name][idx]
            rmse, ci = _bootstrap_rmse(yi, pi)
            row[name] = {"rmse": rmse, "ci_low": ci[0], "ci_high": ci[1]}
        results[sub] = row
    return results


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qm", default="data/raw/misato/QM.hdf5")
    ap.add_argument("--md", default="data/raw/misato/MD.hdf5")
    ap.add_argument("--qm-mode", choices=["qm", "md"], default="qm",
                    help="target source: qm=QM.hdf5 ligand scalar, md=MD.hdf5 binding energy")
    ap.add_argument("--prefer-key", default=None,
                    help="force a specific target key (QM mol_properties or MD scores)")
    ap.add_argument("--splits", default="data/processed/misato_ood_splits.json")
    ap.add_argument("--rcsb_cache", default="data/processed/misato_rcsb_cache.json")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--feat", choices=["hash", "qmprops", "rdkit"], default="hash",
                    help="ligand featurization: hash (ID baseline), qmprops (QM.hdf5 "
                         "ligand scalars, local), rdkit (Morgan fp+descriptors, needs --smiles)")
    ap.add_argument("--smiles", default="data/processed/misato_ligand_smiles.json",
                    help="ligand_code -> SMILES map for --feat rdkit")
    ap.add_argument("--out", default="experiments/misato_ood_benchmark.json")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.md if args.qm_mode == "md" else args.qm, mode=args.qm_mode)
        return

    # load OOD splits + RCSB ligand/protein maps
    sp = json.load(open(args.splits))
    rcsb = json.load(open(args.rcsb_cache))

    if args.qm_mode == "md":
        targets, chosen = extract_targets_md(args.md, prefer_key=args.prefer_key)
    else:
        targets, chosen = extract_targets(args.qm, prefer_key=args.prefer_key)
    print(f"[load] targets extracted ({args.qm_mode}): {len(targets)} complexes (target='{chosen}')")

    # build per-complex ligand/protein from rcsb cache
    ligand_of, protein_of = {}, {}
    usable = []
    for pdb, meta in rcsb.items():
        if pdb not in targets:
            continue
        ligs = meta.get("ligands", [])
        prots = meta.get("proteins", [])
        if not ligs or not prots:
            continue
        ligand_of[pdb] = ligs[0]
        protein_of[pdb] = prots[0]
        usable.append(pdb)
    print(f"[load] usable complexes (target+meta+lig+prot): {len(usable)}")

    pdb_to_i = {p: i for i, p in enumerate(usable)}

    # ligand semantic features (depending on --feat)
    smiles_map = {}
    qmprops_map = None
    if args.feat == "rdkit":
        import json as _json
        try:
            smiles_map = _json.load(open(args.smiles))
            print(f"[feat] rdkit: loaded {len(smiles_map)} ligand SMILES from {args.smiles}")
        except FileNotFoundError:
            print(f"[feat] rdkit: WARNING {args.smiles} not found -> all ligand vecs zero")
    elif args.feat == "qmprops":
        qmprops_map = load_ligand_qmprops(args.qm)
        print(f"[feat] qmprops: {len(qmprops_map)} complexes have QM ligand scalars")

    y = np.array([targets[p] for p in usable], dtype=np.float32)

    # load actual OOD id lists (recomputed from cache, no re-fetch)
    from data.misato_ood_splits import load_id_lists  # noqa
    idl = load_id_lists()
    def _idx(pid_list):
        return [pdb_to_i[p] for p in pid_list if p in pdb_to_i]

    splits = {
        "train": _idx(idl["train"]),
        "id_val": _idx(idl["val"]),
        "id_test": _idx(idl["test"]),
        "ligand_ood": _idx(idl["ood_ligand"]),
        "protein_ood": _idx(idl["ood_protein"]),
        "combo_ood": _idx(idl["ood_combo"]),
    }
    print("[splits] sizes:", {k: len(v) for k, v in splits.items()})

    # build features AFTER splits so semantic columns are standardized on train only
    X = build_X(usable, ligand_of, protein_of, args.feat, smiles_map, qmprops_map,
                dim=args.dim, train_idx=np.array(splits["train"], dtype=int))

    models = {
        "additive_ridge": lambda Xt, yt, Xe: _fit_predict_ridge(Xt, yt, Xe),
        "blackbox_mlp": lambda Xt, yt, Xe: _fit_predict_mlp(Xt, yt, Xe),
        "knn": lambda Xt, yt, Xe: _fit_predict_knn(Xt, yt, Xe),
        "mean": None,
    }
    results = evaluate(X, y, splits, models)
    out_path = args.out
    if args.qm_mode == "md":
        out_path = out_path.replace(".json", "_md.json")
    if args.feat != "hash":
        out_path = out_path.replace(".json", f"_{args.feat}.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[done] wrote {out_path}")
    for sub in ["train", "id_test", "ligand_ood", "protein_ood", "combo_ood"]:
        if sub in results:
            r = results[sub]
            print(f"  {sub:12s} ridge={r['additive_ridge']['rmse']:.3f}  "
                  f"mlp={r['blackbox_mlp']['rmse']:.3f}  "
                  f"knn={r['knn']['rmse']:.3f}  mean={r['mean']['rmse']:.3f}")

    # headline OOD-degradation read
    if "combo_ood" in results and "id_test" in results:
        r_c = results["combo_ood"]["additive_ridge"]["rmse"]
        r_i = results["id_test"]["additive_ridge"]["rmse"]
        print(f"[read] additive_ridge combo_ood/id_test RMSE ratio = {r_c / max(r_i,1e-9):.3f} "
              f"(>1 => OOD degrades)")


if __name__ == "__main__":
    main()
