"""Fetch ligand SMILES for MISATO complexes — RUN ON A MACHINE WITH INTERNET.

The benchmark sandbox cannot reach external APIs (PubChem/RCSB bodies are
blocked), so this must be executed on the user's own machine (normal internet).
It reads data/processed/misato_rcsb_cache.json (PDB ligand 3-letter codes),
queries RCSB chemcomp API (fallback PubChem PUG-REST) for each unique ligand,
and writes data/processed/misato_ligand_smiles.json = {ligand_code: SMILES}.

Usage (on your machine, in this project dir):
  D:/soft/python/python.exe code/fetch_misato_smiles.py
(or any Python 3 with urllib — no pip install needed)
"""
import json
import os
import time
import urllib.request
import urllib.error

# Resolve paths relative to the project root (parent of this script's dir),
# so the script works no matter which directory it is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "processed", "misato_rcsb_cache.json")
OUT = os.path.join(ROOT, "data", "processed", "misato_ligand_smiles.json")


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "misato-smiles-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def smiles_from_rcsb(code):
    try:
        txt = _get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}")
        d = json.loads(txt)
        desc = d.get("rcsb_chem_comp_descriptor", {})
        for k in ("smiles_canonical", "smiles", "smiles_isomeric"):
            if desc.get(k):
                return desc[k]
    except Exception:
        pass
    return None


def smiles_from_pubchem(code):
    try:
        txt = _get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{code}"
            f"/property/IsomericSMILES/JSON"
        )
        d = json.loads(txt)
        props = d.get("PropertyTable", {}).get("Properties", [])
        if props and props[0].get("IsomericSMILES"):
            return props[0]["IsomericSMILES"]
    except Exception:
        pass
    return None


def main():
    with open(CACHE, encoding="utf-8") as f:
        rcsb = json.load(f)
    codes = set()
    for meta in rcsb.values():
        for lig in meta.get("ligands", []):
            if lig:
                codes.add(lig)
    print(f"[fetch] unique ligand codes: {len(codes)}")
    out, miss = {}, []
    for i, code in enumerate(sorted(codes)):
        smi = smiles_from_rcsb(code) or smiles_from_pubchem(code)
        if smi:
            out[code] = smi
        else:
            miss.append(code)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(codes)}  got {len(out)}")
        time.sleep(0.03)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {OUT}: {len(out)}/{len(codes)} codes have SMILES; missing={len(miss)}")
    if miss:
        print("missing sample:", miss[:20])


if __name__ == "__main__":
    main()
