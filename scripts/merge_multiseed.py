#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并多种子结果 → aggregate_all.json（seeds=[0,1,2,3,4] 统一格式）。

输入：
  experiments/multiseed/seed{0,1,2,3,4}/2026*-benchmark.json   ← 各种子 canonical（RMSE/flags）
  experiments/multiseed/aggregate_seed0.json                    ← seed=0 recalibration（recal_seed0.py 产出）
  experiments/multiseed/aggregate_{1}_{2}_{3}_{4}.json          ← 批次 recalibration（multiseed_recalibrate.py 产出）

输出：
  experiments/multiseed/aggregate_all.json
  （stdout 打印跨种子 RMSE 表 + ood_action flag 计数 + recalibration 表）
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "experiments", "multiseed")

# 1) 各种子 canonical（RMSE + flags）
reports = {}
for d in sorted(glob.glob(os.path.join(OUT, "seed*"))):
    js = glob.glob(os.path.join(d, "2026*-benchmark.json"))
    if not js:
        print(f"[warn] {d} 无 canonical json"); continue
    seed = int(os.path.basename(d).replace("seed", ""))
    with open(js[0], encoding="utf-8") as f:
        reports[seed] = json.load(f)
seeds = sorted(reports)
print(f"[merge] 收集到种子: {seeds}")

subsets = reports[seeds[0]]["subsets"]
models = list(reports[seeds[0]]["results"][subsets[0]].keys())

# 2) RMSE 跨种子
rmse = {}
for s in subsets:
    rmse[s] = {}
    for m in models:
        vals = [reports[seed]["results"][s][m]["rmse"] for seed in seeds]
        rmse[s][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "per_seed": [round(float(v), 4) for v in vals]}

# 3) flags：ood_action「疑似仅记忆」在几个种子触发（任一孪生触发即计）
ood_action_mem = 0
per_seed_flags = []
for seed in seeds:
    fs = [{"subset": f["subset"], "twin": f["twin"], "verdict": f["verdict"]}
          for f in reports[seed]["flags"]]
    per_seed_flags.append({"seed": seed, "flags": fs})
    if any(f["subset"] == "ood_action" and "疑似" in f["verdict"] for f in fs):
        ood_action_mem += 1

# 4) recalibration 合并（模型固定顺序）
RECAL_MODELS = ["compositional_twin", "compositional_interaction_twin", "virtual_twin", "mlp"]
def _meanstd(xs):
    xs = [float(v) for v in xs]
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "per_seed": [round(v, 4) for v in xs]}

# 收集 per-model per-subset 的各种子值
recal_vals = {m: {s: {"ece_before": [], "ece_after": [], "cov95_after": [], "s": []} for s in subsets}
              for m in RECAL_MODELS}

def _pull(model, key, val):
    """val 可能是标量（seed0 格式）或 {mean,std,per_seed}（批次格式）。返回 per_seed 列表。"""
    if isinstance(val, dict) and "per_seed" in val:
        return val["per_seed"]
    return [float(val)]

# 4a) seed=0（aggregate_seed0.json，单值格式）
p0 = os.path.join(OUT, "aggregate_seed0.json")
if os.path.exists(p0):
    with open(p0, encoding="utf-8") as f:
        a0 = json.load(f)
    for m in RECAL_MODELS:
        if m not in a0["recalibration"]:
            continue
        rm_ = a0["recalibration"][m]
        for s in subsets:
            v = rm_["subsets"].get(s)
            if not v:
                continue
            recal_vals[m][s]["ece_before"] += [float(v["ece_before"])]
            recal_vals[m][s]["ece_after"] += [float(v["ece_after"])]
            recal_vals[m][s]["cov95_after"] += [float(v.get("cov_after_95", v.get("cov95_after")))]
            recal_vals[m][s]["s"] += [float(rm_["s"])]
else:
    print("[warn] aggregate_seed0.json 缺失")

# 4b) 批次（multiseed_recalibrate.py 的 aggregate_* 格式：{mean,std,per_seed}）
for p in sorted(glob.glob(os.path.join(OUT, "aggregate_*.json"))):
    if os.path.basename(p) in ("aggregate_seed0.json", "aggregate_all.json"):
        continue
    with open(p, encoding="utf-8") as f:
        ab = json.load(f)
    for m in RECAL_MODELS:
        if m not in ab.get("recalibration", {}):
            continue
        rm_ = ab["recalibration"][m]
        for s in subsets:
            v = rm_["subsets"].get(s)
            if not v:
                continue
            recal_vals[m][s]["ece_before"] += _pull(m, "ece_before", v["ece_before"])
            recal_vals[m][s]["ece_after"] += _pull(m, "ece_after", v["ece_after"])
            recal_vals[m][s]["cov95_after"] += _pull(m, "cov95_after", v["cov95_after"])
            recal_vals[m][s]["s"] += _pull(m, "s", rm_["s"])

recal = {}
for m in RECAL_MODELS:
    recal[m] = {"s": _meanstd(recal_vals[m][subsets[0]]["s"]) if recal_vals[m][subsets[0]]["s"] else None,
                "subsets": {}}
    for s in subsets:
        recal[m]["subsets"][s] = {
            "ece_before": _meanstd(recal_vals[m][s]["ece_before"]),
            "ece_after": _meanstd(recal_vals[m][s]["ece_after"]),
            "cov95_after": _meanstd(recal_vals[m][s]["cov95_after"]),
        }

from datetime import datetime, timezone
agg = {
    "seeds": seeds,
    "source": "norman",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "note": "合并自各种子 canonical JSON + aggregate_seed0.json + 批次 aggregate_*.json",
    "rmse": rmse,
    "models": models,
    "subsets": subsets,
    "flags": {"ood_action_mem_seeds": ood_action_mem,
              "ood_action_mem_count": f"{ood_action_mem}/{len(seeds)}",
              "per_seed_detail": per_seed_flags},
    "recalibration": recal,
}
out_path = os.path.join(OUT, "aggregate_all.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(agg, f, ensure_ascii=False, indent=2)
print(f"[merge] 落盘 {out_path}\n")

# 摘要
print("=== RMSE 跨种子 mean±std ===")
for s in subsets:
    row = "  ".join(f"{m}={rmse[s][m]['mean']:.3f}±{rmse[s][m]['std']:.3f}" for m in models)
    print(f"  {s:10s} {row}")
print(f"\nood_action 疑似仅记忆: {agg['flags']['ood_action_mem_count']}")
print("\n=== recalibration ECE 前→后（跨种子 mean±std）===")
for m in RECAL_MODELS:
    line = []
    for s in subsets:
        eb = recal[m]["subsets"][s]["ece_before"]["mean"]
        ea = recal[m]["subsets"][s]["ece_after"]["mean"]
        line.append(f"{s}:{eb:.3f}→{ea:.3f}")
    print(f"  {m:30s} " + "  ".join(line))
