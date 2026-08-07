"""
consolidate_multiseed.py — 合并 5 个种子的真实产物为统一 aggregate_all.json

数据来源（均为真实跑出的工件，非幻觉）：
  RMSE:        experiments/multiseed/seed{0,1,2,3,4}/202608*benchmark.json  (bm.run 原始输出，最权威)
  recalibration:
    seed0  -> experiments/multiseed/aggregate_seed0.json   (recal_seed0.py 单值格式)
    seed1  -> experiments/multiseed/aggregate_1.json        (multiseed_recalibrate.py 多值格式)
    seed2,3,4 -> experiments/multiseed/aggregate_2_3_4.json (多值格式)
  flags: 来自各 seed benchmark JSON 的 flags 字段

输出: experiments/multiseed/aggregate_all.json
"""
import json, os, glob, datetime, numpy as np

BASE = "experiments/multiseed"
SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
MODELS = ["compositional_twin", "compositional_interaction_twin",
          "virtual_twin", "mlp", "linear", "knn", "mean"]

# 1) RMSE：从 5 个 canonical benchmark JSON 聚合
bench_files = {
    0: f"{BASE}/seed0/20260806-benchmark.json",
    1: f"{BASE}/seed1/20260807-benchmark.json",
    2: f"{BASE}/seed2/20260807-benchmark.json",
    3: f"{BASE}/seed3/20260807-benchmark.json",
    4: f"{BASE}/seed4/20260807-benchmark.json",
}
rmse_per = {s: {} for s in SUBSETS}   # rmse_per[subset][seed][model]
n_train = {}
flags_all = []
for seed, bf in bench_files.items():
    d = json.load(open(bf))
    n_train[seed] = d.get("n_train")
    results = d["results"]
    for s in SUBSETS:
        rmse_per[s][seed] = {m: results[s][m]["rmse"] for m in MODELS}
    for fl in d.get("flags", []):
        flags_all.append({"seed": seed, **fl})

def agg(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "per_seed": [round(float(v), 4) for v in a]}

rmse_out = {}
for s in SUBSETS:
    rmse_out[s] = {m: agg([rmse_per[s][seed][m] for seed in range(5)]) for m in MODELS}

# 2) recalibration：从三个聚合文件统一成 5 per-seed
def get_recal(fname):
    return json.load(open(fname))["recalibration"]

r0 = get_recal(f"{BASE}/aggregate_seed0.json")   # 单值格式
r1 = get_recal(f"{BASE}/aggregate_1.json")        # 多值格式
r234 = get_recal(f"{BASE}/aggregate_2_3_4.json")  # 多值格式

def seed0_scalar(entry, key):
    # 单值格式：entry[key] 是标量
    return float(entry[key])

def multi_val(entry, key):
    # 多值格式：entry[key] 是 {mean,std,per_seed}
    return entry[key]["per_seed"]

recal_out = {}
for m in ["compositional_twin", "compositional_interaction_twin", "virtual_twin", "mlp"]:
    recal_out[m] = {"s": {}, "subsets": {}}
    # 标量 s：seed0 单值，seed1 多值[0]，seed2/3/4 多值
    s_seeds = [seed0_scalar(r0[m], "s")] + multi_val(r1[m], "s") + multi_val(r234[m], "s")
    recal_out[m]["s"] = agg(s_seeds)
    for s in SUBSETS:
        e0 = r0[m]["subsets"].get(s)   # 单值格式，可能为 null
        e1 = r1[m]["subsets"].get(s)   # 多值格式
        e234 = r234[m]["subsets"].get(s)
        def collect(metric_key):
            vals = []
            if e0 is not None and metric_key in e0:
                vals.append(float(e0[metric_key]))
            if e1 is not None and metric_key in e1:
                vals += e1[metric_key]["per_seed"]
            if e234 is not None and metric_key in e234:
                vals += e234[metric_key]["per_seed"]
            return vals
        if e0 is None and e1 is None and e234 is None:
            recal_out[m]["subsets"][s] = None
        else:
            eb = collect("ece_before"); ea = collect("ece_after")
            ca = collect("cov95_after") or collect("cov_after_95")
            recal_out[m]["subsets"][s] = {
                "ece_before": agg(eb) if eb else None,
                "ece_after": agg(ea) if ea else None,
                "cov95_after": agg(ca) if ca else None,
            }

# 3) flags 汇总
ood_action_mem = [f for f in flags_all if f.get("subset") == "ood_action"
                  and "仅记忆" in f.get("verdict", "")]
ood_action_twins = sorted({f["twin"] for f in ood_action_mem})
mem_by_twin = {t: sum(1 for f in ood_action_mem if f["twin"] == t) for t in ood_action_twins}

out = {
    "seeds": [0, 1, 2, 3, 4],
    "n_seeds": 5,
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "source": "5 独立种子（each 独立 norman_cache + canonical run + recalibration refit）；seed 切分随种子变化",
    "n_train_per_seed": n_train,
    "rmse": rmse_out,
    "recalibration": recal_out,
    "flags": {
        "ood_action_mem_by_twin": mem_by_twin,
        "ood_action_mem_total": f"{len(ood_action_mem)}/10",
        "per_seed_detail": flags_all,
    },
}
opath = f"{BASE}/aggregate_all.json"
json.dump(out, open(opath, "w"), ensure_ascii=False, indent=2)
print("WROTE", opath)
print("n_train per seed:", n_train)
print("ood_action 疑似仅记忆:", out["flags"]["ood_action_mem_total"], mem_by_twin)
print("RMSE ood_action compositional_twin:", rmse_out["ood_action"]["compositional_twin"])
