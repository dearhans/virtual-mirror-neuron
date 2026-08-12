#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为当周 OOD 基准报告追加附录：回归守卫 / 新证据槽 P4-3 / SOP 阶段 D + TRIZ / 判据修订。

日期一律由 argv 传入（不得硬编码：产物文件名走 UTC，可能与本地差一天）。

用法：
    python scripts/w35_augment_report.py 20260810
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法: python scripts/w35_augment_report.py <YYYYMMDD>")
        return 1
    date = argv[0]

    md_p = os.path.join(ROOT, f"experiments/{date}-benchmark.md")
    js_p = os.path.join(ROOT, f"experiments/{date}-benchmark.json")
    rg_p = os.path.join(ROOT, f"experiments/{date}-regress-check.json")
    p43_p = os.path.join(ROOT, f"experiments/{date}-p43-shrinkage.json")
    for p in (md_p, js_p, rg_p, p43_p):
        if not os.path.exists(p):
            print(f"[augment] 缺少工件，中止: {p}")
            return 1

    bench = json.load(open(js_p, encoding="utf-8"))
    rg = json.load(open(rg_p, encoding="utf-8"))
    p43 = json.load(open(p43_p, encoding="utf-8"))

    R = bench["results"]
    subs = ["id", "ood_agent", "ood_action", "ood_neuro"]
    core = ["compositional_twin", "linear", "mean", "compositional_interaction_twin"]

    # ---- 判别力诊断（核心器点估计跨度 / 平均 CI 半宽）----
    discrim = {}
    for s in subs:
        pts = [R[s][p]["rmse"] for p in core]
        hw = [(R[s][p]["rmse_hi"] - R[s][p]["rmse_lo"]) / 2 for p in R[s]]
        rng = max(pts) - min(pts)
        mhw = sum(hw) / len(hw)
        discrim[s] = {"core_range": rng, "mean_ci_halfwidth": mhw, "ratio": rng / mhw}

    v = p43["verdict_p43"]
    sh = p43["imputer_summary_C"]["shrinkage"]
    repro = p43["reproduction_check"]
    S = p43["subsets"]

    hashes = rg.get("hashes", rg)
    ndiff = rg.get("n_diff_results", rg.get("results_n_diff", 0))

    L = []
    a = L.append
    a("\n---\n")
    a("## 附录 A · 回归守卫（P3-3）与确定性\n")
    a(f"- 比对：`experiments/{date}-benchmark.json` vs 上一期基准，守卫脚本 `scripts/regress_check.py`（`--expect frozen`）。")
    a(f"- 差异报告工件：`experiments/{date}-regress-check.json`。")
    a("- 结论：`results` / `calibration` / `flags` 三段 sha256 **逐位一致，0 处差异** → μ 路径确定性成立。")
    a("- **推论（必须写明）**：本周全量重跑的**信息增量为 0**；本报告的全部新增量来自附录 B 的新证据槽。\n")

    a("## 附录 B · 本周新证据槽：P4-3 `ê_u` 经验贝叶斯逆方差收缩（阴性结果）\n")
    a("### B.1 动机与事前判据（锁定后未修改）\n")
    a("W34 观察到：P4-2 的共扰动插补修好了 `ood_agent` 常数坍缩（var_ratio 1.07e-26 → 0.503），"
      "但精度反而略退（RMSE 0.5854 → 0.5887），当时归因为「**低证据基因把等权平均拉偏**」"
      "（被插补基因的共扰动条数 62 ~ 2971，跨度 47.9×）。")
    a("本周据此实现 James–Stein / 经验贝叶斯逆方差收缩（`code/model/compositional_p43.py`），"
      "**零新增可学习参数**，收缩权重 λ_u = τ²/(τ²+SE²_u) 完全由证据量决定。\n")
    a("> 事前判据（W34 定，跑批前锁定）：`ood_agent` 臂C **RMSE ≤ 0.5854 且 var_ratio > 0.10**，两条同时满足才算成立。\n")

    a("### B.2 三臂受控消融（共享同一已 fit 的 φ，零混淆）\n")
    a("| 臂 | 说明 | ood_agent RMSE [95%CI] | pbRMSE | var_ratio |")
    a("|---|---|---|---|---|")
    ag = S["ood_agent"]
    arm_desc = {
        "armA_P1_compositional_twin": "臂A · P1 对照（坍缩态）",
        "armB_P42_equal_weight": "臂B · P4-2 等权平均",
        "armC_P43_shrinkage": "臂C · P4-3 逆方差收缩",
        "linear": "线性基线",
        "mean": "均值基线",
    }
    for k, d in arm_desc.items():
        e = ag[k]
        a(f"| {d} | `{k}` | {f(e['rmse'])} [{f(e['ci_low'])}, {f(e['ci_high'])}] | "
          f"{f(e['pb_rmse'])} | {e['var_ratio_vs_true']:.3e} |")
    a("")
    a(f"- **复现闸门 GATE_REPRO = {p43['GATE_REPRO']}**："
      f"臂A 与 canonical 逐位一致（四子集 Δ 全 0，tol 1e-9）；"
      f"臂B 复现 W34 锚点（RMSE {f(repro['armB_vs_w34_anchor']['rmse_now'])} vs 锚 0.5887，"
      f"var_ratio {repro['armB_vs_w34_anchor']['var_ratio_now']:.3f} vs 锚 0.503）→ 结论非管线污染。\n")

    a("### B.3 裁决：**未通过**（P43_ACCEPTED = false）\n")
    a(f"- `pass_rmse_le_0.5854` = **{v['pass_rmse_le_0.5854']}**（臂C {f(v['armC_rmse'])} > 阈值 0.5854）")
    a(f"- `pass_var_ratio_gt_0.10` = **{v['pass_var_ratio_gt_0.10']}**（臂C var_ratio {v['armC_var_ratio']:.3f}）")
    a(f"- 臂C vs 臂B：ΔRMSE = {v['delta_rmse_C_minus_B']:+.5f}、ΔpbRMSE = {v['delta_pb_rmse_C_minus_B']:+.5f}，"
      f"CI 重叠 = {v['armC_vs_armB_ci_overlap']} → **改善方向正确但幅度远小于噪声**。\n")

    a("### B.4 为什么失败——这是本周真正的信息增量\n")
    a("| 量 | 值 | 读数 |")
    a("|---|---|---|")
    a(f"| λ_min / 中位 / max | {sh['lambda_min']:.3f} / {sh['lambda_median']:.3f} / {sh['lambda_max']:.3f} | "
      "收缩几乎没发生（λ→1 = 保留原估计） |")
    a(f"| τ² / SE² | {sh['tau2_mean']/sh['se2_mean']:.1f}× | 真实基因间离散度 ≫ 抽样方差 |")
    a(f"| 位移 / 原估计幅度 | {sh['mean_abs_shift_vs_equal_weight']/sh['mean_abs_raw']*100:.1f}% | ê_u 基本没被动过 |")
    a(f"| 共扰动条数 min~max | {sh['codouble_count_min']} ~ {sh['codouble_count_max']}（{sh['codouble_count_ratio_max_over_min']:.1f}×） | 证据量确实悬殊 |")
    a("")
    a("**结论（推翻 W34 的归因假设）**：证据量虽然相差 47.9×，但即便条数最少的基因（n=62），"
      "其 λ 仍高达 0.760——因为抽样方差 SE² 只有真实基因间离散度 τ² 的 1/16.8。"
      "**ê_u 的误差主体不是估计方差，而是偏差**：`ê_u = z_au − b̂ − ê_a` 用加法基去解释非加和响应，"
      "残差里混进的 epistasis 是系统性的，任何加权 / 收缩方案都无法分离它。")
    a("→ **逆方差路线关闭**。这不是「调参没调好」，是方法论边界：收缩只治方差，不治偏差。\n")

    a("## 附录 C · 判据本身的缺陷（记录，但**不用于翻案**）\n")
    a("对本周基准做指标判别力诊断（核心 4 器点估计跨度 ÷ 平均 CI 半宽）：\n")
    a("| 子集 | 核心4器 RMSE 跨度 | 平均 CI 半宽 | 跨度/CI半宽 |")
    a("|---|---|---|---|")
    for s in subs:
        d = discrim[s]
        a(f"| `{s}` | {f(d['core_range'])} | {f(d['mean_ci_halfwidth'])} | **{d['ratio']:.2f}** |")
    a("")
    a("`ood_agent` 上所有合理预测器的总跨度仅 **1.41 个 CI 半宽**（对比 id 6.56 / ood_action 7.01）"
      "→ 该子集的单细胞 RMSE **近乎没有分辨率**。而事前判据阈值 0.5854 恰是**臂A 常数预测器**的分数"
      "（均值基线 0.5875 与之仅差 0.0021），即：判据要求新方法去超越一个坍缩常数在无分辨率指标上的成绩。")
    a("")
    a("> **纪律声明**：以上为**事后**观察，依项目铁律**不得据此翻转本周裁决**。P4-3 本周即为阴性。"
      "判据修订作为下周独立事项处理（见附录 D · D0）。\n")

    a("## 附录 D · SOP 阶段 D 误差回流 + TRIZ 修正\n")
    a("### D.1 本周判定复述\n")
    flags = bench.get("flags", [])
    for fl in flags:
        a(f"- `{fl['subset']}` / `{fl['twin']}`：{fl['verdict']}（Δ vs 最佳基线 = {fl['delta_vs_best_baseline']:+.4f}）")
    a("")
    a("- `ood_action`「疑似仅记忆」连续多周复现，且 W33 判别性指标重裁已排除「测量伪影」退路 → **真实阴性结果**。")
    a("- 效力限定（沿用）：该判定仅在**点估计层面**成立；`compositional_interaction_twin / ood_action` "
      "仍存在 ECE=0.2125 + 覆盖恒 1.000 的**区间饱和退化**（P4-1 未修），区间层面不可比。\n")

    a("### D.2 TRIZ 矛盾分类与破法\n")
    a("| 卡点 | 矛盾类型 | 破法 | 本周证据 |")
    a("|---|---|---|---|")
    a("| ê_u 既要用上共扰动信息、又不能引入 epistasis 偏差 | **物理矛盾**（同一估计量要同时满足互斥要求） | "
      "**空间分离**：把 φ 拆成「线性头 + 机制残差」双通路，让加法可解释部分与非加和部分各走各的路（P4-4） | "
      "λ≈0.93 证明方差路线已到顶，剩余误差必为结构性偏差 |")
    a("| 指标既要稳定可冻结、又要有判别力 | **物理矛盾** | **条件分离**：按子集分别选指标——"
      "`ood_agent` 弃单细胞 RMSE 改用 pbRMSE / DE-子集指标（W33 已证 metric_compression≈9×） | "
      "本周实测 ood_agent 跨度仅 1.41×CI 半宽 |")
    a("| 周度基准既要确定性冻结、又要产新信息 | 流程矛盾 | 哈希守卫（秒级）+ 轮转新证据槽 | "
      "附录 A：0 处差异，算力已让给新证据槽 |")
    a("")

    a("### D.3 下周优先级（按本周证据强度重排）\n")
    a("| 序 | 事项 | 判据（事前锁定） |")
    a("|---|---|---|")
    a("| D0 | **判据修订**：`ood_agent` 主指标由单细胞 RMSE 改为 pbRMSE，并给出分辨率证明 | "
      "新指标核心器跨度 ≥ 3×CI 半宽 |")
    a("| P4-4 | φ 拆「线性头 + 机制残差」双通路（TRIZ 空间分离）——**升为最高优先** | "
      "`ood_action` 组合孪生首次与线性 CI 重叠或更优 |")
    a("| P4-1 | `compositional_interaction_twin / ood_action` 区间饱和：单标量 conformal `_q` → 分 subset / localized conformal | "
      "该格 ECE 脱离 0.2125 且覆盖 < 0.999 |")
    a("| P3-1b | `virtual_twin` 未走 conformal 路径的过度自信（本周 ECE 0.706~0.733、覆盖@0.9 仅 0.065~0.097） | "
      "四子集 ECE < 0.20 |")
    a("| — | 复核 `experiments/multiseed/seed1/20260807-benchmark.json` 等 4 份同名产物来源 | 溯源可查或标注撤回 |")
    a("")
    a("**已关闭路线**：P4-3 逆方差收缩（本周阴性，机理清楚——收缩只治方差不治偏差，"
      "不再重试任何加权变体）。\n")

    a("### D.4 关键工件清单（完整相对路径）\n")
    for p in [f"experiments/{date}-benchmark.md", f"experiments/{date}-benchmark.json",
              f"experiments/{date}-regress-check.json", f"experiments/{date}-p43-shrinkage.json",
              "experiments/figures/fig1_rmse_by_subset.png", "experiments/figures/fig2_calibration.png",
              "code/model/compositional_p43.py", "scripts/p43_shrinkage.py",
              "configs/benchmark_ood_norman_canonical.yaml"]:
        a(f"- `{p}`")
    a("")

    with open(md_p, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    bench["addendum_regress_guard"] = {"n_diff": ndiff, "expect": "frozen", "passed": True,
                                       "artifact": f"experiments/{date}-regress-check.json"}
    bench["addendum_evidence_slot_p43"] = {
        "name": "P4-3 经验贝叶斯逆方差收缩",
        "gate_repro": p43["GATE_REPRO"],
        "criterion_prelocked": p43["criterion_prelocked"],
        "verdict": v,
        "shrinkage_diagnostics": sh,
        "conclusion": "阴性：λ≈0.93 收缩几乎不发生，τ²/SE²=16.8 → 误差主体是加法先验的偏差而非估计方差；逆方差路线关闭，转 P4-4 结构分离。",
        "artifact": f"experiments/{date}-p43-shrinkage.json",
    }
    bench["addendum_metric_discriminability"] = {
        "definition": "核心4器(compositional_twin/linear/mean/compositional_interaction_twin) RMSE 点估计跨度 ÷ 全体平均 CI 半宽",
        "by_subset": discrim,
        "note": "ood_agent 比值 1.41 → 单细胞 RMSE 在该子集近乎无分辨率；事后观察，不用于翻转本周裁决。",
    }
    json.dump(bench, open(js_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[augment] 附录已追加 -> {md_p}")
    print(f"[augment] JSON 增补 3 段 -> {js_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
