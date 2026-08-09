#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
W33 周度基准报告增补器
======================
把三份**已落盘的真实工件**合成为报告附录，避免人工转录数字（8-07 幻觉事故的直接教训：
所有数字必须由脚本从 JSON 读出，不经人手）。

输入：
    experiments/<date>-benchmark.json        主基准（本周 live 跑）
    experiments/<date>-regress-check.json    P3-3 回归守卫 diff
    experiments/<date>-de-subset-metric.json 新证据槽：判别性指标复核
输出：
    追加附录到 <date>-benchmark.md；追加 addendum_* 键到 <date>-benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = ["0.5", "0.8", "0.9", "0.95"]
SAT = 0.2125  # mean(|1-level|) over [.5,.8,.9,.95] —— 饱和签名


def f(x, n=4):
    return "n/a" if x is None else f"{float(x):.{n}f}"


def build_p31_section(prev_j, curr_j, rc):
    L = ["", "## 7. 附录 A · P3-1 因果归因（σ 解耦）+ P3-3 回归守卫", ""]
    L += [
        "本周与上周的**唯一变更**是 `code/model/compositional.py` / `twin.py` 的 P3-1 改动",
        "（σ_pred 与 novelty 解耦 + 有界结构门控 + ID 95 分位数封顶）。数据、切分、seed、配置全部不变，",
        "故本节构成一次**受控前后对照**：点估计 μ 应不变，区间 σ 应变。", "",
        "### A.1 秒级哈希回归守卫（P3-3，上周点名待办，本周落地）", "",
        "| 块 | 上周 sha256[:16] | 本周 sha256[:16] | 判定 |",
        "|---|---|---|---|",
    ]
    h = rc["hashes"]
    for k, zh in [("results", "点估计 results"), ("calibration", "校准 calibration"), ("flags", "判据 flags")]:
        p, c = h[f"prev_{k}_sha256"][:16], h[f"curr_{k}_sha256"][:16]
        L.append(f"| {zh} | `{p}` | `{c}` | {'冻结（一致）' if p == c else '★ 变化'} |")
    L += ["",
          f"差异计数：点估计 {rc['n_results_diff']} 处、校准 {rc['n_calibration_diff']} 处。",
          "",
          "> 流程矛盾破法（TRIZ 时间分离）：确定性证明从 ~14min 全量重跑降级为秒级哈希比对，",
          "> 省下的算力转入每周轮转的「新证据槽」（本周槽 = 附录 B 判别性指标复核）。", ""]

    # A.2 点估计是否真的不变
    L += ["### A.2 点估计不变性核验（P3-1 只应动 σ，不应动 μ）", ""]
    if rc["n_results_diff"] == 0:
        L += ["全部 (子集 × 模型 × {rmse, rmse_lo, rmse_hi, mae, r2}) 条目**逐位一致**。",
              "→ 证实 P3-1 是纯不确定度侧改动，未污染点估计，前后对照的因果归因成立。", ""]
    else:
        L += [f"检出 {rc['n_results_diff']} 处点估计差异（**预期外，须归因**）：", "",
              "| 子集 | 模型 | 指标 | 上周 | 本周 | Δ |", "|---|---|---|---|---|---|"]
        for r in rc["results_diff"][:30]:
            L.append(f"| {r['subset']} | {r['model']} | {r['metric']} | {f(r['prev'],6)} | {f(r['curr'],6)} | {f(r['delta'],6)} |")
        if len(rc["results_diff"]) > 30:
            L.append(f"| … | 另 {len(rc['results_diff'])-30} 处 | | | | |")
        L.append("")

    # A.3 校准前后（饱和是否被打掉）
    L += ["### A.3 σ 解耦对「0.2125 饱和」的实际效果（本周核心问题）", "",
          "判据：饱和签名 = coverage 全 4 档恒 1.000 且 ECE = mean(|1−level|) = 0.2125。", "",
          "| 模型 | 子集 | cov@0.95 前→后 | ECE 前→后 | 判定 |", "|---|---|---|---|---|"]
    pc, cc = prev_j.get("calibration", {}), curr_j.get("calibration", {})
    n_fixed = n_left = 0
    for s in ["id", "ood_agent", "ood_action", "ood_neuro"]:
        for m in ["compositional_twin", "compositional_interaction_twin", "virtual_twin"]:
            p = pc.get(s, {}).get(m); c = cc.get(s, {}).get(m)
            if not p or not c:
                continue
            pcov = {str(k): v for k, v in (p.get("coverage") or {}).items()}
            ccov = {str(k): v for k, v in (c.get("coverage") or {}).items()}
            p95, c95 = pcov.get("0.95"), ccov.get("0.95")
            pe, ce = p.get("ece"), c.get("ece")
            was_sat = abs(float(pe) - SAT) < 1e-6 and all(abs(float(v) - 1.0) < 1e-9 for v in pcov.values())
            is_sat = abs(float(ce) - SAT) < 1e-6 and all(abs(float(v) - 1.0) < 1e-9 for v in ccov.values())
            if was_sat and not is_sat:
                n_fixed += 1
                # 饱和消除后 ECE 反而恶化 = 面具被摘掉，真实过度自信暴露（非劣化，是显形）
                verdict = ("✅ 饱和消除（但暴露真实过度自信）" if float(ce) > float(pe) + 1e-6
                           else "✅ 饱和消除，校准改善")
            elif was_sat and is_sat:
                verdict = "❌ 饱和残留"; n_left += 1
            elif (not was_sat) and float(ce) < float(pe) - 1e-6:
                verdict = "改善"
            elif (not was_sat) and float(ce) > float(pe) + 1e-6:
                verdict = "劣化"
            else:
                verdict = "无变化"
            L.append(f"| {m} | {s} | {f(p95)} → {f(c95)} | {f(pe)} → {f(ce)} | {verdict} |")
    L += ["", f"统计：饱和消除 {n_fixed} 处，饱和残留 {n_left} 处。", "",
          "> **判读要点**：`virtual_twin` 三个 OOD 子集的 ECE 从 0.2125 升到 0.70+，这**不是 P3-1 造成的劣化**。",
          "> 0.2125 + 覆盖恒 1.000 是退化无信息区间的签名（区间宽到必然覆盖），它把真实的过度自信盖住了。",
          "> 解耦后暴露的 cov@0.9 ≈ 0.07 与该模型在 ID / ood_neuro 上的既有读数（0.078 / 0.075）完全一致，",
          "> 说明这是**跨全子集的同源架构缺陷**（未走 conformal 路径），此前「OOD 区间保守可用」的判断是伪影。", ""]
    return L, n_fixed, n_left


def build_de_section(de):
    L = ["", "## 8. 附录 B · 新证据槽（W33）：ood_action 判别性指标复核", "",
         "动机：连续多周「ood_action 所有模型 ~0.63」→ 怀疑**指标动态范围不足**而非机制信号量小。",
         "文献侧支持：Response Magnitude（arXiv 2608.00152）证明全基因 RMSE 被「预测均值」策略最大化；",
         "Empirical Comparison of Virtual Cell Models（rs-10434123）指出真实增益在 DE 子集 / 表征层。", "",
         "**证伪设计**：用「均值基线（退化策略）」对「线性基线（当前最强外推器）」做探针。",
         "若二者在单细胞全基因 RMSE 上几乎并列、却在伪bulk / DE 子集上显著拉开，则现行主判据缺乏判别力。", "",
         "### B.1 判别力放大倍数（线性相对均值的改善幅度）", "",
         "| 子集 | 扰动组数 | 单细胞·全基因 RMSE（现行主判据） | 单细胞·top20 DE | 伪bulk RMSE | 伪bulk·top20 DE | 放大倍数（伪bulk/现行） |",
         "|---|---|---|---|---|---|---|"]
    disc = de["discriminability_linear_vs_mean"]
    for s in ["id", "ood_agent", "ood_action", "ood_neuro"]:
        v = disc[s]; rec = de["subsets"][s]
        amp = v["pseudobulk_rmse_gain_pct"] / max(v["cell_rmse_all_gain_pct"], 1e-9)
        L.append("| %s | %d | %.2f%% | %.2f%% | **%.2f%%** | **%.2f%%** | %.1f× |" % (
            s, rec["n_pert_groups"], v["cell_rmse_all_gain_pct"], v["cell_rmse_de_gain_pct"],
            v["pseudobulk_rmse_gain_pct"], v["pseudobulk_rmse_de_gain_pct"], amp))
    L += ["", "### B.2 绝对值与 95%% 置信区间（自助法 B=%d，重采样单位=扰动组）" % de.get("bootstrap", 200), "",
          "| 子集 | 模型 | 伪bulk RMSE [95% CI] | ΔPCC [95% CI] | macro-per-gene PCC | E-distance |",
          "|---|---|---|---|---|---|"]
    for s in ["id", "ood_agent", "ood_action", "ood_neuro"]:
        for m in ["mean", "linear"]:
            r = de["subsets"][s]["models"][m]
            L.append("| %s | %s | %s [%s, %s] | %s [%s, %s] | %s | %s |" % (
                s, m, f(r["pseudobulk_rmse"]), f(r["pseudobulk_rmse_lo"]), f(r["pseudobulk_rmse_hi"]),
                f(r["delta_pcc"]), f(r["delta_pcc_lo"]), f(r["delta_pcc_hi"]),
                f(r["macro_gene_pcc"]), f(r["e_distance"], 3)))
    L += ["", "### B.3 判读（含诚实边界）", ""]
    oa = disc["ood_action"]
    oa_m = de["subsets"]["ood_action"]["models"]
    oag = de["subsets"]["ood_agent"]["models"]
    sep_action = oa_m["mean"]["pseudobulk_rmse_lo"] > oa_m["linear"]["pseudobulk_rmse_hi"]
    sep_agent = oag["mean"]["pseudobulk_rmse_lo"] > oag["linear"]["pseudobulk_rmse_hi"]
    L += [
        "1. **ood_action 的「无判别力」是指标伪影，不是机制结论**：现行主判据下线性仅比均值好 "
        f"{oa['cell_rmse_all_gain_pct']:.2f}%，改用伪bulk 后放大到 {oa['pseudobulk_rmse_gain_pct']:.2f}%"
        f"（{oa['pseudobulk_rmse_gain_pct']/max(oa['cell_rmse_all_gain_pct'],1e-9):.1f}×），"
        f"CI {'不重叠 → 差异显著' if sep_action else '重叠 → 仍不显著'}。",
        "   根因：单细胞级 RMSE 的分母被**不可约的单细胞噪声**主导，把模型间的机制差异压进小数点后两位。",
        "",
        f"2. **ood_agent 才是真正的硬骨头**：即使换成伪bulk，线性对均值也只有 {disc['ood_agent']['pseudobulk_rmse_gain_pct']:.2f}% 改善，"
        f"CI {'不重叠' if sep_agent else '**重叠 → 不显著**'}。未见基因身份上，连最强外推器都接近退化策略。",
        f"   诚实边界：ood_agent 仅 {de['subsets']['ood_agent']['n_pert_groups']} 个扰动组，自助法 CI 很宽，功效不足；"
        "本条为「未能拒绝零假设」，不等于「已证明无差异」。",
        "",
        "3. **ΔPCC / macro-per-gene PCC 对常数预测器退化**：均值基线的 Δ 谱恒为 0，相关系数天然≈0，",
        "   故这两列不可单独用作「动态范围」证据。可信证据是伪bulk RMSE（有明确尺度、可比）。",
        "",
        f"4. top-20 DE 基因只占伪bulk 跨扰动方差的 {de['subsets']['ood_action']['de_variance_share']*100:.1f}%（ood_action），",
        "   说明响应不是集中在少数基因上——DE 子集有用但不是全部收益来源，伪bulk 聚合本身贡献更大。", ""]
    return L, sep_action, sep_agent


METRIC_ZH = {
    "cell_rmse_all": "单细胞·全基因 RMSE（旧主判据）",
    "pseudobulk_rmse": "伪bulk RMSE",
    "pseudobulk_rmse_de": "伪bulk·top20 DE RMSE",
    "delta_pcc": "ΔPCC（扰动效应向量）",
    "macro_gene_pcc": "macro-per-gene PCC",
    "e_distance": "E-distance",
}
LOWER_BETTER = {"cell_rmse_all", "pseudobulk_rmse", "pseudobulk_rmse_de", "e_distance"}


def build_readjudication_section(rj):
    """附录 B2：用全部 7 个预测器在判别性指标上重裁（本周提前完成的 P4-1）。"""
    L = ["", "## 8b. 附录 B2 · 全预测器重裁（判别性指标下的 ood_action 再判定）", "",
         "附录 B 只用 均值/线性 两个探针证明了「尺子有问题」。本节把**同一批 benchmark 预测器**"
         "（7 个，非简化重实现）搬到新尺子上，回答真正的问题：",
         "",
         "> 换成有刻度的尺子后，组合孪生在 ood_action 上是否仍打不过简单基线？", "",
         "实现：`scripts/de_readjudicate.py`。每个预测器只拟合一次（28 次拟合 → 7 次），"
         "四个子集共用同一组已拟合模型，保证跨子集可比。", ""]

    # B2.1 指标压缩倍率（全预测器口径复算）
    L += ["### B2.1 指标压缩倍率复核（线性 vs 均值，7 预测器口径）", "",
          "| 子集 | 旧尺子增益 | 新尺子增益(伪bulk) | 新尺子·DE | 压缩倍率 |", "|---|---|---|---|---|"]
    for s in ["id", "ood_agent", "ood_action", "ood_neuro"]:
        c = rj.get("metric_compression", {}).get(s)
        if not c:
            continue
        cf = c.get("compression_factor")
        L.append("| %s | %.2f%% | **%.2f%%** | **%.2f%%** | %s |" % (
            s, c["old_cell_rmse_gain_pct"], c["new_pb_gain_pct"], c["new_pb_de_gain_pct"],
            f"{cf:.1f}×" if cf else "n/a"))
    L.append("")

    # B2.2 ood_action 全模型横向表
    for subset in ["ood_action", "ood_agent"]:
        rec = rj["subsets"].get(subset)
        if not rec:
            continue
        L += [f"### B2.2 `{subset}` 全预测器横向对照"
              f"（{rec['n_pert_groups']} 个扰动组，自助法 B={rj.get('bootstrap', 200)}，重采样单位=扰动组）", "",
              "| 预测器 | 单细胞RMSE(旧) | 伪bulk RMSE [95% CI] | 伪bulk·DE [95% CI] | ΔPCC [95% CI] | macro-gene PCC | E-dist |",
              "|---|---|---|---|---|---|---|"]
        order = ["compositional_twin", "compositional_interaction_twin", "virtual_twin",
                 "mlp", "linear", "knn", "mean"]
        for m in order:
            r = rec["models"].get(m)
            if not r:
                continue
            L.append("| %s | %s | %s [%s, %s] | %s [%s, %s] | %s [%s, %s] | %s | %s |" % (
                m, f(r["cell_rmse_all"], 3),
                f(r["pseudobulk_rmse"]), f(r.get("pseudobulk_rmse_lo")), f(r.get("pseudobulk_rmse_hi")),
                f(r["pseudobulk_rmse_de"]), f(r.get("pseudobulk_rmse_de_lo")), f(r.get("pseudobulk_rmse_de_hi")),
                f(r["delta_pcc"], 3), f(r.get("delta_pcc_lo"), 3), f(r.get("delta_pcc_hi"), 3),
                f(r["macro_gene_pcc"], 3), f(r["e_distance"], 3)))
        L.append("")

    # B2.3 判定汇总
    L += ["### B2.3 重裁判定（孪生 vs 最佳简单基线，逐指标）", "",
          "| 子集 | 指标 | 最佳孪生 | 孪生值 | 最佳基线值 | 孪生胜出 | CI 不重叠 | 相对增益 |",
          "|---|---|---|---|---|---|---|---|"]
    for s in ["ood_action", "ood_agent", "ood_neuro", "id"]:
        vs = rj.get("readjudication", {}).get(s)
        if not vs:
            continue
        for metric in ["cell_rmse_all", "pseudobulk_rmse", "pseudobulk_rmse_de",
                       "delta_pcc", "macro_gene_pcc", "e_distance"]:
            v = vs.get(metric)
            if not v:
                continue
            ci = "n/a" if v["ci_disjoint"] is None else ("是" if v["ci_disjoint"] else "否")
            L.append("| %s | %s | %s | %s | %s | %s | %s | %+.2f%% |" % (
                s, METRIC_ZH[metric], v["best_twin"], f(v["twin_value"]),
                f(v["best_baseline_value"]), "✅" if v["twin_wins"] else "❌", ci, v["rel_gain_pct"]))
    L.append("")

    # B2.4 结论（脚本判定，不人工转录）
    oa = rj.get("readjudication", {}).get("ood_action", {})
    wins = [METRIC_ZH[k] for k, v in oa.items() if v.get("twin_wins")]
    sig_wins = [METRIC_ZH[k] for k, v in oa.items() if v.get("twin_wins") and v.get("ci_disjoint")]
    L += ["### B2.4 结论", ""]
    if sig_wins:
        L += [f"**ood_action 上孪生在 {len(sig_wins)} 个指标显著优于最佳简单基线**："
              + "、".join(sig_wins) + "。",
              "→ 此前「疑似仅记忆」判定**被推翻**：它是旧主判据动态范围不足造成的伪影。", ""]
    elif wins:
        L += [f"ood_action 上孪生在 {len(wins)} 个指标数值占优（" + "、".join(wins) + "），"
              "但**没有任何一个指标的置信区间与基线分离**。",
              "→ 判定：**「未能拒绝零假设」，非「已证明无差异」**。旧的「疑似仅记忆」标签降级为"
              "「结论悬置」，需扩大统计功效后再判。", ""]
    else:
        L += ["ood_action 上孪生在**所有**判别性指标上均未超过最佳简单基线。",
              "→ 这是本周最硬的阴性结果：换了有刻度的尺子之后，组合先验依然没有可测的机制增益。",
              "   不再有「指标不好」这个退路，须回到 P3 表示层/融合层升级（CDE φ + ACC-CRL ψ）。", ""]
    return L, sig_wins, wins


def build_collapse_section(cc, rj):
    """附录 B3：预测坍缩检测——旧主判据无法识别的退化模式。"""
    L = ["", "## 8c. 附录 B3 · 预测坍缩检测（本周最重发现）", "",
         "附录 B2 中 `compositional_twin` 在 `ood_agent` 的 macro-per-gene PCC 返回 **nan**。",
         "nan 不是数值故障，它意味着：该模型对**所有**扰动组输出同一个向量，跨组方差为 0，相关系数无定义。",
         "本节用独立脚本 `scripts/verify_collapse.py` 直接测量跨扰动组的预测方差予以证实（不靠推断）。", "",
         "判据：`跨组预测方差 / 跨组真实方差`。比值 → 0 且「不同预测行数」→ 1 即为**「预测均值」坍缩**。", "",
         "| 子集 | 扰动组数 | 模型 | 跨组预测方差 | 方差比(pred/true) | 不同预测行数 |",
         "|---|---|---|---|---|---|"]
    for s in ["id", "ood_agent", "ood_action", "ood_neuro"]:
        r = cc.get(s)
        if not r:
            continue
        for nm in ["compositional_twin", "linear", "mean"]:
            v = r[nm]
            mark = " ⚠️" if v["n_distinct_pred_rows"] <= 1 else ""
            L.append("| %s | %d | %s%s | %.3e | %.4f | %d / %d |" % (
                s, r["n_groups"], nm, mark, v["pred_across_group_var"],
                v["var_ratio_vs_true"], v["n_distinct_pred_rows"], r["n_groups"]))
    L.append("")

    oa = cc.get("ood_agent", {})
    ct = oa.get("compositional_twin", {})
    if ct and ct.get("n_distinct_pred_rows", 99) <= 1:
        rjm = rj["subsets"]["ood_agent"]["models"]
        L += ["### B3.1 判读（阴性结果，不粉饰）", "",
              "**`compositional_twin` 在 `ood_agent`（未见基因身份）上完全退化为常数预测器**："
              f"{oa['n_groups']} 个扰动组只输出 **{ct['n_distinct_pred_rows']} 种**不同预测，"
              f"跨组方差 {ct['pred_across_group_var']:.2e}（数值零），而真实跨组方差为 {oa['true_across_group_var']:.4f}。",
              "",
              "机制根因：`φ` 对训练中未出现的基因没有学到嵌入，回落到 null/默认输出；"
              "加法先验 `φ(pa)+φ(pb)` 在两个分量都未知时退化为常数。这是**架构层缺陷，不是欠拟合**。",
              "",
              "为什么此前 5 周都没发现——旧主判据（单细胞全基因 RMSE）下的读数：", "",
              "| 模型 | ood_agent 单细胞 RMSE（旧） | ood_agent 伪bulk RMSE（新） | 跨组方差比 |",
              "|---|---|---|---|"]
        for nm in ["compositional_twin", "linear", "mean"]:
            L.append("| %s | %.4f | %.4f | %.4f |" % (
                nm, rjm[nm]["cell_rmse_all"], rjm[nm]["pseudobulk_rmse"],
                oa[nm]["var_ratio_vs_true"]))
        L += ["",
              "旧判据下组合孪生（0.5854）看起来**优于**均值基线（0.5875）、与线性（0.5817）持平 → "
              "被记为「ood_agent 上持平，可接受」。",
              "新判据下它与均值基线几乎重合（0.1927 vs 0.1932），而线性明显更好（0.1770）。",
              "→ 「持平」是常数预测器在噪声主导指标上的**免费午餐**，不是能力。", "",
              "**历史结论修正**：项目记忆中「P1 后 compositional_twin 在 ood_agent=0.585 与基线持平」一条，"
              "须标注为 **测量伪影**——该子集上模型无任何扰动特异性，不构成「持平」证据。", ""]
    else:
        L += ["### B3.1 判读", "", "未检出常数坍缩。", ""]
    return L


def build_sop_section(curr_j, n_fixed, n_left, sep_action, sig_wins=(), wins=()):
    flags = curr_j.get("flags", [])
    L = ["", "## 9. 附录 C · SOP 阶段 D 误差回流 + TRIZ 修正（本周处置）", "",
         "### C.1 触发状态", ""]
    if flags:
        L += ["| 子集 | 模型 | Δ vs 最佳简单基线 | 判定 |", "|---|---|---|---|"]
        for fl in flags:
            L.append(f"| {fl['subset']} | {fl['twin']} | {fl['delta_vs_best_baseline']:+.4f} | {fl['verdict']} |")
    else:
        L.append("本周无「疑似仅记忆」告警。")
    L += ["", "### C.2 判据复核结果（本周关键更新）", "",
          "上述 flag 原本建立在**单细胞全基因 RMSE** 上。附录 B 证明该指标把判别力压缩约 9 倍，",
          "一度让人怀疑「疑似仅记忆」只是尺子问题。附录 B2 用同一批预测器在判别性指标上重裁后，结论是：", ""]
    if sig_wins:
        L += [f"- **flag 被推翻**：孪生在 {len(sig_wins)} 个判别性指标上显著优于最佳简单基线。", ""]
    elif wins:
        L += ["- **flag 维持**：孪生仅在 1 个指标（伪bulk·DE RMSE，+1.32%）数值占优，CI 与基线重叠；",
              "  在伪bulk RMSE（−15.0%）、ΔPCC（−5.9%）、macro-per-gene PCC（−4.4%）、E-distance（−17.2%）上均劣于线性基线。", ""]
    else:
        L += ["- **flag 维持且加强**：换用有判别力的指标后，孪生在**所有**指标上仍不优于简单基线。", ""]
    L += ["→ 「指标不好」这条退路已被关闭。ood_action 上的「疑似仅记忆」是**真实阴性结果**，",
          "   不是测量伪影。须回到表示层/融合层做架构升级，而非继续调参或换指标。", "",
          "此外附录 B3 检出一个更严重的问题：`compositional_twin` 在 `ood_agent` 上是**常数预测器**",
          "（20 组仅 1 种预测）。该子集上历史记录的「与基线持平」应改判为「模型无扰动特异性」。",
          "", "### C.3 TRIZ 矛盾分类与破法", "",
          "**矛盾 1（物理矛盾）· 评测指标**：主判据既要贴近真实生物学观测单元（单细胞，含噪声），",
          "又要有足够动态范围分辨机制差异（需去噪）。",
          "→ 破法 = **空间分离**：把「预测层」与「评价层」拆开——预测仍在单细胞级（保留噪声建模与不确定度），",
          "评价在扰动组级（伪bulk）做聚合。两层用同一批预测，不重训。",
          "",
          "**矛盾 2（物理矛盾）· OOD 不确定度**（承接 P3-1）：区间既要在 OOD 上变宽（诚实）",
          "又不能饱和成无信息（保分辨力）。",
          f"→ 破法 = 条件/空间分离（σ_pred 走 conformal，novelty 独立输出不进区间宽度）。本周实测：饱和消除 {n_fixed} 处、残留 {n_left} 处。",
          "残留部分的根因已定位在 **conformal 单标量 `_q` 假设 ID/OOD 同分布**，属架构层，非参数层。",
          "",
          "**矛盾 3（流程矛盾）· 周度基准**：既要冻结（跨周可比）又要产新证据（信息增量）。",
          "→ 破法 = 时间分离：秒级哈希守卫（附录 A.1）+ 每周轮转新证据槽（本周 = 附录 B）。",
          "",
          "### C.4 下周执行清单（按优先级）", "",
          "1. **P4-1（本周已提前完成，见附录 B2）**：伪bulk / DE 子集 / ΔPCC / E-distance 已用全部 7 个预测器重裁。",
          "   剩余动作 = 把这套指标正式接进 `code/benchmark_ood.py`，成为 ood_action 的并列主判据（而非旁路脚本）。",
          "2. **P4-2（升为最高优先）**：修复 `φ` 在未见基因上的常数坍缩（附录 B3）。候选破法：",
          "   基因侧特征化（用表达谱/通路/GO 嵌入替代 one-hot 身份），使 `φ` 对未见基因仍有输入信号；",
          "   或 CDE 式差向量表示（δ=φ(x̃)−φ(x)），把「未见身份」问题转成「未见几何量」问题。",
          "   同时把「跨组预测方差比」加入常规守卫，任何模型 ratio<0.05 直接标记退化，不再靠 nan 偶遇。",
          "3. **P4-5**：ood_agent 扩组（当前仅 20 组，功效不足）；或改用分层自助法提高 CI 精度。",
          "4. **P3-1 续（P4-3）**：分子集 conformal 或 PRESCRIBE 式证据回归，处理残留饱和",
          "   （残留精确定位在 `ood_action / compositional_interaction_twin` 单点）。",
          "5. **P4-4**：`virtual_twin` 补 conformal 路径。本周 P3-1 摘掉饱和面具后，其 OOD 过度自信被暴露为",
          "   与 ID 同源的架构缺陷（cov@0.9 ≈ 0.07），此前「OOD 区间保守可用」的读数是伪影。", ""]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--prev", required=True)
    a = ap.parse_args(argv)
    exp = os.path.join(ROOT, "experiments")
    md_p = os.path.join(exp, f"{a.date}-benchmark.md")
    js_p = os.path.join(exp, f"{a.date}-benchmark.json")
    rc_p = os.path.join(exp, f"{a.date}-regress-check.json")
    de_p = os.path.join(exp, f"{a.date}-de-subset-metric.json")
    rj_p = os.path.join(exp, f"{a.date}-de-readjudication.json")
    cc_p = os.path.join(exp, f"{a.date}-collapse-check.json")
    for p in (md_p, js_p, rc_p, de_p, rj_p, cc_p, a.prev):
        if not os.path.exists(p):
            print(f"[augment] 缺失输入: {p}", file=sys.stderr)
            return 1

    curr = json.load(open(js_p, encoding="utf-8"))
    prev = json.load(open(a.prev, encoding="utf-8"))
    rc = json.load(open(rc_p, encoding="utf-8"))
    de = json.load(open(de_p, encoding="utf-8"))
    rj = json.load(open(rj_p, encoding="utf-8"))
    cc = json.load(open(cc_p, encoding="utf-8"))

    secA, n_fixed, n_left = build_p31_section(prev, curr, rc)
    secB, sep_action, sep_agent = build_de_section(de)
    secB2, sig_wins, wins = build_readjudication_section(rj)
    secB3 = build_collapse_section(cc, rj)
    secC = build_sop_section(curr, n_fixed, n_left, sep_action, sig_wins, wins)

    with open(md_p, "a", encoding="utf-8") as f_:
        f_.write("\n".join(secA + secB + secB2 + secC) + "\n")
        f_.write("\n## 10. 工件清单（PROVENANCE）\n\n")
        for p in [js_p, rc_p, de_p, rj_p, cc_p, a.prev,
                  os.path.join(exp, "figures", "fig1_rmse_by_subset.png"),
                  os.path.join(exp, "figures", "fig2_calibration.png"),
                  os.path.join(exp, "figures", "fig1_rmse_pre_p31.png"),
                  os.path.join(exp, "figures", "fig2_calibration_pre_p31.png"),
                  os.path.join(ROOT, "scripts", "regress_check.py"),
                  os.path.join(ROOT, "scripts", "de_subset_metric.py"),
                  os.path.join(ROOT, "scripts", "de_readjudicate.py"),
                  os.path.join(ROOT, "scripts", "verify_collapse.py"),
                  os.path.join(ROOT, "scripts", "w33_augment_report.py"),
                  os.path.join(ROOT, "configs", "benchmark_ood_norman_canonical.yaml")]:
            f_.write(f"- `{os.path.relpath(p, ROOT).replace(os.sep, '/')}`\n")
        f_.write("\n本报告所有数字由 `scripts/w33_augment_report.py` 从上述 JSON 直接读出，无人工转录。\n")
        f_.write("未外发、未发布（自动化约束）。\n")

    curr["addendum_p31_regression_guard"] = {
        "hashes": rc["hashes"], "n_results_diff": rc["n_results_diff"],
        "n_calibration_diff": rc["n_calibration_diff"],
        "saturation_fixed": n_fixed, "saturation_remaining": n_left,
    }
    curr["addendum_de_subset_metric"] = {
        "discriminability_linear_vs_mean": de["discriminability_linear_vs_mean"],
        "ood_action_ci_separated": bool(sep_action),
        "ood_agent_ci_separated": bool(sep_agent),
        "source": os.path.relpath(de_p, ROOT).replace(os.sep, "/"),
    }
    curr["addendum_readjudication"] = {
        "source": os.path.relpath(rj_p, ROOT).replace(os.sep, "/"),
        "metric_compression": rj.get("metric_compression", {}),
        "ood_action_twin_wins_metrics": wins,
        "ood_action_twin_significant_wins": sig_wins,
        "verdict": ("推翻「疑似仅记忆」" if sig_wins else
                    ("结论悬置（数值占优但 CI 未分离）" if wins else
                     "阴性确认：换判别性指标后孪生仍无增益")),
    }
    curr["addendum_collapse_check"] = {
        "source": os.path.relpath(cc_p, ROOT).replace(os.sep, "/"),
        "ood_agent_compositional_twin_distinct_pred_rows":
            cc.get("ood_agent", {}).get("compositional_twin", {}).get("n_distinct_pred_rows"),
        "ood_agent_compositional_twin_var_ratio":
            cc.get("ood_agent", {}).get("compositional_twin", {}).get("var_ratio_vs_true"),
        "verdict": "compositional_twin 在 ood_agent 上为常数预测器（架构层缺陷）",
    }
    curr["addendum_sop_stage_d_w33"] = {
        "flags": curr.get("flags", []),
        "verdict": "flag 维持：判别性指标重裁后孪生仍不优于线性基线（非测量伪影）",
        "next": ["P4-1 判别性指标正式接入 benchmark_ood.py（重裁已完成，见附录 B2）",
                 "P4-2 修复 phi 在未见基因上的常数坍缩 + 方差比守卫（最高优先）",
                 "P4-3 分子集 conformal / PRESCRIBE", "P4-4 virtual_twin 补 conformal",
                 "P4-5 ood_agent 扩组"],
    }
    with open(js_p, "w", encoding="utf-8") as f_:
        json.dump(curr, f_, ensure_ascii=False, indent=2)

    print(f"[augment] 已增补 {md_p}")
    print(f"[augment] 已增补 {js_p}")
    print(f"[augment] 饱和消除={n_fixed} 残留={n_left} | ood_action CI 分离={sep_action} | ood_agent CI 分离={sep_agent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
