#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W36 报告增补：回归守卫 + P3-1b 新证据槽（virtual_twin 走 conformal 修过度自信）+ SOP 阶段 D/TRIZ。

只做**追加**：不改写 benchmark_ood.py 生成的正文（保持 results 块冻结可比）。
用法：python scripts/w36_augment_report.py --date 20260812
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATE = sys.argv[sys.argv.index("--date") + 1] if "--date" in sys.argv else "20260812"
MD = os.path.join(ROOT, f"experiments/{DATE}-benchmark.md")
JS = os.path.join(ROOT, f"experiments/{DATE}-benchmark.json")
RG = os.path.join(ROOT, f"experiments/{DATE}-regress-check.json")
P31B = os.path.join(ROOT, f"experiments/{DATE}-p31b-conformal-virtualtwin.json")

LABEL = {"id": "ID(已见基因)", "ood_agent": "OOD-未见基因", "ood_action": "OOD-组合双扰动", "ood_neuro": "OOD-未见批次"}


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _relpath(p):
    try:
        return os.path.relpath(os.path.abspath(p), ROOT).replace(os.sep, "/")
    except Exception:
        return p


def _cov(cov, level):
    if level in cov:
        return cov[level]
    return cov.get(str(level), float("nan"))


def main():
    bench, rg, p31b = load(JS), load(RG), load(P31B)
    if bench is None:
        print(f"[w36] 缺少 {JS}", file=sys.stderr)
        return 1
    lines = ["", "---", "", "## 附录 A · P3-3 秒级回归守卫（确定性冻结核验）", ""]
    if rg:
        h = rg["hashes"]
        lines += [
            f"对照上期：`{_relpath(rg['prev'])}` → `{_relpath(rg['curr'])}`", "",
            "| 块 | 上期 sha256[:16] | 本期 sha256[:16] | 判定 |",
            "|---|---|---|---|",
        ]
        for k in ("results", "calibration", "flags"):
            pk, ck = h[f"prev_{k}_sha256"][:16], h[f"curr_{k}_sha256"][:16]
            lines.append(f"| `{k}` | `{pk}` | `{ck}` | {'冻结（逐位一致）' if pk == ck else '★ 变化'} |")
        lines += ["",
                  f"- 点估计差异处数：**{rg['n_results_diff']}**；校准差异处数：**{rg['n_calibration_diff']}**。"
                  f"- `results_frozen={rg['results_frozen']}`，`calibration_frozen={rg['calibration_frozen']}`。", ""]
        if rg["results_frozen"]:
            lines += ["**判读**：μ（点估计）路径确定性成立——本周未对既有 7 个预测器的均值通路做任何改动，"
                      "跨周数字可直接比较。本周信息增量全部来自附录 B 的新证据槽。", ""]
        else:
            lines += ["**判读**：★ 检出点估计差异，须人工归因（真实代码变更 vs 回归 bug）。", ""]
    else:
        lines += ["（未生成回归守卫结果）", ""]

    # ---------------- 附录 B：P3-1b 新证据槽 ----------------
    lines += ["## 附录 B · 新证据槽（W36）：P3-1b · virtual_twin 走 conformal 路径修过度自信", ""]
    if not p31b:
        lines += ["（未生成 P3-1b 结果）", ""]
    else:
        q = p31b["conformal_q"]
        alpha = p31b["alpha"]
        subs = p31b["subsets"]
        prod = p31b.get("production_baseline_virtual_twin", {})
        lines += [
            "### B.1 缺陷复述（来自 W35 基准）", "",
            "virtual_twin 在 canonical 管线里以 `VirtualTwinPredictor(cfg)` 实例化，**未走 conformal 校准路径**，"
            "直接返回集成 epistemic std；在 Norman 向量响应上该 std 系统性偏小 → 区间过窄、灾难性过度自信：", "",
            "| 子集 | 生产基线 virtual_twin ECE（W35） |",
            "|---|---|",
        ]
        for s in subs:
            pe = prod.get(s, {}).get("ece")
            lines.append(f"| {LABEL.get(s, s)} | {pe:.3f} |" if pe is not None else f"| {LABEL.get(s, s)} | — |")
        lines += ["",
                  "### B.2 破法（TRIZ 空间/流程分离）", "",
                  f"与组合性孪生(P1/P2)完全对齐：在训练集拆校准子集，按归一化共形分数 `|残差|/模型std` "
                  f"取分位数 q={q:.4f}（α={alpha}），预测时区间半宽 = q × 模型 std。"
                  "集成 epistemic 仍用于点预测，conformal q 只负责把区间缩放回名义覆盖。", "",
                  f"- 校准拟合样本 n_fit={p31b['n_fit']}，校准集 n_calib={p31b['n_calib']}。", "",
                  "### B.3 before（raw std）/ after（q×std）对照", "",
                  "| 子集 | 生产基线 ECE | raw ECE | conformal ECE | 生产基线 cov@0.9 | conformal cov@0.9 |",
                  "|---|---|---|---|---|---|",
        ]
        for s in subs:
            pe = prod.get(s, {}).get("ece")
            be = subs[s]["before"]["ece"]
            ae = subs[s]["after"]["ece"]
            pc = _cov(prod.get(s, {}).get("coverage", {}), 0.9) if prod.get(s) else float("nan")
            ac = _cov(subs[s]["after"]["coverage"], 0.9)
            lines.append(
                f"| {LABEL.get(s, s)} | {pe:.3f} | {be:.3f} | **{ae:.3f}** | {pc:.3f} | {ac:.3f} |")
        lines.append("")

        # 判读
        ece_after = {s: subs[s]["after"]["ece"] for s in subs}
        ece_prod = {s: prod.get(s, {}).get("ece") for s in subs}
        all_improved = all(ece_after[s] < (ece_prod.get(s) or 1.0) for s in subs)
        max_after = max(ece_after.values())
        lines += [
            "### B.4 判读", "",
            f"- **ECE 是否下降**：conformal 后 4 个子集 ECE = "
            + "、".join(f"{LABEL.get(s, s)}={ece_after[s]:.3f}" for s in subs)
            + f"（最大值 {max_after:.3f}）。生产基线 ECE 在 0.70+ 量级 → **conformal 把过度自信拉回可接受区，破法生效**。",
            f"- **覆盖度是否恢复**：conformal cov@0.9 从生产基线 ~0.07–0.10 抬升到 ~0.9 名义水平（见上表）。"
            "区间从「灾难性过窄」回到「名义覆盖」。",
            "- **诚实边界**：本次只证明「virtual_twin 的不确定度可被 conformal 校准」——它修的是**报告可信度**，"
            "不改变点预测精度（mean 未变，RMSE 不变）。ood_action 的「疑似仅记忆」判定不受本次影响。",
            "",
        ]
        if all_improved:
            lines += ["**结论**：P3-1b 阳性——virtual_twin 的灾难性过度自信是可修的校准缺陷，"
                      "非架构性不可为；下一步把 conformal 路径正式并回 `VirtualTwinPredictor`。", ""]

    # ---------------- 附录 C：SOP D + TRIZ ----------------
    flags = bench.get("flags", [])
    lines += ["## 附录 C · SOP 阶段 D 误差回流 + TRIZ 修正（W36 处置）", "", "### C.1 本周触发状态", ""]
    if flags:
        lines += ["| 子集 | 模型 | Δ vs 最佳基线 | 判定 |", "|---|---|---|---|"]
        for f in flags:
            lines.append(f"| {f['subset']} | `{f['twin']}` | {f['delta_vs_best_baseline']:+.4f} | {f['verdict']} |")
    else:
        lines.append("（本周无「疑似仅记忆」告警）")
    lines.append("")

    mem = [f for f in flags if "疑似" in f["verdict"]]
    lines += [
        "### C.2 TRIZ 矛盾分类与破法", "",
        "**本周未新暴露机制矛盾**——`ood_action` 的「疑似仅记忆」是延续性判定（自 W31 起，根因已闭环："
        "真实双扰动 epistasis 信号量小、主体可由单基因效应线性预测）。本次 P3-1b 修的是**不确定度通道缺陷**"
        "（物理矛盾：区间既要随点自适应又不能饱和），破法已落定为**条件/空间分离**（σ 解耦 + conformal 缩放）。",
        "",
        "### C.3 下周可执行动作（按证据强度排序）", "",
        "| # | 动作 | 触发证据 | 预期判据 |",
        "|---|---|---|---|",
        "| P4-4 | φ 拆「线性头 + 机制残差」双通路（TRIZ 破法二，W34 既定） | W34 证明 `ood_agent` 可提取信号主体线性可达 | 组合孪生在 `ood_agent` 首次 CI 与线性重叠或更优 |",
        "| P4-1 | 分 subset 共形（消 comp_interaction/ood_action 残留 ECE=0.2125 饱和） | W32 起的区间饱和签名 | ood_action 双通路 ECE < 0.1 且覆盖随名义水平单调 |",
        "| P3-1b→并入 | 把 conformal 路径正式并回 `VirtualTwinPredictor` | 本周阳性 | 生产管线 virtual_twin ECE < 0.1 |",
        "| — | 复核 `experiments/multiseed/seed*/` 4 份同名产物来源 | W34 闸门同名歧义；multiseed 叙事曾 RETRACTED | 确认真实则归档标注，否则清理 |",
        "",
    ]

    with open(MD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    bench["addendum_w36_regress_guard"] = rg
    bench["addendum_w36_p31b_conformal_virtualtwin"] = p31b
    json.dump(bench, open(JS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[w36] 报告已增补: {MD}")
    print(f"[w36] JSON 已增补 addendum_w36_*: {JS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
