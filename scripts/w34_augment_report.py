#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W34 报告增补：把回归守卫 + P4-2 新证据槽 + SOP 阶段 D/TRIZ 写进当周基准报告。

只做**追加**：不改写 benchmark_ood.py 生成的正文（保持 results 块冻结可比）。
用法：python scripts/w34_augment_report.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATE = "20260808"
MD = os.path.join(ROOT, f"experiments/{DATE}-benchmark.md")
JS = os.path.join(ROOT, f"experiments/{DATE}-benchmark.json")
RG = os.path.join(ROOT, f"experiments/{DATE}-regress-check.json")
P42 = os.path.join(ROOT, f"experiments/{DATE}-p42-solo-imputation.json")

SAT = 0.2125
ARM_A = "armA_compositional_twin_P1"
ARM_B = "armB_compositional_twin_P42"
LABEL = {"id": "ID(已见基因)", "ood_agent": "OOD-未见基因", "ood_action": "OOD-组合双扰动", "ood_neuro": "OOD-未见批次"}


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    bench, rg, p42 = load(JS), load(RG), load(P42)
    if bench is None:
        print(f"[w34] 缺少 {JS}", file=sys.stderr)
        return 1
    lines = ["", "---", "",
             "## 附录 A · P3-3 秒级回归守卫（确定性冻结核验）", ""]

    # ---------------- 附录 A：回归守卫 ----------------
    if rg:
        h = rg["hashes"]
        lines += [
            f"对照上期：`{os.path.basename(rg['prev'])}` → `{os.path.basename(rg['curr'])}`", "",
            "| 块 | 上期 sha256[:16] | 本期 sha256[:16] | 判定 |",
            "|---|---|---|---|",
        ]
        for k in ("results", "calibration", "flags"):
            pk, ck = h[f"prev_{k}_sha256"][:16], h[f"curr_{k}_sha256"][:16]
            lines.append(f"| `{k}` | `{pk}` | `{ck}` | {'冻结（逐位一致）' if pk == ck else '★ 变化'} |")
        lines += ["",
                  f"- 点估计差异处数：**{rg['n_results_diff']}**；校准差异处数：**{rg['n_calibration_diff']}**。",
                  f"- `results_frozen={rg['results_frozen']}`，`calibration_frozen={rg['calibration_frozen']}`。", ""]
        if rg["results_frozen"]:
            lines += ["**判读**：μ（点估计）路径确定性成立——本周未对既有 7 个预测器的均值通路做任何改动，",
                      "跨周数字可直接比较。这也意味着**全量重跑本身的信息增量为 0**，本周的信息增量全部来自附录 B 的新证据槽。", ""]
        else:
            lines += ["**判读**：★ 检出点估计差异，须人工归因（真实代码变更 vs 回归 bug），",
                      "在归因完成前不得对外声称任何跨周比较结论。", ""]
    else:
        lines += ["（未生成回归守卫结果）", ""]

    # ---------------- 附录 B：P4-2 新证据槽 ----------------
    lines += ["## 附录 B · 新证据槽（W34）：P4-2 共扰动→单独效应插补，修 `ood_agent` 常数坍缩", ""]
    if not p42:
        lines += ["（未生成 P4-2 结果）", ""]
    else:
        s = p42["imputer_summary"]
        chk = p42.get("arm_a_reproduction_check", {})
        lines += [
            "### B.1 缺陷复述与根因定位", "",
            "W33 坍缩检测发现：`compositional_twin` 在 `ood_agent` 上**完全坍缩为常数预测器**",
            "（跨扰动组 `var_ratio=1.07e-26`，20 组只输出 1 种预测）。本周定位到精确根因——", "",
            "1. `ood_agent` = 「held-out 基因的**单**扰动」；",
            "2. 但切分规则把「**含 held-out 基因的双扰动**」划进了 `train`（作为额外信号）",
            "   → 该基因在训练集中**出现过**，只是从未以「单独扰动」形式出现；",
            "3. 而 P1 的 φ **只在单扰动行上监督**（`single_mask = pb.sum(1)==0`），双扰动行完全不进 φ 的训练集",
            "   → held-out 基因对应的 one-hot 坐标在 φ 的训练输入里恒为 0",
            "   → `φ(p_unseen) ≡ φ(0) = baseline` → **必然输出常数**。", "",
            "**结论：这不是容量问题，是监督口径问题——信号在数据里，被 P1 的口径丢掉了。**", "",
            "### B.2 破法（TRIZ 自服务 / 预先作用）", "",
            "用系统自身的加法先验，为缺失坐标制造监督：对训练双扰动 (a=已见, u=未见)，",
            "由 `z_au = φ(a) + φ(u) − b` 反解 `e_u = z_au − b̂ − ê_a`，对该基因的所有共扰动取均值得 `ê_u`；",
            "预测时把 `φ(p_u)` 覆写为 `b̂ + ê_u`。", "",
            "> 镜像神经元语义：**推断一个只在合作中被观察过的主体，单独行动时的效应**——",
            "> 共享潜空间（加法基 φ）提供了把「合作观察」投影回「单独执行」的通道。", "",
            f"- 基因总数 {s['n_genes_total']}，有单扰动直接监督 {s['n_genes_with_single_supervision']}，",
            f"  无单扰动监督 {s['n_genes_without_single_supervision']}，其中**可由共扰动插补 {s['n_genes_imputed_from_codoubles']}**，",
            f"  真正不可识别 {s['n_genes_unidentifiable']}（训练集中连共扰动都没有 → 如实报告，不插补）。",
            f"- 共形分位数：臂A(P1) q={p42['conformal_q']['armA_P1']:.4f}，臂B(P4-2) q={p42['conformal_q']['armB_P42']:.4f}。", "",
            "### B.3 受控消融（两臂共享同一个已 fit 的 φ，零混淆）", "",
        ]
        if chk.get("performed"):
            ok = chk["armA_matches_canonical"]
            dd = ", ".join(f"{k}: {v:+.2e}" for k, v in chk["deltas"].items())
            lines += [
                f"**臂 A 复现核验：{'✅ 通过' if ok else '❌ 未通过'}** —— 臂 A 与当周基准 `compositional_twin` 的 RMSE 差 = {dd}。",
                ("消融管线可信，臂 B 的差异可归因于插补本身。" if ok
                 else "**消融管线不可信，以下臂 B 数字不得作为结论使用。**"), "",
            ]
        lines += [
            "| 子集 | 预测器 | RMSE [95% CI] | pbRMSE | var_ratio | 不同预测行 | 坍缩 | ECE |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for sub, row in p42["subsets"].items():
            for name in (ARM_A, ARM_B, "linear", "mean"):
                e = row.get(name)
                if not e:
                    continue
                ece = f"{e['ece']:.4f}" + ("★饱和" if e.get("saturated") else "") if "ece" in e else "—"
                lines.append(
                    f"| {LABEL.get(sub, sub)} | `{name}` | {e['rmse']:.4f} [{e['ci_low']:.4f}, {e['ci_high']:.4f}] | "
                    f"{e['pb_rmse']:.4f} | {e['var_ratio_vs_true']:.2e} | {e['n_distinct_pred_rows']}/{e['n_groups']} | "
                    f"{'**是**' if e['collapsed'] else '否'} | {ece} |")
        lines.append("")

        # 判读
        ag = p42["subsets"].get("ood_agent", {})
        if ag:
            A, B = ag.get(ARM_A), ag.get(ARM_B)
            ln, mn = ag.get("linear"), ag.get("mean")
            best_base = min([x for x in (ln, mn) if x], key=lambda e: e["rmse"])
            best_name = "linear" if best_base is ln else "mean"
            d_rmse = B["rmse"] - best_base["rmse"]
            d_pb = B["pb_rmse"] - best_base["pb_rmse"]
            sig = not (B["ci_low"] <= best_base["rmse"] <= B["ci_high"])
            lines += [
                "### B.4 判读（`ood_agent`，本周核心）", "",
                f"- **坍缩是否修复**：臂A `var_ratio={A['var_ratio_vs_true']:.2e}`（{A['n_distinct_pred_rows']}/{A['n_groups']} 种预测）"
                f" → 臂B `var_ratio={B['var_ratio_vs_true']:.2e}`（{B['n_distinct_pred_rows']}/{B['n_groups']} 种预测）。"
                + ("**坍缩已修复：模型恢复扰动特异性。**" if not B["collapsed"] else "**仍坍缩：破法未生效。**"),
                f"- **点估计**：臂B RMSE={B['rmse']:.4f} [{B['ci_low']:.4f}, {B['ci_high']:.4f}]；"
                f"最佳简单基线 `{best_name}`={best_base['rmse']:.4f}，Δ={d_rmse:+.4f}"
                f"（{'CI 不含基线值 → 显著' if sig else 'CI 覆盖基线值 → 不显著'}）。",
                f"- **判别性指标（pbRMSE，W33 已证实抗「预测均值」陷阱）**：臂B={B['pb_rmse']:.4f} vs `{best_name}`={best_base['pb_rmse']:.4f}，Δ={d_pb:+.4f}。",
                "- **诚实边界**：`ê_u` 用加法基去解释真实（含上位性的）响应，对强上位性基因对有偏；",
                "  且被插补基因的证据量差异很大（共扰动条数不等），故已把插补标准误按方差加进 σ_pred。", "",
            ]

    # ---------------- 附录 C：SOP D + TRIZ ----------------
    flags = bench.get("flags", [])
    lines += ["## 附录 C · SOP 阶段 D 误差回流 + TRIZ 修正（W34 处置）", "", "### C.1 本周触发状态", ""]
    if flags:
        lines += ["| 子集 | 模型 | Δ vs 最佳基线 | 判定 |", "|---|---|---|---|"]
        for f in flags:
            lines.append(f"| {f['subset']} | `{f['twin']}` | {f['delta_vs_best_baseline']:+.4f} | {f['verdict']} |")
    else:
        lines.append("（本周无「疑似仅记忆」告警）")
    lines.append("")

    json.dump({"appendix_generated": True}, open(os.devnull, "w"))  # no-op guard

    # 追加写 md
    with open(MD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 回写 json addenda
    bench["addendum_w34_regress_guard"] = rg
    bench["addendum_w34_p42_solo_imputation"] = p42
    json.dump(bench, open(JS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[w34] 报告已增补: {MD}")
    print(f"[w34] JSON 已增补 addendum_w34_*: {JS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
