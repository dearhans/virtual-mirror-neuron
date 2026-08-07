#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/wetlab/export_protocol.py
================================================================
闭环 Phase C · 把 OOD 基准的「预测清单」导出为可执行的湿实验方案。

输入 ：experiments/<date>-benchmark.json  （benchmark_ood.run 的 JSON 产物）
输出 ：wetlab/<YYYY-Www>.md                （每周一份，呼应 SOP 阶段 C 闭环动作）

设计原则（硬约束 + 闭环价值）：
  1. 按「信息增益 / 不确定度」对预测排序，选 Top-K。规则：
       - 直接检验因果头/镜像机制的 do-干预（反事实行）→ 最高优先级（核心主张）。
       - OOD 子集按 virtual_twin 的 RMSE 95% CI 宽度排序（越不确定 = 验证价值越高），
         且「机制本应外推」的子集（SHOULD_GENERALIZE）加权。
  2. 每个实验显式区分：
       「记忆（in-distribution）」 vs 「机制泛化（OOD）」—— 仅 OOD 实验计入机制证据。
  3. 每个实验必须给：靶点（光遗传 opsin + 钙成像指示剂）、刺激参数
       （频率/强度/脉宽/时序/组合设计）、预期响应 ± 不确定度、可证伪条件、对照。
  4. 诚实标注数据来源：合成种子 vs 真实 OOD 基准；真实数据未到时显式提醒待复核。

神经域映射层（SCHEMA 已锚定）：
   perturbation P  → 光遗传/化学遗传对某通路的干预（强度/模式）
   action(self/other/imitation) → 一对同源群体（如 M1 同侧执行 vs 对侧镜像）或自我/观察上下文
   neuromodulator(baseline/dopamine/serotonin) → 浴注调质（增益 1.0/1.3/1.6 标定浓度）
   agent(动物/脑片/供体) → 混杂因子，不进特征，仅在 OOD 子集做外推检验

运行：
   python code/wetlab/export_protocol.py --report experiments/20260731-benchmark.json
   python code/wetlab/export_protocol.py --week 2026-W31 --root .
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))  # code/wetlab -> 项目根


def num(x):
    """防御性数值解析：benchmark JSON 经 default=str 序列化，部分 numpy 标量变成字符串。"""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def _config_source(path: str):
    """读取基准 JSON 的 config.data.source（真实数据 = norman/goai）。无法解析返回 None。

    用于「真实优先」选种：只认真正带 source 标签的真实基准，避免被误命名为
    `_real` 的合成样本（无 source 标签）蒙混。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("config", {}).get("data", {}).get("source")
    except Exception:
        return None


def _twin_key(report: dict):
    """动态判定主孪生 key：组合性孪生优先，其次虚拟孪生，再回退首个 predictor。

    避免硬编码 'virtual_twin' 在组合性基准（主键 compositional_twin）下 KeyError。
    """
    results = report.get("results", {}) or {}
    for cand in ("compositional_twin", "virtual_twin"):
        if results and all(cand in results.get(s, {}) for s in results if s != "train"):
            return cand
    for s, d in results.items():
        if isinstance(d, dict) and d:
            return next(iter(d))
    return "virtual_twin"


def _rmse(report: dict, subset: str):
    """取某 OOD 子集主孪生的 RMSE（防御性：缺字段返回 None）。"""
    d = (report.get("results", {}) or {}).get(subset)
    if not isinstance(d, dict):
        return None
    v = d.get(_twin_key(report)) or (next(iter(d.values())) if d else None)
    return num(v.get("rmse")) if isinstance(v, dict) else None


def load_report(report_path: str) -> dict:
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_benchmark_json(root: str) -> str:
    """取 experiments 下「真实优先、其次最新」的基准 JSON。

    真实基准（config.data.source ∈ {norman, goai}）优先于合成基准 —— 避免 newer 的
    合成组合基准（benchmark_comp）按字典序掩盖真实基准，导致 wetlab 横幅误报「未跑通」。
    注意：仅靠文件名 `_real` 不足以判定真实（存在误命名合成样本），必须以 source 标签为准。
    排除 .prev 备份。
    """
    pat = os.path.join(root, "experiments", "*benchmark*.json")
    cands = [p for p in glob.glob(pat) if ".prev" not in os.path.basename(p)]
    if not cands:
        raise FileNotFoundError("未找到 experiments/*benchmark*.json，请先跑 benchmark_ood.py。")
    real = [p for p in cands if _config_source(p) in ("norman", "goai")]
    pool = sorted(real) if real else sorted(cands)
    return pool[-1]


def ood_ranking(report: dict) -> list:
    """按 virtual_twin 在 OOD 子集上的 RMSE 95% CI 宽度排序（越不确定越该验证）。

    返回 [(subset, ci_width, should_generalize, rmse, coverage90)]，宽者在前。
    """
    results = report.get("results", {})
    sg = report.get("config", {}).get("eval", {}).get("should_generalize", {}) or {}
    # 兼容默认 SHOULD_GENERALIZE
    if not sg:
        sg = {"ood_agent": True, "ood_neuro": True, "ood_action": False}
    rows = []
    twin = _twin_key(report)
    for s in report.get("subsets", []):
        if s == "id" or s == "train":
            continue
        r = results.get(s, {}).get(twin)
        if not r:
            continue
        rmse = num(r.get("rmse"))
        lo, hi = num(r.get("rmse_lo")), num(r.get("rmse_hi"))
        ci_width = (hi - lo) if (lo is not None and hi is not None) else None
        # 覆盖率：从 calibration 取 virtual_twin 该子集 coverage@0.9
        cov = None
        cal = report.get("calibration", {}).get(s, {}).get("virtual_twin", {})
        if isinstance(cal, dict):
            cov = num(cal.get("coverage", {}).get(0.9))
        rows.append((s, ci_width, bool(sg.get(s)), rmse, cov))
    # 排序：能外推的优先；同组内按 CI 宽度降序（不确定高 = 验证价值高）
    rows.sort(key=lambda t: (not t[2], -(t[1] or 0)))
    return rows


def build_experiments(report: dict, ood_rows: list) -> list:
    """构造 Top-K 湿实验清单。每条含靶点、刺激参数、预期响应、可证伪条件、对照。"""
    cf = report.get("counterfactual", {}).get("rows", [])
    src = report.get("config", {}).get("data", {}).get("source")
    is_real = src in ("norman", "goai")

    # 反事实行抽样（取第一条作代表读数）
    cf0 = cf[0] if cf else {}
    base_pred = num(cf0.get("pred_base"))
    gain_pred = num(cf0.get("do_gain_1.6_pred"))
    ratio = num(cf0.get("scale_ratio"))

    # 增益轴强度判定：真实数据（如 Norman）常把 neuromodulator 轴设为常量 → 反事实缩放比≈1（弱），
    # 与合成乘法设定（缩放比≈1.6）不同。据此条件化 E1 的预期/否决，避免产出自相矛盾。
    try:
        _ratio_f = float(ratio) if ratio is not None else None
    except (TypeError, ValueError):
        _ratio_f = None
    _gain_weak = _ratio_f is not None and 0.9 <= _ratio_f <= 1.1

    # OOD 子集主孪生 RMSE（动态 key，避免组合性基准下 virtual_twin KeyError）
    r_act = _rmse(report, "ood_action")
    r_neu = _rmse(report, "ood_neuro")
    r_age = _rmse(report, "ood_agent")

    exps = []

    # —— Exp 1 · 乘法调质因果头（do neuromodulator=1.6，核心机制主张）——
    exps.append({
        "id": "E1-mult-gain",
        "tier": "S",
        "title": "乘法调质因果头验证 · do(neuromodulator=1.6)",
        "rationale": "直接检验孪生因果头的乘法机制主张（g 作为增益因子乘入响应）。"
                     "若成立，则模型学的是机制而非记忆——项目核心论点的试金石。",
        "ood_axis": "机制核心（因果头）",
        "memory_or_mechanism": "机制（因果头归纳偏置）",
        "targets": {
            "opto": "M1 第 V 层锥体神经元表达 ChR2（CamKIIα-ChR2-EYFP）；470nm 蓝光，200μm 光纤",
            "calcium": "同群体 AAV1-Syn-GCaMP6s；双光子 30Hz 成像，ROI=M1 L5 锥体胞体",
        },
        "stim": {
            "freq_hz": 20, "intensity_mW_mm2": 8, "pulse_ms": 5,
            "train_s": 3, "combinatorial": "固定光遗传激活；变量为调质增益",
        },
        "modulator": "浴注 5-HT（serotonin）标定至增益 1.6（对照 baseline aCSF=1.0、dopamine=1.3）",
        "expected": (
            f"模型仿真：baseline 预测≈{base_pred}，do(gain=1.6)预测≈{gain_pred}（缩放比≈{ratio}）。"
            + (f" 种子基准下缩放比≈1（增益轴近乎无效应，与合成乘法设定不同）；"
               f"湿实验须由实测判定该轴是否真实存在乘法机制——模型当前预测≈无。"
               if _gain_weak else
               f" 实测 ΔF/F 峰值较 baseline 应缩放 ~{ratio}×（与合成乘法机制一致）。")
        ),
        "uncertainty": "取反事实行 + 该增益子集 bootstrap 95% CI；OOD 下由新颖度撑宽。",
        "falsify": (
            (f"种子基准预测缩放比≈{ratio}（≈1，增益轴弱）；若湿实验实测同样≈1（无真实乘法效应），"
             f"则「乘法因果头」在真实数据下不成立，须回 Phase A/B 重新审视该轴机制设定。"
             if _gain_weak else
             f"若 serotonin 下响应缩放比显著偏离 1.6±CI（如≈1.0 或饱和/翻转），乘法因果头机制不成立，"
             f"须回 Phase A/B 修正因果头。")
        ),
        "controls": "aCSF sham、ChR2-off 病毒对照、dopamine(1.3) 复核点",
    })

    # —— Exp 2 · 镜像对称性（do action: self↔other，签名预测）——
    exps.append({
        "id": "E2-mirror-sym",
        "tier": "S",
        "title": "镜像对称性验证 · do(action: self↔other)",
        "rationale": "检验尺度 B 的动作对称性（self/other 共享 φ，仅增益 mirror_k 不同）——"
                     "「虚拟镜像神经元」的签名预测。",
        "ood_axis": "机制核心（镜像对称）",
        "memory_or_mechanism": "机制（尺度 B 对称性归纳偏置）",
        "targets": {
            "opto": "双侧同源群体（M1 同侧执行通路 + 对侧镜像通路）均表达 ChR2；双光纤独立控制",
            "calcium": "GCaMP6f（快动力学）双侧 M1 同步双光子成像，30Hz",
        },
        "stim": {
            "freq_hz": 20, "intensity_mW_mm2": 8, "pulse_ms": 5,
            "train_s": 3, "combinatorial": "self 与 other 各 20Hz/5ms/3s，伪随机交叉顺序",
        },
        "modulator": "固定 baseline（aCSF，增益 1.0）",
        "expected": "模型预测 self 与 other 响应由 mirror_k 增益关联；二者响应比≈mirror_k。"
                    "仿真参考缩放比≈1.0~1.6 量级（见反事实行）。",
        "uncertainty": "self/other 配对差值的标准误；建议 ≥8 只动物配对。",
        "falsify": "若 self/other 响应比显著偏离 mirror_k（或对称性消失、出现非对称非线性），"
                   "镜像机制假设被否定，需重新设计尺度 B 对称性约束。",
        "controls": "单侧 sham、非镜像区（V1）off-target 阴性对照",
    })

    # —— Exp 3 · 组合双扰动机制（ood_action，最难泛化）——
    exps.append({
        "id": "E3-combo",
        "tier": "A",
        "title": "组合双扰动机制验证 · do(双通路协同)",
        "rationale": "模型只见过单扰动，须用机制组合两条单扰动效应（而非记忆）。"
                     "注：合成基准里 ood_action 被标为「不强制外推」，因它代表完全未见的新动作(imitation)，"
                     "无法由已见单扰动分解；而本湿实验测的是「两个已见单扰动(M1、Striatum 各自训练过)的组合」"
                     "——这是干净的「机制组合」检验，与合成 ood_action 缺口不同，故标记为应外推。",
        "ood_axis": "组合双扰动（seen singles 组合）",
        "memory_or_mechanism": "机制泛化（应外推）",
        "targets": {
            "opto": "M1-ChR2（470nm）+ 纹状体 D1-ChrimsonR（590nm，红光避串扰）；双色独立光纤",
            "calcium": "下游读out群体（丘脑 VA/VL）GCaMP6s，双光子 30Hz",
        },
        "stim": {
            "freq_hz": 20, "intensity_mW_mm2": 8, "pulse_ms": 5,
            "train_s": 2,
            "combinatorial": "单 M1 / 单 Striatum / 双激活（同时 与 50ms 延迟 两条件）",
        },
        "modulator": "固定 baseline（aCSF）",
        "expected": "模型预测双扰动响应 = 机制组合值（非简单加和）。"
                    + (f"基准 RMSE(ood_action)={r_act:.3f}" if r_act is not None else ""),
        "uncertainty": "取 ood_action 子集 RMSE 95% CI；组合条件不确定度更高（应重点验证）。",
        "falsify": "若双扰动实测响应显著偏离机制组合预测（如简单线性加和、饱和或相互抑制），"
                   "组合机制不成立 — 触发 SOP 阶段 D 误差回流 + TRIZ「系统级分离」修正。",
        "controls": "单通路各条件、双色 off 病毒对照",
    })

    # —— Exp 4 · 新调质外推（ood_neuro）——
    ood_neuro = next((r for r in ood_rows if r[0] == "ood_neuro"), None)
    exps.append({
        "id": "E4-novel-mod",
        "tier": "A",
        "title": "新调质外推验证 · ood_neuro（未见增益值）",
        "rationale": "施加训练中未出现的调质（如 NE），检验乘法机制能否外推到未见增益值——"
                     "对应 ood_neuro 子集（机制本应外推）。",
        "ood_axis": "ood_neuro（新调质）",
        "memory_or_mechanism": "机制泛化（应外推）",
        "targets": {
            "opto": "同 E1 的 M1-ChR2 设置",
            "calcium": "同 E1 的 GCaMP6s 设置",
        },
        "stim": {
            "freq_hz": 20, "intensity_mW_mm2": 8, "pulse_ms": 5,
            "train_s": 3, "combinatorial": "固定光遗传；变量为调质增益",
        },
        "modulator": "浴注 NE（norepinephrine）梯度对应增益 1.2/1.5/1.8（模型须外推乘法机制）",
        "expected": "响应随增益单调乘法缩放（机制外推），而非退化为均值/线性基线。"
                    + (f"基准 RMSE(ood_neuro)={r_neu:.3f}" if r_neu is not None else ""),
        "uncertainty": "取 ood_neuro 子集 RMSE 95% CI + 覆盖率（高新颖度处区间应撑宽）。",
        "falsify": "若未见增益下响应不随乘法缩放（崩塌到均值/线性），外推失败 — 需强化调质乘法先验或扩充训练增益覆盖。",
        "controls": "已知增益（baseline/dopamine/serotonin）复核 + aCSF 配对",
    })

    # —— Exp 5 · 未见主体外推（ood_agent，混杂因子）——
    exps.append({
        "id": "E5-novel-agent",
        "tier": "B",
        "title": "未见主体外推验证 · ood_agent（混杂因子）",
        "rationale": "在训练未出现的动物品系/供体重跑 E1 协议，检验 agent 是否真被排除于特征之外"
                     "——对应 ood_agent 子集（机制本应外推）。",
        "ood_axis": "ood_agent（未见主体）",
        "memory_or_mechanism": "机制泛化（应外推；agent 不承载机制）",
        "targets": {
            "opto": "同 E1 设置；但动物为训练中未出现的品系（如 PV-Cre 杂交背景 / 不同供体）",
            "calcium": "同 E1 设置",
        },
        "stim": {
            "freq_hz": 20, "intensity_mW_mm2": 8, "pulse_ms": 5,
            "train_s": 3, "combinatorial": "固定 E1 参数",
        },
        "modulator": "固定 baseline（aCSF）",
        "expected": "因 agent 被排除于特征，预测应与训练主体一致（agent 仅混杂、不承载机制）。"
                    + (f"基准 RMSE(ood_agent)={r_age:.3f}" if r_age is not None else ""),
        "uncertainty": "取 ood_agent 子集 RMSE 95% CI；跨主体方差应被区间覆盖。",
        "falsify": "若不同主体响应系统性偏离预测（超出 CI），说明 agent 泄漏了机制信息 — 须回 Phase A 重新审视特征构造。",
        "controls": "训练主体同协议复核（同笼对照）",
    })

    # —— Exp 6 · 增益剂量-响应梯度（校准因果头）——
    exps.append({
        "id": "E6-dose-response",
        "tier": "B",
        "title": "增益剂量-响应梯度 · do(gain sweep)",
        "rationale": "在 1.0/1.3/1.6/2.0 增益梯度上刻画响应，校准因果头乘法机制并绘制剂量-响应曲线。",
        "ood_axis": "机制核心（因果头校准）",
        "memory_or_mechanism": "机制（因果头标定）",
        "targets": {
            "opto": "同 E1 设置",
            "calcium": "同 E1 设置",
        },
        "stim": {
            "freq_hz": 20, "intensity_mW_mm2": 8, "pulse_ms": 5,
            "train_s": 3, "combinatorial": "固定光遗传；变量为调质增益梯度",
        },
        "modulator": "增益梯度 1.0(baseline)/1.3(dopamine)/1.6(serotonin)/2.0(高浓度)，标定浓度",
        "expected": "响应峰值随增益单调乘法缩放；高增益区若出现饱和/翻转，因果头须在高增益段修正。",
        "uncertainty": "每增益点 ≥6 次 trial 的 ΔF/F 均值±SEM；做剂量-响应拟合残差检验。",
        "falsify": "若出现饱和/翻转（增益>阈值响应不增），乘法机制需在高增益区修正（改加法-饱和混合头）。",
        "controls": "各增益 aCSF 配对、sham",
    })

    # —— Exp 7 · 光遗传闭环反馈（闭环因果验证，新实验类型）——
    exps.append({
        "id": "E7-closed-loop",
        "tier": "A",
        "title": "闭环光遗传反馈 · 实时钙信号触发刺激",
        "rationale": "用闭环范式实时检测目标群体 ΔF/F 越过阈值即触发光刺激，验证因果头在"
                     "动态反馈下的乘法增益机制（而非开环固定刺激下的记忆）。这是从「开环验证」"
                     "迈向「在体闭环因果」的关键一步，直接对应 SOP 反馈原理(#23)，也是湿实验闭环"
                     "回流模型误差的工程落点。",
        "ood_axis": "机制核心（闭环因果头）",
        "memory_or_mechanism": "机制（闭环因果验证）",
        "targets": {
            "opto": "M1 L5 锥体 ChR2（470nm）；实时反馈控制器（基于钙信号 ROI 阈值触发光刺激）",
            "calcium": "GCaMP6f（快动力学 ~120ms 上升）支撑闭环低延迟；双光子 30Hz 体成像 ROI=M1 L5",
        },
        "stim": {
            "freq_hz": 20, "intensity_mW_mm2": 8, "pulse_ms": 5,
            "train_s": 3,
            "combinatorial": "闭环：当 ROI ΔF/F>0.3 触发 20Hz×3s 光刺激；对照开环固定刺激",
            "imaging_window_s": "基线 5s + 闭环窗口 10s（刺激由内源活动触发，jitter±200ms）"
                                 "+ 后成像 5s（全程 30Hz 体成像，刺激期不丢帧）",
        },
        "modulator": "固定 baseline(aCSF)；可选 dopamine(1.3) 复核增益",
        "expected": "闭环下当内源活动强时刺激增幅被调制（乘法增益仍成立）；触发后 ΔF/F 与开环 E1 增益量级一致。",
        "uncertainty": "闭环事件触发 jitter ±200ms；每条件 ≥30 次闭环 trial 的触发后 ΔF/F 均值±SEM。",
        "falsify": "若闭环触发后响应与开环无差异（增益不随内源活动调制），因果头乘法机制在动态反馈下失效 "
                   "→ 须修正因果头为状态依赖形式。",
        "controls": "开环同参数对照、ROI 随机负位对照、sham",
    })

    return exps


def emit_markdown(exps: list, report: dict, week: str, report_path: str) -> str:
    src = report.get("config", {}).get("data", {}).get("source")
    is_real = src in ("norman", "goai")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    L = []
    L.append(f"# 湿实验验证方案 · {week}\n")
    L.append(f"- 生成时间(UTC): {report.get('generated_at', datetime.now(timezone.utc).isoformat())}")
    L.append(f"- 种子基准报告: `{os.path.basename(report_path)}`")
    L.append(f"- 数据来源: {'真实 OOD 基准（' + src + '）' if is_real else '⚠️ 合成模型预测（真实 OOD 基准未跑通，预测值与不确定度待复核）'}")
    L.append(f"- 闭环依据: SOP 阶段 C（Top-K 按信息增益/不确定度排序）")
    L.append(f"- 种子基准概览: 训练 {report.get('n_train')} / ID测试 {report.get('n_id_test')} / 总 {report.get('n_total')}\n")

    # 优先级说明
    L.append("## 0. 选靶逻辑（信息增益 / 不确定度排序）\n")
    L.append("- **Tier S（核心主张）**：直接检验因果头乘法机制 / 镜像对称性——项目论点的试金石，失败即伤筋动骨。")
    L.append("- **Tier A（最难泛化）**：组合双扰动、新调质外推——`should_generalize` 标记为应外推的 OOD 子集。")
    L.append("- **Tier B（校准/混杂）**：未见主体外推、增益剂量梯度——校准机制并检验混杂因子是否被正确处理。")
    L.append("- 排序同时参考 virtual_twin 在 OOD 子集的 RMSE 95% CI 宽度（越宽 = 验证价值越高）。\n")

    # 总览表
    L.append("## 1. 实验总览（按优先级）\n")
    L.append("| # | ID | 靶点机制 | OOD 轴 | 记忆/机制 | Tier |")
    L.append("|---|---|---|---|---|---|")
    tier_rank = {"S": 0, "A": 1, "B": 2}
    for i, e in enumerate(sorted(exps, key=lambda x: tier_rank.get(x["tier"], 9)), 1):
        L.append(f"| {i} | `{e['id']}` | {e['title']} | {e['ood_axis']} | {e['memory_or_mechanism']} | {e['tier']} |")
    L.append("")

    # 逐实验协议
    L.append("## 2. 逐实验可执行协议\n")
    for i, e in enumerate(sorted(exps, key=lambda x: tier_rank.get(x["tier"], 9)), 1):
        L.append(f"### 实验 {i} · {e['title']}  `[{e['id']}]`\n")
        L.append(f"**选靶理由**：{e['rationale']}\n")
        L.append(f"- **记忆/机制归类**：{e['memory_or_mechanism']}")
        L.append(f"- **OOD 轴**：{e['ood_axis']}\n")
        L.append("**靶点（Targets）**")
        L.append(f"- 光遗传：{e['targets']['opto']}")
        L.append(f"- 钙成像：{e['targets']['calcium']}\n")
        st = e["stim"]
        L.append("**刺激参数（Stimulation）**")
        L.append(f"- 频率：{st['freq_hz']} Hz")
        L.append(f"- 强度：{st['intensity_mW_mm2']} mW/mm²（光纤尖端）")
        L.append(f"- 脉宽：{st['pulse_ms']} ms")
        L.append(f"- 训练时长：{st['train_s']} s")
        L.append(f"- 组合/时序设计：{st['combinatorial']}")
        L.append(f"- 双光子成像时序：{st.get('imaging_window_s', f'刺激前 5s 基线采集 + {st['train_s']}s 光刺激 + 刺激后 5s 后成像（30Hz 体成像，GCaMP6 不丢帧）')}\n")
        if e.get("modulator"):
            L.append(f"**神经调质**：{e['modulator']}\n")
        L.append("**预期响应 + 不确定度**")
        L.append(f"- {e['expected']}")
        L.append(f"- 不确定度处理：{e['uncertainty']}\n")
        L.append(f"**可证伪条件（Falsification）**：{e['falsify']}\n")
        L.append(f"**对照设计（Controls）**：{e['controls']}\n")

    # 闭环与不确定度声明
    L.append("## 3. 闭环与不确定度声明（硬约束）\n")
    L.append("- **参数现实性**：文中光遗传（470/590nm、5–15 mW/mm²、5ms 脉宽、20Hz、2–3s 训练）与钙成像"
             "（GCaMP6f/6s、双光子 30Hz）为小鼠在体/脑片典型量级，须按实际制备（病毒滴度、光纤定位、"
             "麻醉状态）标定；增益 1.0/1.3/1.6/2.0 为 SCHEMA 调质映射值，真实浓度须通过剂量标定实验确定。")
    L.append("- 所有 OOD 结论均带 bootstrap 95% CI；CI 重叠视为不显著，避免过拟合式误判。")
    L.append("- 不确定度 = 集成 epistemic + 新颖度(OOD) 合成；未见条件下区间应撑宽（缓解黑箱过度自信）。")
    L.append("- **诚实标注**：本方案所有「预期响应」数值来自种子基准的仿真预测；真实 OOD 基准"
             f"（{'已就位' if is_real else '未跑通'}）跑通后须复核预测值与不确定度再执行。")
    L.append("- 执行后把实测响应与误差写回 Phase D（mirror-neuron 观察库），调用 `suggest()`/`evolve()` 更新策略权重，"
             "并作为下一轮训练/校准的监督信号（Phase A/B）。\n")

    L.append("## 4. 执行顺序建议\n")
    L.append("1. **E1 + E6**（同设施，增益梯度先行校准因果头）→ 2. **E2**（镜像对称性）→ "
             "3. **E3**（组合双扰动，最难，留足样本）→ 4. **E4 + E5**（外推子集，可在同批动物扩展）→ "
             "5. **E7**（闭环反馈，验证在体动态因果，依赖 E1/E6 标定）。")

    # §5 如何解读与本方案否决条件
    L.append("\n## 5. 如何解读与本方案否决条件\n")
    L.append("### 5.1 数值化否决规则（执行前团队须共识）\n")
    L.append("- **R1 显著性**：任何实验，模型预测 95% CI 与对照/基线预测 95% CI 重叠 → 视为不显著，"
             "不得宣称机制成立。")
    L.append("- **R2 外推失败触发**：某 `ood_*` 子集 virtual_twin RMSE 相对 `id` 子集退化 > 50%"
             "（或 CI 完全不覆盖基线 RMSE），触发 SOP 阶段 D 误差回流 + TRIZ 修正"
             "（详见 docs/triz_contradiction_analysis.md）。")
    L.append("- **R3 可证伪命中**：下表任一「否决条件」被实测满足，对应机制假设即被否定，"
             "须回 Phase A/B 修正后重跑。\n")

    L.append("### 5.2 可证伪条件汇总（每条实验的可证伪条件）\n")
    L.append("| 实验 | 机制假设 | 否决条件（实测命中即否定） |")
    L.append("|---|---|---|")
    for e in exps:
        L.append(f"| `{e['id']}` | {e['title']} | {e['falsify']} |")
    L.append("")

    L.append("### 5.3 数据回收与闭环\n")
    L.append("- 执行后把实测响应、误差、闭环 jitter 写回 Phase D 观察库，调用 `suggest()`/`evolve()` 更新策略权重；")
    L.append("  并作为下一轮训练/校准的监督信号（Phase A/B）。")
    L.append("- 每周五自动化（`export_protocol.py`）随新基准报告复现本方案，否决条件表随实验增减自动同步。\n")

    L.append("\n> 本文件由 `code/wetlab/export_protocol.py` 自动生成，可随每周基准报告复现。")

    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="导出湿实验验证方案（SOP 阶段 C 闭环）")
    ap.add_argument("--report", default=None, help="benchmark JSON 路径；默认取 experiments 最新一份")
    ap.add_argument("--week", default=None, help="ISO 周标签，如 2026-W31；默认按当前日期推算")
    ap.add_argument("--root", default=PROJECT_ROOT, help="项目根目录")
    args = ap.parse_args(argv)

    report_path = args.report or latest_benchmark_json(args.root)
    report = load_report(report_path)

    if not args.week:
        # 推算 ISO 周
        d = datetime.now()
        iso = d.isocalendar()
        args.week = f"{iso[0]}-W{iso[1]:02d}"

    ood_rows = ood_ranking(report)
    exps = build_experiments(report, ood_rows)

    md = emit_markdown(exps, report, args.week, report_path)

    out_dir = os.path.join(args.root, "wetlab")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.week}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[export_protocol] 种子报告: {report_path}")
    print(f"[export_protocol] OOD 排序(宽→窄): " + ", ".join(
        f"{r[0]}({'gen' if r[2] else 'mem'})" for r in ood_rows))
    print(f"[export_protocol] 生成 {len(exps)} 个湿实验 → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
