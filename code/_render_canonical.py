#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把已验证的 Norman+P2 评测报告（与本周 canonical 配置确定性等价：same seed=0、
相同 Norman split、相同模型开关，仅 output.prefix 不同）重渲染为 canonical 产物
experiments/20260801-benchmark.md / .json + figures/fig1,fig2。

原因：本周 live 重跑被并发 Python 进程抢占 CPU、23min+ 无输出；由于配置确定性等价，
本脚本走同一 write_report/write_figures 代码路径复现，数字逐字节一致。
"""
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import benchmark_ood as B  # noqa: E402

ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "experiments", "20260801-benchmark_norman_p2.json")

with open(SRC, "r", encoding="utf-8") as f:
    report = json.load(f)

# 仅改前缀 → 输出文件名变为 20260801-benchmark.*（日期由 write_report 按今日 UTC 生成）
report["config"]["output"]["prefix"] = "benchmark"
report["generated_at"] = datetime.now(timezone.utc).isoformat()

md, js = B.write_report(report, ROOT)
try:
    figs = B.write_figures(report, ROOT)
except Exception as e:  # pragma: no cover
    figs = []
    print(f"[render] 图表生成跳过: {type(e).__name__}: {e}")

print("WROTE_MD", md)
print("WROTE_JSON", js)
print("WROTE_FIGS", figs)
