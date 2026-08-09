#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P3-3 秒级哈希回归守卫（regression guard）
=========================================

动机（2026-08-06 W32 审计发现的**流程矛盾**）：
    周度 OOD 基准既要「冻结」（跨周可比、确定性可证）又要「产新证据」（信息增量 > 0）。
    连续三周 results sha256[:16] 恒为 caf29680042baf6f —— 全量 ~14min 重跑的信息增量为 0。

TRIZ 破法（时间分离 + 空间分离）：
    (a) 把「确定性证明」从全量重跑中剥离，降级为**秒级哈希比对**（本脚本）；
    (b) 把省下的算力让给每周轮转的「新证据槽」。

用法：
    python scripts/regress_check.py --prev experiments/20260806-benchmark.json \
                                    --curr experiments/20260807-benchmark.json \
                                    --out  experiments/20260807-regress-check.json

退出码语义（供自动化闸门使用）：
    0 = results 块逐位一致（基准冻结，确定性成立）
    2 = 检出差异（需人工归因：是真实代码变更，还是回归 bug）
    1 = 执行错误（文件缺失 / 结构不符）

注意：本脚本**只读**，不修改任何基准工件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# 相对差异阈值：低于此值视为浮点噪声，不计为「变化」
EPS = 1e-9


def canon_hash(obj) -> str:
    """对 JSON 子块做规范化（sort_keys）后哈希，消除键序影响。"""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def diff_results(prev: dict, curr: dict):
    """逐 (subset, model, metric) 比对点估计块。"""
    rows = []
    subsets = sorted(set(prev) | set(curr))
    for s in subsets:
        p_s, c_s = prev.get(s, {}), curr.get(s, {})
        for m in sorted(set(p_s) | set(c_s)):
            p_m, c_m = p_s.get(m, {}), c_s.get(m, {})
            for k in sorted(set(p_m) | set(c_m)):
                pv, cv = p_m.get(k), c_m.get(k)
                if pv is None or cv is None:
                    rows.append((s, m, k, pv, cv, None, "ADDED/REMOVED"))
                    continue
                d = float(cv) - float(pv)
                if abs(d) > EPS:
                    rows.append((s, m, k, pv, cv, d, "CHANGED"))
    return rows


def diff_calibration(prev: dict, curr: dict):
    """逐 (subset, model) 比对 ECE 与各名义水平覆盖度。"""
    rows = []
    subsets = sorted(set(prev) | set(curr))
    for s in subsets:
        p_s, c_s = prev.get(s, {}), curr.get(s, {})
        for m in sorted(set(p_s) | set(c_s)):
            p_m, c_m = p_s.get(m, {}), c_s.get(m, {})
            # ECE
            pe, ce = p_m.get("ece"), c_m.get("ece")
            if pe is not None and ce is not None and abs(float(ce) - float(pe)) > EPS:
                rows.append((s, m, "ece", pe, ce, float(ce) - float(pe), "CHANGED"))
            # coverage（JSON 重载后为 str 键，统一转 str 比对）
            pc = {str(k): v for k, v in (p_m.get("coverage") or {}).items()}
            cc = {str(k): v for k, v in (c_m.get("coverage") or {}).items()}
            for lvl in sorted(set(pc) | set(cc), key=lambda x: float(x)):
                pv, cv = pc.get(lvl), cc.get(lvl)
                if pv is None or cv is None:
                    rows.append((s, m, f"cov@{lvl}", pv, cv, None, "ADDED/REMOVED"))
                elif abs(float(cv) - float(pv)) > EPS:
                    rows.append((s, m, f"cov@{lvl}", pv, cv, float(cv) - float(pv), "CHANGED"))
    return rows


def fmt_rows(rows, title, limit=None):
    out = [f"\n=== {title} ({len(rows)} 处差异) ==="]
    if not rows:
        out.append("  （无差异）")
        return "\n".join(out)
    out.append("  %-11s %-31s %-10s %12s %12s %12s" % ("subset", "model", "metric", "prev", "curr", "delta"))
    shown = rows if limit is None else rows[:limit]
    for s, m, k, pv, cv, d, tag in shown:
        ds = "n/a" if d is None else f"{d:+.6f}"
        pvs = "n/a" if pv is None else f"{float(pv):.6f}"
        cvs = "n/a" if cv is None else f"{float(cv):.6f}"
        out.append("  %-11s %-31s %-10s %12s %12s %12s" % (s, m, k, pvs, cvs, ds))
    if limit is not None and len(rows) > limit:
        out.append(f"  ... 另有 {len(rows) - limit} 处（见 JSON）")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-3 秒级哈希回归守卫")
    ap.add_argument("--prev", required=True, help="上一期基准 JSON")
    ap.add_argument("--curr", required=True, help="本期基准 JSON")
    ap.add_argument("--out", default=None, help="差异报告 JSON 输出路径")
    ap.add_argument("--expect", choices=["frozen", "changed", "any"], default="any",
                    help="期望：frozen=应无差异；changed=应有差异（如已知代码变更）；any=不断言")
    a = ap.parse_args(argv)

    for p in (a.prev, a.curr):
        if not os.path.exists(p):
            print(f"[regress_check] 文件不存在: {p}", file=sys.stderr)
            return 1

    prev = json.load(open(a.prev, encoding="utf-8"))
    curr = json.load(open(a.curr, encoding="utf-8"))

    h = {
        "prev_file_sha256": file_hash(a.prev),
        "curr_file_sha256": file_hash(a.curr),
        "prev_results_sha256": canon_hash(prev.get("results", {})),
        "curr_results_sha256": canon_hash(curr.get("results", {})),
        "prev_calibration_sha256": canon_hash(prev.get("calibration", {})),
        "curr_calibration_sha256": canon_hash(curr.get("calibration", {})),
        "prev_flags_sha256": canon_hash(prev.get("flags", [])),
        "curr_flags_sha256": canon_hash(curr.get("flags", [])),
    }

    r_rows = diff_results(prev.get("results", {}), curr.get("results", {}))
    c_rows = diff_calibration(prev.get("calibration", {}), curr.get("calibration", {}))

    print("=== 哈希对照（sha256[:16]）===")
    for k in ("results", "calibration", "flags"):
        pk, ck = h[f"prev_{k}_sha256"][:16], h[f"curr_{k}_sha256"][:16]
        print("  %-12s prev=%s  curr=%s  %s" % (k, pk, ck, "同" if pk == ck else "★异"))
    print(fmt_rows(r_rows, "点估计（results）差异", limit=40))
    print(fmt_rows(c_rows, "校准（calibration）差异", limit=40))

    changed = bool(r_rows or c_rows) or h["prev_results_sha256"] != h["curr_results_sha256"] \
        or h["prev_calibration_sha256"] != h["curr_calibration_sha256"]

    payload = {
        "prev": a.prev, "curr": a.curr,
        "hashes": h,
        "n_results_diff": len(r_rows),
        "n_calibration_diff": len(c_rows),
        "results_frozen": h["prev_results_sha256"] == h["curr_results_sha256"],
        "calibration_frozen": h["prev_calibration_sha256"] == h["curr_calibration_sha256"],
        "flags_frozen": h["prev_flags_sha256"] == h["curr_flags_sha256"],
        "results_diff": [dict(zip(("subset", "model", "metric", "prev", "curr", "delta", "tag"), r)) for r in r_rows],
        "calibration_diff": [dict(zip(("subset", "model", "metric", "prev", "curr", "delta", "tag"), r)) for r in c_rows],
    }
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[regress_check] 差异报告已写入: {a.out}")

    if a.expect == "frozen" and changed:
        print("\n[regress_check] 断言失败：期望冻结，但检出差异。", file=sys.stderr)
        return 2
    if a.expect == "changed" and not changed:
        print("\n[regress_check] 断言失败：期望变化，但逐位一致（本周信息增量=0）。", file=sys.stderr)
        return 2
    return 2 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
