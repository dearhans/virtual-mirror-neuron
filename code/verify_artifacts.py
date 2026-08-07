#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_artifacts.py —— 声明↔工件强绑定校验器

动机（2026-08-07 事故）：
    一份 automation 叙事声称产出了 multiseed sweep / recalibration 脚本与三份 JSON，
    并据此给出「5/5 种子稳健」「ECE 0.212→0.056 且迁移到 OOD」等结论。
    人工 find 后发现：scripts/ 目录不存在、三份 JSON 不存在、全库连字符串提及都没有。
    即报告文本被生成，但后半段计算从未执行。

    本项目硬约束「任何预测性结论须附复现脚本与随机种子」在此被违反，且
    违反方式是**沉默的**——没有报错，只有一段读起来很像真的 markdown。

修复思路（TRIZ · 物理矛盾）：
    矛盾：报告既要能自由陈述结论（表达力），又必须不能陈述未发生的事（可信性）。
    破法 = 空间分离：把「结论文本」与「工件证据」拆成两层，用本脚本在二者之间
    架一道机器可判的闸门——凡报告中出现的本地路径，必须在磁盘上存在并可哈希。
    破法 = 时间分离：校验发生在**交付前**（秒级），而不是事后人工审计。

用法：
    # 校验单份报告中声称的所有本地路径
    python code/verify_artifacts.py --report experiments/20260806-benchmark.md

    # 同时校验配套 JSON，并写出 PROVENANCE 清单
    python code/verify_artifacts.py --report experiments/20260806-benchmark.md \
        --json experiments/20260806-benchmark.json \
        --manifest experiments/20260806-PROVENANCE.json

    # 显式断言某些工件必须存在（用于 CI / automation 收尾）
    python code/verify_artifacts.py --require scripts/multiseed_ood_sweep.py \
        --require experiments/2026-W32-ood-benchmark.md

退出码：
    0 = 全部 PASS
    1 = 存在 MISSING（有声明无工件）—— 交付必须中止
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- 路径抽取

# 反引号内、看起来像相对路径的字符串：含 / 且带已知扩展名
_BACKTICK = re.compile(r"`([^`\n]+)`")
# markdown 链接 [text](path)
_MDLINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

_EXTS = {
    ".py", ".md", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".png", ".jpg", ".svg", ".pdf", ".npz", ".npy", ".log",
    ".txt", ".zip", ".h5", ".pt", ".ckpt",
}

# 这些前缀视为外部资源，不做磁盘校验
_SKIP_PREFIX = ("http://", "https://", "ftp://", "mailto:", "doi:", "arXiv", "10.")


def _looks_like_path(s: str) -> bool:
    s = s.strip()
    if not s or s.startswith(_SKIP_PREFIX):
        return False
    if any(ch in s for ch in "<>|*?\n"):
        return False
    # 必须带已知扩展名，避免把普通代码片段当路径
    suffix = Path(s).suffix.lower()
    if suffix not in _EXTS:
        return False
    # 排除纯变量名 / 函数调用残片
    if "(" in s or "=" in s or " " in s.strip():
        return False
    return True


def extract_claimed_paths(text: str) -> list[str]:
    """从报告文本中抽出所有被声称的本地工件路径。"""
    found: list[str] = []
    for m in _BACKTICK.finditer(text):
        cand = m.group(1).strip()
        # 反引号里可能是 "a.md / b.json" 这种并列
        for part in re.split(r"[、,，]|\s+/\s+", cand):
            part = part.strip().strip("。，,;；:：")
            if _looks_like_path(part):
                found.append(part)
    for m in _MDLINK.finditer(text):
        cand = m.group(1).strip()
        if _looks_like_path(cand):
            found.append(cand)
    # 去重保序
    seen: set[str] = set()
    uniq = []
    for p in found:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def extract_paths_from_json(obj, acc: list[str] | None = None) -> list[str]:
    """递归扫描 JSON 中的字符串值，捞出路径型工件声明。"""
    if acc is None:
        acc = []
    if isinstance(obj, dict):
        for v in obj.values():
            extract_paths_from_json(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            extract_paths_from_json(v, acc)
    elif isinstance(obj, str) and _looks_like_path(obj):
        acc.append(obj)
    seen: set[str] = set()
    return [p for p in acc if not (p in seen or seen.add(p))]


# ---------------------------------------------------------------- 校验

def sha256_of(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


_SEARCH_EXCLUDE = {".git", ".pylibs", "__pycache__", "node_modules", ".venv"}


def _resolve_bare(root: Path, name: str) -> Path | None:
    """裸文件名（不含目录分量）在报告里很常见，例如正文写 `20260806-benchmark.json`
    而文件实际位于 experiments/ 下。若直接按项目根解析会产生**假阳性 MISSING**，
    使闸门误杀真实报告。故对裸名做一次全库唯一性查找。"""
    hits = []
    for p in root.rglob(name):
        if any(part in _SEARCH_EXCLUDE for part in p.parts):
            continue
        if p.is_file():
            hits.append(p)
            if len(hits) > 1:
                break
    return hits[0] if len(hits) == 1 else None


def verify(root: Path, paths: list[str]) -> list[dict]:
    rows = []
    for rel in paths:
        p = Path(rel)
        target = p if p.is_absolute() else (root / rel)
        resolved_as = None

        # 裸文件名兜底：仅当按根解析失败且路径不含目录分量时才触发
        if not target.exists() and not p.is_absolute() and "/" not in rel and "\\" not in rel:
            alt = _resolve_bare(root, rel)
            if alt is not None:
                target = alt
                resolved_as = str(alt.relative_to(root)).replace("\\", "/")

        if target.exists() and target.is_file():
            rows.append({
                "path": rel,
                "status": "PASS" if resolved_as is None else "PASS*",
                "resolved_as": resolved_as,
                "bytes": target.stat().st_size,
                "sha256_16": sha256_of(target),
            })
        elif target.exists() and target.is_dir():
            rows.append({"path": rel, "status": "PASS_DIR", "resolved_as": None,
                         "bytes": None, "sha256_16": None})
        else:
            rows.append({"path": rel, "status": "MISSING", "resolved_as": None,
                         "bytes": None, "sha256_16": None})
    return rows


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="校验报告中声称的工件是否真实落盘")
    ap.add_argument("--report", action="append", default=[],
                    help="待校验的报告 .md（可重复）")
    ap.add_argument("--json", action="append", default=[],
                    help="待校验的指标 .json（可重复）")
    ap.add_argument("--require", action="append", default=[],
                    help="显式断言必须存在的工件路径（可重复）")
    ap.add_argument("--ignore", action="append", default=[],
                    help="豁免校验的路径（可重复）。用于撤回记录等"
                         "「刻意引用不存在工件」的文档。")
    ap.add_argument("--manifest", default=None,
                    help="将校验结果写出为 PROVENANCE JSON 的路径")
    ap.add_argument("--root", default=None, help="项目根（默认取本文件上一级）")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent

    claimed: list[str] = []
    sources: list[str] = []

    for rp in args.report:
        f = (root / rp) if not Path(rp).is_absolute() else Path(rp)
        if not f.exists():
            print(f"[FATAL] 报告本身不存在: {rp}", file=sys.stderr)
            return 1
        sources.append(str(rp))
        claimed += extract_claimed_paths(f.read_text(encoding="utf-8"))

    for jp in args.json:
        f = (root / jp) if not Path(jp).is_absolute() else Path(jp)
        if not f.exists():
            print(f"[FATAL] 指标 JSON 不存在: {jp}", file=sys.stderr)
            return 1
        sources.append(str(jp))
        claimed += extract_paths_from_json(json.load(f.open(encoding="utf-8")))

    claimed += list(args.require)

    # 去重保序 + 豁免
    ignored = set(args.ignore)
    seen: set[str] = set()
    claimed = [p for p in claimed
               if p not in ignored and not (p in seen or seen.add(p))]

    if ignored:
        print(f"[INFO] 已豁免 {len(ignored)} 条路径: {sorted(ignored)}\n")

    if not claimed:
        print("[WARN] 未从输入中抽出任何工件路径声明。")
        return 0

    rows = verify(root, claimed)
    missing = [r for r in rows if r["status"] == "MISSING"]

    width = max(len(r["path"]) for r in rows)
    print(f"{'STATUS':8s}  {'PATH'.ljust(width)}  {'SHA256[:16]':16s}  BYTES")
    print("-" * (8 + 2 + width + 2 + 16 + 2 + 10))
    for r in rows:
        print(f"{r['status']:8s}  {r['path'].ljust(width)}  "
              f"{(r['sha256_16'] or '-'):16s}  {r['bytes'] if r['bytes'] is not None else '-'}")
        if r.get("resolved_as"):
            print(f"{'':8s}  └─ 裸名解析至: {r['resolved_as']}")

    print()
    print(f"合计 {len(rows)} 条声明 | PASS {len(rows) - len(missing)} | MISSING {len(missing)}")
    if any(r["status"] == "PASS*" for r in rows):
        print("  (PASS* = 报告写的是裸文件名，已在库内唯一定位；建议报告改写完整相对路径)")

    if args.manifest:
        out = (root / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "root": str(root),
            "sources": sources,
            "n_claimed": len(rows),
            "n_missing": len(missing),
            "verdict": "PASS" if not missing else "FAIL",
            "artifacts": rows,
        }, out.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"PROVENANCE 已写出: {out}")

    if missing:
        print()
        print("[FAIL] 存在「有声明无工件」条目 —— 依据项目硬约束（结论须附可复现脚本），"
              "该报告不得进入决策或提交叙事：")
        for r in missing:
            print(f"   - {r['path']}")
        return 1

    print("[PASS] 所有声明均有工件支撑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
