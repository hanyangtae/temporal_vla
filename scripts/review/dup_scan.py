#!/usr/bin/env python3
"""중복·고아 탐지 — 레포 검토 스테이지 카드용 기계 근거.

"비슷해 보인다"가 아니라 수치로 판정하기 위한 도구.
설계: docs/superpowers/specs/2026-07-28-repo-review-design.md §5

  python3 scripts/review/dup_scan.py scripts/safe/groot_n15/robocasa/collect ...
  python3 scripts/review/dup_scan.py --min-ratio 0.6 --json out.json <paths...>

세 가지를 본다:
  1. 함수/클래스 완전 중복  — 본문 정규화(주석·docstring·공백 제거) 후 해시 일치
  2. 파일 쌍 유사도          — difflib ratio (라운드별 복사-수정 패턴 검출)
  3. 고아 파일               — import 그래프에서 아무도 참조하지 않는 모듈
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# ---------- 수집 ----------

def collect(paths: list[str], exts=(".py", ".sh")) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = (REPO / raw) if not Path(raw).is_absolute() else Path(raw)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [q for q in p.rglob("*") if q.suffix in exts and "__pycache__" not in q.parts]
    return sorted(set(out))


# ---------- 1. 함수/클래스 완전 중복 ----------

def _norm_src(src: str) -> str:
    """주석·docstring·공백을 지운 정규화 본문. 이름은 유지(같은 로직 다른 이름도 잡되 표시)."""
    lines = []
    for ln in src.splitlines():
        s = re.sub(r"#.*$", "", ln).strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


def dup_defs(files: list[Path], min_lines: int = 8) -> list[dict]:
    """본문 해시가 같은 def/class 묶음. min_lines 미만은 사소해서 제외."""
    buckets: dict[str, list[tuple[Path, str, int, int]]] = defaultdict(list)
    for f in files:
        if f.suffix != ".py":
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            n = end - node.lineno + 1
            if n < min_lines:
                continue
            body = "\n".join(lines[node.lineno - 1:end])
            # docstring 제거
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                ds = node.body[0]
                ds_end = getattr(ds, "end_lineno", ds.lineno)
                body = "\n".join(lines[node.lineno - 1:ds.lineno - 1] + lines[ds_end:end])
            h = hashlib.sha1(_norm_src(body).encode()).hexdigest()[:12]
            buckets[h].append((f, node.name, node.lineno, n))
    groups = []
    for h, members in buckets.items():
        if len({m[0] for m in members}) < 2:  # 서로 다른 파일에 걸친 것만
            continue
        groups.append({
            "hash": h,
            "lines": members[0][3],
            "members": [{"file": str(m[0].relative_to(REPO)), "name": m[1], "line": m[2]} for m in members],
        })
    return sorted(groups, key=lambda g: -g["lines"] * len(g["members"]))


# ---------- 2. 파일 쌍 유사도 ----------

def similar_files(files: list[Path], min_ratio: float = 0.55, min_lines: int = 30) -> list[dict]:
    norm: dict[Path, list[str]] = {}
    for f in files:
        try:
            body = _norm_src(f.read_text(encoding="utf-8", errors="replace")).splitlines()
        except OSError:
            continue
        if len(body) >= min_lines:
            norm[f] = body
    items = sorted(norm.items())
    out = []
    for i, (fa, a) in enumerate(items):
        for fb, b in items[i + 1:]:
            if fa.suffix != fb.suffix:
                continue
            # 길이 차가 크면 건너뛴다 (ratio 상한이 낮음)
            if min(len(a), len(b)) / max(len(a), len(b)) < min_ratio:
                continue
            r = difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio()
            if r < min_ratio:
                continue
            r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            if r >= min_ratio:
                out.append({"a": str(fa.relative_to(REPO)), "b": str(fb.relative_to(REPO)),
                            "ratio": round(r, 3), "lines_a": len(a), "lines_b": len(b)})
    return sorted(out, key=lambda d: -d["ratio"])


# ---------- 3. 고아 (아무도 참조 안 함) ----------

def orphans(files: list[Path], scope: list[Path] | None = None) -> list[str]:
    """scope 안의 어떤 파일에서도 import/실행되지 않는 파일. scope 기본 = repo 전체 py/sh."""
    if scope is None:
        scope = [p for p in REPO.rglob("*")
                 if p.suffix in (".py", ".sh") and "__pycache__" not in p.parts
                 and not any(x in p.parts for x in ("src/policies", "src/benchmarks", "lerobot", ".git"))]
    blobs = []
    for p in scope:
        try:
            blobs.append((p, p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass
    out = []
    for f in files:
        stem, name = f.stem, f.name
        if name in ("__init__.py", "conftest.py"):
            continue
        referenced = False
        for p, txt in blobs:
            if p == f:
                continue
            if re.search(rf"\b(import|from)\s+[\w.]*\b{re.escape(stem)}\b", txt) or name in txt:
                referenced = True
                break
        if not referenced:
            out.append(str(f.relative_to(REPO)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="검사할 파일/디렉토리 (repo 상대경로)")
    ap.add_argument("--min-ratio", type=float, default=0.55)
    ap.add_argument("--min-def-lines", type=int, default=8)
    ap.add_argument("--json", help="결과를 JSON 으로도 저장")
    ap.add_argument("--no-orphans", action="store_true", help="고아 검사 생략(느림)")
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        print("대상 파일 없음", file=sys.stderr)
        return 1
    print(f"대상 {len(files)}개 파일\n")

    dups = dup_defs(files, args.min_def_lines)
    print(f"=== 1. 함수/클래스 완전 중복 ({len(dups)}군) ===")
    for g in dups[:30]:
        print(f"  [{g['lines']}줄 × {len(g['members'])}곳]")
        for m in g["members"]:
            print(f"      {m['file']}:{m['line']}  {m['name']}()")
    if len(dups) > 30:
        print(f"  ... 외 {len(dups)-30}군 (전체는 --json)")

    sims = similar_files(files, args.min_ratio)
    print(f"\n=== 2. 파일 쌍 유사도 ≥{args.min_ratio} ({len(sims)}쌍) ===")
    for s in sims[:30]:
        print(f"  {s['ratio']:.2f}  {s['a']} ({s['lines_a']})  ~  {s['b']} ({s['lines_b']})")
    if len(sims) > 30:
        print(f"  ... 외 {len(sims)-30}쌍")

    orph = [] if args.no_orphans else orphans(files)
    if not args.no_orphans:
        print(f"\n=== 3. 고아 파일 ({len(orph)}개) ===")
        for o in orph:
            print(f"  {o}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"files": [str(f.relative_to(REPO)) for f in files],
             "dup_defs": dups, "similar": sims, "orphans": orph},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
