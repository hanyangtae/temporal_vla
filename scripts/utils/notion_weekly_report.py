#!/usr/bin/env python3
"""Notion Weekly Report 헬퍼 CLI.

Notion `Weekly Report` 페이지 (연도 → 월 → 주차 toggle heading) 를 탐색·읽기·작성한다.
`/weekly-report` skill 이 이 CLI 를 통해 Notion 과 통신한다 (stdlib 만 사용).

서브커맨드:
  resolve        --year Y --month M [--week N] [--create]  : 주차 heading id 탐색/생성
  latest-report                                            : 가장 최근 내용이 있는 주차 heading 탐색
  read-week      --block-id ID                             : 주차 heading 아래 블록을 markdown 으로 덤프
  upload         --block-id ID --draft FILE                : markdown draft 를 Notion 블록으로 변환·append

토큰은 NOTION_TOKEN env → repo 루트 .env 순으로 로드한다.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
WEEKLY_REPORT_PAGE_ID = "1ac63918-d42a-804e-b403-e37471a837c6"

# 주차 heading 표기 변형: "2째주", "2주차", "둘째주" 등
_WEEK_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*째\s*주\s*$"),
    re.compile(r"^\s*(\d+)\s*주\s*차\s*$"),
]
_KO_ORDINALS = {"첫": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5}
_KO_WEEK_PATTERN = re.compile(r"^\s*(첫|둘|셋|넷|다섯)\s*째\s*주\s*$")


def load_token():
    import os

    token = os.environ.get("NOTION_TOKEN")
    if token:
        return token
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("NOTION_TOKEN="):
                return line.split("=", 1)[1].strip()
    sys.exit("NOTION_TOKEN 을 env 또는 .env 에서 찾지 못함")


def api(method, path, payload=None):
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {load_token()}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"Notion API {method} {path} 실패 ({e.code}): {e.read().decode()[:500]}")


def list_children(block_id):
    results, cursor = [], None
    while True:
        path = f"/blocks/{block_id}/children?page_size=100"
        if cursor:
            path += f"&start_cursor={cursor}"
        d = api("GET", path)
        results.extend(d["results"])
        if not d.get("has_more"):
            return results
        cursor = d["next_cursor"]


def block_text(block):
    content = block.get(block["type"], {})
    if block["type"] in ("child_page", "child_database"):
        return content.get("title", "")
    return "".join(t.get("plain_text", "") for t in content.get("rich_text", []))


def parse_week_no(title):
    for pat in _WEEK_PATTERNS:
        m = pat.match(title)
        if m:
            return int(m.group(1))
    m = _KO_WEEK_PATTERN.match(title)
    if m:
        return _KO_ORDINALS[m.group(1)]
    return None


def find_child_page(parent_id, title):
    for b in list_children(parent_id):
        if b["type"] == "child_page" and block_text(b).strip() == title:
            return b["id"]
    return None


def find_week_headings(month_page_id):
    """월 페이지 안의 주차 heading 목록 [(week_no, block_id, title)]."""
    weeks = []
    for b in list_children(month_page_id):
        if b["type"].startswith("heading"):
            n = parse_week_no(block_text(b))
            if n is not None:
                weeks.append((n, b["id"], block_text(b)))
    return weeks


# ---------------------------------------------------------------- resolve

def cmd_resolve(args):
    year_id = find_child_page(WEEKLY_REPORT_PAGE_ID, str(args.year))
    if year_id is None:
        sys.exit(f"연도 페이지 '{args.year}' 없음 (생성은 Notion 에서 수동으로)")
    month_title = f"{args.month}월"
    month_id = find_child_page(year_id, month_title)
    if month_id is None:
        if not args.create:
            sys.exit(f"월 페이지 '{month_title}' 없음 (--create 로 생성 가능)")
        d = api("POST", "/pages", {
            "parent": {"page_id": year_id},
            "properties": {"title": {"title": [{"text": {"content": month_title}}]}},
        })
        month_id = d["id"]
    out = {"year_page_id": year_id, "month_page_id": month_id}
    if args.week is not None:
        weeks = find_week_headings(month_id)
        hit = [(n, bid, t) for n, bid, t in weeks if n == args.week]
        if hit:
            out["week_block_id"], out["week_title"] = hit[0][1], hit[0][2]
        elif args.create:
            title = f"{args.week}째주"
            d = api("PATCH", f"/blocks/{month_id}/children", {
                "children": [{
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": title}}],
                        "is_toggleable": True,
                    },
                }],
            })
            out["week_block_id"], out["week_title"] = d["results"][0]["id"], title
        else:
            sys.exit(f"{args.month}월에 {args.week}째주 heading 없음 (--create 로 생성 가능)")
    print(json.dumps(out, ensure_ascii=False, indent=2))


# ---------------------------------------------------------- latest-report

def cmd_latest_report(args):
    """연도→월→주차 를 역순으로 훑어 내용이 있는 가장 최근 주차 heading 을 찾는다."""
    years = sorted(
        ((int(block_text(b)), b["id"]) for b in list_children(WEEKLY_REPORT_PAGE_ID)
         if b["type"] == "child_page" and block_text(b).strip().isdigit()),
        reverse=True,
    )
    for year, year_id in years:
        months = sorted(
            ((int(m.group(1)), b["id"], block_text(b)) for b in list_children(year_id)
             if b["type"] == "child_page"
             and (m := re.match(r"^\s*(\d+)\s*월\s*$", block_text(b)))),
            reverse=True,
        )
        for month, month_id, _ in months:
            for week_no, block_id, title in sorted(find_week_headings(month_id), reverse=True):
                if list_children(block_id):
                    print(json.dumps({
                        "year": year, "month": month, "week": week_no,
                        "week_title": title, "week_block_id": block_id,
                    }, ensure_ascii=False, indent=2))
                    return
    sys.exit("내용이 있는 주차 heading 을 찾지 못함")


# ------------------------------------------------------------- read-week

def _dump_blocks(block_id, depth=0):
    lines = []
    for b in list_children(block_id):
        t, txt = b["type"], block_text(b)
        indent = "  " * depth
        if t == "heading_1":
            lines.append(f"{indent}# {txt}")
        elif t == "heading_2":
            lines.append(f"{indent}## {txt}")
        elif t == "heading_3":
            lines.append(f"{indent}### {txt}")
        elif t == "bulleted_list_item":
            lines.append(f"{indent}- {txt}")
        elif t == "numbered_list_item":
            lines.append(f"{indent}1. {txt}")
        elif t == "toggle":
            lines.append(f"{indent}> toggle: {txt}")
        elif t == "table":
            lines.append(f"{indent}[표]")
        elif t in ("image", "video", "file"):
            lines.append(f"{indent}[{t}]")
        elif t == "divider":
            lines.append(f"{indent}---")
        elif txt.strip():
            lines.append(f"{indent}{txt}")
        if b.get("has_children") and t != "table":
            lines.extend(_dump_blocks(b["id"], depth + 1))
    return lines


def cmd_read_week(args):
    print("\n".join(_dump_blocks(args.block_id)))


# ---------------------------------------------------------------- upload

def _rt(text):
    return [{"type": "text", "text": {"content": text}}]


def _placeholder(text):
    return {"type": "paragraph", "paragraph": {
        "rich_text": [{"type": "text", "text": {"content": text},
                       "annotations": {"color": "gray", "italic": True}}],
    }}


_IMG_LINE = re.compile(r"^\[(이미지|비디오|표|영상)\s*:\s*(.+)\]$")
_IMG_PATH = re.compile(r"([\w./\-]+\.(?:png|jpg|jpeg|gif|webp))", re.IGNORECASE)


def upload_file_to_notion(path):
    """Notion File Upload API (single_part) 로 로컬 파일 업로드 → file_upload id 반환."""
    import mimetypes
    import uuid

    name = Path(path).name
    info = api("POST", "/file_uploads", {"filename": name})
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{API_BASE}/file_uploads/{info['id']}/send",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {load_token()}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"파일 업로드 실패 {name} ({e.code}): {e.read().decode()[:300]}")
    return info["id"]


def _media_block(desc):
    """'[이미지: 설명 — 경로]' 라인 → 경로가 실재하면 업로드 후 image 블록, 아니면 placeholder."""
    m = _IMG_PATH.search(desc)
    if m:
        p = Path(m.group(1))
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            fid = upload_file_to_notion(p)
            caption = desc[: m.start()].strip().rstrip("—-– ").strip()
            return {"type": "image", "image": {
                "type": "file_upload", "file_upload": {"id": fid},
                "caption": _rt(caption) if caption else [],
            }}
    return _placeholder(f"[{desc}] ← 수동 삽입")


def _table_block(rows):
    width = max(len(r) for r in rows)
    return {"type": "table", "table": {
        "table_width": width,
        "has_column_header": True,
        "has_row_header": False,
        "children": [
            {"type": "table_row", "table_row": {
                "cells": [_rt(c) for c in r] + [_rt("")] * (width - len(r)),
            }} for r in rows
        ],
    }}


def md_to_blocks(md_text):
    """마크다운 draft → Notion 블록 리스트.

    지원: #/##/### heading, -/1. 리스트, 표, '> toggle: 제목'(직후 들여쓰기 줄이 toggle 내용),
    [이미지: ...]/[비디오: ...] placeholder, 일반 단락. 그 외 문법(굵게 등)은 plain text 로 들어감.
    """
    blocks = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("### "):
            blocks.append({"type": "heading_3", "heading_3": {"rich_text": _rt(s[4:]), "color": "blue_background"}})
        elif s.startswith("## "):
            blocks.append({"type": "heading_2", "heading_2": {"rich_text": _rt(s[3:]), "color": "green_background"}})
        elif s.startswith("# "):
            blocks.append({"type": "heading_1", "heading_1": {"rich_text": _rt(s[2:]), "color": "red_background"}})
        elif s == "---":
            blocks.append({"type": "divider", "divider": {}})
        elif s.startswith("> toggle:"):
            title = s[len("> toggle:"):].strip()
            children = []
            while i + 1 < len(lines) and (lines[i + 1].startswith(("  ", "\t")) or not lines[i + 1].strip()):
                nxt = lines[i + 1].strip()
                i += 1
                if not nxt:
                    continue
                im = _IMG_LINE.match(nxt)
                if im:
                    children.append(_media_block(im.group(2)))
                else:
                    children.append({"type": "paragraph", "paragraph": {"rich_text": _rt(nxt)}})
            blocks.append({"type": "toggle", "toggle": {"rich_text": _rt(title), "children": children}})
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):  # 구분선 행 스킵
                    rows.append(cells)
                i += 1
            i -= 1
            if rows:
                blocks.append(_table_block(rows))
        elif (im := _IMG_LINE.match(s)):
            blocks.append(_media_block(im.group(2)))
        elif re.match(r"^\d+\.\s", s):
            blocks.append({"type": "numbered_list_item",
                           "numbered_list_item": {"rich_text": _rt(re.sub(r"^\d+\.\s+", "", s))}})
        elif s.startswith("- "):
            blocks.append({"type": "bulleted_list_item",
                           "bulleted_list_item": {"rich_text": _rt(s[2:])}})
        else:
            blocks.append({"type": "paragraph", "paragraph": {"rich_text": _rt(s)}})
        i += 1
    return blocks


def cmd_upload(args):
    md = Path(args.draft).read_text()
    blocks = md_to_blocks(md)
    appended = 0
    for start in range(0, len(blocks), 100):  # API 한도: 100 블록/요청
        api("PATCH", f"/blocks/{args.block_id}/children",
            {"children": blocks[start:start + 100]})
        appended += len(blocks[start:start + 100])
    print(json.dumps({"appended_blocks": appended, "week_block_id": args.block_id},
                     ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="연/월/주차 heading id 탐색·생성")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--month", type=int, required=True)
    r.add_argument("--week", type=int)
    r.add_argument("--create", action="store_true")
    r.set_defaults(func=cmd_resolve)

    l = sub.add_parser("latest-report", help="가장 최근 내용이 있는 주차 heading 탐색")
    l.set_defaults(func=cmd_latest_report)

    rd = sub.add_parser("read-week", help="주차 heading 아래 블록을 markdown 으로 덤프")
    rd.add_argument("--block-id", required=True)
    rd.set_defaults(func=cmd_read_week)

    u = sub.add_parser("upload", help="markdown draft 를 주차 heading 에 append")
    u.add_argument("--block-id", required=True)
    u.add_argument("--draft", required=True)
    u.set_defaults(func=cmd_upload)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
