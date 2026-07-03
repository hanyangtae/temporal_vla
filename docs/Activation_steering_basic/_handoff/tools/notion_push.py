#!/usr/bin/env python3
"""Notion push helper (stdlib only).

Converts a restricted Markdown subset -> Notion blocks and pushes to a page.
Reads NOTION_TOKEN from environment (or --token).

Supported markdown:
  # / ## / ###        headings (colored bg: h1 red / h2 green / h3 blue)
  paragraph text      paragraph
  - / *               bulleted list item
  1.                   numbered list item
  > text              callout (or blockquote)
  ```lang ... ```     code block
  ---                 divider
  | a | b |           table (first row = header)
  <details><summary>제목</summary> ... </details>   toggle (children = 내부 마크다운)
  inline: **bold**  *italic*  ==highlight==  `code`  [text](url)
        (_underscore_ 이탤릭은 미지원: C_success 같은 snake_case 수식·파일명 보호)

Soft-wrapped source markdown is normalized first: a line that does not start a
new block (heading/bullet/number/callout/table/fence/divider) is merged into
the previous line, so multi-line paragraphs/bullets become one Notion block.

CLI:
  python notion_push.py smoke                    # create+append+archive throwaway page
  python notion_push.py create <parent_id> "Title"        -> prints new page id
  python notion_push.py push <page_id> <markdown_file>    # append md as blocks
  python notion_push.py page <parent_id> "Title" <md_file># create child page + fill
  python notion_push.py archive <page_id>
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
HEADING_COLOR = {1: "red_background", 2: "green_background", 3: "blue_background"}
MAX_RT = 1900          # chars per rich_text object (< 2000 hard limit)
MAX_CHILDREN = 90      # blocks per append request (< 100 hard limit)


def _token(tok=None):
    t = tok or os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
    if not t:
        sys.exit("ERROR: NOTION_TOKEN not set")
    return t


def req(method, path, token, payload=None):
    url = path if path.startswith("http") else f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {token}")
    r.add_header("Notion-Version", NOTION_VERSION)
    r.add_header("Content-Type", "application/json")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(r) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429 and attempt < 4:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise SystemExit(f"HTTP {e.code} on {method} {path}: {body}")
        except urllib.error.URLError as e:
            if attempt < 4:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise SystemExit(f"URLError on {method} {path}: {e}")


# ---------- inline markdown -> rich_text ----------
_INLINE = re.compile(
    r"(\*\*(?P<b>.+?)\*\*)"
    r"|(==(?P<h>.+?)==)"
    r"|(?<!\*)\*(?P<i>[^*].*?)\*(?!\*)"
    r"|(`(?P<c>.+?)`)"
    r"|(\[(?P<lt>.+?)\]\((?P<lu>.+?)\))"
)
HIGHLIGHT_COLOR = "yellow_background"


def rich_text(s):
    """Parse a line of inline markdown into Notion rich_text objects."""
    out = []
    pos = 0
    for m in _INLINE.finditer(s):
        if m.start() > pos:
            out.extend(_rt_plain(s[pos:m.start()]))
        if m.group("b") is not None:
            out.append(_rt(m.group("b"), bold=True))
        elif m.group("h") is not None:
            out.append(_rt(m.group("h"), color=HIGHLIGHT_COLOR))
        elif m.group("i") is not None:
            out.append(_rt(m.group("i"), italic=True))
        elif m.group("c") is not None:
            out.append(_rt(m.group("c"), code=True))
        elif m.group("lt") is not None:
            out.append(_rt(m.group("lt"), link=m.group("lu")))
        pos = m.end()
    if pos < len(s):
        out.extend(_rt_plain(s[pos:]))
    return out or [_rt("")]


def _rt_plain(s):
    # chunk long plain text under the per-object limit
    return [_rt(s[i:i + MAX_RT]) for i in range(0, len(s), MAX_RT)] or [_rt("")]


def _rt(content, bold=False, italic=False, code=False, link=None, color="default"):
    txt = {"content": content}
    if link:
        txt["link"] = {"url": link}
    return {
        "type": "text",
        "text": txt,
        "annotations": {"bold": bold, "italic": italic, "code": code,
                        "color": color},
    }


# ---------- block builders ----------
def h_block(level, text):
    key = f"heading_{level}"
    return {"object": "block", "type": key,
            key: {"rich_text": rich_text(text), "color": HEADING_COLOR[level]}}


def p_block(text):
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": rich_text(text)}}


def bullet_block(text):
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": rich_text(text)}}


def numbered_block(text):
    return {"object": "block", "type": "numbered_list_item",
            "numbered_list_item": {"rich_text": rich_text(text)}}


def callout_block(text):
    return {"object": "block", "type": "callout",
            "callout": {"rich_text": rich_text(text), "icon": {"emoji": "💡"}}}


def toggle_block(text, children):
    return {"object": "block", "type": "toggle",
            "toggle": {"rich_text": rich_text(text), "children": children}}


def divider_block():
    return {"object": "block", "type": "divider", "divider": {}}


def code_block(text, lang="plain text"):
    lang = (lang or "plain text").strip().lower() or "plain text"
    aliases = {"py": "python", "sh": "shell", "bash": "shell", "": "plain text"}
    lang = aliases.get(lang, lang)
    return {"object": "block", "type": "code",
            "code": {"rich_text": [_rt(text[:MAX_RT])], "language": lang}}


def table_block(rows):
    width = max(len(r) for r in rows)
    trows = []
    for r in rows:
        cells = [rich_text(c.strip()) for c in r] + [[_rt("")]] * (width - len(r))
        trows.append({"object": "block", "type": "table_row",
                      "table_row": {"cells": cells}})
    return {"object": "block", "type": "table",
            "table": {"table_width": width, "has_column_header": True,
                      "has_row_header": False, "children": trows}}


# ---------- markdown document -> blocks ----------
_BLOCK_START = re.compile(
    r"^\s*(#{1,6}\s|[-*]\s|\d+\.\s|>\s?|\||```|-{3,}\s*$|\*{3,}\s*$|_{3,}\s*$)"
)


def merge_continuations(md):
    """Join soft-wrapped continuation lines into their parent block line."""
    out = []
    in_code = False
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or not s:
            out.append(line)
            continue
        prev = out[-1].strip() if out else ""
        # HTML tag lines (<details>/<summary>/</details>) never merge
        if s.startswith("<") or prev.startswith("<"):
            out.append(line)
            continue
        # consecutive "> quote" lines collapse into one callout
        if s.startswith(">") and prev.startswith(">"):
            out[-1] = out[-1].rstrip() + " " + s.lstrip(">").strip()
            continue
        starts_block = bool(_BLOCK_START.match(line))
        # never merge into: nothing, blank, heading, divider, table row, fence
        prev_open = (prev and not re.match(r"^#{1,6}\s", prev)
                     and not re.fullmatch(r"-{3,}|\*{3,}|_{3,}", prev)
                     and not prev.startswith("|") and not prev.startswith("```"))
        if starts_block or not prev_open:
            out.append(line)
        else:
            out[-1] = out[-1].rstrip() + " " + s
    return "\n".join(out)


def md_to_blocks(md):
    lines = merge_continuations(md).split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        # code fence
        if s.startswith("```"):
            lang = s[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(code_block("\n".join(buf), lang))
            continue
        # toggle: <details><summary>title</summary> ... </details>
        # (summary may be inline with <details> or on its own line)
        if s.startswith("<details"):
            buf, depth = [line], 1
            i += 1
            while i < len(lines) and depth > 0:
                ls = lines[i].strip()
                if ls.startswith("<details"):
                    depth += 1
                if ls.startswith("</details>"):
                    depth -= 1
                buf.append(lines[i])
                i += 1
            full = "\n".join(buf)
            msum = re.search(r"<summary>(.*?)</summary>", full, re.S)
            title = msum.group(1).strip() if msum else "상세"
            body = re.sub(r"<summary>.*?</summary>", "", full, flags=re.S)
            body = re.sub(r"</?details[^>]*>", "", body)
            blocks.append(toggle_block(title, md_to_blocks(body)))
            continue
        # table (consecutive pipe rows)
        if s.startswith("|") and "|" in s[1:]:
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip().strip("|")
                if re.fullmatch(r"[\s:|-]+", row):  # separator row
                    i += 1
                    continue
                tbl.append([c.strip() for c in row.split("|")])
                i += 1
            if tbl:
                blocks.append(table_block(tbl))
            continue
        # divider
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", s):
            blocks.append(divider_block())
            i += 1
            continue
        # headings
        m = re.match(r"(#{1,3})\s+(.*)", s)
        if m:
            lvl = len(m.group(1))
            blocks.append(h_block(lvl, m.group(2).strip()))
            i += 1
            continue
        if s.startswith("#### "):  # h4+ -> bold paragraph
            blocks.append(p_block("**" + s[5:].strip() + "**"))
            i += 1
            continue
        # callout / blockquote
        if s.startswith("> "):
            blocks.append(callout_block(s[2:].strip()))
            i += 1
            continue
        # bullets
        m = re.match(r"[-*]\s+(.*)", s)
        if m:
            blocks.append(bullet_block(m.group(1).strip()))
            i += 1
            continue
        # numbered
        m = re.match(r"\d+\.\s+(.*)", s)
        if m:
            blocks.append(numbered_block(m.group(1).strip()))
            i += 1
            continue
        blocks.append(p_block(s))
        i += 1
    return blocks


# ---------- API ops ----------
def create_page(parent_id, title, token):
    payload = {"parent": {"type": "page_id", "page_id": parent_id},
               "properties": {"title": [{"type": "text", "text": {"content": title}}]}}
    return req("POST", "/pages", token, payload)["id"]


def append_blocks(block_id, blocks, token):
    for k in range(0, len(blocks), MAX_CHILDREN):
        req("PATCH", f"/blocks/{block_id}/children", token,
            {"children": blocks[k:k + MAX_CHILDREN]})


def archive_page(page_id, token):
    req("PATCH", f"/pages/{page_id}", token, {"archived": True})


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    tok = _token()
    cmd = args[0]
    if cmd == "smoke":
        pid = create_page("39063918d42a801ab093f046f22701b4", "[smoke test - delete me]", tok)
        append_blocks(pid, md_to_blocks(
            "## Smoke\nhello **bold** and `code` and [link](https://example.com)\n"
            "- a\n- b\n\n| x | y |\n|---|---|\n| 1 | 2 |\n"), tok)
        print("created", pid)
        archive_page(pid, tok)
        print("archived", pid, "-> smoke OK")
    elif cmd == "create":
        print(create_page(args[1], args[2], tok))
    elif cmd == "push":
        with open(args[2]) as f:
            append_blocks(args[1], md_to_blocks(f.read()), tok)
        print("pushed to", args[1])
    elif cmd == "page":
        pid = create_page(args[1], args[2], tok)
        with open(args[3]) as f:
            append_blocks(pid, md_to_blocks(f.read()), tok)
        print(pid)
    elif cmd == "archive":
        archive_page(args[1], tok)
        print("archived", args[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
