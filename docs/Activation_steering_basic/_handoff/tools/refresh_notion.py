#!/usr/bin/env python3
"""Clean-rebuild the Notion mirror as a SINGLE rich page from the survey markdown.

Archives all existing children under the parent page, then pushes the whole survey
as one page: `#`→h1(red), `##`→h2(green), `###`→h3(blue), `<details>`→toggle,
`==..==`→highlight, `**..**`→bold, tables/callouts as-is. Relative md links are
flattened to plain text (Notion has no repo file tree)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from notion_push import _token, req, append_blocks, md_to_blocks  # noqa

PARENT = "39063918d42a801ab093f046f22701b4"
SURVEY = "/home/dongkyu/pkt_ws/temporal_vla/docs/Activation_steering_basic/00_activation_steering_survey.md"


def strip_relative_links(text):
    def repl(m):
        label, url = m.group(1), m.group(2)
        return label if not url.startswith("http") else m.group(0)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, text)


def list_children(pid, tok):
    out, cursor = [], None
    while True:
        q = f"/blocks/{pid}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        r = req("GET", q, tok)
        out += r["results"]
        if not r.get("has_more"):
            break
        cursor = r["next_cursor"]
    return out


def main():
    tok = _token()
    # 1) archive all existing children (old child pages + intro blocks)
    kids = list_children(PARENT, tok)
    for b in kids:
        req("DELETE", f"/blocks/{b['id']}", tok)
    print(f"archived {len(kids)} existing blocks/pages")

    # 2) rebuild as one page
    raw = strip_relative_links(open(SURVEY).read())
    blocks = md_to_blocks(raw)
    append_blocks(PARENT, blocks, tok)
    kinds = {}
    for b in blocks:
        kinds[b["type"]] = kinds.get(b["type"], 0) + 1
    print(f"pushed {len(blocks)} top-level blocks: {kinds}")


if __name__ == "__main__":
    main()
