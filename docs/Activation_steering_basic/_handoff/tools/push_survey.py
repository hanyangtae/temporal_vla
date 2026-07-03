#!/usr/bin/env python3
"""Split the survey markdown into per-section Notion child pages under the target page."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from notion_push import _token, create_page, append_blocks, md_to_blocks  # noqa

PARENT = "39063918d42a801ab093f046f22701b4"
SURVEY = "/home/dongkyu/pkt_ws/temporal_vla/docs/Activation_steering_basic/00_activation_steering_survey.md"


def strip_relative_links(text):
    # [label](url): keep http(s), otherwise drop to plain label
    def repl(m):
        label, url = m.group(1), m.group(2)
        return label if not url.startswith("http") else m.group(0)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, text)


def main():
    tok = _token()
    with open(SURVEY) as f:
        raw = f.read()
    raw = strip_relative_links(raw)
    lines = raw.split("\n")

    # split: leading block (title + blockquote intro) until first '## '
    sec_idx = [i for i, l in enumerate(lines) if l.startswith("## ")]
    intro = lines[:sec_idx[0]]
    sections = []
    for k, start in enumerate(sec_idx):
        end = sec_idx[k + 1] if k + 1 < len(sec_idx) else len(lines)
        title = lines[start][3:].strip()
        body = lines[start + 1:end]
        # strip leading/trailing blank + divider lines
        while body and (not body[0].strip() or body[0].strip() == "---"):
            body.pop(0)
        while body and (not body[-1].strip() or body[-1].strip() == "---"):
            body.pop()
        sections.append((title, "\n".join(body)))

    # parent intro: blockquote lines -> callout/paragraphs, + orientation
    intro_md = "\n".join(l for l in intro if not l.startswith("# "))
    intro_blocks = md_to_blocks(intro_md)
    intro_blocks += md_to_blocks(
        f"아래 하위 페이지 {len(sections)}개를 위→아래 순서(§1→§7 + 부록)로 읽으세요. "
        "각 논문 정독 노트(41편)는 repo `docs/Activation_steering_basic/notes/` 에 있습니다.")
    append_blocks(PARENT, intro_blocks, tok)
    print(f"parent intro pushed ({len(intro_blocks)} blocks)")

    for title, body in sections:
        pid = create_page(PARENT, title, tok)
        blocks = md_to_blocks(body)
        append_blocks(pid, blocks, tok)
        print(f"  child: {title[:40]:40s} -> {len(blocks)} blocks  {pid}")


if __name__ == "__main__":
    main()
