#!/usr/bin/env python3
"""seq-steer Notion 페이지(38e63918d42a80698ac2f193716c03a3) 업데이트 헬퍼.

사람이 보는 결과 페이지 — 간략한 결과만. NOTION_TOKEN 은 .env (repo root).
사용:
  python3 scripts/utils/notion_seq_steer.py append "한 줄 결과"      # bullet 추가
  python3 scripts/utils/notion_seq_steer.py fill                     # 초기 채움(1회)
"""
import json
import os
import sys
import urllib.request

PAGE = "38e63918d42a80698ac2f193716c03a3"
API = "https://api.notion.com/v1"


def _tok():
    if os.environ.get("NOTION_TOKEN"):
        return os.environ["NOTION_TOKEN"]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for line in open(os.path.join(root, ".env")):
        if line.startswith("NOTION_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("NOTION_TOKEN not found")


def req(method, path, body=None):
    r = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {_tok()}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
        method=method,
    )
    return json.load(urllib.request.urlopen(r))


def rt(text, bold=False):
    return [{"type": "text", "text": {"content": text}, "annotations": {"bold": bold}}]


def append_bullet(text):
    req("PATCH", f"/blocks/{PAGE}/children", {
        "children": [{"type": "bulleted_list_item",
                      "bulleted_list_item": {"rich_text": rt(text)}}]})
    print("appended:", text[:60])


def fill():
    blocks = req("GET", f"/blocks/{PAGE}/children?page_size=100")["results"]
    para_setup = para_base = table = None
    for b in blocks:
        t = b["type"]
        txt = "".join(x.get("plain_text", "") for x in b[t].get("rich_text", [])) if "rich_text" in b[t] else ""
        if "instruction" in txt:
            para_setup = b["id"]
        elif txt.startswith("baseline"):
            para_base = b["id"]
        elif t == "table":
            table = b["id"]
    # 1) setup 문단
    req("PATCH", f"/blocks/{para_setup}", {"paragraph": {"rich_text": rt(
        "하나의 PnP instruction(\"Pick the bread from the counter and place it in the cabinet.\"), "
        "하나의 환경 seed(scene 고정 100084; diffusion noise seed만 rollout마다 다름)에서 "
        "15개 rollout(11성공/4실패)으로 contrastive conceptor(C_succ∧¬C_fail)를 fit, "
        "β=0.3 곱셈 게이트로 전 스텝 주입. 평가는 같은 scene에서 30 rollout(SR)로.")}})
    # 2) baseline
    req("PATCH", f"/blocks/{para_base}", {"paragraph": {"rich_text": rt(
        "baseline 성공률: 22/30 = 0.733 (ep0-29; 참고: 새 seed ep30-89에선 49/60 = 0.817)")}})
    # 3) 테이블 (ΔSR)
    rows = req("GET", f"/blocks/{table}/children?page_size=10")["results"]
    fills = [
        None,  # header 유지
        ["task 단위(global)", "−0.10 (L4)", "+0.20 (L4+8+12) ※"],
        ["phase 구분(transport)", "−0.067 (L4)", "+0.167 (L4+8+12) ※"],
    ]
    for row, cells in zip(rows, fills):
        if cells is None:
            continue
        req("PATCH", f"/blocks/{row['id']}", {"table_row": {"cells": [rt(c) for c in cells]}})
    # 4) 결과 요약 블록 append
    children = []
    def para(text, bold=False):
        children.append({"type": "paragraph", "paragraph": {"rich_text": rt(text, bold)}})
    def bullet(text):
        children.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rt(text)}})
    para("※ 위 표는 fit에 쓴 diffusion seed가 평가에 섞인(in-sample) 수치 — 아래 반전 참고.", bold=True)
    para("★ 반전 (held-out 확증, n=90):", bold=True)
    bullet("multi-layer +0.20은 in-sample 암기 효과였음. fit이 본 실패 4판은 전부 구제했지만, "
           "새 diffusion seed 60판(ep30-89)에선 −0.067로 효과 소멸.")
    bullet("새 seed에서의 flip(구제 8/파괴 12)은 '무작위 re-roll'과 통계적으로 구분 불가 — "
           "즉 못 본 seed에선 성공 방향이 아니라 큰 섭동으로 작동. 개입 채널 자체는 강력(궤적을 확실히 바꿈).")
    bullet("레이어 7개 전부 동시 주입 = SR 0.000 (정책 완전 붕괴). {4,8} 2개 = 효과 0. → 과다/과소 주입 모두 무효, "
           "조합 ablation 진행 중.")
    para("부수 발견 (drop-aware 라벨러):", bold=True)
    bullet("bread 실패의 실체는 'transport 정체'가 아니라 잡았다 떨어뜨린 뒤 재파지 실패(실패 12판 중 11판). "
           "기존 transport 정체 프레임은 monotone 라벨러의 아티팩트.")
    para("진행 중:", bold=True)
    bullet("layer 조합 ablation / drop-aware 엄밀 phase(5-phase: pre-grasp·pre-place 포함) conceptor steer 큐 — "
           "승자는 held-out(ep60-89)에서 재판정 예정.")
    req("PATCH", f"/blocks/{PAGE}/children", {"children": children})
    print("filled.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fill"
    if cmd == "append":
        append_bullet(sys.argv[2])
    else:
        fill()
