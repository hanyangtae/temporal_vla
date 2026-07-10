#!/usr/bin/env python3
"""최종 scene-seed 매트릭스 집계 + Notion append. BREAD_ONLY=1 이면 bread 4 cell만(중간 보고)."""
import glob
import json
import os
import urllib.request

E = "outputs/eval/robocasa/groot_n15/steer_eval"
CELLS = {
    "ppcc_bread": (5, "PickPlaceCounterToCabinet"), "ppcc_bread_s300028": (5, "PickPlaceCounterToCabinet"),
    "ppcc_bread_s300033": (5, "PickPlaceCounterToCabinet"), "ppcc_bread_s400020": (5, "PickPlaceCounterToCabinet"),
    "ppcs_apple": (1, "PickPlaceCounterToStove"), "ppcs_apple_s100050": (1, "PickPlaceCounterToStove"),
    "ppcs_apple_s100084": (1, "PickPlaceCounterToStove"), "ppcs_apple_s100104": (1, "PickPlaceCounterToStove"),
}
bread_only = os.environ.get("BREAD_ONLY") == "1"
cells = [c for c in CELLS if (not bread_only) or c.startswith("ppcc")]


def _episodes(cell, tag):
    """(ep→succ) 맵. 로컬 pkl 우선, worker2 직송 arm은 manifest_w2.txt 폴백(파일명만 로컬 보관)."""
    ci, T = CELLS[cell]
    d = f"{E}/{cell}/{tag}/raw_rollouts/{T}/{cell}"
    eps = {}
    # pkl 원본 또는 아카이브 후 경량 마커(.csv 등) — 파일명이 판정 출처라 확장자 무관
    for ep in range(60, 120):
        if glob.glob(f"{d}/task{ci}--ep{ep}--succ1.*"):
            eps[ep] = 1
        elif glob.glob(f"{d}/task{ci}--ep{ep}--succ0.*"):
            eps[ep] = 0
    mf = f"{E}/{cell}/{tag}/manifest_w2.txt"
    if os.path.exists(mf):
        import re
        for line in open(mf):
            m = re.search(rf"task{ci}--ep(\d+)--succ([01])\.pkl", line)
            if m and int(m.group(1)) not in eps:
                eps[int(m.group(1))] = int(m.group(2))
    return eps


def machine_of(cell, tag):
    """MACHINE.txt 기반 산출 머신 표기: L=로컬, W=worker2(A100), LW=혼합, ?=기록없음."""
    p = f"{E}/{cell}/{tag}/MACHINE.txt"
    if not os.path.exists(p):
        return "L?"  # 큐 이전(마스터 시대) 산출 = 로컬
    txt = open(p).read()
    has_l = "local" in txt
    has_w = "worker2" in txt
    return "LW" if (has_l and has_w) else ("W" if has_w else "L")


def sr(cell, tag):
    eps = _episodes(cell, tag)
    s = sum(eps.values())
    n = len(eps)
    return f"{s}/{n}" if n else "-"


rows = []
for cell in cells:
    xn = "xb" if cell.startswith("ppcc") else "xa"
    r = {"cell": cell, "base": sr(cell, "ho_base")}
    for N in [15, 30, 60]:
        r[f"perm{N}"] = sr(cell, f"ho_permps{N}")
        r[f"gated{N}"] = sr(cell, f"ho_gatedps{N}")
        r[f"xp{N}"] = sr(cell, f"ho_perm{xn}{N}")
        r[f"xg{N}"] = sr(cell, f"ho_gated{xn}{N}")
        r[f"gp{N}"] = sr(cell, f"ho_permgx{N}")
        r[f"gg{N}"] = sr(cell, f"ho_gatedgx{N}")
    r["L4"] = sr(cell, "ho_gatedL4")
    r["L812"] = sr(cell, "ho_gatedL812")
    r["xL4"] = sr(cell, f"ho_gated{xn}L4")
    r["xL812"] = sr(cell, f"ho_gated{xn}L812")
    r["gxL4"] = sr(cell, "ho_gatedgxL4")
    r["gxL812"] = sr(cell, "ho_gatedgxL812")
    # 산출 머신 요약 (arm별 상세는 각 arm 디렉토리 MACHINE.txt / work_queue/ledger.tsv)
    machines = {machine_of(cell, t) for t in [f"ho_perm{k}{N}" for k in ("ps", xn, "gx") for N in (15, 30, 60)]
                + [f"ho_gated{k}{N}" for k in ("ps", xn, "gx") for N in (15, 30, 60)] + ["ho_base"]}
    r["machine"] = "+".join(sorted(m for m in machines if m != "L?")) or "L"
    rows.append(r)

out = f"{E}/RESULTS_final_scene{'_bread' if bread_only else ''}.json"
json.dump(rows, open(out, "w"), indent=1)
print("saved:", out)
for r in rows:
    print(r)

TOK = os.environ.get("NOTION_TOKEN")
if TOK:
    def rt(t, b=False):
        return [{"type": "text", "text": {"content": t}, "annotations": {"bold": b}}]

    def row(c, h=False):
        return {"type": "table_row", "table_row": {"cells": [rt(x, h) for x in c]}}

    def j(r, pre):
        return "/".join(str(r.get(f"{pre}{N}", "-")) for N in [15, 30, 60])

    hdr = ["cell", "base", "per_scene-perm(15/30/60)", "per_scene-gated", "cross_scene-perm", "cross_scene-gated",
           "grand-perm", "grand-gated", "L4/L812(per_scene)", "L4/L812(cross_scene)", "L4/L812(grand)", "machine"]
    body = [row([r["cell"], r["base"], j(r, "perm"), j(r, "gated"), j(r, "xp"), j(r, "xg"),
                 j(r, "gp"), j(r, "gg"), f"{r.get('L4', '-')} · {r.get('L812', '-')}",
                 f"{r.get('xL4', '-')} · {r.get('xL812', '-')}",
                 f"{r.get('gxL4', '-')} · {r.get('gxL812', '-')}", r.get("machine", "L")]) for r in rows]
    title = "★ scene-seed 매트릭스 " + ("(bread 중간)" if bread_only else "(최종 8cell)") + " — eval ep60-119"
    ch = [{"type": "heading_3", "heading_3": {"rich_text": rt(title)}},
          {"type": "table", "table": {"table_width": 12, "has_column_header": True,
                                      "children": [row(hdr, True)] + body}}]
    req = urllib.request.Request(
        "https://api.notion.com/v1/blocks/38e63918d42a80698ac2f193716c03a3/children",
        data=json.dumps({"children": ch}).encode(),
        headers={"Authorization": f"Bearer {TOK}", "Notion-Version": "2022-06-28",
                 "Content-Type": "application/json"}, method="PATCH")
    urllib.request.urlopen(req)
    print("notion appended")
