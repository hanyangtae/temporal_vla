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


def sr(cell, tag):
    ci, T = CELLS[cell]
    d = f"{E}/{cell}/{tag}/raw_rollouts/{T}/{cell}"
    s = f = 0
    for ep in range(60, 120):
        if glob.glob(f"{d}/task{ci}--ep{ep}--succ1.pkl"):
            s += 1
        elif glob.glob(f"{d}/task{ci}--ep{ep}--succ0.pkl"):
            f += 1
    return f"{s}/{s+f}" if s + f else "-"


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
    if cell.startswith("ppcc"):
        r["L4"] = sr(cell, "ho_gatedL4")
        r["L812"] = sr(cell, "ho_gatedL812")
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

    hdr = ["cell", "base", "perm(15/30/60)", "gated", "xseed-perm", "xseed-gated", "grand-perm", "grand-gated", "L4", "L812"]
    body = [row([r["cell"], r["base"], j(r, "perm"), j(r, "gated"), j(r, "xp"), j(r, "xg"),
                 j(r, "gp"), j(r, "gg"), r.get("L4", "-"), r.get("L812", "-")]) for r in rows]
    title = "★ scene-seed 매트릭스 " + ("(bread 중간)" if bread_only else "(최종 8cell)") + " — eval ep60-119"
    ch = [{"type": "heading_3", "heading_3": {"rich_text": rt(title)}},
          {"type": "table", "table": {"table_width": 10, "has_column_header": True,
                                      "children": [row(hdr, True)] + body}}]
    req = urllib.request.Request(
        "https://api.notion.com/v1/blocks/38e63918d42a80698ac2f193716c03a3/children",
        data=json.dumps({"children": ch}).encode(),
        headers={"Authorization": f"Bearer {TOK}", "Notion-Version": "2022-06-28",
                 "Content-Type": "application/json"}, method="PATCH")
    urllib.request.urlopen(req)
    print("notion appended")
