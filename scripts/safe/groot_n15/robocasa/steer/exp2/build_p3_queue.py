"""P3 held-out 본실험 큐 빌더 (ep60-119, n=60/arm).

cell당 arm (docs/steering/17 + D2 원안 + Gate1):
  base(신규 3 cell만 수집 — 재사용 5 cell은 구 ho_base 참조) / null(위약: per_scene_fit15_null1,
  선택 config 기계식) / pos(positive-only: C_success, per_scene_fit15) /
  {perm,gated} × {per_scene,cross_scene,grand} × {fit15,fit30} (신규 apple 3 cell은
  per_scene_fit30 구조적 N/A → 10). (ℓ,β)는 granularity별 P2 선택값.

큐 행 = exp2_cell_runner 가 그대로 소비할 env 조각 (NPZ 는 repo 상대경로 — 러너가 자기
머신 경로로 확장). 형식: KEY=VAL 공백연결 한 줄 + '|retry' 접미.

usage: build_p3_queue.py --out <queue.tsv> [--selection-dir exp2_analysis]
  선행조건: selection_report_<scene>.json (per_scene 8) + selection_report_pool_<scope>.json
  (cross_scene_bread/apple, grand) — 없으면 해당 arm 생성 보류하고 경고.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
G15 = REPO / "outputs/eval/robocasa/groot_n15"
CONC = "outputs/eval/robocasa/groot_n15/exp2_analysis/conceptors"

BREADS = ["ppcc_bread", "ppcc_bread_s300028", "ppcc_bread_s300033", "ppcc_bread_s400020"]
APPLES = ["ppcs_apple", "ppcs_apple_s100353", "ppcs_apple_s100395", "ppcs_apple_s100422"]
NEW_APPLES = {"ppcs_apple_s100353", "ppcs_apple_s100395", "ppcs_apple_s100422"}  # fit30 N/A·base 수집
FIT15_ONLY = NEW_APPLES


def sel_of(report: Path):
    """selection_report → (layers, beta). 예: p2_multi_4_8_12_b01 → ("4,8,12", 0.1)."""
    r = json.loads(report.read_text())
    pick = r.get("selected")
    if not pick:
        return None
    m = re.match(r"p2_(single_L(\d+)|multi_([\d_]+))_b(\d+)", pick["tag"])
    layers = m.group(2) if m.group(2) else m.group(3).replace("_", ",")
    beta = float("0." + m.group(4).lstrip("0"))
    return layers, beta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(G15 / "work_queue_p3/queue.tsv"))
    ap.add_argument("--selection-dir", default=str(G15 / "exp2_analysis"))
    args = ap.parse_args()
    sdir = Path(args.selection_dir)
    sel_ps, warn = {}, []
    for c in BREADS + APPLES:
        p = sdir / f"selection_report_{c}.json"
        sel_ps[c] = sel_of(p) if p.exists() else None
        if not sel_ps[c]:
            warn.append(f"per_scene 선택 없음: {c}")
    sel_pool = {}
    for scope in ("cross_scene_bread", "cross_scene_apple", "grand"):
        p = sdir / f"selection_report_pool_{scope}.json"
        sel_pool[scope] = sel_of(p) if p.exists() else None
        if not sel_pool[scope]:
            warn.append(f"pool 선택 없음: {scope}")

    rows = []

    def emit(cell, tag, mode, npz_rel, layers, beta):
        rows.append(f"CELL={cell} TAG={tag} MODE={mode} NPZ={npz_rel} LAYERS={layers} BETA={beta}|0")

    for cell in BREADS + APPLES:
        instr_scope = "cross_scene_bread" if cell in BREADS else "cross_scene_apple"
        ps = sel_ps.get(cell)
        if cell in NEW_APPLES:
            emit(cell, "ho_base", "base", "-", "-", "-")
        if ps:
            L, B = ps
            emit(cell, "ho_null_per_scene_fit15", "null",
                 f"{CONC}/{cell}/per_scene_fit15_null1/global", L, B)
            emit(cell, "ho_pos_per_scene_fit15", "pos",
                 f"{CONC}/{cell}/per_scene_fit15/global", L, B)
            for fitn in (15,) if cell in FIT15_ONLY else (15, 30):
                emit(cell, f"ho_perm_per_scene_fit{fitn}", "perm",
                     f"{CONC}/{cell}/per_scene_fit{fitn}/global", L, B)
                emit(cell, f"ho_gated_per_scene_fit{fitn}", "gated",
                     f"{CONC}/{cell}/per_scene_fit{fitn}", L, B)
        xp = sel_pool.get(instr_scope)
        if xp:
            L, B = xp
            pool = instr_scope.replace("cross_scene_", "cross_scene_")
            stem = f"_pools/{instr_scope.replace('cross_scene_', 'cross_scene_')}"
            base = f"{CONC}/_pools/{instr_scope}"
            for fitn in (15, 30):
                emit(cell, f"ho_perm_cross_scene_fit{fitn}", "perm", f"{base}_fit{fitn}/global", L, B)
                emit(cell, f"ho_gated_cross_scene_fit{fitn}", "gated", f"{base}_fit{fitn}", L, B)
        gx = sel_pool.get("grand")
        if gx:
            L, B = gx
            base = f"{CONC}/_pools/grand"
            for fitn in (15, 30):
                emit(cell, f"ho_perm_grand_fit{fitn}", "perm", f"{base}_fit{fitn}/global", L, B)
                emit(cell, f"ho_gated_grand_fit{fitn}", "gated", f"{base}_fit{fitn}", L, B)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "running").mkdir(exist_ok=True)
    (out.parent / "done").mkdir(exist_ok=True)
    (out.parent / "failed").mkdir(exist_ok=True)
    out.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"P3 큐 {len(rows)} arm → {out}")
    per_cell = {}
    for r in rows:
        c = r.split()[0].split("=")[1]
        per_cell[c] = per_cell.get(c, 0) + 1
    for c, n in per_cell.items():
        print(f"  {c}: {n} arm")
    for w in warn:
        print(f"  [경고] {w}")


if __name__ == "__main__":
    main()
