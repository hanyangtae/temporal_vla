#!/usr/bin/env python3
"""grid rollout 인덱스(v6) → scene-local LOKO(leave-one-jitter-out) 평가 셀 선정.

단위는 **instruction**(= `grid_instruction` 열, 예 "PPCC/apple", "OpenDrawer/left")이다.
`slug` 은 경로에 쓰기 위한 파생 이름일 뿐 — grid_instruction 의 '/' 와 공백을 '_' 로
바꾼 것이고, 선정 단위가 아니다.

⚠ 지터 좌표는 반드시 **`jitter_idx`** 열이다. `jitter_reset_idx` 는 oven/washer 계열
키에서 전부 0 이라 그 열로 묶으면 지터 축이 통째로 사라진다(셀 1개로 붕괴). 이 스크립트는
`jitter_idx` 열이 없거나, 어떤 (grid_instruction, scene_idx) 조합의 서로 다른 jitter 값이
2개 미만이면 즉시 실패한다.

규칙 — 각 (grid_instruction, scene_idx) 안에서 그 scene 에 존재하는 jitter j 마다:
  tgt_fail  = (instr, scene, j) 의 success==0 판 수      ← 평가 대상(구제 시도할 실패판)
  tgt_succ  = (instr, scene, j) 의 success==1 판 수
  pool_succ = 같은 (instr, scene) 의 j 이외 jitter 중 success==1 판 수
  pool_fail = 같은 (instr, scene) 의 j 이외 jitter 중 success==0 판 수
  n_pool    = pool_succ + pool_fail + tgt_fail   (LOKO fit 풀 = 나머지 j 전부 + 대상 j 실패)

선정 조건:  pool_fail >= --min-pool-fail (기본 3)  AND  tgt_fail >= --min-tgt-fail (기본 1)
  · pool_fail 하한이 없으면 연산자가 "대상 j = 실패 / 나머지 j = 성공", 즉 지터 좌표
    자체를 학습해 버린다.
  · pool_succ < 9 인 셀은 검출기 보정 표본이 얇아 매니페스트에 flag_succ_lt9=1 로 표시된다
    (제외는 하지 않는다).

산출물:
  --out         선정된 셀의 **실패 에피소드 1판 = 1행** 매니페스트 tsv
  --cells-out   (선택) 평가한 모든 (instr, scene, j) 셀의 집계 tsv (선정/미선정 + 사유)
  stdout        (instr, scene) 별 SR·선정 j·실패판 수, machine 별 합계, 총계

산출물에는 절대경로를 쓰지 않는다 (인덱스의 rel_path 를 그대로 옮긴다).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

JITTER_COL = "jitter_idx"
SUCC_TRUE = ("1", "true", "yes", "t")
SUCC_FALSE = ("0", "false", "no", "f")

MANIFEST_COLS = [
    "grid_instruction", "slug", "machine", "scene_idx", "jitter_idx", "noise_idx",
    "jitter_reset_idx", "base_lat", "base_back", "env_seed", "inference_seed",
    "sig", "rel_path", "pool_succ", "pool_fail", "tgt_fail", "tgt_succ", "flag_succ_lt9",
]
CELL_COLS = [
    "grid_instruction", "slug", "machine", "scene_idx", "jitter_idx", "n_total",
    "tgt_succ", "tgt_fail", "pool_succ", "pool_fail", "n_pool", "selected", "reason",
]


def slug_of(instr: str) -> str:
    """grid_instruction → 경로 안전 이름 ('/' 와 공백을 '_' 로)."""
    return instr.replace("/", "_").replace(" ", "_")


def as_int(raw: str | None, what: str) -> int:
    v = (raw or "").strip()
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"[select_loko_cells] {what} 값을 정수로 읽을 수 없습니다: {v!r}")


def as_success(raw: str | None) -> int:
    v = (raw or "").strip().lower()
    if v in SUCC_TRUE:
        return 1
    if v in SUCC_FALSE:
        return 0
    raise SystemExit(f"[select_loko_cells] success 열 값이 0/1 이 아닙니다: {raw!r}")


def parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def parse_scenes(raw: str | None) -> tuple[set[int], set[tuple[str, int]]]:
    """--scenes 문자열 → (전역 scene 집합, (instr, scene) 쌍 집합).

    "instr:scene" 형태는 그 instruction 에만, 맨 정수는 모든 instruction 에 적용.
    """
    bare: set[int] = set()
    pairs: set[tuple[str, int]] = set()
    for tok in parse_list(raw):
        if ":" in tok:
            instr, _, scene = tok.rpartition(":")
            if not instr:
                raise SystemExit(f"[select_loko_cells] --scenes 항목이 잘못됨: {tok!r}")
            pairs.add((instr, as_int(scene, "--scenes scene")))
        else:
            bare.add(as_int(tok, "--scenes scene"))
    return bare, pairs


def read_index(path: Path) -> list[dict]:
    """인덱스 tsv → 행 리스트. jitter_idx 열 부재는 즉시 실패."""
    if not path.exists():
        raise SystemExit(f"[select_loko_cells] 인덱스 파일이 없습니다: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = reader.fieldnames or []
        if JITTER_COL not in header:
            raise SystemExit(
                f"[select_loko_cells] 인덱스에 '{JITTER_COL}' 열이 없습니다 "
                f"(jitter_reset_idx 로 대체 금지 — oven/washer 키는 전부 0). 열: {header}"
            )
        for need in ("grid_instruction", "scene_idx", "success"):
            if need not in header:
                raise SystemExit(f"[select_loko_cells] 인덱스에 '{need}' 열이 없습니다. 열: {header}")
        rows = []
        for raw in reader:
            if not (raw.get("grid_instruction") or "").strip():
                continue
            row = dict(raw)
            row["_instr"] = raw["grid_instruction"].strip()
            row["_scene"] = as_int(raw.get("scene_idx"), "scene_idx")
            row["_jit"] = as_int(raw.get(JITTER_COL), JITTER_COL)
            row["_noise"] = as_int(raw.get("noise_idx"), "noise_idx") if (raw.get("noise_idx") or "").strip() else 0
            row["_succ"] = as_success(raw.get("success"))
            rows.append(row)
    if not rows:
        raise SystemExit(f"[select_loko_cells] 인덱스에 유효한 행이 없습니다: {path}")
    return rows


def check_jitter_axis(rows: list[dict]) -> None:
    """(instr, scene) 마다 서로 다른 jitter 값이 2개 이상인지 검사 (지터 축 붕괴 방지)."""
    jits: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        jits[(row["_instr"], row["_scene"])].add(row["_jit"])
    bad = sorted((k, sorted(v)) for k, v in jits.items() if len(v) < 2)
    if bad:
        lines = ", ".join(f"{instr}:s{scene}(j={vals})" for (instr, scene), vals in bad[:8])
        raise SystemExit(
            f"[select_loko_cells] '{JITTER_COL}' 축이 붕괴한 (instruction, scene) 이 있습니다 "
            f"— LOKO 불가: {lines}"
            + (f" ... 총 {len(bad)}개" if len(bad) > 8 else "")
        )


def compute_cells(
    rows: list[dict],
    min_pool_fail: int,
    min_tgt_fail: int,
    min_pool_succ: int,
    instructions: list[str],
    exclude_instructions: list[str],
    scenes_bare: set[int],
    scenes_pairs: set[tuple[str, int]],
) -> list[dict]:
    """(instr, scene, j) 셀별 LOKO 집계 + 선정 판정. 필터로 빠진 셀은 reason='excluded'."""
    keep = set(instructions)
    drop = set(exclude_instructions)
    by_scene: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_scene[(row["_instr"], row["_scene"])].append(row)

    cells: list[dict] = []
    for (instr, scene), scene_rows in sorted(by_scene.items()):
        excluded = False
        pkl_missing = [r for r in scene_rows if str(r.get("has_pkl", "1")).strip() not in ("1", "True", "true")]
        if pkl_missing:  # 이관 미완(meta 는 있으나 pkl 없음) — 셀표에 넣으면 shard 결손으로 이어진다
            print(f"[warn] {instr} s{scene}: has_pkl!=1 {len(pkl_missing)}행 → scene 전체 제외(pkl_missing)", file=sys.stderr)
            excluded = True
        if keep and instr not in keep:
            excluded = True
        if instr in drop:
            excluded = True
        if (scenes_bare or scenes_pairs) and not (
            scene in scenes_bare or (instr, scene) in scenes_pairs
        ):
            excluded = True

        scene_succ = sum(r["_succ"] for r in scene_rows)
        for jit in sorted({r["_jit"] for r in scene_rows}):
            tgt = [r for r in scene_rows if r["_jit"] == jit]
            other = [r for r in scene_rows if r["_jit"] != jit]
            tgt_succ = sum(r["_succ"] for r in tgt)
            tgt_fail = len(tgt) - tgt_succ
            pool_succ = sum(r["_succ"] for r in other)
            pool_fail = len(other) - pool_succ
            machines = Counter(r.get("machine", "") for r in tgt)
            machine = machines.most_common(1)[0][0] if machines else ""

            if excluded:
                reason, selected = ("pkl_missing" if pkl_missing else "excluded"), 0
            elif pool_fail < min_pool_fail:
                reason, selected = f"pool_fail<{min_pool_fail}", 0
            elif tgt_fail < min_tgt_fail:
                reason, selected = f"tgt_fail<{min_tgt_fail}", 0
            elif pool_succ < min_pool_succ:
                reason, selected = f"pool_succ<{min_pool_succ}", 0
            else:
                reason, selected = "", 1

            cells.append({
                "grid_instruction": instr,
                "slug": slug_of(instr),
                "machine": machine,
                "scene_idx": scene,
                "jitter_idx": jit,
                "n_total": len(tgt),
                "tgt_succ": tgt_succ,
                "tgt_fail": tgt_fail,
                "pool_succ": pool_succ,
                "pool_fail": pool_fail,
                "n_pool": pool_succ + pool_fail + tgt_fail,
                "selected": selected,
                "reason": reason,
                "_rows": tgt,
                "_scene_succ": scene_succ,
                "_scene_total": len(scene_rows),
            })
    return cells


def write_manifest(cells: list[dict], out: Path) -> list[dict]:
    """선정 셀의 실패 에피소드를 1행씩 기록. 반환값 = 기록한 행들."""
    recs: list[dict] = []
    for cell in cells:
        if not cell["selected"]:
            continue
        flag = 1 if cell["pool_succ"] < 9 else 0
        for row in cell["_rows"]:
            if row["_succ"] != 0:
                continue
            recs.append({
                "grid_instruction": cell["grid_instruction"],
                "slug": cell["slug"],
                "machine": row.get("machine", ""),
                "scene_idx": cell["scene_idx"],
                "jitter_idx": cell["jitter_idx"],
                "noise_idx": row["_noise"],
                "jitter_reset_idx": row.get("jitter_reset_idx", ""),
                "base_lat": row.get("base_lat", ""),
                "base_back": row.get("base_back", ""),
                "env_seed": row.get("env_seed", ""),
                "inference_seed": row.get("inference_seed", ""),
                "sig": row.get("sig", ""),
                "rel_path": row.get("rel_path", ""),
                "pool_succ": cell["pool_succ"],
                "pool_fail": cell["pool_fail"],
                "tgt_fail": cell["tgt_fail"],
                "tgt_succ": cell["tgt_succ"],
                "flag_succ_lt9": flag,
            })
    recs.sort(key=lambda r: (
        r["machine"], r["grid_instruction"], r["scene_idx"], r["jitter_idx"], r["noise_idx"]
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(recs)
    return recs


def write_cells(cells: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CELL_COLS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            writer.writerow({k: cell[k] for k in CELL_COLS})


def print_summary(cells: list[dict], recs: list[dict]) -> None:
    per_scene: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for cell in cells:
        per_scene[(cell["grid_instruction"], cell["scene_idx"])].append(cell)

    print("== LOKO 선정 요약 ==")
    for (instr, scene), group in sorted(per_scene.items()):
        sel = [c for c in group if c["selected"]]
        succ, total = group[0]["_scene_succ"], group[0]["_scene_total"]
        eps = sum(c["tgt_fail"] for c in sel)
        js = ",".join(str(c["jitter_idx"]) for c in sel) if sel else "-"
        detail = " ".join(
            f"j{c['jitter_idx']}(tf{c['tgt_fail']}/ps{c['pool_succ']}/pf{c['pool_fail']}"
            + ("*" if c["pool_succ"] < 9 else "") + ")"
            for c in sel
        )
        print(f"  {instr:<26} s{scene}  SR {succ}/{total}  sel_j=[{js}]  eps={eps:<3} {detail}")

    by_machine_cells: Counter = Counter()
    for cell in cells:
        if cell["selected"]:
            by_machine_cells[cell["machine"]] += 1
    by_machine_eps = Counter(r["machine"] for r in recs)
    print("-- machine 별 --")
    for machine in sorted(set(by_machine_cells) | set(by_machine_eps)):
        print(f"  {machine:<10} cells={by_machine_cells[machine]:<4} eps={by_machine_eps[machine]}")
    n_sel = sum(1 for c in cells if c["selected"])
    n_flag = sum(1 for r in recs if r["flag_succ_lt9"])
    print(f"-- 총계 -- cells={n_sel}/{len(cells)}  eps={len(recs)}  flag_succ_lt9={n_flag}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="grid v6 인덱스 → scene-local LOKO 평가 셀 선정 (단위 = instruction)"
    )
    ap.add_argument("--index", type=Path, required=True, help="index_rollouts_v6.tsv 경로")
    ap.add_argument("--out", type=Path, required=True, help="선정 실패 에피소드 매니페스트 tsv")
    ap.add_argument("--cells-out", type=Path, default=None, help="전체 셀 집계 tsv (선택)")
    ap.add_argument("--min-pool-fail", type=int, default=3, help="LOKO 풀의 최소 실패 수 (기본 3)")
    ap.add_argument("--min-tgt-fail", type=int, default=1, help="대상 j 의 최소 실패 수 (기본 1)")
    ap.add_argument("--instructions", default="", help="쉼표 목록 — 이 grid_instruction 만 사용")
    ap.add_argument("--min-pool-succ", type=int, default=0, help="pool(타 j) 성공 하한 — 성공 평균·detector calib 성립용 (사용자 규칙 미확정: 기본 0)")
    ap.add_argument("--exclude-instructions", default="", help="쉼표 목록 — 제외할 grid_instruction")
    ap.add_argument("--scenes", default="", help='쉼표 목록 — "instr:scene" 쌍 또는 맨 scene 정수')
    args = ap.parse_args()

    rows = read_index(args.index)
    check_jitter_axis(rows)
    scenes_bare, scenes_pairs = parse_scenes(args.scenes)
    cells = compute_cells(
        rows,
        min_pool_fail=args.min_pool_fail,
        min_tgt_fail=args.min_tgt_fail,
        min_pool_succ=args.min_pool_succ,
        instructions=parse_list(args.instructions),
        exclude_instructions=parse_list(args.exclude_instructions),
        scenes_bare=scenes_bare,
        scenes_pairs=scenes_pairs,
    )
    recs = write_manifest(cells, args.out)
    if args.cells_out is not None:
        write_cells(cells, args.cells_out)
    print_summary(cells, recs)
    print(f"[out] manifest rows={len(recs)} -> {args.out}")
    if args.cells_out is not None:
        print(f"[out] cells rows={len(cells)} -> {args.cells_out}")
    if not recs:
        print("[warn] 선정된 에피소드가 없습니다 — 조건(--min-pool-fail/--min-tgt-fail) 확인", file=sys.stderr)


if __name__ == "__main__":
    main()
