#!/usr/bin/env python3
"""grid index(rollouts.tsv)에서 fit_setm용 manifest tsv 생성.

행 = pkl_path \t label(success 0/1) \t scene. instruction당 machine은
extract_grid_matrix와 같은 규칙(셀 수 최대, 동수면 fail-loud)으로 1개 선택.
pkl 부모 디렉토리는 grid 규약상 'base' → fit_setm --cell base 로 소비.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-tsv", type=Path, required=True)
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--instruction", required=True, help="예: OvenRack/out")
    ap.add_argument("--machine", default=None, help="동수 tie 시 수동 지정")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lines = args.index_tsv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    col = {name: i for i, name in enumerate(header)}
    for need in ("grid_instruction", "machine", "armsig", "scene_idx", "noise_idx",
                 "success", "has_pkl", "rel_path"):
        if need not in col:
            raise SystemExit(f"index에 {need} 열 없음")

    rows = []
    for ln in lines[1:]:
        p = ln.split("\t")
        if p[col["grid_instruction"]] != args.instruction or p[col["armsig"]] != "base":
            continue
        if p[col["has_pkl"]] != "1":
            continue
        rows.append(p)
    if not rows:
        raise SystemExit(f"instruction={args.instruction} base 행 없음")

    by_machine = collections.Counter(p[col["machine"]] for p in rows)
    if args.machine:
        pick = args.machine
        if pick not in by_machine:
            raise SystemExit(f"machine={pick} 행 없음 (있는 것: {dict(by_machine)})")
    else:
        top = by_machine.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            raise SystemExit(f"machine 동수 tie {dict(by_machine)} — --machine 지정 필요")
        pick = top[0][0]

    seen = set()
    out_lines = [f"# instruction={args.instruction} machine={pick} (index={args.index_tsv.name})"]
    n_succ = n_fail = 0
    for p in rows:
        if p[col["machine"]] != pick:
            continue
        key = (p[col["scene_idx"]], p[col["noise_idx"]])
        if key in seen:
            raise SystemExit(f"중복 셀 {key} (machine={pick})")
        seen.add(key)
        succ = p[col["success"]]
        if succ not in ("0", "1"):
            raise SystemExit(f"success 값 이상: {succ!r} @ {key}")
        pkl = args.grid_root / p[col["rel_path"]]
        out_lines.append(f"{pkl}\t{succ}\ts{p[col['scene_idx']]}")
        n_succ += succ == "1"
        n_fail += succ == "0"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[manifest] {args.out} cells={len(seen)} succ={n_succ} fail={n_fail} machine={pick}")


if __name__ == "__main__":
    main()
