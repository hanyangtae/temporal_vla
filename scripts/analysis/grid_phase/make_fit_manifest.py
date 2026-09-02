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
    ap.add_argument("--scenes", default=None,
                    help="fit scene_idx 콤마목록 (예: 0,1,2,3,4). 미지정=전체")
    ap.add_argument("--noises", default=None,
                    help="fit noise_idx 콤마목록. 부족 시 같은 scene 안에서 다음 noise 를 "
                         "순서대로 추가해 클래스 최소를 채운다 (--min-class)")
    ap.add_argument("--min-class", type=int, default=3,
                    help="성공/실패 episode 최소 수 (부족 시 noise 확장, 그래도 부족이면 에러)")
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

    cells = {}
    for p in rows:
        if p[col["machine"]] != pick:
            continue
        key = (int(p[col["scene_idx"]]), int(p[col["noise_idx"]]))
        if key in cells:
            raise SystemExit(f"중복 셀 {key} (machine={pick})")
        succ = p[col["success"]]
        if succ not in ("0", "1"):
            raise SystemExit(f"success 값 이상: {succ!r} @ {key}")
        cells[key] = (succ, p[col["rel_path"]])

    fit_scenes = ([int(x) for x in args.scenes.split(",")] if args.scenes
                  else sorted({s for s, _ in cells}))
    base_noises = ([int(x) for x in args.noises.split(",")] if args.noises
                   else sorted({n for _, n in cells}))
    all_noises = sorted({n for _, n in cells})
    chosen = [(s, n) for s in fit_scenes for n in base_noises if (s, n) in cells]
    extended = []
    if args.noises:
        # 클래스 최소 미달이면 fit scene 안에서 다음 noise 를 순서대로 추가 (결정적)
        def count(cls):
            return sum(1 for k in chosen + extended if cells[k][0] == cls)
        extra = [n for n in all_noises if n not in base_noises]
        for n in extra:
            if count("1") >= args.min_class and count("0") >= args.min_class:
                break
            for s in fit_scenes:
                k = (s, n)
                if k in cells and k not in chosen:
                    extended.append(k)
    sel = chosen + extended
    n_succ = sum(1 for k in sel if cells[k][0] == "1")
    n_fail = sum(1 for k in sel if cells[k][0] == "0")
    if n_succ < args.min_class or n_fail < args.min_class:
        raise SystemExit(f"클래스 최소 미달: succ={n_succ} fail={n_fail} "
                         f"(scenes={fit_scenes} 전 noise 확장 후에도)")

    out_lines = [f"# instruction={args.instruction} machine={pick} (index={args.index_tsv.name})",
                 f"# fit_scenes={fit_scenes} base_noises={base_noises} "
                 f"extended_cells={extended} min_class={args.min_class}"]
    for s, n in sorted(sel):
        succ, rel = cells[(s, n)]
        pkl = args.grid_root / rel
        if pkl.suffix != ".pkl":          # rel_path 는 arm 디렉토리(.../base)까지만
            pkl = pkl / "rollout.pkl"
        out_lines.append(f"{pkl}\t{succ}\ts{s}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[manifest] {args.out} cells={len(sel)} (+ext {len(extended)}) "
          f"succ={n_succ} fail={n_fail} machine={pick} scenes={fit_scenes}")


if __name__ == "__main__":
    main()
