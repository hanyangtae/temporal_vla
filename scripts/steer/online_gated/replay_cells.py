#!/usr/bin/env python3
"""replay eval 의 셀 표 생성 — grid 수집 셀을 **그대로 재생**하기 위한 좌표 조회.

기존 `scene_table.py` 는 "grid 와 같은 scene + 새 inference_seed 대역" 이었다.
replay 모드는 그게 아니라 **(scene s, noise m) 셀의 수집 당시 env_seed·inference_seed
를 그대로** 다시 돌린다. 같은 머신이면 base arm 은 수집 결과와 (거의) 동일하게
재현되므로, 개입 arm 과의 차이를 셀 단위 **구제 / 파손** 으로 읽을 수 있다.

한 줄 = 셀 하나:

    scene_idx  noise_idx  env_seed  inference_seed  collection_success  task  env_name  instruction_text

출처
----
- `--index-tsv` (grid 인덱스 회수본): `grid_instruction`/`machine`/`armsig=base` 행에서
  `scene_idx`/`noise_idx`/`env_seed`/`inference_seed`/`success` 를 읽는다.
  **실제로 수집된 셀만** 나온다 (계획엔 있지만 미수집인 셀은 재생 대상이 아니다).
- `--plan-json` (collection_plan.json): env_name·instruction 문자열의 정본
  (`extra.env_names` / `extra.instruction_text`), 그리고 env_seed 교차검증.

machine
-------
수집은 머신 분할이라 instruction 하나는 보통 machine 하나다. 두 개 이상이면
`--machine` 으로 명시해야 한다 (자동 선택 시 hostname 과 일치하는 것을 우선).
"머신이 다르면 base 재현이 깨진다" — memory `machine-repro-fresh-gate`.

사용:
    python scripts/steer/online_gated/replay_cells.py --slug OvenRack_out \
        --index-tsv outputs/steer/online_pipe/manifests/index_rollouts.tsv \
        --plan-json configs/collect/n15_grid_v1/collection_plan.json \
        --scenes 0-9 --noises 0,1,5,6
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scene_table import derive_task, from_plan  # noqa: E402

NEED_COLS = ("grid_instruction", "machine", "scene_idx", "noise_idx", "armsig",
             "env_seed", "inference_seed", "success")


def parse_int_list(spec: str) -> list[int]:
    """"0,1,5,6" 또는 "0-9" 또는 "0-4,8" → 정수 목록 (정렬·중복제거)."""
    out: set[int] = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def read_index(index_tsv: Path, instr: str, machine: str | None) -> tuple[dict, str]:
    """(scene, noise) → dict(env_seed, inference_seed, success). 반환 = (셀, 쓰인 machine)."""
    lines = index_tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"{index_tsv}: 빈 파일")
    col = {n: i for i, n in enumerate(lines[0].split("\t"))}
    for need in NEED_COLS:
        if need not in col:
            raise SystemExit(f"{index_tsv}: '{need}' 열 없음")

    rows = []
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) <= max(col.values()):
            continue
        if p[col["grid_instruction"]] != instr or p[col["armsig"]] != "base":
            continue
        rows.append(p)
    if not rows:
        raise SystemExit(f"{index_tsv}: instruction={instr} 의 base 행 없음")

    machines = sorted({p[col["machine"]] for p in rows})
    if machine is None:
        if len(machines) == 1:
            machine = machines[0]
        else:
            here = socket.gethostname()
            if here in machines:
                machine = here
                print(f"[replay] machine 후보 {machines} → hostname {here} 선택",
                      file=sys.stderr)
            else:
                raise SystemExit(
                    f"instruction={instr} 의 machine 이 여러 개 {machines} 이고 hostname"
                    f"({here}) 과도 안 맞는다 — --machine 으로 명시할 것")
    elif machine not in machines:
        raise SystemExit(f"instruction={instr} 에 machine={machine} 없음 (있는 것: {machines})")

    cells: dict[tuple[int, int], dict] = {}
    for p in rows:
        if p[col["machine"]] != machine:
            continue
        raw_seed, raw_inf = p[col["env_seed"]], p[col["inference_seed"]]
        if raw_seed in ("", "None") or raw_inf in ("", "None"):
            continue
        key = (int(p[col["scene_idx"]]), int(p[col["noise_idx"]]))
        rec = {"env_seed": int(raw_seed), "inference_seed": int(raw_inf),
               "success": p[col["success"]]}
        prev = cells.setdefault(key, rec)
        if prev != rec:
            raise SystemExit(
                f"{index_tsv}: {instr} s{key[0]}/n{key[1]} 의 좌표가 두 값 ({prev}, {rec})")
    if not cells:
        raise SystemExit(f"{index_tsv}: instruction={instr} machine={machine} 셀 0")
    return cells, machine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="예: OvenRack_out")
    ap.add_argument("--index-tsv", type=Path, required=True)
    ap.add_argument("--plan-json", type=Path, required=True)
    ap.add_argument("--machine", default=None, help="비우면 단일 machine 자동 / hostname")
    ap.add_argument("--scenes", default="0-9", help='"0-9" 또는 "0,1,2"')
    ap.add_argument("--noises", default="0,1,5,6")
    ap.add_argument("--out", type=Path, default=None, help="미지정 시 stdout")
    args = ap.parse_args()

    instr, plan_seeds, env_name, text = from_plan(args.plan_json, args.slug)
    cells, machine = read_index(args.index_tsv, instr, args.machine)

    scenes = parse_int_list(args.scenes)
    noises = parse_int_list(args.noises)
    task = derive_task(env_name)

    rows, missing = [], []
    for si in scenes:
        for ni in noises:
            rec = cells.get((si, ni))
            if rec is None:
                missing.append((si, ni))
                continue
            # plan 과의 env_seed 교차검증 (다른 scene 을 조용히 도는 사고 방지)
            if si in plan_seeds and plan_seeds[si] != rec["env_seed"]:
                raise SystemExit(
                    f"s{si}: index env_seed {rec['env_seed']} != plan {plan_seeds[si]}")
            rows.append((si, ni, rec["env_seed"], rec["inference_seed"],
                         rec["success"] or "NA"))
    if missing:
        raise SystemExit(
            f"수집 index 에 없는 셀 {missing} (instr={instr}, machine={machine}) — "
            "--scenes/--noises 를 수집 범위 안으로 줄여야 한다")

    body = "".join(
        f"{si}\t{ni}\t{es}\t{inf}\t{cs}\t{task}\t{env_name}\t{text}\n"
        for si, ni, es, inf, cs in rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        n_fail = sum(1 for r in rows if r[4] == "0")
        print(f"[replay] {args.slug}: {len(rows)} 셀 (machine={machine}, instr={instr}, "
              f"scenes={len(scenes)}×noises={len(noises)}, 수집실패 {n_fail}) → {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
