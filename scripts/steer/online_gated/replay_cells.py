#!/usr/bin/env python3
"""replay eval 의 셀 표 생성 — grid 수집 셀을 **그대로 재생**하기 위한 좌표 조회.

기존 `scene_table.py` 는 "grid 와 같은 scene + 새 inference_seed 대역" 이었다.
replay 모드는 그게 아니라 **(scene s, noise m) 셀의 수집 당시 env_seed·inference_seed
를 그대로** 다시 돌린다. 같은 머신이면 base arm 은 수집 결과와 (거의) 동일하게
재현되므로, 개입 arm 과의 차이를 셀 단위 **구제 / 파손** 으로 읽을 수 있다.

한 줄 = 셀 하나:

    scene_idx  noise_idx  env_seed  inference_seed  collection_success  task  env_name  instruction_text

v4 지터 축 (docs/04 §3.1.1)
--------------------------
index 헤더에 `jitter_reset_idx` 열이 있으면 **v4 모드**로 동작한다 (없으면 v1/v2 그대로 —
열 수·의미 불변). v4 에서는 셋째 축 k(reset_idx)가 붙고, 좌표는 기존 2축 인프라를 그대로
쓰기 위해 **평탄 cell id** 로 접는다:

    cell_key = cell_si = base_scene*100 + k   (지터 행)
    cell_key = scene_idx                      (base 행 = v2 재사용분, jitter_reset_idx="base")

- 첫 열(scene_idx 자리)에는 **cell_key(평탄값)** 를 싣는다 — 러너가 이 값으로
  `ep = cell_key*EP_IDX_STRIDE + noise` 를 만들기 때문. base scene 복원은
  `cell_key//100` (cell_key>=100 일 때). base scene 0 의 지터 셀은 cell_key 0..3 이라
  이 나눗셈으로는 복원되지 않으므로, **base scene 은 10번째 열로 따로 싣는다**.
- v4 모드에서만 뒤에 두 열이 더 붙는다: `jitter_reset_idx`("base" 또는 정수 k) 와
  `base_scene_idx`(0-4). 기존 8열의 위치·의미는 그대로다.
- `--scenes` 는 cell_key 위에서 고른다 (예: `0-3,100-103,200-203,300-303,400-403`).
  ★ base 행(cell_key 0-4)과 base scene 0 의 지터 행(cell_key 0-3)은 평탄값이 겹친다 —
  둘을 한 판에 섞으면 좌표 충돌로 fail-loud 한다 (수집측이 cell_si 로 분리해 주지 않는 한).

출처
----
- `--index-tsv` (grid 인덱스 회수본): `grid_instruction`/`machine`/`armsig=base` 행에서
  `scene_idx`/`noise_idx`/`env_seed`/`inference_seed`/`success` 를 읽는다.
  **실제로 수집된 셀만** 나온다 (계획엔 있지만 미수집인 셀은 재생 대상이 아니다).
- `--plan-json` (collection_plan.json): env_name·instruction 문자열의 정본
  (`extra.env_names` / `extra.instruction_text`), 그리고 env_seed 교차검증.
  v4 plan 은 seeds 리스트 index 가 평탄 si 이므로 교차검증 키는 `base_scene*100`
  (한 scene 의 모든 k 행이 base env_seed 를 공유한다).

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

    # v4 (지터) — --scenes 는 평탄 cell id
    python scripts/steer/online_gated/replay_cells.py --slug OvenRack_out \
        --index-tsv .../index_rollouts_v4.tsv \
        --plan-json configs/collect/n15_grid_v4/collection_plan.json \
        --scenes 0-3,100-103,200-203,300-303,400-403 --noises 0,1,2,3,4
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


def cell_key_of(p: list[str], col: dict[str, int]) -> tuple[int, str, int]:
    """index 한 행 → (cell_key, jitter_reset_idx, base_scene).

    v4 규약(docs/04 §3.1.1): `cell_si` 열이 있으면 그것이 정본 평탄값이고, 없으면
    지터 행은 base_scene*100+k, base 행은 scene_idx 로 접는다. `cell_si` 가
    scene_idx 와 같은 base 행(수집측이 평탄화하지 않은 경우)은 scene_idx 를 쓴다.
    """
    base_scene = int(p[col["scene_idx"]])
    jit = (p[col["jitter_reset_idx"]] or "base").strip() if "jitter_reset_idx" in col else "base"
    raw_si = p[col["cell_si"]].strip() if "cell_si" in col else ""
    if jit in ("", "base", "NA", "None"):
        jit = "base"
        key = int(raw_si) if raw_si not in ("", "NA", "None") else base_scene
    else:
        k = int(jit)
        key = int(raw_si) if raw_si not in ("", "NA", "None") else base_scene * 100 + k
    return key, jit, base_scene


def read_index(index_tsv: Path, instr: str, machine: str | None) -> tuple[dict, str, bool]:
    """(cell_key, noise) → dict(env_seed, inference_seed, success, jitter, base_scene).

    반환 = (셀, 쓰인 machine, v4 여부). v4 여부는 헤더에 `jitter_reset_idx` 열이 있는지로
    판정한다 (v1/v2 index 에서는 cell_key = scene_idx 라 기존과 동일).
    """
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
        cell_key, jit, base_scene = cell_key_of(p, col)
        key = (cell_key, int(p[col["noise_idx"]]))
        rec = {"env_seed": int(raw_seed), "inference_seed": int(raw_inf),
               "success": p[col["success"]], "jitter": jit, "base_scene": base_scene}
        prev = cells.setdefault(key, rec)
        if prev != rec:
            hint = ""
            if prev.get("jitter") != rec.get("jitter"):
                # base 행 cell_key(0-4)와 base scene 0 의 지터 행 cell_key(0-3) 충돌
                hint = (" — base 행과 지터 행의 평탄 cell id 가 겹쳤다 (docstring 참조): "
                        "두 종류를 한 판에 섞지 말거나 index 의 cell_si 로 분리할 것")
            raise SystemExit(
                f"{index_tsv}: {instr} cell{key[0]}/n{key[1]} 의 좌표가 두 값 "
                f"({prev}, {rec}){hint}")
    if not cells:
        raise SystemExit(f"{index_tsv}: instruction={instr} machine={machine} 셀 0")
    return cells, machine, "jitter_reset_idx" in col


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
    cells, machine, is_v4 = read_index(args.index_tsv, instr, args.machine)

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
            # plan 과의 env_seed 교차검증 (다른 scene 을 조용히 도는 사고 방지).
            # v4 plan 은 seeds 리스트 index 가 평탄 si 라 base scene 의 대표 자리
            # (base_scene*100) 로 조회한다 — 한 scene 의 모든 k 행이 base seed 공유.
            pk = rec["base_scene"] * 100 if is_v4 else si
            if pk in plan_seeds and plan_seeds[pk] != rec["env_seed"]:
                raise SystemExit(
                    f"cell{si}: index env_seed {rec['env_seed']} != plan {plan_seeds[pk]}")
            rows.append((si, ni, rec["env_seed"], rec["inference_seed"],
                         rec["success"] or "NA", rec["jitter"], rec["base_scene"]))
    if missing:
        raise SystemExit(
            f"수집 index 에 없는 셀 {missing} (instr={instr}, machine={machine}) — "
            "--scenes/--noises 를 수집 범위 안으로 줄여야 한다")

    # v4 모드에서만 뒤에 2열(jitter_reset_idx, base_scene_idx) 추가 — 기존 8열 불변.
    body = "".join(
        f"{si}\t{ni}\t{es}\t{inf}\t{cs}\t{task}\t{env_name}\t{text}"
        + (f"\t{jit}\t{bs}\n" if is_v4 else "\n")
        for si, ni, es, inf, cs, jit, bs in rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        n_fail = sum(1 for r in rows if r[4] == "0")
        n_jit = sum(1 for r in rows if r[5] != "base")
        print(f"[replay] {args.slug}: {len(rows)} 셀 (machine={machine}, instr={instr}, "
              f"cells={len(scenes)}×noises={len(noises)}, 수집실패 {n_fail}"
              + (f", v4 지터행 {n_jit}" if is_v4 else "") + f") → {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
