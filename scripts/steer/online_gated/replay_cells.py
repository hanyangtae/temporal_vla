#!/usr/bin/env python3
"""replay eval 의 셀 표 생성 — grid 수집 셀을 **그대로 재생**하기 위한 좌표 조회.

기존 `scene_table.py` 는 "grid 와 같은 scene + 새 inference_seed 대역" 이었다.
replay 모드는 그게 아니라 **(scene s, jitter k, noise m) 셀의 수집 당시
env_seed·inference_seed 를 그대로** 다시 돌린다. 같은 머신이면 base arm 은 수집 결과와
(거의) 동일하게 재현되므로, 개입 arm 과의 차이를 셀 단위 **구제 / 파손** 으로 읽을 수 있다.

한 줄 = 셀 하나 (9열 고정):

    scene_idx  noise_idx  env_seed  inference_seed  collection_success
    task  env_name  instruction_text  jitter_reset_idx

- `scene_idx` = **base scene**(plan `instructions[instr][scene_idx]` 가 env_seed). 평탄
  인코딩(구 `cell_si`)은 폐지됐다 — docs/04 §3.1.1.
- `jitter_reset_idx` = 정수 k, 또는 지터 축이 없는 legacy(2축) 행이면 `"base"`.

3축 좌표 (docs/04 §3.1.1)
------------------------
좌표는 `s<i>/k<r>/n<j>` 3축 폴더층이고, 인덱스도 `scene_idx`·`jitter_reset_idx`·
`noise_idx` 3열을 직접 갖는다. 이 스크립트는 그 3열을 **그대로** 읽는다 (평탄 접기 없음).

- `--scenes` 는 **base scene**(0-4) 위에서 고른다.
- `--jitters` 로 k 를 고른다: `"all"`(기본, 인덱스에 있는 k 전부) / `"1,4"` / `"base"`.
- legacy(2축) 인덱스에는 `jitter_reset_idx` 열이 없거나 값이 비어 있다 → k = `"base"`.

러너(`run_online_gated_eval.sh`)는 이 표에서 episode_idx 를
`(scene*100 + k)*EP_IDX_STRIDE + noise` (legacy 는 `scene*EP_IDX_STRIDE + noise`) 로
만든다. 그 평탄값이 겹치는 셀을 한 표에 담으면 판이 덮어써지므로 **여기서 fail-loud** 한다
(legacy base 행과 지터 행을 한 판에 섞는 경우가 대표적이다).

구 v4 인덱스(`cell_si` 열 보유)는 **거부**한다 — 평탄 si 규약은 폐지됐고, 그 표의
`scene_idx` 열 의미가 판마다 달라 조용히 다른 셀을 도는 사고가 난다.

출처
----
- `--index-tsv` (grid 인덱스 회수본): `grid_instruction`/`machine`/`armsig=base` 행에서
  `scene_idx`/`jitter_reset_idx`/`noise_idx`/`env_seed`/`inference_seed`/`success` 를 읽는다.
  **실제로 수집된 셀만** 나온다 (계획엔 있지만 미수집인 셀은 재생 대상이 아니다).
- `--plan-json` (collection_plan.json): env_name·instruction 문자열의 정본
  (`extra.env_names` / `extra.instruction_text`), 그리고 env_seed 교차검증.
  3축 plan 은 `instructions[instr]` 가 **base scene seed 목록**(길이 5)이므로 대조는
  `plan.instructions[instr][scene_idx] == env_seed` (한 scene 의 모든 k 행이 base
  env_seed 를 공유한다).

machine
-------
수집은 머신 분할이라 instruction 하나는 보통 machine 하나다. 두 개 이상이면
`--machine` 으로 명시해야 한다 (자동 선택 시 hostname 과 일치하는 것을 우선).
"머신이 다르면 base 재현이 깨진다" — memory `machine-repro-fresh-gate`.

사용:
    # legacy 2축 (v1/v2)
    python scripts/steer/online_gated/replay_cells.py --slug OvenRack_out \
        --index-tsv outputs/steer/online_pipe/manifests/index_rollouts.tsv \
        --plan-json configs/collect/n15_grid_v1/collection_plan.json \
        --scenes 0-9 --noises 0,1,5,6

    # 3축 (지터 k) — --scenes 는 base scene, --jitters 로 k 선택
    python scripts/steer/online_gated/replay_cells.py --slug OvenRack_out \
        --index-tsv .../index_rollouts_v5.tsv \
        --plan-json configs/collect/n15_grid_v5_scenario/collection_plan.json \
        --scenes 0-4 --jitters all --noises 0,1,2,3,4
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
BASE_K = "base"                 # 지터 축 없는 행의 k 표기
EMPTY = ("", "NA", "None")      # 빈 값으로 취급하는 문자열


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


def parse_jitter_spec(spec: str) -> list | None:
    """`--jitters` 문자열 → k 목록. `"all"`(또는 빈 값) 은 None = 인덱스에 있는 k 전부.

    토큰은 정수 k 또는 `"base"`(지터 축 없는 행). 범위 표기(`0-3`)도 받는다.
    """
    s = (spec or "").strip().lower()
    if s in ("", "all", "*"):
        return None
    out: list = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in (BASE_K, "none", "na"):
            if BASE_K not in out:
                out.append(BASE_K)
        elif "-" in tok[1:]:
            a, b = tok.split("-", 1)
            out.extend(k for k in range(int(a), int(b) + 1) if k not in out)
        else:
            k = int(tok)
            if k not in out:
                out.append(k)
    if not out:
        raise SystemExit(f"--jitters={spec!r} 해석 결과가 비었다")
    return sorted(out, key=k_sort_key)


def k_sort_key(k) -> tuple[int, int]:
    """정수 k 먼저, `"base"` 는 뒤."""
    return (1, 0) if k == BASE_K else (0, int(k))


def flat_cell_id(scene: int, k) -> int:
    """러너 episode_idx 산식의 평탄 좌표 (충돌 검사·TRIGGER_TSV 조회 키와 동일 규약).

    지터 행 = `scene*100 + k`, legacy(base) 행 = `scene`. 저장 좌표가 아니라 **판 번호
    유일성** 용 파생값이다 (docs/04 §3.1.1 — 평탄 cell_si 열 자체는 폐지).
    """
    return scene if k == BASE_K else scene * 100 + int(k)


def read_index(index_tsv: Path, instr: str, machine: str | None
               ) -> tuple[dict, str, bool]:
    """(scene, k, noise) → dict(env_seed, inference_seed, success).

    반환 = (셀, 쓰인 machine, 지터 축 유무). 지터 축 유무 = 헤더에 `jitter_reset_idx`
    열이 있는지 (없으면 모든 행 k = "base").
    """
    lines = index_tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"{index_tsv}: 빈 파일")
    col = {n: i for i, n in enumerate(lines[0].split("\t"))}
    if "cell_si" in col:
        raise SystemExit(
            f"{index_tsv}: 구 v4 인덱스(`cell_si` 평탄 열)다 — 평탄 si 규약은 폐지됐다"
            " (docs/04 §3.1.1). scene_idx(base)·jitter_reset_idx·noise_idx 3열을 갖는"
            " 인덱스(build_grid_index.py 재생성본)를 쓸 것")
    for need in NEED_COLS:
        if need not in col:
            raise SystemExit(f"{index_tsv}: '{need}' 열 없음")
    has_jitter = "jitter_reset_idx" in col

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

    cells: dict[tuple, dict] = {}
    for p in rows:
        if p[col["machine"]] != machine:
            continue
        raw_seed, raw_inf = p[col["env_seed"]], p[col["inference_seed"]]
        if raw_seed in ("", "None") or raw_inf in ("", "None"):
            continue
        raw_k = p[col["jitter_reset_idx"]].strip() if has_jitter else ""
        k = BASE_K if raw_k in EMPTY or raw_k.lower() == BASE_K else int(raw_k)
        key = (int(p[col["scene_idx"]]), k, int(p[col["noise_idx"]]))
        rec = {"env_seed": int(raw_seed), "inference_seed": int(raw_inf),
               "success": p[col["success"]]}
        prev = cells.setdefault(key, rec)
        if prev != rec:
            raise SystemExit(
                f"{index_tsv}: {instr} s{key[0]}/k{key[1]}/n{key[2]} 의 좌표가 두 값 "
                f"({prev}, {rec})")
    if not cells:
        raise SystemExit(f"{index_tsv}: instruction={instr} machine={machine} 셀 0")
    return cells, machine, has_jitter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="예: OvenRack_out")
    ap.add_argument("--index-tsv", type=Path, required=True)
    ap.add_argument("--plan-json", type=Path, required=True)
    ap.add_argument("--machine", default=None, help="비우면 단일 machine 자동 / hostname")
    ap.add_argument("--scenes", default="0-9", help='base scene: "0-9" 또는 "0,1,2"')
    ap.add_argument("--jitters", default="all",
                    help='지터 축 k: "all"(기본) / "1,4" / "0-3" / "base"')
    ap.add_argument("--noises", default="0,1,5,6")
    ap.add_argument("--out", type=Path, default=None, help="미지정 시 stdout")
    args = ap.parse_args()

    instr, plan_seeds, env_name, text = from_plan(args.plan_json, args.slug)
    cells, machine, has_jitter = read_index(args.index_tsv, instr, args.machine)

    scenes = parse_int_list(args.scenes)
    noises = parse_int_list(args.noises)
    want_k = parse_jitter_spec(args.jitters)
    if want_k is not None and not has_jitter and any(k != BASE_K for k in want_k):
        raise SystemExit(
            f"{args.index_tsv}: `jitter_reset_idx` 열이 없는 legacy 2축 인덱스인데 "
            f"--jitters={args.jitters!r} 로 정수 k 를 요구했다 (all|base 만 가능)")
    task = derive_task(env_name)
    # scene 별로 인덱스에 실제 있는 k (--jitters all 일 때 쓰인다).
    k_by_scene: dict[int, list] = {}
    for (si, k, _ni) in cells:
        if k not in k_by_scene.setdefault(si, []):
            k_by_scene[si].append(k)

    rows, missing, flat_seen = [], [], {}
    for si in scenes:
        ks = want_k if want_k is not None else sorted(k_by_scene.get(si, []), key=k_sort_key)
        if not ks:
            missing.append((si, "*", "*"))
            continue
        for k in ks:
            for ni in noises:
                rec = cells.get((si, k, ni))
                if rec is None:
                    missing.append((si, k, ni))
                    continue
                # plan 과의 env_seed 교차검증 (다른 scene 을 조용히 도는 사고 방지).
                # 3축 plan 의 instructions[instr] 는 base scene seed 목록이므로 키는
                # scene_idx (한 scene 의 모든 k 행이 base env_seed 를 공유한다).
                if si in plan_seeds and plan_seeds[si] != rec["env_seed"]:
                    raise SystemExit(
                        f"s{si}: index env_seed {rec['env_seed']} != plan {plan_seeds[si]}")
                # 러너 episode_idx 유일성 — 평탄값 충돌은 판이 덮어써진다.
                flat = (flat_cell_id(si, k), ni)
                if flat in flat_seen:
                    raise SystemExit(
                        f"episode_idx 충돌: (s{si},k{k},n{ni}) 와 {flat_seen[flat]} 의 "
                        f"평탄 좌표가 {flat[0]} 로 같다 — legacy base 행과 지터 행을 한 "
                        "판에 섞지 말 것 (--jitters 로 분리)")
                flat_seen[flat] = (si, k, ni)
                rows.append((si, ni, rec["env_seed"], rec["inference_seed"],
                             rec["success"] or "NA", k))
    if missing:
        raise SystemExit(
            f"수집 index 에 없는 셀 {missing} (instr={instr}, machine={machine}) — "
            "--scenes/--jitters/--noises 를 수집 범위 안으로 줄여야 한다")

    body = "".join(
        f"{si}\t{ni}\t{es}\t{inf}\t{cs}\t{task}\t{env_name}\t{text}\t{k}\n"
        for si, ni, es, inf, cs, k in rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        n_fail = sum(1 for r in rows if r[4] == "0")
        n_jit = sum(1 for r in rows if r[5] != BASE_K)
        print(f"[replay] {args.slug}: {len(rows)} 셀 (machine={machine}, instr={instr}, "
              f"scenes={len(scenes)}×jitters={args.jitters}×noises={len(noises)}, "
              f"수집실패 {n_fail}"
              + (f", 지터행 {n_jit}" if has_jitter else "") + f") → {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
