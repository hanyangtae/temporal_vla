#!/usr/bin/env python3
"""replay eval 의 셀 표 생성 — grid 수집 셀을 **그대로 재생**하기 위한 좌표 조회.

기존 `scene_table.py` 는 "grid 와 같은 scene + 새 inference_seed 대역" 이었다.
replay 모드는 그게 아니라 **(scene s, jitter j, noise n) 셀의 수집 당시
env_seed·inference_seed 를 그대로** 다시 돌린다. 같은 머신이면 base arm 은 수집 결과와
(거의) 동일하게 재현되므로, 개입 arm 과의 차이를 셀 단위 **구제 / 파손** 으로 읽을 수 있다.

한 줄 = 셀 하나 (11열 고정 — 러너·collect_results·final_agg_condg 공용 계약):

    scene_idx  noise_idx  env_seed  inference_seed  collection_success
    task  env_name  instruction_text  jitter_idx  jitter_reset_idx  lang

- `scene_idx` = plan `scenes[instr][scene_idx]` = `instructions[instr][scene_idx]` 가 env_seed.
- `jitter_idx` = v6 지터 인덱스 j (경로 `j<jid>`). legacy(k 층·2축) 인덱스에서는 `"NA"`.
- `jitter_reset_idx` = 연속 reset 횟수. v6 는 plan `jitters[..].reset_idx` 가 인덱스에 실려
  온 값이고, legacy k 층에서는 k 자체다. 지터 축이 없는 2축 legacy 행은 `"base"`.
- `instruction_text` = plan 기본 문장, `lang` = 그 scene 의 실제 문장.
  **collector 에 넘길 canonical instruction 은 `lang` 우선**(비면 instruction_text).

3축 좌표 (v6 — 핸드오프 §4, docs/04 §3.1.1)
-------------------------------------------
좌표는 `s<sid>/j<jid>/n<nid>` 3축 폴더층이고, 인덱스도 `scene_idx`·`jitter_idx`·
`noise_idx` 3열을 직접 갖는다. 이 스크립트는 그 3열을 **그대로** 읽는다 (평탄 접기 없음).

- `--scenes` 는 scene 인덱스(sid) 위에서 고른다.
- `--jitters` 는 **jitter_idx(j)** 위에서 고른다: `"all"`(기본) / `"1,4"` / `"base"`.

구 인덱스 처리 방침
-------------------
- **v6 인덱스 안의 legacy 행**: `jitter_idx` 가 빈 칸이거나 `legacy=1` 인 행은 **행 단위로**
  legacy 취급 — 좌표는 `jitter_reset_idx`(k), 출력 `jitter_idx` 열은 `"NA"`.
- **v5 (k 층)**: `jitter_reset_idx` 는 있고 `jitter_idx` 가 **없는** 인덱스. legacy 모드로
  읽는다 — 지터 축 좌표로 k 를 쓰고(`--jitters` 도 k 위), 출력의 `jitter_idx` 열은 `"NA"`,
  `jitter_reset_idx` 열에 k 를 싣는다. 러너는 이 표를 보고 v6 의 `--jitter-idx` 가 아니라
  기존 `--jitter-reset-idx` 를 collector 에 넘긴다(무음 오동작 없음). stderr 에 legacy
  경고를 찍는다.
- **2축 legacy (v1/v2)**: 지터 열 자체가 없다 → 모든 행 `"base"`.
- **구 v4 (`cell_si` 평탄 열)**: 거부한다 — 평탄 si 규약은 폐지됐고 `scene_idx` 열 의미가
  판마다 달라 조용히 다른 셀을 도는 사고가 난다.

러너(`run_online_gated_eval.sh`)는 이 표에서 episode_idx 를
`(scene*100 + j)*EP_IDX_STRIDE + noise` (2축 legacy 는 `scene*EP_IDX_STRIDE + noise`) 로
만든다. 그 평탄값이 겹치는 셀을 한 표에 담으면 판이 덮어써지므로 **여기서 fail-loud** 한다.

출처
----
- `--index-tsv` (grid 인덱스 회수본): `grid_instruction`/`machine`/`armsig=base` 행에서
  `scene_idx`/`jitter_idx`/`jitter_reset_idx`/`noise_idx`/`env_seed`/`inference_seed`/
  `success`/`lang` 을 읽는다. **실제로 수집된 셀만** 나온다.
- `--plan-json` (collection_plan.json): env_name·instruction 문자열의 정본
  (`extra.env_names` / `extra.instruction_text`), 그리고 env_seed 교차검증
  `plan.instructions[instr][scene_idx] == env_seed` (한 scene 의 모든 j 행이 base
  env_seed 를 공유한다).

machine
-------
수집은 머신 분할이라 instruction 하나는 보통 machine 하나다. 두 개 이상이면
`--machine` 으로 명시해야 한다 (자동 선택 시 hostname 과 일치하는 것을 우선).
"머신이 다르면 base 재현이 깨진다" — memory `machine-repro-fresh-gate`.

사용:
    # v6 (지터 j)
    python scripts/steer/online_gated/replay_cells.py --slug OpenDrawer_left \
        --index-tsv .../index_rollouts_v6.tsv \
        --plan-json configs/collect/n15_grid_v6/collection_plan.json \
        --scenes 0-2 --jitters all --noises 0,1,2,3,4

    # legacy v5 (k 층) / 2축
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
BASE_K = "base"                 # 지터 축 없는 행의 좌표 표기
NA = "NA"                       # 출력 빈 칸
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
    """`--jitters` 문자열 → 지터 좌표 목록. `"all"`(또는 빈 값) 은 None = 인덱스에 있는 것 전부.

    토큰은 정수(v6 = jitter_idx, legacy = k) 또는 `"base"`(지터 축 없는 행).
    범위 표기(`0-3`)도 받는다.
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
    """정수 좌표 먼저, `"base"` 는 뒤."""
    return (1, 0) if k == BASE_K else (0, int(k))


def flat_cell_id(scene: int, j) -> int:
    """러너 episode_idx 산식의 평탄 좌표 (충돌 검사·TRIGGER_TSV 조회 키와 동일 규약).

    지터 행 = `scene*100 + j`, 2축 legacy(base) 행 = `scene`. 저장 좌표가 아니라
    **판 번호 유일성** 용 파생값이다.
    """
    return scene if j == BASE_K else scene * 100 + int(j)


def read_index(index_tsv: Path, instr: str, machine: str | None
               ) -> tuple[dict, str, bool, bool]:
    """(scene, j, noise) → dict(env_seed, inference_seed, success, reset_idx, lang).

    반환 = (셀, 쓰인 machine, 지터 축 유무, v6 여부).
    - v6: 헤더에 `jitter_idx` 열이 있다 → 지터 좌표 = jitter_idx, reset_idx 는 별도 열.
    - legacy k 층(v5): `jitter_reset_idx` 만 있다 → 지터 좌표 = k, reset_idx = 같은 값.
    - 2축 legacy: 지터 열 없음 → 모든 행 좌표 `"base"`.
    """
    lines = index_tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"{index_tsv}: 빈 파일")
    col = {n: i for i, n in enumerate(lines[0].split("\t"))}
    if "cell_si" in col:
        raise SystemExit(
            f"{index_tsv}: 구 v4 인덱스(`cell_si` 평탄 열)다 — 평탄 si 규약은 폐지됐다"
            " (docs/04 §3.1.1). scene_idx·jitter_idx·noise_idx 3열을 갖는"
            " 인덱스(build_grid_index.py 재생성본)를 쓸 것")
    for need in NEED_COLS:
        if need not in col:
            raise SystemExit(f"{index_tsv}: '{need}' 열 없음")
    is_v6 = "jitter_idx" in col
    has_reset = "jitter_reset_idx" in col
    has_jitter = is_v6 or has_reset
    if has_reset and not is_v6:
        print(f"[replay] LEGACY: {index_tsv.name} 에 `jitter_idx` 열이 없다 — v5 k 층 "
              "인덱스로 읽는다 (지터 좌표 = jitter_reset_idx k, 출력 jitter_idx=NA; "
              "러너는 --jitter-reset-idx 경로를 탄다)", file=sys.stderr)

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

    def field(p, name) -> str:
        i = col.get(name)
        return p[i].strip() if (i is not None and i < len(p)) else ""

    cells: dict[tuple, dict] = {}
    for p in rows:
        if p[col["machine"]] != machine:
            continue
        raw_seed, raw_inf = p[col["env_seed"]], p[col["inference_seed"]]
        if raw_seed in ("", "None") or raw_inf in ("", "None"):
            continue
        raw_reset = field(p, "jitter_reset_idx") if has_reset else ""
        raw_j = field(p, "jitter_idx") if is_v6 else ""
        # v6 인덱스 안에도 legacy 행(jitter_idx 빈 칸 + legacy=1)이 섞일 수 있다 —
        # 그 행은 **행 단위로** legacy 취급해 k 를 좌표로 쓰고 러너도 legacy 경로를 탄다.
        row_v6 = is_v6 and raw_j not in EMPTY and field(p, "legacy") != "1"
        if row_v6:
            j = int(raw_j)
            reset = NA if raw_reset in EMPTY else raw_reset   # plan 유래 출처 열
        else:
            j = BASE_K if raw_reset in EMPTY or raw_reset.lower() == BASE_K else int(raw_reset)
            reset = BASE_K if j == BASE_K else str(j)
        lang = field(p, "lang")
        key = (int(p[col["scene_idx"]]), j, int(p[col["noise_idx"]]))
        rec = {"env_seed": int(raw_seed), "inference_seed": int(raw_inf),
               "success": p[col["success"]], "reset_idx": reset,
               "lang": lang or NA, "v6": row_v6}
        prev = cells.setdefault(key, rec)
        if prev != rec:
            raise SystemExit(
                f"{index_tsv}: {instr} s{key[0]}/j{key[1]}/n{key[2]} 의 좌표가 두 값 "
                f"({prev}, {rec})")
    if not cells:
        raise SystemExit(f"{index_tsv}: instruction={instr} machine={machine} 셀 0")
    return cells, machine, has_jitter, is_v6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="예: OpenDrawer_left")
    ap.add_argument("--index-tsv", type=Path, required=True)
    ap.add_argument("--plan-json", type=Path, required=True)
    ap.add_argument("--machine", default=None, help="비우면 단일 machine 자동 / hostname")
    ap.add_argument("--scenes", default="0-9", help='scene 인덱스: "0-2" 또는 "0,1,2"')
    ap.add_argument("--jitters", default="all",
                    help='지터 축(v6 jitter_idx / legacy k): "all"(기본) / "1,4" / "base"')
    ap.add_argument("--noises", default="0,1,5,6")
    ap.add_argument("--out", type=Path, default=None, help="미지정 시 stdout")
    args = ap.parse_args()

    instr, plan_seeds, env_name, text = from_plan(args.plan_json, args.slug)
    cells, machine, has_jitter, is_v6 = read_index(args.index_tsv, instr, args.machine)

    scenes = parse_int_list(args.scenes)
    noises = parse_int_list(args.noises)
    want_j = parse_jitter_spec(args.jitters)
    if want_j is not None and not has_jitter and any(j != BASE_K for j in want_j):
        raise SystemExit(
            f"{args.index_tsv}: 지터 열이 없는 legacy 2축 인덱스인데 "
            f"--jitters={args.jitters!r} 로 정수 좌표를 요구했다 (all|base 만 가능)")
    task = derive_task(env_name)
    # scene 별로 인덱스에 실제 있는 지터 좌표 (--jitters all 일 때 쓰인다).
    j_by_scene: dict[int, list] = {}
    for (si, j, _ni) in cells:
        if j not in j_by_scene.setdefault(si, []):
            j_by_scene[si].append(j)

    rows, missing, flat_seen = [], [], {}
    for si in scenes:
        js = want_j if want_j is not None else sorted(j_by_scene.get(si, []), key=k_sort_key)
        if not js:
            missing.append((si, "*", "*"))
            continue
        for j in js:
            for ni in noises:
                rec = cells.get((si, j, ni))
                if rec is None:
                    missing.append((si, j, ni))
                    continue
                # plan 과의 env_seed 교차검증 (다른 scene 을 조용히 도는 사고 방지).
                # 한 scene 의 모든 j 행이 그 scene 의 env_seed 를 공유한다.
                if si in plan_seeds and plan_seeds[si] != rec["env_seed"]:
                    raise SystemExit(
                        f"s{si}: index env_seed {rec['env_seed']} != plan {plan_seeds[si]}")
                # 러너 episode_idx 유일성 — 평탄값 충돌은 판이 덮어써진다.
                flat = (flat_cell_id(si, j), ni)
                if flat in flat_seen:
                    raise SystemExit(
                        f"episode_idx 충돌: (s{si},j{j},n{ni}) 와 {flat_seen[flat]} 의 "
                        f"평탄 좌표가 {flat[0]} 로 같다 — legacy base 행과 지터 행을 한 "
                        "판에 섞지 말 것 (--jitters 로 분리)")
                flat_seen[flat] = (si, j, ni)
                rows.append((si, ni, rec["env_seed"], rec["inference_seed"],
                             rec["success"] or NA, j, rec["reset_idx"], rec["lang"],
                             rec["v6"]))
    if missing:
        raise SystemExit(
            f"수집 index 에 없는 셀 {missing} (instr={instr}, machine={machine}) — "
            "--scenes/--jitters/--noises 를 수집 범위 안으로 줄여야 한다")

    # 출력 11열. v6 행이 아니면 jitter_idx 열은 NA (러너가 legacy 경로를 타는 신호).
    body = "".join(
        f"{si}\t{ni}\t{es}\t{inf}\t{cs}\t{task}\t{env_name}\t{text}\t"
        f"{j if row_v6 else NA}\t{reset}\t{lang}\n"
        for si, ni, es, inf, cs, j, reset, lang, row_v6 in rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        n_fail = sum(1 for r in rows if r[4] == "0")
        n_jit = sum(1 for r in rows if r[5] != BASE_K)
        n_v6 = sum(1 for r in rows if r[8])
        print(f"[replay] {args.slug}: {len(rows)} 셀 (machine={machine}, instr={instr}, "
              f"mode={'v6' if is_v6 else 'legacy'}(v6행 {n_v6}/{len(rows)}), "
              f"scenes={len(scenes)}×jitters={args.jitters}×noises={len(noises)}, "
              f"수집실패 {n_fail}"
              + (f", 지터행 {n_jit}" if has_jitter else "") + f") → {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
