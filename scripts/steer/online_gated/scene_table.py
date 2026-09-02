#!/usr/bin/env python3
"""online-gated eval 의 scene 표 생성 — grid 와 **같은 scene** 을 쓰기 위한 좌표 조회.

한 줄 = scene 하나:

    scene_idx <TAB> env_seed <TAB> task <TAB> env_name <TAB> instruction_text

출처
----
- `--index-tsv` (grid 인덱스 rollouts.tsv): `grid_instruction`/`scene_idx`/`env_seed`
  열에서 **실제로 수집된** scene 만 읽는다 (계획엔 있지만 미수집인 scene 을 평가하면
  scene-matched 가 아니다).
- `--plan-json` (collection_plan.json): env_name·instruction 문자열의 정본
  (`extra.env_names` / `extra.instruction_text`), 그리고 index 가 없을 때의 env_seed 폴백.

둘 다 주면 env_seed 를 교차검증하고 불일치는 fail-loud (조용히 다른 scene 을 도는 사고 방지).
instruction 은 slug (`OvenRack_out`) 로 지정한다 — `extract_grid_matrix.instr_slug` 규약.

사용:
    python scripts/steer/online_gated/scene_table.py --slug OvenRack_out \
        --plan-json configs/collect/n15_grid_v1/collection_plan.json \
        [--index-tsv outputs/collect/.../rollouts.tsv] [--n-scenes 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def instr_slug(instruction: str) -> str:
    """`PPCC/bread` → `PPCC_bread` (extract_grid_matrix.instr_slug 와 동일 규약)."""
    out = instruction.strip()
    for ch in ("/", " ", "\\", "\t"):
        out = out.replace(ch, "_")
    return out


def derive_task(env_name: str) -> str:
    leaf = env_name.split("/")[-1]
    return leaf.split("_PandaOmron")[0].split("_Env")[0]


def from_plan(plan_json: Path, slug: str) -> tuple[str, dict[int, int], str, str]:
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    matches = [i for i in plan["instructions"] if instr_slug(i) == slug]
    if len(matches) != 1:
        raise SystemExit(
            f"{plan_json.name}: slug={slug!r} 에 맞는 instruction 이 {len(matches)}개 "
            f"(있는 것: {[instr_slug(i) for i in plan['instructions']]})")
    instr = matches[0]
    extra = plan.get("extra") or {}
    env_name = (extra.get("env_names") or {}).get(instr)
    if not env_name:
        raise SystemExit(f"{plan_json.name}: extra.env_names 에 {instr!r} 없음")
    text = (extra.get("instruction_text") or {}).get(instr, instr)
    seeds = {i: int(s) for i, s in enumerate(plan["instructions"][instr])}
    return instr, seeds, env_name, text


def from_index(index_tsv: Path, instr: str) -> dict[int, int]:
    lines = index_tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"{index_tsv}: 빈 파일")
    col = {n: i for i, n in enumerate(lines[0].split("\t"))}
    for need in ("grid_instruction", "scene_idx", "env_seed", "armsig"):
        if need not in col:
            raise SystemExit(f"{index_tsv}: {need} 열 없음")
    seeds: dict[int, int] = {}
    for ln in lines[1:]:
        p = ln.split("\t")
        if p[col["grid_instruction"]] != instr or p[col["armsig"]] != "base":
            continue
        raw = p[col["env_seed"]]
        if raw in ("", "None"):
            continue
        si, es = int(p[col["scene_idx"]]), int(raw)
        prev = seeds.setdefault(si, es)
        if prev != es:
            raise SystemExit(
                f"{index_tsv}: instruction={instr} s{si} 의 env_seed 가 두 값 ({prev}, {es})")
    if not seeds:
        raise SystemExit(f"{index_tsv}: instruction={instr} base 행 없음")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="예: OvenRack_out")
    ap.add_argument("--plan-json", type=Path, required=True)
    ap.add_argument("--index-tsv", type=Path, default=None)
    ap.add_argument("--n-scenes", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None, help="미지정 시 stdout")
    args = ap.parse_args()

    instr, plan_seeds, env_name, text = from_plan(args.plan_json, args.slug)
    seeds = plan_seeds
    source = "plan"
    if args.index_tsv is not None:
        idx_seeds = from_index(args.index_tsv, instr)
        bad = {si: (plan_seeds.get(si), es) for si, es in idx_seeds.items()
               if si in plan_seeds and plan_seeds[si] != es}
        if bad:
            raise SystemExit(
                f"index 와 plan 의 env_seed 불일치 {bad} — 같은 scene 을 도는지 확인 필요")
        seeds = idx_seeds
        source = "index"

    rows = sorted(seeds.items())[: args.n_scenes]
    if len(rows) < args.n_scenes:
        raise SystemExit(
            f"scene {len(rows)}개 < 요청 {args.n_scenes} (source={source}, instr={instr}) "
            "— 수집이 덜 됐거나 --n-scenes 를 낮춰야 한다")
    task = derive_task(env_name)
    body = "".join(f"{si}\t{es}\t{task}\t{env_name}\t{text}\n" for si, es in rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        print(f"[scenes] {args.slug}: {len(rows)} scene (source={source}, instr={instr}) "
              f"→ {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
