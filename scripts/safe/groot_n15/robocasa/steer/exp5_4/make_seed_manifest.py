#!/usr/bin/env python3
"""exp5-4 Phase A 신규 noise seed manifest 생성 (사전 등록·재현 가능).

Codex Gate1 리뷰 반영: 8e6~15e6 같은 "scene 공유 8종" 을 폐기하고 **scene 별 고유
base seed** 를 쓴다. 공유 seed 는 draw 가 scene 을 가로질러 동일해져 "이 노이즈가 다른
scene 에서 이미 나빴다" 는 암기 경로를 남긴다 (계획서 §0 헤드라인 주의).

계약
  · RNG = numpy default_rng(424101) 고정 → 같은 인자면 같은 manifest.
  · base seed 는 [20_000_000, 2**31 - 10**6) 에서 추첨.
  · 한 rollout 은 call 마다 inference_seed = base + record 를 쓰므로 base 가 점유하는
    구간은 [base, base + GUARD). 모든 신규 구간은 (a) 서로, (b) 옛 seed 구간
    {0, 1e6, ..., 7e6} + [0, GUARD) 와 겹치지 않는다.
  · 출력 TSV 열: scene_idx scene cand_idx base_seed episode_idx probe_order rollout_order
      - probe_order   : scene 내 probe 호출 순서 (0..k-1 순열)
      - rollout_order : 전체 rollout 실행 순서 (0..N-1 전역 순열)
      - episode_idx   : scene_idx * k + cand_idx (출력 스템 고정 — 셔플과 무관)
  · 마지막에 파일 sha256 을 stdout 에 출력 (러너·문서에 봉인용).

사용:
  python scripts/safe/groot_n15/robocasa/steer/exp5_4/make_seed_manifest.py \
      --out outputs/eval/robocasa/groot_n15/exp5_4/seed_manifest.tsv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

RNG_SEED = 424101
GUARD = 200               # rollout 당 최대 record 수(720/5=144) + 여유
LO = 20_000_000
HI = 2**31 - 10**6
OLD_BASES = [k * 1_000_000 for k in range(8)]   # 기존 scene-matched 수집 (0~7e6)

# drawer_right 20 scene — exp5_3/beta_sweep.sh SCENES 와 동일 (라틴 40셀 근거 목록)
DRAWER_SCENES = [
    100000, 100003, 100005, 100006, 100009, 100010, 100011, 100012, 100016, 100018,
    100020, 100022, 100023, 100025, 100026, 100031, 100033, 100034, 100035, 100039,
]


def _overlaps(base: int, taken: list[int]) -> bool:
    """[base, base+GUARD) 가 taken 의 어떤 구간과도 겹치는지."""
    return any(abs(base - other) < GUARD for other in taken)


def draw_bases(n: int, rng: np.random.Generator) -> list[int]:
    taken = list(OLD_BASES)
    out: list[int] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 100 * n:
            raise RuntimeError("base seed 추첨 실패 (제약 과다)")
        base = int(rng.integers(LO, HI))
        if _overlaps(base, taken):
            continue
        taken.append(base)
        out.append(base)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-cand", type=int, default=8, help="scene 당 후보 draw 수 (기본 8)")
    ap.add_argument(
        "--scenes",
        default=None,
        help="콤마 구분 scenario_seed 목록 (기본 = drawer_right 20 scene)",
    )
    args = ap.parse_args()

    scenes = (
        DRAWER_SCENES
        if args.scenes is None
        else [int(tok) for tok in args.scenes.split(",") if tok.strip() != ""]
    )
    k = args.n_cand
    n = len(scenes) * k
    rng = np.random.default_rng(RNG_SEED)
    bases = draw_bases(n, rng)

    # 순서 셔플: scene 내 probe 순서 + 전역 rollout 순서
    probe_orders = [rng.permutation(k) for _ in scenes]
    rollout_order = rng.permutation(n)

    rows = []
    for s_idx, scene in enumerate(scenes):
        # probe_orders[s_idx][j] = j 번째 후보가 몇 번째로 probe 되는지
        for c_idx in range(k):
            flat = s_idx * k + c_idx
            rows.append(
                (
                    s_idx,
                    scene,
                    c_idx,
                    bases[flat],
                    flat,                       # episode_idx
                    int(probe_orders[s_idx][c_idx]),
                    int(rollout_order[flat]),
                )
            )

    # 사후 검증 (사전등록 계약이 실제로 지켜졌는지 — 산출 직후 자기검사)
    all_bases = [r[3] for r in rows]
    assert len(set(all_bases)) == n, "base seed 중복"
    ordered = sorted(all_bases + OLD_BASES)
    for a, b in zip(ordered, ordered[1:]):
        assert b - a >= GUARD, f"seed 구간 겹침: {a} {b}"
    for s_idx in range(len(scenes)):
        assert sorted(r[5] for r in rows if r[0] == s_idx) == list(range(k))
    assert sorted(r[6] for r in rows) == list(range(n))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["scene_idx\tscene\tcand_idx\tbase_seed\tepisode_idx\tprobe_order\trollout_order"]
    lines += ["\t".join(str(v) for v in row) for row in rows]
    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {args.out}: scenes={len(scenes)} cand={k} rows={n}")
    print(f"rng_seed={RNG_SEED} guard={GUARD} range=[{LO},{HI})")
    print(f"sha256={sha}")


if __name__ == "__main__":
    main()
