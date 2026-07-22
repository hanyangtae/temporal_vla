"""patchceil 대상 manifest 생성 — 실패(target)·성공(donor 후보) 목록.

두 소스를 각각 manifest 로 만든다 (재실행 안전, stdlib 전용 — 로컬 python3 numpy 부재):

1. **fit 수집분 ep0-59 (주 대상, targets_fit.tsv)** — 승준 HDD 에 pkl.zst 60판/cell 실존
   확인(2026-07-16, NOTICE "유실 5-cell" 판정 정정 — `.pkl` 만 세고 `.pkl.zst` 를 놓친
   검증 구멍). actions(full chunk)·feature_phases·phase_timeline 동봉이라 발산점 라벨·
   action-replay arm 재료가 이미 있음 → GPU 재수집(구 pass A) 불필요.
   입력: patchceil/remote_fit_listing.txt (원격 ls 스냅샷, "cell task5--epN--succS" 행).
2. **ho_base ep60-119 (보조, targets.tsv)** — per-episode 판정은 남았으나 pkl purge 로
   actions 소실. 입력: steer_eval/<cell>/ho_base/raw_rollouts/collection_summary.tsv.

출력 컬럼: cell, episode_idx, scenario_seed, inference_seed(=ep*1000 — heldout_round_cell.sh
STRIDE=1000, fit 수집 HANDOFF §39-44 동일 공식·원격 pkl 실측 22000=ep22 확인), succ,
role(target|donor_pool).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import re

REPO = Path(__file__).resolve().parents[6]
GROOT = REPO / "outputs/eval/robocasa/groot_n15"
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]
STRIDE = 1000  # heldout_round_cell.sh: --inference-seed $((ep * STRIDE))
SEEDS = {"ppcc_bread_s300033": 300033, "ppcc_bread_s400020": 400020}
# ho_base(ep60-119) 실측 기대값 (steer_eval_pq2/aggregate_v2/matrix.md 대조) — 어긋나면 중단
EXPECT = {"ppcc_bread_s300033": (25, 60), "ppcc_bread_s400020": (21, 60)}
# fit 수집분(ep0-59) 실측 기대값 (승준 HDD 파일명 집계, 2026-07-16)
EXPECT_FIT = {"ppcc_bread_s300033": (20, 60), "ppcc_bread_s400020": (23, 60)}
STEM_RE = re.compile(r"^task5--ep(\d+)--succ([01])$")


def write_manifest(cell: str, out_name: str, out_rows: list[dict], expect: tuple[int, int]) -> None:
    n_succ = sum(r["succ"] for r in out_rows)
    assert (n_succ, len(out_rows)) == expect, (
        f"{cell}/{out_name}: SR {n_succ}/{len(out_rows)} != 기대 {expect}"
    )
    out = GROOT / "patchceil" / cell / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out_rows.sort(key=lambda r: r["episode_idx"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(
        f"{cell}: targets={len(out_rows) - n_succ} donor_pool={n_succ} "
        f"-> {out.relative_to(REPO)}"
    )


def make_fit_manifests() -> None:
    """주 대상: fit 수집분 ep0-59 (승준 원격 listing 스냅샷 기반)."""
    listing = GROOT / "patchceil" / "remote_fit_listing.txt"
    rows_by_cell: dict[str, list[dict]] = {c: [] for c in CELLS}
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        cell, stem = line.split()
        m = STEM_RE.match(stem)
        assert m, f"stem 파싱 실패: {line}"
        ep, succ = int(m.group(1)), int(m.group(2))
        rows_by_cell[cell].append(
            {
                "cell": cell,
                "episode_idx": ep,
                "scenario_seed": SEEDS[cell],
                "inference_seed": ep * STRIDE,
                "succ": succ,
                "role": "donor_pool" if succ else "target",
            }
        )
    for cell in CELLS:
        write_manifest(cell, "targets_fit.tsv", rows_by_cell[cell], EXPECT_FIT[cell])


def main() -> int:
    for cell in CELLS:
        src = GROOT / "steer_eval" / cell / "ho_base/raw_rollouts/collection_summary.tsv"
        rows = list(csv.DictReader(open(src), delimiter="\t"))
        out_rows = []
        n_succ = 0
        for r in rows:
            stem = Path(r["pkl"]).name
            assert "--succ0" in stem or "--succ1" in stem, stem
            succ = 1 if "--succ1" in stem else 0
            n_succ += succ
            ep = int(r["episode_idx"])
            out_rows.append(
                {
                    "cell": cell,
                    "episode_idx": ep,
                    "scenario_seed": int(r["seed"]),
                    "inference_seed": ep * STRIDE,
                    "succ": succ,
                    "role": "donor_pool" if succ else "target",
                }
            )
        seeds = {r["scenario_seed"] for r in out_rows}
        assert seeds == {SEEDS[cell]}, f"{cell}: scenario_seed 불일치 {seeds}"
        del n_succ  # write_manifest 가 재계산·검증
        write_manifest(cell, "targets.tsv", out_rows, EXPECT[cell])
    make_fit_manifests()
    return 0


if __name__ == "__main__":
    sys.exit(main())
