"""patchceil pass B 대상 선정 (사전등록 규칙, 결정적) — donor 4 + placebo/sham 4 per cell.

규칙 (PROTOCOL.md §donor 선정과 동일, 결과 보기 전 동결):
- donor(성공) 4판/cell: grasp·insert-settle 모두 도달한 성공 중 n_records 가 성공 중앙값에
  가까운 순 4판 (동률은 episode_idx 오름차순). 근거: '전형적' 성공 재생 + Gate 1 §채택6
  donor 다양성 (개별 배정은 러너에서 target ep 순환).
- placebo-fail 4판/cell: 실패 중 episode_idx 균등 간격 4판 (index 0, n/3, 2n/3, n-1 위치).
  이 4판이 sham 자기-donor 후보를 겸한다 (sham 은 이 중 앞 2판).

출력: patchceil/<cell>/passB_manifest.tsv (cell, episode_idx, succ, role, inference_seed)
stdlib 전용.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
GROOT = REPO / "outputs/eval/robocasa/groot_n15"
META = GROOT / "patchceil/patchceil_meta"
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]
SEEDS = {"ppcc_bread_s300033": 300033, "ppcc_bread_s400020": 400020}
N_DONOR = 4
N_PLACEBO = 4


def load(cell: str) -> list[dict]:
    eps = []
    for f in sorted((META / cell).glob("ep*.json")):
        if "_actions" in f.name:
            continue
        eps.append(json.loads(f.read_text()))
    assert len(eps) == 60, f"{cell}: meta {len(eps)} != 60"
    return eps


def main() -> None:
    for cell in CELLS:
        eps = load(cell)
        succ = [d for d in eps if int(d["episode_success"]) == 1]
        fail = [d for d in eps if int(d["episode_success"]) == 0]

        # donor: grasp·insert-settle 도달 성공, 중앙값 길이 근접 4
        ok = [
            d for d in succ
            if "grasp" in d["feature_phases"] and "insert-settle" in d["feature_phases"]
        ]
        lens = sorted(d["n_records"] for d in ok)
        med = lens[len(lens) // 2]
        donors = sorted(ok, key=lambda d: (abs(d["n_records"] - med), d["episode_idx"]))[:N_DONOR]

        # placebo: 실패 episode_idx 오름차순에서 균등 간격 4
        fail_sorted = sorted(fail, key=lambda d: d["episode_idx"])
        n = len(fail_sorted)
        idxs = sorted({0, n // 3, (2 * n) // 3, n - 1})
        placebos = [fail_sorted[i] for i in idxs]

        rows = []
        for d in donors:
            rows.append((d["episode_idx"], 1, "donor"))
        for j, d in enumerate(placebos):
            role = "placebo+sham" if j < 2 else "placebo"
            rows.append((d["episode_idx"], 0, role))
        out = GROOT / "patchceil" / cell / "passB_manifest.tsv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["cell", "episode_idx", "succ", "role", "inference_seed"])
            for ep, s, role in sorted(rows):
                w.writerow([cell, ep, s, role, ep * 1000])
        print(f"{cell}: donor={[d['episode_idx'] for d in donors]} "
              f"(len med={med}) placebo={[d['episode_idx'] for d in placebos]} -> {out.name}")


if __name__ == "__main__":
    main()
