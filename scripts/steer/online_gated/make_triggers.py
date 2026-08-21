#!/usr/bin/env python3
"""러너 job 의 per_episode.tsv → TRIGGER_TSV (scene/noise/trigger 3열) 투영기.

`run_online_gated_eval.sh` 는 replay oracle arm 에서 셀별 발화 시점을 TRIGGER_TSV
(헤더 없는 `scene\\tnoise\\ttrigger` 3열) 로 읽는다 (해당 파일의 awk 소비 지점 참조).
그런데 이 TSV 를 만드는 작성기가 레포에 없어 그동안 매번 즉석 awk/손편집으로
때웠다 — 이 스크립트가 그 빈틈을 메운다.

입력은 `collect_results.py` 가 쓴 per_episode.tsv (열 이름 정본도 그 파일의
COLUMNS). 여기서 scene_idx / noise_idx / trigger_step 만 뽑아 정렬해 내보낸다.
발화가 없는 행(trigger_step 이 빈칸·NA·None)은 제외 — 그 셀은 트리거 없음이므로
러너의 awk 조회가 빈 문자열을 받게 두는 것이 맞다.

사용:
    python3 scripts/steer/online_gated/make_triggers.py \
        --per-episode <job>/per_episode.tsv --out <logs>/triggers_<slug>.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

MISSING = {"", "NA", "None", "none", "nan"}


def build_rows(per_episode: Path) -> tuple[list[tuple[int, int, str]], int]:
    """per_episode.tsv → [(scene, noise, trigger)] (정렬됨) 과 전체 행 수."""
    with per_episode.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for col in ("scene_idx", "noise_idx", "trigger_step"):
            if reader.fieldnames is None or col not in reader.fieldnames:
                raise SystemExit(f"per_episode.tsv 에 {col} 열이 없음 "
                                 f"(열: {reader.fieldnames})")
        rows: list[tuple[int, int, str]] = []
        total = 0
        for rec in reader:
            total += 1
            trig = (rec.get("trigger_step") or "").strip()
            if trig in MISSING:
                continue
            scene = (rec.get("scene_idx") or "").strip()
            noise = (rec.get("noise_idx") or "").strip()
            if scene in MISSING or noise in MISSING:
                continue
            rows.append((int(scene), int(noise), trig))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-episode", type=Path, required=True,
                    help="collect_results.py 산출 per_episode.tsv")
    ap.add_argument("--out", type=Path, required=True,
                    help="출력 TRIGGER_TSV (헤더 없는 scene/noise/trigger 3열)")
    args = ap.parse_args()

    if not args.per_episode.exists():
        raise SystemExit(f"per_episode.tsv 없음: {args.per_episode}")
    rows, total = build_rows(args.per_episode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerows(rows)
    print(f"발화 {len(rows)}/총 {total} 셀 → {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
