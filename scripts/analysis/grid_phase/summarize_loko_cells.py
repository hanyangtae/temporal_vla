#!/usr/bin/env python
"""loko-cell detector 산출물 → **셀 한 줄** 요약 TSV (집계 조인용).

`failure_detector_sim.py --arm loko-cell` 이 slug 별로 흩어 놓은 `sim_summary.tsv`
(셀당 eval_set 3행 × α + timer 행)와 `cell_registry.tsv` 를 셀 키로 접어
(instruction, scene, jitter) 하나에 한 행을 만든다.

지표 읽는 법 (이 파일을 쓰는 쪽이 반드시 알아야 하는 것):
  - `td10_holdout` = **무편향** 셀 내 판별력. 타 j 만으로 학습한 2차 모델로 대상 j 를
    채점한 값(action phase 의 leave-one-jitter-out 과 같은 층위). **이게 정본이다.**
  - `td10_insample` = 배포 모델(대상 j 실패판이 학습에 포함)의 같은 지표. 낙관 편향이
    있으니 성능으로 인용하지 말 것 — 두 값의 차이가 곧 편향 크기다.
  - `tpr_target_fail` 은 in-sample 이라 언제나 1.0 에 가깝다. 성능 아님.
  - `fpr_target_succ` = 대상 j 성공판 오경보. 밴드가 타 j 성공으로만 보정되므로
    j 간 분포 차가 여기서 드러난다(1.00 이면 사실상 j 감지기 = 게이팅 가치 없음).
  - `n_target_succ` 0 이면 판별력·오경보 둘 다 측정 불가(빈 값).

사용:
    python scripts/analysis/grid_phase/summarize_loko_cells.py \
        --root outputs/analysis/grid_phase/detector_v6 \
        --alpha 0.1 --out outputs/analysis/grid_phase/detector_v6/cells_summary.tsv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

COLS = ["instruction", "stem", "scene", "jitter", "registered", "reason",
        "n_pool_other", "n_pool_fail", "n_target_fail", "n_target_succ",
        "n_succ_calib", "td10_holdout", "td10_insample", "auroc_target_j_max",
        "fire_p25", "fire_p50", "fire_p75", "n_fired_fail",
        "fpr_target_succ", "tpr_target_fail", "timer_fire_p50",
        "auroc_jonly_scene", "length_auroc", "ckpt_rel"]


def _v(row: dict, key: str) -> str:
    v = row.get(key, "")
    return "" if v in ("None", None) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="detector_v6 루트 (slug 디렉터리들의 부모)")
    ap.add_argument("--alpha", default="0.1", help="요약에 쓸 CP 유의수준 (기본 0.1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    want_a = f"{float(args.alpha):g}"

    # 1) registry 를 셀 키로 모은다 (미등록 셀도 포함 — 무음 탈락 금지)
    cells: dict[tuple, dict] = {}
    for p in sorted(glob.glob(str(root / "*" / "cell_registry.tsv"))):
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            if _v(r, "reason") == "shard_not_loaded":
                continue          # 그 slug 실행에서 안 본 셀 — 다른 파일이 채운다
            key = (_v(r, "slug") or _v(r, "instruction"), _v(r, "scene"), _v(r, "jitter"))
            cells[key] = {
                "instruction": _v(r, "instruction"), "stem": _v(r, "slug"),
                "scene": _v(r, "scene"), "jitter": _v(r, "jitter"),
                "registered": _v(r, "registered"), "reason": _v(r, "reason"),
                "n_pool_other": _v(r, "n_pool_other"), "n_pool_fail": _v(r, "n_pool_fail"),
                "n_target_fail": _v(r, "n_target_fail"),
                "n_target_succ": _v(r, "n_target_succ"),
                "n_succ_calib": _v(r, "n_succ_calib"), "ckpt_rel": _v(r, "ckpt_rel"),
            }

    # 2) sim_summary 에서 셀 단위 지표를 채운다
    for p in sorted(glob.glob(str(root / "*" / "sim_summary.tsv"))):
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            if _v(r, "eval_set") == "" or _v(r, "skip_reason"):
                continue
            key = (_v(r, "task"), _v(r, "scene"), _v(r, "jitter"))
            c = cells.get(key)
            if c is None:
                continue
            model, eset = _v(r, "model"), _v(r, "eval_set")
            a = _v(r, "alpha")
            if model == "timer":
                if eset == "target_j_fail":
                    c["timer_fire_p50"] = _v(r, "t_fire_p50")
                continue
            if model != "lstm" or (a and f"{float(a):g}" != want_a):
                continue
            # 셀 단위 상수(어느 eval_set 행에서 읽어도 같다)
            c.setdefault("td10_holdout", _v(r, "auroc_target_j_holdout_td10"))
            c.setdefault("td10_insample", _v(r, "auroc_target_j_td10"))
            c.setdefault("auroc_target_j_max", _v(r, "auroc_target_j"))
            c.setdefault("auroc_jonly_scene", _v(r, "auroc_jonly_scene"))
            if eset == "target_j_fail":
                c["fire_p25"], c["fire_p50"], c["fire_p75"] = (
                    _v(r, "t_fire_p25"), _v(r, "t_fire_p50"), _v(r, "t_fire_p75"))
                c["n_fired_fail"] = _v(r, "n_fired")
                c["tpr_target_fail"] = _v(r, "tpr")
                c["length_auroc"] = _v(r, "length_auroc")
            elif eset == "target_j_succ":
                c["fpr_target_succ"] = _v(r, "fpr")

    rows = [ {k: c.get(k, "") for k in COLS}
             for _, c in sorted(cells.items(),
                                key=lambda kv: (kv[0][0], int(kv[0][1] or 0),
                                                int(kv[0][2] or 0))) ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    n_reg = sum(1 for r in rows if r["registered"] == "1")
    n_meas = sum(1 for r in rows if r["td10_holdout"])
    print(f"[cells] {len(rows)} 셀 → {out} (등록 {n_reg} / 무편향 지표 측정가능 {n_meas})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
