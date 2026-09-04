#!/usr/bin/env python3
"""grid 인덱스(index_rollouts_v6.tsv 등) → (task, scene) 별 fit episode 매니페스트.

셀 좌표는 3축 `scene_idx`·**지터 좌표**·`noise_idx` 다 (docs/04 §3.1.1).
지터 좌표는 인덱스 판마다 다르다:
  - **v6**: `jitter_idx`(j) — 경로 `j<jid>` 층. `jitter_reset_idx` 는 plan 유래 출처 열일
    뿐 좌표가 아니다.
  - **legacy(v5 k 층)**: `jitter_reset_idx`(k) 가 좌표. 2축 legacy 는 빈 값 = base.
plan_id 디렉토리가 여럿 섞인 격자에서 좌표 튜플만으로는 판을 특정할 수 없으므로,
fit 입력은 index 의 rel_path(plan_id 포함 → 유일)로 명시 나열한다.
출력 = fit_cond_guidance --episode-manifest 계약(헤더 기반 파싱):
pkl_path·label·base_scene·cell_si·noise_idx·jitter_idx·jitter_reset_idx.
`cell_si` 는 **저장 좌표가 아니라 파생 평탄값** (`scene*100 + 지터좌표`, base = `+99`)
으로, 구 매니페스트와 값이 같도록 유지한다 (fit_cond_guidance 의 provenance 열).

사용:
    python3 make_fit_manifest.py --index outputs/.../index_rollouts_v5.tsv \
        --instruction OpenDrawer/left --out-dir outputs/steer/online_pipe/manifests/v4 \
        [--min-per-class 8] [--scenes 3,4]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

NEED = ("grid_instruction", "scene_idx", "noise_idx", "success", "rel_path",
        "machine", "plan_id")
JIT_COLS = ("jitter_idx", "jitter_reset_idx")   # 둘 중 하나는 있어야 한다
TRUE = ("1", "True", "true")
BASE_K = 99          # 지터 축 없는 행의 좌표 자리 — 파생 평탄값 규약 (docs/04 §3.1.1)


def _int_or_base(raw: str | None) -> int:
    v = (raw or "").strip()
    return BASE_K if v == "" or v.lower() in ("base", "na", "none") else int(v)


def k_of(row: dict, is_v6: bool) -> int:
    """index 행 → 지터 좌표 정수 (v6 = jitter_idx j, legacy = jitter_reset_idx k).

    지터 축이 없는 행(빈 값/`"base"`)은 99.
    """
    if is_v6:
        j = _int_or_base(row.get("jitter_idx"))
        # v6 인덱스 안의 legacy 행(j 없음)은 reset_idx 를 좌표로 되돌려 쓴다.
        return j if j != BASE_K else _int_or_base(row.get("jitter_reset_idx"))
    return _int_or_base(row.get("jitter_reset_idx"))


def reset_of(row: dict, is_v6: bool) -> str:
    """reset 횟수 출처 열. v6 = plan 유래 값(없으면 "base"), legacy = 구 매니페스트와
    같은 정수 표기(base=99)를 유지한다."""
    if not is_v6:
        return str(k_of(row, False))
    return (row.get("jitter_reset_idx") or "").strip() or "base"


def cell_si_of(row: dict, is_v6: bool) -> int:
    """파생 평탄 cell id = scene*100 + 지터좌표 (base = scene*100 + 99)."""
    return int(row["scene_idx"]) * 100 + k_of(row, is_v6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--instruction", required=True, help="grid_instruction 값 (예 OpenDrawer/left)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-per-class", type=int, default=8, help="클래스당 경성 하한 (45 §3)")
    ap.add_argument("--scenes", default=None, help="base scene 제한 (예 3,4). 기본=하한 통과 전부")
    args = ap.parse_args()

    with args.index.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("grid_instruction") == args.instruction]
    if not rows:
        raise SystemExit(f"instruction={args.instruction} 행 없음: {args.index}")
    missing = [c for c in NEED if c not in rows[0]]
    if missing:
        raise SystemExit(f"index 열 누락: {missing}")
    if not any(c in rows[0] for c in JIT_COLS):
        raise SystemExit(f"index 에 지터 열이 없다 (필요: {JIT_COLS[0]} 또는 {JIT_COLS[1]})")
    is_v6 = "jitter_idx" in rows[0]
    print(f"[index] {'v6 (jitter_idx=j 좌표)' if is_v6 else 'legacy (jitter_reset_idx=k 좌표)'}")

    want = None if args.scenes is None else {int(x) for x in args.scenes.split(",") if x.strip()}
    by_scene: dict[int, list[dict]] = {}
    for r in rows:
        sc = int(r["scene_idx"])
        if want is not None and sc not in want:
            continue
        by_scene.setdefault(sc, []).append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.instruction.replace("/", "_")
    kept = []
    for sc in sorted(by_scene):
        eps = by_scene[sc]
        succ = sum(1 for r in eps if r["success"] in TRUE)
        fail = len(eps) - succ
        if min(succ, fail) < args.min_per_class:
            print(f"[skip] {slug} s{sc}: succ={succ} fail={fail} < 하한 {args.min_per_class}")
            continue
        out = args.out_dir / f"{slug}_s{sc}.tsv"
        with out.open("w", encoding="utf-8") as fh:
            fh.write("pkl_path\tlabel\tbase_scene\tcell_si\tnoise_idx"
                     "\tjitter_idx\tjitter_reset_idx\n")
            for r in sorted(eps, key=lambda x: (k_of(x, is_v6), int(x["noise_idx"]))):
                lab = 1 if r["success"] in TRUE else 0
                fh.write(f"{r['rel_path'].rstrip('/')}/rollout.pkl\t{lab}\t{sc}"
                         f"\t{cell_si_of(r, is_v6)}\t{r['noise_idx']}"
                         f"\t{k_of(r, is_v6)}\t{reset_of(r, is_v6)}\n")
        kept.append((sc, len(eps), succ, fail, out))
        print(f"[ok] {slug} s{sc}: {len(eps)}판 (succ={succ} fail={fail}) → {out}")
    if not kept:
        raise SystemExit(f"{slug}: 하한 통과 scene 없음")
    print(f"총 {len(kept)} scene 매니페스트 생성")


if __name__ == "__main__":
    main()
