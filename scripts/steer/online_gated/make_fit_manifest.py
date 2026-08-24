#!/usr/bin/env python3
"""index_rollouts_v4.tsv → (task, base scene) 별 fit episode 매니페스트.

v4 격자는 plan_id 디렉토리가 여럿(v4 지터 + v2/v1 base 재사용)이라 (scene_idx, noise_idx)
튜플로 셀을 고르면 **평탄 si 충돌**로 조용히 섞인다 (v2 base scene1 = s1, v4 scene0/k1 = s1).
그래서 fit 입력은 index 의 rel_path(plan_id 포함 → 유일)로 명시 나열한다.
출력 = fit_cond_guidance --episode-manifest 계약: pkl_path·label·base_scene·cell_si·noise_idx.

사용:
    python3 make_fit_manifest.py --index outputs/.../index_rollouts_v4.tsv \
        --instruction OpenDrawer/left --out-dir outputs/steer/online_pipe/manifests/v4 \
        [--min-per-class 8] [--scenes 3,4]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

NEED = ("grid_instruction", "scene_idx", "noise_idx", "cell_si", "jitter_reset_idx",
        "success", "rel_path", "machine", "plan_id")
TRUE = ("1", "True", "true")


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
            fh.write("pkl_path\tlabel\tbase_scene\tcell_si\tnoise_idx\n")
            for r in sorted(eps, key=lambda x: (int(x["cell_si"]), int(x["noise_idx"]))):
                lab = 1 if r["success"] in TRUE else 0
                fh.write(f"{r['rel_path'].rstrip('/')}/rollout.pkl\t{lab}\t{sc}"
                         f"\t{r['cell_si']}\t{r['noise_idx']}\n")
        kept.append((sc, len(eps), succ, fail, out))
        print(f"[ok] {slug} s{sc}: {len(eps)}판 (succ={succ} fail={fail}) → {out}")
    if not kept:
        raise SystemExit(f"{slug}: 하한 통과 scene 없음")
    print(f"총 {len(kept)} scene 매니페스트 생성")


if __name__ == "__main__":
    main()
