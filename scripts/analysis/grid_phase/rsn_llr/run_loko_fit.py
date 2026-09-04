"""LOKO fit 증분 러너 — 중추 셀표(v6_loko_cells.tsv)를 읽어 (키, scene)별로 fit 을 돌린다.

셀표 행 = 선정 셀의 실패 판이라 `(grid_instruction, scene_idx, jitter_idx)` distinct 가
fit 대상이다. 이 스크립트는 그 조합을 (키, scene) 로 묶어 지터 목록을 만들고, shard 가
올라온 키만 골라 원격(승준)에서 `loko_fit.py` 를 실행한다.

- 기본은 **--only-gt**: GT phase 라벨은 AE 번들과 무관하므로, 완료 instruction 의 shard
  로 만든 gt 산출물은 임시가 아니라 최종이다. ck8 은 정식 번들(`ae_k8/`) 이후 `--ck8` 로.
- 완주 판정은 로그의 `V5_LOKO_DONE <slug> s<scene>` 센티넬(셀 수 대조).
- 원격 CPU cap 8(승준 규약). 코드는 ssh stdin 으로 넣으므로 원격 repo 를 건드리지 않는다.

usage:
  python run_loko_fit.py --cells configs/collect/n15_grid_v6_scene_jitter/v6_loko_cells.tsv
                         [--only-slugs PPCC_bread,OpenDrawer_left] [--ck8] [--dry-run]
"""
import argparse
import collections
import csv
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
FIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loko_fit.py")
SSH = ["ssh", "-p", "11112", "kimseungjun@166.104.146.37"]
SEG = "~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/segA"
SENTINEL = "V5_LOKO_DONE"        # loko_fit.py 가 (키, scene) 완주 시 찍는 문자열


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="중추 셀표 TSV")
    ap.add_argument("--only-slugs", default=None, help="쉼표 목록으로 한정")
    ap.add_argument("--ck8", action="store_true",
                    help="ck8 도 산출(정식 AE 번들 이후에만). 기본은 gt 전용")
    ap.add_argument("--bundle", default=None, help="AE 번들 경로 override(임시본 사용 시)")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--coord", default="j")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    rows = list(csv.DictReader(open(args.cells), delimiter="\t"))
    if not rows:
        raise SystemExit(f"{args.cells}: 행 0")
    need = ("grid_instruction", "slug", "scene_idx", "jitter_idx")
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise SystemExit(f"{args.cells}: 열 없음 {miss} (있는 열 {sorted(rows[0])})")

    cells = collections.defaultdict(set)
    for r in rows:
        cells[(r["slug"], int(r["scene_idx"]))].add(int(r["jitter_idx"]))
    only = set(args.only_slugs.split(",")) if args.only_slugs else None

    # shard 가 올라온 것만
    have = subprocess.run(SSH + [f"ls {SEG}/*.npz 2>/dev/null"],
                          capture_output=True, text=True).stdout.split()
    have = {os.path.basename(p)[:-4] for p in have}
    print(f"[shard] 승준 segA 에 있는 slug {len(have)}개: {sorted(have)}")

    todo, skip = [], []
    for (slug, scene), js in sorted(cells.items()):
        if only and slug not in only:
            continue
        (todo if slug in have else skip).append((slug, scene, sorted(js)))
    for slug, scene, js in skip:
        print(f"[대기] {slug} s{scene} j{js} — shard 미도착")
    if not todo:
        print("[run] 돌릴 셀 없음")
        return 0

    fail = 0
    for slug, scene, js in todo:
        cmd = (f"export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; "
               f"~/anaconda3/bin/python - {slug} {scene} --tag {args.tag} "
               f"--coord {args.coord} --jitters {','.join(map(str, js))}")
        if not args.ck8:
            cmd += " --only-gt"
        elif args.bundle:
            cmd += f" --bundle {args.bundle}"
        print(f"\n=== {slug} s{scene} j{js} ({'gt+ck8' if args.ck8 else 'gt'}) ===")
        if args.dry_run:
            print("  [dry-run]", cmd)
            continue
        with open(FIT) as f:
            p = subprocess.run(SSH + [cmd], stdin=f, capture_output=True, text=True)
        out = p.stdout + p.stderr
        print(out.rstrip())
        if f"{SENTINEL} {slug} s{scene}" not in out:
            print(f"  [fail] 센티넬 없음 — rc={p.returncode}")
            fail += 1
    print(f"\n[run] 대상 {len(todo)} · 실패 {fail} · 대기 {len(skip)}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
