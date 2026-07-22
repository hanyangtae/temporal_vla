#!/bin/bash
# P1 잔여 체인 (승준에서 실행): 신규 apple 3 scene split(--pkl-meta-labels) → per_scene fits
# → pool manifests(cross_scene apple/bread, grand ×fit{15,30}, null1) → pool fits → 후보 JSON.
# env: PYBIN(=REMOTE_PYTHON)
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$REPO"
PYBIN=${PYBIN:-${REMOTE_PYTHON:-python3}}
G=scripts/safe/groot_n15/robocasa/steer/exp2/make_fit_manifests.py
B=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts
M=outputs/eval/robocasa/groot_n15/pq2_manifests  # 원격(승준) 실행 — 원격은 구명(pq) 유지
NEW="ppcs_apple_s100353 ppcs_apple_s100395 ppcs_apple_s100422"

echo "=== 1) 신규 scene split (pkl-meta 라벨, discordant 제외) ==="
for c in $NEW; do
  "$PYBIN" "$G" scene --scene-dir "$B/PickPlaceCounterToStove/$c" --pkl-meta-labels \
    --out-dir "$M/$c" --null-seed 1 || exit 1
done

echo "=== 2) pool manifests ==="
mkdir -p "$M/_pools"
"$PYBIN" "$G" pool --scope cross_scene --fit-n 15 --scene-min 3 --null-seed 1 \
  --scene-manifest-dirs $M/ppcs_apple $M/ppcs_apple_s100353 $M/ppcs_apple_s100395 $M/ppcs_apple_s100422 \
  --out "$M/_pools/cross_scene_apple_fit15.tsv" || exit 1
"$PYBIN" "$G" pool --scope cross_scene --fit-n 30 --scene-min 3 --null-seed 1 \
  --scene-manifest-dirs $M/ppcs_apple $M/ppcs_apple_s100353 $M/ppcs_apple_s100395 $M/ppcs_apple_s100422 \
  --out "$M/_pools/cross_scene_apple_fit30.tsv" || exit 1
"$PYBIN" "$G" pool --scope cross_scene --fit-n 15 --scene-min 3 --null-seed 1 \
  --scene-manifest-dirs $M/ppcc_bread $M/ppcc_bread_s300028 $M/ppcc_bread_s300033 $M/ppcc_bread_s400020 \
  --out "$M/_pools/cross_scene_bread_fit15.tsv" || exit 1
"$PYBIN" "$G" pool --scope cross_scene --fit-n 30 --scene-min 3 --null-seed 1 \
  --scene-manifest-dirs $M/ppcc_bread $M/ppcc_bread_s300028 $M/ppcc_bread_s300033 $M/ppcc_bread_s400020 \
  --out "$M/_pools/cross_scene_bread_fit30.tsv" || exit 1
ALL8="$M/ppcc_bread $M/ppcc_bread_s300028 $M/ppcc_bread_s300033 $M/ppcc_bread_s400020 $M/ppcs_apple $M/ppcs_apple_s100353 $M/ppcs_apple_s100395 $M/ppcs_apple_s100422"
"$PYBIN" "$G" pool --scope grand --fit-n 15 --scene-min 1 --null-seed 1 --scene-manifest-dirs $ALL8 \
  --out "$M/_pools/grand_fit15.tsv" || exit 1
"$PYBIN" "$G" pool --scope grand --fit-n 30 --scene-min 1 --null-seed 1 --scene-manifest-dirs $ALL8 \
  --out "$M/_pools/grand_fit30.tsv" || exit 1

echo "=== 3) fits (신규 scene + pools, JOBS=1) ==="
MFS=""
for c in $NEW; do for f in "$M/$c"/per_scene_fit*.tsv; do case "$f" in *_null*) ;; *) MFS="$MFS $f";; esac; done; done
for f in "$M/_pools"/*.tsv; do case "$f" in *_null*) ;; *) MFS="$MFS $f";; esac; done
PYBIN="$PYBIN" JOBS=1 OMP_NUM_THREADS=4 bash scripts/safe/groot_n15/robocasa/steer/exp2/p1_screen.sh $MFS || exit 1

echo "=== 4) 후보 JSON (전체) ==="
STEMS=""
for c in ppcc_bread ppcc_bread_s300028 ppcc_bread_s300033 ppcc_bread_s400020 ppcs_apple; do
  STEMS="$STEMS $c/per_scene_fit15 $c/per_scene_fit30"; done
for c in $NEW; do STEMS="$STEMS $c/per_scene_fit15"; done
for p in cross_scene_apple_fit15 cross_scene_apple_fit30 cross_scene_bread_fit15 cross_scene_bread_fit30 grand_fit15 grand_fit30; do
  STEMS="$STEMS _pools/$p"; done
"$PYBIN" scripts/safe/groot_n15/robocasa/steer/exp2/p1_candidates.py \
  --out outputs/eval/robocasa/groot_n15/pq2_analysis/p1_candidates_full.json $STEMS  # 원격은 구명(pq) 유지
echo "[p1_remote_chain] DONE"
