#!/bin/bash
# directed pair 전이 배치 — action-phase 세션의 cluster 공유도↔전이 상관용 (잠정 목록).
# 각 run = --arm loto over 지정 shard 집합; 판독 시 필요한 held-out fold 행만 사용.
# 사용: bash run_transfer_pairs.sh <shard_dir> <out_root>   (승준, OMP cap 호출측)
set -e
SHARD=${1:?shard_dir}
OUT=${2:?out_root}
PY=~/anaconda3/bin/python

# name|shards  (양방향 쌍은 한 run의 두 fold로 산출)
RUNS=(
  "bread_candle|PPCC_bread,PPCC_candle"
  "bread_marsh|PPCC_bread,PPCC_marshmallow"
  "candle_marsh|PPCC_candle,PPCC_marshmallow"
  "apple_bread|PPCC_apple,PPCC_bread"
  "jug_marsh|PPCC_jug,PPCC_marshmallow"
  "drawerL_bread|OpenDrawer_left,PPCC_bread"
  "drawerL_marsh|OpenDrawer_left,PPCC_marshmallow"
  "candle_drawerL|PPCC_candle,OpenDrawer_left"
  "ppcc4_drawerL|PPCC_bread,PPCC_candle,PPCC_jug,PPCC_apple,OpenDrawer_left"
  "drawer2_bread|OpenDrawer_left,OpenDrawer_right,PPCC_bread"
  "drawer2_marsh|OpenDrawer_left,OpenDrawer_right,PPCC_marshmallow"
)

for S in 0 1 2; do
  for M in phase-gt none; do
    for spec in "${RUNS[@]}"; do
      NAME="${spec%%|*}"; SHARDS="${spec#*|}"
      D="$OUT/pairs_s${S}/${M}/${NAME}"
      if [ -f "$D/sim_summary.tsv" ]; then echo "[skip] $D"; continue; fi
      $PY scripts/analysis/grid_phase/failure_detector_sim.py \
        --shard-dir "$SHARD" --out "$D" --arm loto --shards "$SHARDS" \
        --truncate-train "$M" --models lstm,mlp --threads 8 --seed "$S" --quiet \
        && echo "PAIR_s${S}_${M}_${NAME}_DONE"
    done
  done
done
echo PAIRS_ALL_DONE
