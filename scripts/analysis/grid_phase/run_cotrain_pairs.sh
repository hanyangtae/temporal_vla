#!/bin/bash
# co-training 이득 측정 — 쌍 합동학습(mixed) vs pertask(기존 detector_trunc) Δ용.
# 각 run = --arm mixed --shards A,B (A+B 1 detector, slug별 test scene 평가, CP per-task).
# baseline과 동일 seed·split·hyperparam 전제 (split은 seed×task-crc32 결정적).
# 사용: bash run_cotrain_pairs.sh <shard_dir> <out_root>
set -e
SHARD=${1:?shard_dir}
OUT=${2:?out_root}
PY=~/anaconda3/bin/python

RUNS=(
  "bread_candle|PPCC_bread,PPCC_candle"
  "candle_marsh|PPCC_candle,PPCC_marshmallow"
  "bread_marsh|PPCC_bread,PPCC_marshmallow"
  "jug_marsh|PPCC_jug,PPCC_marshmallow"
  "candle_drawerL|PPCC_candle,OpenDrawer_left"
  "coffee_bread|CoffeeSetupMug,PPCC_bread"
  "bread_drawerR|PPCC_bread,OpenDrawer_right"
  "marsh_drawerR|PPCC_marshmallow,OpenDrawer_right"
  "candle_oven|PPCC_candle,OvenRack_out"
  "drawerL_oven|OpenDrawer_left,OvenRack_out"
  "drawerL_drawerR|OpenDrawer_left,OpenDrawer_right"
)

for S in 0 1 2; do
  for spec in "${RUNS[@]}"; do
    NAME="${spec%%|*}"; SHARDS="${spec#*|}"
    D="$OUT/cotrain_s${S}/${NAME}"
    if [ -f "$D/sim_summary.tsv" ]; then echo "[skip] $D"; continue; fi
    $PY scripts/analysis/grid_phase/failure_detector_sim.py \
      --shard-dir "$SHARD" --out "$D" --arm mixed --shards "$SHARDS" \
      --truncate-train phase-gt --models lstm,mlp --threads 8 --seed "$S" --quiet \
      && echo "COT_s${S}_${NAME}_DONE"
  done
done
echo COTRAIN_ALL_DONE
