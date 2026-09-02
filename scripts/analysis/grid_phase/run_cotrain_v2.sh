#!/bin/bash
# 라운드3 (grid v2 confirmatory): negative filter 검증 — pertask baseline + mixed 8쌍.
# v2 = scene 15개 → split train 11 / calib 2 / test 2 (명시 필수; v1 기본값은 6/2/2).
# 사용: bash run_cotrain_v2.sh <v2_shard_dir> <out_root>
set -e
SHARD=${1:?v2_shard_dir}
OUT=${2:?out_root}
PY=~/anaconda3/bin/python
SPLIT="--train-scenes 11 --calib-scenes 2 --test-scenes 2"

# 사전등록(46 라운드3): 반전군 4쌍(해악 예측) + 정렬군 4쌍(무해 대조)
RUNS=(
  "candle_oven|PPCC_candle,OvenRack_out"
  "drawerL_oven|OpenDrawer_left,OvenRack_out"
  "bread_drawerR|PPCC_bread,OpenDrawer_right"
  "marsh_drawerR|PPCC_marshmallow,OpenDrawer_right"
  "bread_marsh|PPCC_bread,PPCC_marshmallow"
  "candle_marsh|PPCC_candle,PPCC_marshmallow"
  "coffee_bread|CoffeeSetupMug,PPCC_bread"
  "bread_candle|PPCC_bread,PPCC_candle"
)

for S in 0 1 2; do
  # 1) pertask baseline (전 slug 1 run)
  D="$OUT/v2_pertask_s${S}"
  if [ -f "$D/sim_summary.tsv" ]; then echo "[skip] $D"; else
    $PY scripts/analysis/grid_phase/failure_detector_sim.py \
      --shard-dir "$SHARD" --out "$D" --arm pertask $SPLIT \
      --truncate-train phase-gt --models lstm,mlp --threads 8 --seed "$S" --quiet \
      && echo "V2_PERTASK_s${S}_DONE"
  fi
  # 2) mixed 8쌍
  for spec in "${RUNS[@]}"; do
    NAME="${spec%%|*}"; SHARDS="${spec#*|}"
    D="$OUT/v2_cotrain_s${S}/${NAME}"
    if [ -f "$D/sim_summary.tsv" ]; then echo "[skip] $D"; continue; fi
    $PY scripts/analysis/grid_phase/failure_detector_sim.py \
      --shard-dir "$SHARD" --out "$D" --arm mixed --shards "$SHARDS" $SPLIT \
      --truncate-train phase-gt --models lstm,mlp --threads 8 --seed "$S" --quiet \
      && echo "V2_COT_s${S}_${NAME}_DONE"
  done
done
echo V2_COTRAIN_ALL_DONE
