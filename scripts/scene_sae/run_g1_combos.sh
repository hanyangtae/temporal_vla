#!/bin/bash
# G1 sweep v2: 명시 조합 리스트(L:K 공백구분)를 받아 학습+probe. 순열 30회로 가속판.
# 예: GPU=0 COMBOS="10:64 12:16 12:32 12:64" bash scripts/scene_sae/run_g1_combos.sh
set -euo pipefail

REPO=/home/dongkyu/pkt_ws/temporal_vla
IN=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/pq3_drawer_left/inputs
OUT=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/pq3_drawer_left
PY=/home/dongkyu/miniconda3/envs/vla-safe/bin/python
GPU="${GPU:?}"
COMBOS_STR="${COMBOS:?L:K 리스트 필수}"
M=6144
NPERM="${NPERM:-30}"
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$REPO"

read -ra COMBO_ARR <<< "$COMBOS_STR"
for C in "${COMBO_ARR[@]}"; do
  L="${C%%:*}"; K="${C##*:}"
  TAG="L${L}_m${M}_k${K}_s0"
  CKPT="$OUT/$TAG"
  if [[ -f "$OUT/probe_${TAG}.json" ]]; then echo "[skip] $TAG"; continue; fi
  if [[ ! -f "$CKPT/model.pt" ]]; then
    echo "=== [train] $TAG $(date +%H:%M:%S) gpu=$GPU ==="
    $PY scripts/scene_sae/train_scene_sae.py \
      --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
      --cell pq3_drawer_left --layer "$L" --m $M --k "$K" --seed 0 --device cuda
  fi
  echo "=== [probe] $TAG $(date +%H:%M:%S) gpu=$GPU ==="
  $PY scripts/scene_sae/probe_scene.py \
    --ckpt-dir "$CKPT" \
    --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
    --out "$OUT/probe_${TAG}.json" --label layout_id --n-perm "$NPERM" \
    --seed 0 --device cuda ${WINDOW:+--window $WINDOW}
    # (구 --null-max-iter 60 제거 — 순열 null 은 본 probe 와 동일 max_iter 를 쓴다. 리뷰 #7)
done
echo "=== combos [$COMBOS_STR] 완료 $(date +%H:%M:%S) ==="
