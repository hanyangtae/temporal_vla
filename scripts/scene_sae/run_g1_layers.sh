#!/bin/bash
# run_g1_drawer_left.sh 의 파라미터판 (병렬 분할용): LAYERS·GPU 를 env 로 받는다.
# 예: GPU=1 LAYERS="8 2" bash scripts/scene_sae/run_g1_layers.sh
# 완료 조합(model.pt+probe json 존재)은 skip — 인스턴스 간 중복 방지는 layer 집합을
# 서로 겹치지 않게 주는 것으로 보장한다 (동일 layer 동시 실행 금지).
set -euo pipefail

REPO=/home/dongkyu/pkt_ws/temporal_vla
IN=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/pq3_drawer_left/inputs
OUT=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/pq3_drawer_left
PY=/home/dongkyu/miniconda3/envs/vla-safe/bin/python
GPU="${GPU:?GPU 번호 필수}"
LAYERS_STR="${LAYERS:?layer 목록 필수 (공백 구분)}"
M=6144
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$REPO"

read -ra LAYER_ARR <<< "$LAYERS_STR"
KS=(16 32 64)

for L in "${LAYER_ARR[@]}"; do
  for K in "${KS[@]}"; do
    TAG="L${L}_m${M}_k${K}_s0"
    CKPT="$OUT/$TAG"
    if [[ -f "$CKPT/model.pt" && -f "$OUT/probe_${TAG}.json" ]]; then
      echo "[skip] $TAG (이미 완료)"; continue
    fi
    echo "=== [train] $TAG $(date +%H:%M:%S) gpu=$GPU ==="
    $PY scripts/scene_sae/train_scene_sae.py \
      --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
      --cell pq3_drawer_left --layer "$L" --m $M --k "$K" --seed 0 --device cuda
    echo "=== [probe] $TAG $(date +%H:%M:%S) gpu=$GPU ==="
    $PY scripts/scene_sae/probe_scene.py \
      --ckpt-dir "$CKPT" \
      --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
      --out "$OUT/probe_${TAG}.json" --label layout_id --n-perm 100 --seed 0 --device cuda
  done
done
echo "=== layers [$LAYERS_STR] 완료 $(date +%H:%M:%S) ==="
