#!/bin/bash
# exp5 G1 drawer_left sweep 드라이버: 5 layer × k{16,32,64} SAE 학습 → 각각 scene probe.
# L10(atlas peak) 최우선 순서. GPU 1장(완전히 빈 것) 사용, 완료 판정은 산출물 개수(핸드아웃 §6-6).
set -euo pipefail

REPO=/home/dongkyu/pkt_ws/temporal_vla
IN=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/pq3_drawer_left/inputs
OUT=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/pq3_drawer_left
PY=/home/dongkyu/miniconda3/envs/vla-safe/bin/python
GPU="${GPU:-0}"
M=6144
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$REPO"

LAYERS=(10 12 8 2 0)   # L10 우선
KS=(16 32 64)

for L in "${LAYERS[@]}"; do
  for K in "${KS[@]}"; do
    TAG="L${L}_m${M}_k${K}_s0"
    CKPT="$OUT/$TAG"
    if [[ -f "$CKPT/model.pt" && -f "$OUT/probe_${TAG}.json" ]]; then
      echo "[skip] $TAG (이미 완료)"; continue
    fi
    echo "=== [train] $TAG $(date +%H:%M:%S) ==="
    $PY scripts/scene_sae/train_scene_sae.py \
      --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
      --cell pq3_drawer_left --layer "$L" --m $M --k "$K" --seed 0 --device cuda
    echo "=== [probe] $TAG $(date +%H:%M:%S) ==="
    $PY scripts/scene_sae/probe_scene.py \
      --ckpt-dir "$CKPT" \
      --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
      --out "$OUT/probe_${TAG}.json" --label layout_id --n-perm 100 --seed 0 --device cuda
  done
done

echo "=== 완료 대조: probe json 개수 (기대 15) ==="
ls "$OUT"/probe_*.json | wc -l
