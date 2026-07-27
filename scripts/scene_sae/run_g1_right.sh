#!/bin/bash
# exp5 G1-재판정 (scene-matched drawer_right 160판) + G2용 SAE 학습 드라이버.
# 축 규약(리뷰 #2): SAE 학습 축 == probe 평가 축 (지문 가드가 강제).
#   split_episode 축 → scenario_seed 20클래스 + layout_id probe (G1-재판정)
#   split_scene   축 → G2 잔차화용 SAE (probe 는 layout_id 만 참고 실행)
# 사용: GPU=1 AXIS=split_episode LAYERS="8 10 12" bash scripts/scene_sae/run_g1_right.sh
set -euo pipefail

REPO=/home/dongkyu/pkt_ws/temporal_vla
IN=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/scene_matched_drawer_right/inputs
OUT=$REPO/outputs/eval/robocasa/groot_n15/scene_sae/scene_matched_drawer_right
PY=/home/dongkyu/miniconda3/envs/vla-safe/bin/python
GPU="${GPU:?}"
AXIS="${AXIS:?split_episode|split_scene}"
LAYERS_STR="${LAYERS:?}"
K="${K:-64}"
M=6144
AUXK="${AUXK:-0}"          # 0=끔(기존 drawer_left 비교용), 512=개선판
WINDOW="${WINDOW:-38}"      # exp5-3 길이통제 창 [0,38)
NPERM="${NPERM:-50}"
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$REPO"

# 발사 직전 GPU 소유자 확인 — 완전히 빈 GPU 만 (사용자 규칙)
if nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -q "$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU)"; then
  echo "GPU $GPU 점유 중 — 중단" >&2; exit 3
fi

read -ra LAYER_ARR <<< "$LAYERS_STR"
for L in "${LAYER_ARR[@]}"; do
  TAG="R_L${L}_m${M}_k${K}_aux${AUXK}_${AXIS}"
  CKPT="$OUT/$TAG"
  if [[ ! -f "$CKPT/model.pt" ]]; then
    echo "=== [train] $TAG $(date +%H:%M:%S) gpu=$GPU ==="
    $PY scripts/scene_sae/train_scene_sae.py \
      --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
      --cell scene_matched_drawer_right --layer "$L" --m $M --k "$K" --seed 0 \
      --split-col "$AXIS" --aux-k "$AUXK" --device cuda --out-dir "$CKPT"
  fi
  if [[ "$AXIS" == "split_episode" ]]; then
    for LBL in scenario_seed layout_id; do
      PJ="$OUT/probe_${TAG}_${LBL}.json"
      [[ -f "$PJ" ]] && { echo "[skip] $PJ"; continue; }
      echo "=== [probe] $TAG label=$LBL $(date +%H:%M:%S) gpu=$GPU ==="
      $PY scripts/scene_sae/probe_scene.py \
        --ckpt-dir "$CKPT" \
        --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
        --out "$PJ" --label "$LBL" --split-col "$AXIS" --window "$WINDOW" \
        --n-perm "$NPERM" --seed 0 --device cuda
    done
  else
    PJ="$OUT/probe_${TAG}_layout.json"
    [[ -f "$PJ" ]] || $PY scripts/scene_sae/probe_scene.py \
      --ckpt-dir "$CKPT" \
      --x "$IN/X_L${L}.npz" --stats "$IN/stats_L${L}.npz" --meta "$IN/meta.npz" \
      --out "$PJ" --label layout_id --split-col "$AXIS" --window "$WINDOW" \
      --n-perm "$NPERM" --seed 0 --device cuda
  fi
done
echo "=== AXIS=$AXIS layers[$LAYERS_STR] aux$AUXK 완료 $(date +%H:%M:%S) ==="
