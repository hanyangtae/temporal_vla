#!/bin/bash
# exp5-3 eval 러너 (rudxo_home 전용): within-scene setM steering — drawer_right.
#
# Arms (순차):
#   A0_anchor  : 무개입 40판 (scene 20 × seed 2 라틴 배치) — home 머신 base SR 앵커.
#                srv50 수집 320판과의 cross-machine 비교 각주용 (결정론은 머신-로컬).
#   setM_within_permanent : fold k=0..7 마다 serve 재기동(loo_seed{k}) → scene 20 × seed k.
#                latch = --steer-from-record 0 (phase-mode global → POST "steer").
#   setM_within_gated     : 동일 fold 구조, phase registry(채택 phase만) + --steer-phase-mode current.
#                미등록 phase = identity(off).
#
# serve: host conda(serve_host.sh env 복제) 3개 — ★4개는 OOM (사용자 확정, 2026-07-27).
# in-sample 차단: seed k 평가엔 k 제외 7-seed fit NPZ (fit_within_scene_setM.py 산출).
# NPZ 배치(사전): ~/exp53_npz/deploy/permanent/loo_seed{k}/steer/dit_L12/conceptors.npz
#                ~/exp53_npz/deploy/gated/loo_seed{k}/<phase>/dit_L12/conceptors.npz
set -u
cd "$HOME/workspace/temporal_vla" || exit 1

PORTS=(8600 8601 8602)
LAYER=12
BETA=1.0
DEPLOY="$HOME/exp53_npz/deploy"
OUT_BASE="outputs/eval/robocasa/groot_n15/exp5_3"
PYP="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
LOG="$HOME/exp53_eval.log"
ARMS=(${ARMS:-A0_anchor setM_within_permanent setM_within_gated})

SCENES=(100000 100003 100005 100006 100009 100010 100011 100012 100016 100018 \
        100020 100022 100023 100025 100026 100031 100033 100034 100035 100039)

serve_env() {
  export CUDA_VISIBLE_DEVICES=0
  export VLA_CACHE_ROOT="$HOME/.cache/temporal_vla"
  export HF_HOME="$HOME/.cache/temporal_vla/datasets/huggingface"
  export HF_HUB_CACHE="$HOME/.cache/temporal_vla/datasets/huggingface/hub"
  export HF_TRUST_REMOTE_CODE=1
  export FLASH_ATTN_CUDA_ARCHS=86
  export TORCH_CUDA_ARCH_LIST=8.9
  export PYTHONPATH="$PWD/lerobot/src:$PWD:$PWD/scripts/utils"
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/lerobot_050_groot/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
  HF_TOKEN_VAL="$(grep -E '^HF_TOKEN=' .env | cut -d= -f2)"
  export HF_TOKEN="$HF_TOKEN_VAL"
}

serve_kill() {
  pkill -f "lerobot.py.*--port 860" 2>/dev/null
  sleep 6
}

serve_up() {  # $1=arm $2=fold(steered 전용)
  local arm=$1 fold=${2:-}
  serve_kill
  serve_env
  local PY="$HOME/miniconda3/envs/lerobot_050_groot/bin/python"
  local extra=()
  case "$arm" in
    A0_anchor) ;;
    setM_within_permanent)
      # registry = scene{S}/(per-scene setpoint) + steer/(전역 fallback 참조)
      local pph
      pph=$(ls "$DEPLOY/permanent/loo_seed${fold}" | paste -sd, -)
      extra=(--steering-phase-npz-base "$DEPLOY/permanent/loo_seed${fold}" \
             --steering-phases "$pph" --steering-layers $LAYER --steering-beta $BETA \
             --steering-token-select all) ;;
    setM_within_gated)
      local phases
      phases=$(ls "$DEPLOY/gated/loo_seed${fold}" | paste -sd, -)
      extra=(--steering-phase-npz-base "$DEPLOY/gated/loo_seed${fold}" \
             --steering-phases "$phases" --steering-layers $LAYER --steering-beta $BETA \
             --steering-token-select all) ;;
  esac
  for p in "${PORTS[@]}"; do
    setsid nohup "$PY" scripts/serve/lerobot.py \
      --profile configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml \
      --host '*' --port "$p" --device cuda \
      --groot-dit-token-pool all_token_full --groot-dit-capture-layers 0,2,4,8,10,12,15 \
      "${extra[@]}" > "$HOME/serve53_${p}.log" 2>&1 &
    sleep 8
  done
  for p in "${PORTS[@]}"; do
    local ok=0
    for i in $(seq 1 60); do
      curl -s -m 3 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
      sleep 5
    done
    [ $ok = 1 ] || { echo "[ABORT] port $p health 실패 (arm=$arm fold=$fold)" | tee -a "$LOG"; exit 2; }
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1 >> "$LOG"
}

run_one() {  # $1=arm $2=scene $3=inf $4=epidx $5=port
  local arm=$1 S=$2 inf=$3 epidx=$4 port=$5
  local d="$OUT_BASE/$arm/raw_rollouts/OpenDrawer/pq3_drawer_right"
  if ls "$d/task7--ep${epidx}--succ"*.csv >/dev/null 2>&1; then return 0; fi
  local extra=()
  case "$arm" in
    A0_anchor) ;;
    # per-scene setpoint 스위칭: latch ON 시 phase 이름 scene{S} 를 POST (registry 대응 dir)
    setM_within_permanent) extra=(--steer-from-record 0 --steer-phase-name "scene${S}") ;;
    setM_within_gated) extra=(--steer-from-record 0 --steer-phase-mode current) ;;
  esac
  docker exec -e MUJOCO_GL=egl \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    -e PYTHONPATH="$PYP" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:$port" --task OpenDrawer \
    --env-name "robocasa_panda_omron/OpenDrawer_PandaOmron_Env" \
    --output-dir "/temporal_vla/$OUT_BASE/$arm/raw_rollouts" \
    --cell-id pq3_drawer_right --cell-index 7 \
    --canonical-instruction "Open the right drawer." \
    --episode-start-idx "$epidx" --n-episodes 1 \
    --seed "$S" --inference-seed "$inf" --n-action-steps 5 --max-episode-steps 720 \
    --video-fps 20 --steps-per-render 2 --wait-ready --proximity-phases --no-features \
    "${extra[@]}" > /dev/null 2>&1
  echo "[$([ $? = 0 ] && echo ok || echo FAIL:$?)] $arm S$S inf$inf ep$epidx p$port $(date -u +%T)" >> "$LOG"
}

run_grid() {  # $1=arm, ROWS 배열 사용 (scene inf epidx)
  local arm=$1 n=${#ROWS[@]}
  for ((w=0; w<${#PORTS[@]}; w++)); do
    (
      for ((i=w; i<n; i+=${#PORTS[@]})); do
        run_one "$arm" ${ROWS[$i]} "${PORTS[$w]}"
      done
    ) &
  done
  wait
}

trap 'serve_kill' EXIT

echo "=== exp5-3 eval 시작 $(date -u +%FT%T) arms=${ARMS[*]} ===" | tee -a "$LOG"
for arm in "${ARMS[@]}"; do
  echo "--- arm $arm $(date -u +%FT%T) ---" | tee -a "$LOG"
  if [ "$arm" = A0_anchor ]; then
    ROWS=()
    for i in "${!SCENES[@]}"; do
      for k in $((i % 8)) $(((i + 5) % 8)); do
        ROWS+=("${SCENES[$i]} $((k * 1000000)) $((i * 8 + k))")
      done
    done
    serve_up "$arm"
    run_grid "$arm"
  else
    for k in 0 1 2 3 4 5 6 7; do
      ROWS=()
      for i in "${!SCENES[@]}"; do
        ROWS+=("${SCENES[$i]} $((k * 1000000)) $((i * 8 + k))")
      done
      # fold 전체가 이미 완료면 재기동 생략
      local_done=1
      for i in "${!SCENES[@]}"; do
        ls "$OUT_BASE/$arm/raw_rollouts/OpenDrawer/pq3_drawer_right/task7--ep$((i * 8 + k))--succ"*.csv \
          >/dev/null 2>&1 || { local_done=0; break; }
      done
      [ $local_done = 1 ] && continue
      serve_up "$arm" "$k"
      run_grid "$arm"
    done
  fi
  n_csv=$(find "$OUT_BASE/$arm" -name "*.csv" 2>/dev/null | wc -l)
  echo "--- arm $arm 완료: csv $n_csv ---" | tee -a "$LOG"
done
serve_kill
echo "=== 전체 완료 $(date -u +%FT%T) ===" | tee -a "$LOG"
touch "$HOME/exp53_eval.DONE"
