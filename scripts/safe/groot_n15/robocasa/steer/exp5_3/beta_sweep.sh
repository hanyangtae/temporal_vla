#!/bin/bash
# exp5-3 β sweep (rudxo_home) — exp5-1 요청 (exp5-1_to_exp5-3_beta_sweep_request.txt).
# 목적: β=1.0 파괴적 해악(0.344→0.025)이 (a)용량 과다 vs (b)setpoint 오차/(c)방향 유해 인지 분해.
#   arm 순서: setM_within_permanent_b05 → _b02 (β=0.5 우선, §3) → A0_anchor_kin
#   (A0 재실행 = action_kinematics jerk base 확보 + 머신 결정론 검증 — 기존 A0 json 엔 kin 없음)
# 그리드 = A0_anchor 와 동일 40셀 (scene i → seed {i%8, (i+5)%8} 라틴 배치), paired 비교.
# serve 구성 = A0 때와 동일 (3개, 8600-8602, host conda) — §2 요건.
# fit/NPZ 재사용: ~/exp53_npz/deploy/permanent/loo_seed{k} (per-scene setpoint registry).
set -u
cd "$HOME/workspace/temporal_vla" || exit 1
PORTS=(8600 8601 8602)
LAYER=12
DEPLOY="$HOME/exp53_npz/deploy"
OUT_BASE="outputs/eval/robocasa/groot_n15/exp5_3"
PYP="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
LOG="$HOME/exp53_bsweep.log"
SCENES=(100000 100003 100005 100006 100009 100010 100011 100012 100016 100018 \
        100020 100022 100023 100025 100026 100031 100033 100034 100035 100039)

serve_env() {
  export CUDA_VISIBLE_DEVICES=0 VLA_CACHE_ROOT="$HOME/.cache/temporal_vla"
  export HF_HOME="$HOME/.cache/temporal_vla/datasets/huggingface"
  export HF_HUB_CACHE="$HOME/.cache/temporal_vla/datasets/huggingface/hub"
  export HF_TRUST_REMOTE_CODE=1 FLASH_ATTN_CUDA_ARCHS=86 TORCH_CUDA_ARCH_LIST=8.9
  export PYTHONPATH="$PWD/lerobot/src:$PWD:$PWD/scripts/utils"
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/lerobot_050_groot/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
  export HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | cut -d= -f2)"
}

serve_kill() { pkill -f "lerobot.py.*--port 86[0]" 2>/dev/null; sleep 6; }

serve_up() {  # $1=beta("A0"=무개입) $2=fold $3=registry서브디렉토리(기본 permanent)
  local beta=$1 fold=${2:-} sub=${3:-permanent}
  serve_kill; serve_env
  local PY="$HOME/miniconda3/envs/lerobot_050_groot/bin/python"
  local extra=()
  if [ "$beta" != "A0" ]; then
    local pph
    pph=$(ls "$DEPLOY/$sub/loo_seed${fold}" | paste -sd, -)
    extra=(--steering-phase-npz-base "$DEPLOY/$sub/loo_seed${fold}" \
           --steering-phases "$pph" --steering-layers $LAYER --steering-beta "$beta" \
           --steering-token-select all)
  fi
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
    for i in $(seq 1 90); do
      curl -s -m 3 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
      sleep 5
    done
    [ $ok = 1 ] || { echo "[ABORT] port $p (beta=$beta fold=$fold)" | tee -a "$LOG"; exit 2; }
  done
}

run_one() {  # $1=arm $2=beta $3=scene $4=inf $5=epidx $6=port
  local arm=$1 beta=$2 S=$3 inf=$4 epidx=$5 port=$6
  local d="$OUT_BASE/$arm/raw_rollouts/OpenDrawer/pq3_drawer_right"
  if ls "$d/task7--ep${epidx}--succ"*.json >/dev/null 2>&1; then return 0; fi
  local extra=()
  [ "$beta" != "A0" ] && extra=(--steer-from-record 0 --steer-phase-name "scene${S}")
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
  echo "[$([ $? = 0 ] && echo ok || echo FAIL:$?)] $arm S$S inf$inf ep$epidx $(date -u +%T)" >> "$LOG"
}

# 라틴 40셀: scene i → seed {i%8, (i+5)%8}  (A0_anchor 와 동일)
latin_rows_for_seed() {  # $1=k → "scene inf epidx" 줄들
  local k=$1 i
  for i in "${!SCENES[@]}"; do
    if [ $((i % 8)) -eq "$k" ] || [ $(((i + 5) % 8)) -eq "$k" ]; then
      echo "${SCENES[$i]} $((k * 1000000)) $((i * 8 + k))"
    fi
  done
}

trap 'serve_kill' EXIT
echo "=== β sweep 시작 $(date -u +%FT%T) ===" | tee -a "$LOG"

# spec = "arm beta [registry서브디렉토리]" — ARMS_SPEC env 로 override 가능
#   기본 = β sweep (exp5-1 요청). future-only 라운드는:
#   ARMS_SPEC="setM_within_fut_b10 1.0 permanent_fut;setM_within_fut_b05 0.5 permanent_fut"
DEFAULT_SPEC="setM_within_permanent_b05 0.5;setM_within_permanent_b02 0.2;A0_anchor_kin A0"
IFS=';' read -ra SPECS <<< "${ARMS_SPEC:-$DEFAULT_SPEC}"
for spec in "${SPECS[@]}"; do
  set -- $spec; arm=$1; beta=$2; sub=${3:-permanent}
  echo "--- arm $arm (beta=$beta) $(date -u +%FT%T) ---" | tee -a "$LOG"
  if [ "$beta" = "A0" ]; then
    ROWS=()
    for k in 0 1 2 3 4 5 6 7; do while IFS= read -r r; do ROWS+=("$r"); done < <(latin_rows_for_seed $k); done
    serve_up A0
    WPIDS=()
    for ((w=0; w<${#PORTS[@]}; w++)); do
      ( for ((i=w; i<${#ROWS[@]}; i+=${#PORTS[@]})); do run_one "$arm" A0 ${ROWS[$i]} "${PORTS[$w]}"; done ) &
      WPIDS+=($!)
    done
    wait "${WPIDS[@]}"   # ★워커만 대기 (serve 자식 포함 bare wait 금지 — mixer 러너 교훈)
  else
    for k in 0 1 2 3 4 5 6 7; do
      ROWS=()
      while IFS= read -r r; do ROWS+=("$r"); done < <(latin_rows_for_seed $k)
      done_all=1
      for r in "${ROWS[@]}"; do set -- $r
        ls "$OUT_BASE/$arm/raw_rollouts/OpenDrawer/pq3_drawer_right/task7--ep$3--succ"*.json >/dev/null 2>&1 || { done_all=0; break; }
      done
      [ $done_all = 1 ] && continue
      serve_up "$beta" "$k" "$sub"
      WPIDS=()
      for ((w=0; w<${#PORTS[@]}; w++)); do
        ( for ((i=w; i<${#ROWS[@]}; i+=${#PORTS[@]})); do run_one "$arm" "$beta" ${ROWS[$i]} "${PORTS[$w]}"; done ) &
        WPIDS+=($!)
      done
      wait "${WPIDS[@]}"
    done
  fi
  serve_kill   # ★serve 자식이 살아 있으면 이후 bare wait 가 영구 대기 — 즉시 정리
  echo "--- $arm 완료: json $(find "$OUT_BASE/$arm" -name '*.json' | wc -l) ---" | tee -a "$LOG"
done
echo "=== β sweep 전체 완료 $(date -u +%FT%T) ===" | tee -a "$LOG"
touch "${DONE_FILE:-$HOME/exp53_bsweep.DONE}"
