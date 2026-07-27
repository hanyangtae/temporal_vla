#!/bin/bash
# exp5-3 Phase2: mixer(OpenStandMixerHead) scene-matched 수집 — rudxo_home 전용.
# 같은 scenario_seed(scene) × inference_seed 8종 → 한 scene 안에 succ/fail 혼재 (drawer 규약 동일).
#   scene 20 = mixer_feasibility.json feasible 앞 20 (BLOCKED 100010 제외; 기하 불가 seed 는
#   fit·eval 양쪽 동일 제외 — scene-feasibility-filter 규칙).
#   capture ON(full-token L{0,2,4,8,10,12,15}) · n_action_steps 5 · --proximity-phases.
# serve: host conda 3개 (★4개 OOM — 2026-07-27 사용자 확정).
# 영상: 판마다 vis/annotate_phase_video.py 로 **상단 여백 배너(instruction+phase+step)** 주석본
#   (*--annot.mp4) 생성 — 하단 가림 캡션 대신 이 방식 유지 (2026-07-27 사용자 지시).
# 배송: home→승준 직결 불가(포트 차단) → 승준 측 puller(mixer_sm_pull.sh, 승준 tmux)가
#   pull + 검증 후 home pkl 삭제. resume 마커는 csv (pkl 삭제 후에도 잔존).
set -u
cd "$HOME/workspace/temporal_vla" || exit 1
PORTS=(8600 8601 8602)
OUT_BASE="outputs/eval/robocasa/groot_n15/exp5_3_mixer_sm"
OUT_RAW="$OUT_BASE/raw_rollouts"
PYP="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
LOG="$HOME/mixer_sm.log"
TASK=OpenStandMixerHead
ENVN="robocasa_panda_omron/OpenStandMixerHead_PandaOmron_Env"
CELL_ID=exp53_mixer_sm; CELL_INDEX=0
INSTR="Open the stand mixer head."
N_SEED=8
# mixer_feasibility.json feasible 앞 20 (100010 BLOCKED 제외) — 추출 2026-07-27
SCENES=(100000 100001 100002 100003 100004 100005 100006 100007 100008 100009 \
        100011 100012 100013 100014 100015 100016 100017 100018 100019 100020)

serve_env() {
  export CUDA_VISIBLE_DEVICES=0 VLA_CACHE_ROOT="$HOME/.cache/temporal_vla"
  export HF_HOME="$HOME/.cache/temporal_vla/datasets/huggingface"
  export HF_HUB_CACHE="$HOME/.cache/temporal_vla/datasets/huggingface/hub"
  export HF_TRUST_REMOTE_CODE=1 FLASH_ATTN_CUDA_ARCHS=86 TORCH_CUDA_ARCH_LIST=8.9
  export PYTHONPATH="$PWD/lerobot/src:$PWD:$PWD/scripts/utils"
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/lerobot_050_groot/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
  export HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | cut -d= -f2)"
}

echo "=== mixer scene-matched 수집 시작 $(date -u +%FT%T) — 160판 ===" | tee -a "$LOG"
mkdir -p "$OUT_BASE"
echo "rudxo_home (218.152.144.220) GPU0 4090 serve3 — 수집 $(date -u +%F)" > "$OUT_BASE/MACHINE.txt"

serve_env
PY="$HOME/miniconda3/envs/lerobot_050_groot/bin/python"
pkill -f "lerobot.py.*--port 86[0]" 2>/dev/null; sleep 6
for p in "${PORTS[@]}"; do
  setsid nohup "$PY" scripts/serve/lerobot.py \
    --profile configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml \
    --host '*' --port "$p" --device cuda \
    --groot-dit-token-pool all_token_full --groot-dit-capture-layers 0,2,4,8,10,12,15 \
    > "$HOME/serve_m53_${p}.log" 2>&1 &
  sleep 8
done
for p in "${PORTS[@]}"; do
  ok=0
  for i in $(seq 1 90); do
    curl -s -m 3 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    sleep 5
  done
  [ $ok = 1 ] || { echo "[ABORT] serve $p health 실패" | tee -a "$LOG"; exit 2; }
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1 >> "$LOG"

annot_one() {  # $1=epidx — 상단 배너 주석본 생성 (실패해도 수집은 계속)
  docker exec -e PYTHONPATH="$PYP" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/vis/annotate_phase_video.py \
    --run-dir "/temporal_vla/$OUT_RAW" \
    --out-dir "/temporal_vla/$OUT_RAW/$TASK/$CELL_ID/annot" \
    --max-ep "$1" >> "$HOME/mixer_sm_annot.log" 2>&1 \
    || echo "[annot FAIL] ep$1" >> "$LOG"
}

run_one() {  # $1=scene $2=inf $3=epidx $4=port
  local S=$1 inf=$2 epidx=$3 port=$4
  local d="$OUT_RAW/$TASK/$CELL_ID"
  if ls "$d/task${CELL_INDEX}--ep${epidx}--succ"*.csv >/dev/null 2>&1; then return 0; fi
  docker exec -e MUJOCO_GL=egl \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    -e PYTHONPATH="$PYP" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:$port" --task "$TASK" --env-name "$ENVN" \
    --output-dir "/temporal_vla/$OUT_RAW" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
    --canonical-instruction "$INSTR" --episode-start-idx "$epidx" --n-episodes 1 \
    --seed "$S" --inference-seed "$inf" --n-action-steps 5 --max-episode-steps 720 \
    --video-fps 20 --steps-per-render 2 --wait-ready --proximity-phases \
    > /dev/null 2>&1
  echo "[$([ $? = 0 ] && echo ok || echo FAIL:$?)] S$S inf$inf ep$epidx p$port $(date -u +%T)" >> "$LOG"
}

ROWS=()
for i in "${!SCENES[@]}"; do
  for k in $(seq 0 $((N_SEED-1))); do
    ROWS+=("${SCENES[$i]} $((k * 1000000)) $((i * 8 + k))")
  done
done

trap 'pkill -f "lerobot.py.*--port 86[0]" 2>/dev/null' EXIT
for ((w=0; w<${#PORTS[@]}; w++)); do
  (
    for ((i=w; i<${#ROWS[@]}; i+=${#PORTS[@]})); do
      set -- ${ROWS[$i]}
      run_one "$1" "$2" "$3" "${PORTS[$w]}"
      annot_one "$3"
    done
  ) &
done
wait
pkill -f "lerobot.py.*--port 86[0]" 2>/dev/null
{
echo "=== 수집 완료 $(date -u +%FT%T) ==="
echo "csv: $(find "$OUT_RAW" -name '*.csv' | wc -l) / pkl 잔존: $(find "$OUT_RAW" -name '*.pkl' | wc -l)"
echo "succ 분포: $(find "$OUT_RAW" -name '*.csv' | grep -oE 'succ[01]' | sort | uniq -c | tr '\n' ' ')"
} | tee -a "$LOG"
touch "$HOME/mixer_sm.DONE"
