#!/usr/bin/env bash
# Eagle pre-LLM ep500 자동 chain: N1.5 → N1.6 (target 15-task).
#
# Resumable 설계:
#   - 각 단계의 task-level skip (extract_eagle_*.py:127 의 embeddings.pt 존재 시 skip)
#     으로 chain 도중 죽거나 끊겨도 같은 명령 재실행 시 미완 task 만 추출.
#   - chain 자체가 idempotent: N1.5/N1.6 둘 다 진행 점진적.
#
# 실행 (host shell):
#   LOG=/home/junhyeong/pkt_ws/temporal_vla/outputs/eagle_chain_ep500_$(date +%Y%m%d_%H%M%S).log
#   setsid nohup bash /home/junhyeong/pkt_ws/temporal_vla/scripts/utils/eagle_extract_ep500_chain.sh \
#       > "$LOG" 2>&1 < /dev/null & disown
#
# Env override:
#   MAX_EPISODES=500
#   MAX_CONCURRENT=4
#   BATCH=16
#   CONTAINER=groot_extract       # GPU 1 매핑
#   N1D5_SAVE=/temporal_vla/data/robocasa_eagle_pre_llm_target_ep500_n1d5
#   N1D6_SAVE=/temporal_vla/data/robocasa_eagle_pre_llm_target_ep500_n1d6

set -u

MAX_EPISODES="${MAX_EPISODES:-500}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
BATCH="${BATCH:-16}"
CONTAINER="${CONTAINER:-groot_extract}"
N1D5_SAVE="${N1D5_SAVE:-/temporal_vla/data/robocasa_eagle_pre_llm_target_ep500_n1d5}"
N1D6_SAVE="${N1D6_SAVE:-/temporal_vla/data/robocasa_eagle_pre_llm_target_ep500_n1d6}"
TASKS_N=15

# Host 경로 (chain 이 host 에서 실행되므로 host 관점 cache dir 확인용)
N1D5_HOST="${N1D5_SAVE/#\/temporal_vla\//\/home\/junhyeong\/pkt_ws\/temporal_vla\/}"
N1D6_HOST="${N1D6_SAVE/#\/temporal_vla\//\/home\/junhyeong\/pkt_ws\/temporal_vla\/}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] $*"; }

count_cache() {
  local dir="$1"
  ls "$dir"/*/embeddings.pt 2>/dev/null | wc -l
}

log "=========================================="
log "Eagle ep500 chain start (N1.5 → N1.6)"
log "  CONTAINER     : $CONTAINER"
log "  MAX_EPISODES  : $MAX_EPISODES"
log "  MAX_CONCURRENT: $MAX_CONCURRENT (task 병렬)"
log "  BATCH         : $BATCH"
log "  N1.5 save     : $N1D5_SAVE  (host: $N1D5_HOST)"
log "  N1.6 save     : $N1D6_SAVE  (host: $N1D6_HOST)"
log "=========================================="

# ─── Stage 1: N1.5 추출 ──────────────────────────────────────
n_pre_n1d5=$(count_cache "$N1D5_HOST")
log "Stage 1: N1.5 ep500 (start cache: $n_pre_n1d5/$TASKS_N tasks)"

docker exec \
    -e MAX_EPISODES="$MAX_EPISODES" \
    -e SAVE_PATH="$N1D5_SAVE" \
    -e MAX_CONCURRENT="$MAX_CONCURRENT" \
    -e BATCH="$BATCH" \
    "$CONTAINER" bash /temporal_vla/scripts/extract/extract_eagle_parallel_n1d5.sh
RC=$?
n_post_n1d5=$(count_cache "$N1D5_HOST")
log "Stage 1 done: rc=$RC  cache: $n_post_n1d5/$TASKS_N tasks"
if [ "$n_post_n1d5" -lt "$TASKS_N" ]; then
  log "Stage 1 INCOMPLETE — abort (재실행 시 미완 task 만 추출됨)"
  exit 1
fi

# ─── Stage 2: N1.6 추출 ──────────────────────────────────────
# extract_eagle_parallel.sh (N1.6 launcher) 가 max_episodes / save_path 같은 env
# 변수를 동일하게 받는지 점검: 받으면 그대로, 아니면 inline 호출 fallback.
n_pre_n1d6=$(count_cache "$N1D6_HOST")
log "Stage 2: N1.6 ep500 (start cache: $n_pre_n1d6/$TASKS_N tasks)"

if docker exec "$CONTAINER" bash -lc \
    "grep -q 'MAX_EPISODES' /temporal_vla/scripts/extract/extract_eagle_parallel.sh"; then
  docker exec \
      -e MAX_EPISODES="$MAX_EPISODES" \
      -e SAVE_PATH="$N1D6_SAVE" \
      -e MAX_CONCURRENT="$MAX_CONCURRENT" \
      -e BATCH="$BATCH" \
      "$CONTAINER" bash /temporal_vla/scripts/extract/extract_eagle_parallel.sh
  RC=$?
else
  log "  N1.6 launcher 가 env var 인식 안 함 — script 직접 호출"
  docker exec \
      -e MAX_EPISODES="$MAX_EPISODES" \
      -e SAVE_PATH="$N1D6_SAVE" \
      -e MAX_CONCURRENT="$MAX_CONCURRENT" \
      -e BATCH="$BATCH" \
      "$CONTAINER" bash /temporal_vla/scripts/extract/extract_eagle_parallel.sh
  RC=$?
fi
n_post_n1d6=$(count_cache "$N1D6_HOST")
log "Stage 2 done: rc=$RC  cache: $n_post_n1d6/$TASKS_N tasks"

log "=========================================="
log "CHAIN DONE."
log "  N1.5 cache: $N1D5_SAVE ($n_post_n1d5/$TASKS_N)"
log "  N1.6 cache: $N1D6_SAVE ($n_post_n1d6/$TASKS_N)"
log "=========================================="
