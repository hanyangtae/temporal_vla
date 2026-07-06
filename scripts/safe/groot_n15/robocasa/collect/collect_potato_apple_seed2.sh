#!/usr/bin/env bash
# potato/apple 재수집 (기존 seed가 성공 쏠림 → 다른 seed로 실패 보강).
#   ppcc_potato_s2 (seed 200019), ppcs_apple_s2 (seed 100074), 각 15 rollout.
# 기존 데이터 보존: NEW cell_id(_s2)로 저장하므로 ppcc_potato/ppcs_apple 와 collision 없음.
# 같은 RUN_ID(phase_event_aligned_4cell)라 raw_rollouts 트리에 나란히 쌓임(분석 자동 discover).
#
# 전제: N1.5 capture serve 3개가 이미 포트 8400/8401/8402 에서 /health ready.
#   (serve = lerobot.py --collect --capture-vl --groot-dit-capture-layers 0,2,4,8,10,12,15,
#    layer 포맷을 기존 bread/onion 과 동일하게 유지.)
# 자족: 3 worker 병렬 round-robin → 완료 시 serve kill(포트별) → split.tsv + sentinel.
# detached 실행:
#   cd REPO && setsid nohup bash scripts/safe/groot_n15/robocasa/collect/collect_potato_apple_seed2.sh \
#     > outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/logs/s2_orchestrator.log 2>&1 < /dev/null &
set -uo pipefail  # NOT -e: rollout 에러가 나도 나머지 진행 + cleanup 보장

PORTS=(8400 8401 8402)
N_WORKERS=3
N_ROLLOUTS="${N_ROLLOUTS:-15}"
SEED_STRIDE="${SEED_STRIDE:-1000}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
RUN_ID="${RUN_ID:-phase_event_aligned_4cell}"
CONTAINER="${CONTAINER:-robocasa}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
HOST_OUT_ROOT="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/${RUN_ID}"
CONTAINER_OUT_DIR="/temporal_vla/outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts"
LOGDIR="${HOST_OUT_ROOT}/logs"
mkdir -p "$LOGDIR"

# cell rows: cell_id|task|env_name|cell_index|scenario_seed|canonical_instruction
CELLS=(
  "ppcc_potato_s2|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|4|200019|Pick the potato from the counter and place it in the cabinet."
  "ppcs_apple_s2|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100074|Pick the apple from the plate and place it in the pan."
)
PYTHONPATH_IN="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

run_worker() {
  local w="$1" job=0
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r cell_id task env_name cell_index seed instr <<<"$cell"
    for idx in $(seq 0 $((N_ROLLOUTS - 1))); do
      if [ $((job % N_WORKERS)) -eq "$w" ]; then
        if ls "${HOST_OUT_ROOT}/raw_rollouts/${task}/${cell_id}/task${cell_index}--ep${idx}--succ"*.pkl >/dev/null 2>&1; then
          echo "[w${w}] skip existing ${cell_id} ep${idx}"
        else
          local inf=$((idx * SEED_STRIDE))
          echo "[w${w}] collect ${cell_id} ep${idx} seed=${seed} inf=${inf}"
          docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYTHONPATH_IN" "$CONTAINER" \
            python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
            --vla-server "http://127.0.0.1:${PORTS[$w]}" --task "$task" --env-name "$env_name" \
            --output-dir "$CONTAINER_OUT_DIR" --cell-id "$cell_id" --cell-index "$cell_index" \
            --canonical-instruction "$instr" \
            --episode-start-idx "$idx" --n-episodes 1 \
            --seed "$seed" --inference-seed "$inf" \
            --n-action-steps "$N_ACTION_STEPS" --max-episode-steps "$MAX_EPISODE_STEPS" \
            --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
            | grep -E "^wrote|Error|Traceback" || true
        fi
      fi
      job=$((job + 1))
    done
  done
  echo "[w${w}] DONE"
}

echo "[orch] $(date '+%F %T') start: 2 cells x ${N_ROLLOUTS} = $((2 * N_ROLLOUTS)) jobs, ${N_WORKERS} workers"
for w in $(seq 0 $((N_WORKERS - 1))); do
  run_worker "$w" > "${LOGDIR}/s2_worker${w}.log" 2>&1 &
done
wait
echo "[orch] $(date '+%F %T') all workers done; killing serves"

# serve kill (포트별로 surgical)
for PORT in "${PORTS[@]}"; do
  docker exec "${CONTAINER/robocasa/lerobot}" bash -lc "pkill -f 'serve/lerobot.py.*--port ${PORT}' || true" 2>/dev/null || \
    docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${PORT}' || true" 2>/dev/null || true
done

# succ/fail split for the 2 new cells
SPLIT="${HOST_OUT_ROOT}/succ_fail_split_s2.tsv"
{
  echo -e "cell\ttask\tsucc\tfail\ttotal"
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r cell_id task _ _ _ _ <<<"$cell"
    d="${HOST_OUT_ROOT}/raw_rollouts/${task}/${cell_id}"
    s=$(ls "$d"/*succ1.pkl 2>/dev/null | wc -l)
    f=$(ls "$d"/*succ0.pkl 2>/dev/null | wc -l)
    echo -e "${cell_id}\t${task}\t${s}\t${f}\t$((s + f))"
  done
} > "$SPLIT"
cat "$SPLIT"
touch "${LOGDIR}/S2_DONE"
echo "[orch] $(date '+%F %T') DONE -> ${SPLIT}"
