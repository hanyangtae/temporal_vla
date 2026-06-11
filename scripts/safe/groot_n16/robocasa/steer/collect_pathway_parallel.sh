#!/bin/bash
# Pathway-resolved per-token activation 병렬 수집 (latent steering plan Phase 2).
#
# collect_multilayer_parallel.sh 의 변형:
#   - DiT per-token multilayer + VL(goal) pathway(action_head.vlln seq-mean-pool) 를
#     같은 forward 에서 동시 캡처 (--capture-vl). 두 pathway 가 step 단위로 정렬됨.
#   - GPUS 기본값 = "3 4 5 6" (다른 collection 이 0~2 점유 시 충돌 회피).
#   - BASE_PORT 기본값 = 5611 (5601~5605 과 충돌 회피).
#   - **cleanup 은 이 스크립트가 띄운 포트만 kill** (전역 pkill 금지 → 타 collection 보호).
#   - CAPTURE_LAYERS 로 DiT layer subset 지정 (디스크 절감). 빈 값="" 이면 전체 32 layer.
#
# 호스트에서 실행. 결과: outputs/.../<RUN_ID>/raw_rollouts/<task>/*.pkl
set -u

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
REPO=/temporal_vla
cd "${REPO_HOST}"

GPUS=(${GPUS:-3 4 5 6})
EPISODES_PER_TASK=${EPISODES_PER_TASK:-100}
SEED_START=${SEED_START:-200000}
BASE_PORT=${BASE_PORT:-5611}
PROFILE=${PROFILE:-${REPO}/configs/checkpoints/groot__robocasa365_ckpt120000.yaml}
ENV_SOURCE=robocasa365
RUN_ID=${RUN_ID:-target_atomic_moderate10_pathway_pertoken_100ep}
OUT_ROOT=${REPO}/outputs/eval/robocasa/groot_n16/${RUN_ID}/raw_rollouts
OUT_ROOT_HOST=outputs/eval/robocasa/groot_n16/${RUN_ID}/raw_rollouts
# ep_meta 매니페스트 디렉토리(scene replay 가능 → 향후 pathway 정렬 재수집 대비).
EP_META_ROOT=${REPO}/outputs/eval/robocasa/groot_n16/${RUN_ID}/ep_meta
EP_META_ROOT_HOST=outputs/eval/robocasa/groot_n16/${RUN_ID}/ep_meta
LOG=/tmp/pathway_collect
PYTHON=${PYTHON:-python}
VERIFY_AFTER_COLLECT=${VERIFY_AFTER_COLLECT:-1}
NPROC=${#GPUS[@]}
# DiT per-token capture mode (full=T 전체 보존, K mean). VL 은 항상 seq-mean-pool.
CAPTURE_TOKEN_MODE=${CAPTURE_TOKEN_MODE:-full}
# DiT layer subset (comma-separated). 디스크 절감용. "" = 전체 32 layer.
CAPTURE_LAYERS=${CAPTURE_LAYERS:-0,2,4,8,16,24,31}

# moderate-10 (COAST steering 대상: SR 양극단 제외한 10 task). task_id = 이 리스트 내 index.
TASKS=(CloseToasterOvenDoor NavigateKitchen OpenCabinet OpenDrawer PickPlaceCounterToCabinet
       PickPlaceCounterToStove PickPlaceDrawerToCounter SlideDishwasherRack TurnOnMicrowave
       TurnOnSinkFaucet)
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  TASKS=( ${TASKS_OVERRIDE} )
fi

# cleanup: 이 스크립트가 띄운 포트(BASE_PORT..+NPROC-1)의 서버만 kill + 내 collect client kill.
# 전역 pkill 금지(타 collection 보호). trap 으로 정상 종료·인터럽트 모두에서 실행.
cleanup() {
  for idx in "${!GPUS[@]}"; do
    local port=$((BASE_PORT + idx))
    docker exec groot bash -lc "pkill -f 'feature_server.py.*--port ${port}'" 2>/dev/null
    docker exec robocasa bash -lc "pkill -f 'policy-client-port ${port}'" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

PYPATH="${REPO}/src/policies/Isaac-GR00T:${REPO}/src/benchmarks/robocasa:${REPO}/src/benchmarks/robosuite:${REPO}"
mkdir -p "${OUT_ROOT_HOST}"
mkdir -p "${LOG}"
docker exec groot bash -lc "mkdir -p ${LOG}" 2>/dev/null
echo "[setup] GPUs=${GPUS[*]} eps/task=${EPISODES_PER_TASK} tasks=${#TASKS[@]}"
echo "[setup] token_mode=${CAPTURE_TOKEN_MODE} capture_layers='${CAPTURE_LAYERS}' (--capture-vl) out=${OUT_ROOT}"

ping_server() {  # $1=port
  docker exec groot python3 -c "
import zmq, sys
sys.path.insert(0, '/temporal_vla/src/policies/Isaac-GR00T')
from gr00t.policy.server_client import MsgSerializer
s = zmq.Context().socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 4000); s.setsockopt(zmq.SNDTIMEO, 4000); s.setsockopt(zmq.LINGER, 0)
s.connect('tcp://127.0.0.1:$1')
try:
    s.send(MsgSerializer.to_bytes({'endpoint': 'ping'})); MsgSerializer.from_bytes(s.recv()); print('OK')
except Exception:
    print('FAIL')
" 2>/dev/null | grep -q OK
}

capture_layer_count() {
  local clean="${CAPTURE_LAYERS// /}"
  if [[ -z "${clean}" ]]; then
    echo 32
    return
  fi
  awk -F',' '{print NF}' <<<"${clean}"
}

expected_dit_hidden_shape() {
  local layers
  layers="$(capture_layer_count)"
  case "${CAPTURE_TOKEN_MODE}" in
    full)
      echo "${layers},51,1536"
      ;;
    valid|all)
      echo "${layers},1536"
      ;;
    *)
      echo "ERROR: unsupported CAPTURE_TOKEN_MODE=${CAPTURE_TOKEN_MODE}" >&2
      return 1
      ;;
  esac
}

expected_dit_feature_kind() {
  case "${CAPTURE_TOKEN_MODE}" in
    full)
      echo "groot_n16_dit_block_residual_kmean_perT_multilayer"
      ;;
    valid|all)
      echo "groot_n16_dit_block_residual_pooled_multilayer"
      ;;
    *)
      echo "ERROR: unsupported CAPTURE_TOKEN_MODE=${CAPTURE_TOKEN_MODE}" >&2
      return 1
      ;;
  esac
}

# ── 1. GPU당 feature_server(multilayer + VL) 기동 ──
LAYERS_ARG=""
[[ -n "${CAPTURE_LAYERS}" ]] && LAYERS_ARG="--capture-layers ${CAPTURE_LAYERS}"
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}; port=$((BASE_PORT + idx))
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
    "cd ${REPO} && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
       --profile ${PROFILE} --host '*' --port ${port} --device cuda --feature-slice valid \
       --capture-token-mode ${CAPTURE_TOKEN_MODE} ${LAYERS_ARG} --capture-vl \
       > ${LOG}/fs_port${port}.log 2>&1"
  echo "[server] gpu=${gpu} port=${port} 기동 (token_mode=${CAPTURE_TOKEN_MODE})"
done

# ── 2. 모든 서버 ready 대기 (ZMQ ping) ──
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}; port=$((BASE_PORT + idx))
  ok=0
  for i in $(seq 1 50); do
    if ping_server "${port}"; then ok=1; break; fi
    if docker exec groot bash -lc "grep -qiE 'Error|Traceback' ${LOG}/fs_port${port}.log 2>/dev/null"; then
      echo "[server] gpu=${gpu} port=${port} ERROR"; docker exec groot bash -lc "tail -5 ${LOG}/fs_port${port}.log"; exit 1; fi
    sleep 10
  done
  [ "${ok}" = 1 ] && echo "[server] gpu=${gpu} ready(port ${port})" || { echo "[server] gpu=${gpu} TIMEOUT"; exit 1; }
done

# ── 3. worker (background): round-robin task shard + ep 분배 수집 ──
WORKERS_PER_SERVER=${WORKERS_PER_SERVER:-1}
N_WORKERS=$((NPROC * WORKERS_PER_SERVER))
echo "[setup] workers=${N_WORKERS} (servers=${NPROC} × ${WORKERS_PER_SERVER}/server)"

worker() {
  local widx=$1
  local server_idx=$((widx / WORKERS_PER_SERVER))
  local sub_idx=$((widx % WORKERS_PER_SERVER))
  local gpu=${GPUS[$server_idx]}
  local port=$((BASE_PORT + server_idx))
  for t in "${!TASKS[@]}"; do
    [ $((t % NPROC)) -ne "${server_idx}" ] && continue
    local task="${TASKS[$t]}"; local env="robocasa_panda_omron/${task}_PandaOmron_Env"
    local tdir="${OUT_ROOT}/${task}"; local tdir_host="${OUT_ROOT_HOST}/${task}"
    mkdir -p "${tdir_host}" "${EP_META_ROOT_HOST}/${task}"
    for ep in $(seq 0 $((EPISODES_PER_TASK - 1))); do
      [ $((ep % WORKERS_PER_SERVER)) -ne "${sub_idx}" ] && continue
      local seed=$((SEED_START + ep))
      if ls "${tdir_host}/task${t}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
      docker exec -e MUJOCO_GL=egl -e ROBOCASA_ENV_SOURCE=${ENV_SOURCE} -e PYTHONPATH="${PYPATH}" robocasa \
        python ${REPO}/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py \
          --policy-client-host 127.0.0.1 --policy-client-port ${port} \
          --feature-endpoint get_action_with_multilayer_features \
          --env-name "${env}" --robocasa-env-source ${ENV_SOURCE} \
          --output-dir "${tdir}" --task-id ${t} --episode-start-idx ${ep} \
          --n-episodes 1 --seed ${seed} --inference-seed ${seed} \
          --ep-meta-dir "${EP_META_ROOT}/${task}" \
          >> ${LOG}/collect_port${port}_w${sub_idx}.log 2>&1
    done
    echo "[worker gpu=${gpu} w${sub_idx}] task${t} ${task} done"
  done
}

for widx in $(seq 0 $((N_WORKERS - 1))); do worker "${widx}" & done
wait

# ── 4. cleanup 은 trap(EXIT) 가 수행 (이 스크립트 포트만 kill) ──
echo "[done] 수집 완료 -> ${OUT_ROOT}"
find "${OUT_ROOT_HOST}" -name "*.pkl" | wc -l | xargs echo "pkl 개수:"
if [[ "${VERIFY_AFTER_COLLECT}" == "1" ]]; then
  dit_hidden_shape="$(expected_dit_hidden_shape)"
  dit_feature_kind="$(expected_dit_feature_kind)"
  echo "[verify] DiT+VL pkl contract 확인"
  if ! "${PYTHON}" scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
      "${OUT_ROOT_HOST}" \
      --tasks-override "${TASKS[@]}" \
      --episodes-per-task "${EPISODES_PER_TASK}" \
      --expected-feature-kind "${dit_feature_kind}" \
      --expected-hidden-shape "${dit_hidden_shape}" \
      --expected-model-family groot_n16 \
      --expected-policy-transport zmq \
      --expected-task-suite-name groot_n16_robocasa \
      --expected-video-source groot_upstream_video_recording_wrapper \
      --expected-model-horizon 50 \
      --expected-valid-horizon 16 \
      --require-vl-hidden-states \
      --expected-vl-hidden-shape 2048 \
      --expected-vl-feature-kind groot_n16_vlln_seq_meanpool \
      --expected-vl-feature-dim 2048; then
    echo "[verify] ERROR: DiT+VL pkl contract 검증 실패" >&2
    exit 1
  fi
fi
