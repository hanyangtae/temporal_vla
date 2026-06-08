#!/bin/bash
# Multi-layer activation 병렬 수집 (COAST A.10.2 Stage1 layer sweep 용).
#
# free GPU N개에 (groot 컨테이너 feature_server[multilayer endpoint] + robocasa 컨테이너
# collect_rollout client) 쌍을 띄우고, seen18 18 task를 GPU에 round-robin 분배해 병렬 수집한다.
# task_id는 18-task canonical index(global)로 고정 → shard 간 충돌 없음.
#
# feature_server는 steering 없이 get_action_with_multilayer_features 로 transformer_blocks 전체
# (32 layer) residual stream(1536-dim, K·H mean pooled)을 env step당 [L,D]로 반환.
#
# 호스트에서 실행 (docker 권한 필요). 결과: outputs/.../<RUN_ID>/raw_rollouts/<task>/*.pkl
set -u

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
REPO=/temporal_vla
cd "${REPO_HOST}"

GPUS=(${GPUS:-0 1 2 3 4 5})
EPISODES_PER_TASK=${EPISODES_PER_TASK:-15}
SEED_START=${SEED_START:-200000}
BASE_PORT=${BASE_PORT:-5601}
PROFILE=${PROFILE:-${REPO}/configs/checkpoints/groot__robocasa365_ckpt120000.yaml}
ENV_SOURCE=robocasa365
RUN_ID=${RUN_ID:-target_atomic_seen18_multilayer_15ep}
OUT_ROOT=${REPO}/outputs/eval/robocasa/groot_n16/${RUN_ID}/raw_rollouts
OUT_ROOT_HOST=outputs/eval/robocasa/groot_n16/${RUN_ID}/raw_rollouts
LOG=/tmp/ml_collect
NPROC=${#GPUS[@]}

# seen18 canonical 순서 = global task_id (CloseBlenderLid=0 ... TurnOnSinkFaucet=17)
TASKS=(CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen
       OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet
       PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter
       PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove TurnOnElectricKettle
       TurnOnMicrowave TurnOnSinkFaucet)
# TASKS_OVERRIDE(공백구분 task명)로 subset 수집 가능. task_id = override 리스트 내 index.
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  TASKS=( ${TASKS_OVERRIDE} )
fi

PYPATH="${REPO}/src/policies/Isaac-GR00T:${REPO}/src/benchmarks/robocasa:${REPO}/src/benchmarks/robosuite:${REPO}"
mkdir -p "${OUT_ROOT_HOST}"
mkdir -p "${LOG}"                                  # 호스트: collect_rollout 로그 redirect용
docker exec groot bash -lc "mkdir -p ${LOG}" 2>/dev/null  # 컨테이너: feature_server 로그용
echo "[setup] GPUs=${GPUS[*]} eps/task=${EPISODES_PER_TASK} tasks=${#TASKS[@]} out=${OUT_ROOT}"

# PolicyServer 의 "ready" print 는 flush 안 되므로 ZMQ ping 으로 readiness 판정.
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

# ── 1. GPU당 feature_server(multilayer) 기동 ──
# CAPTURE_TOKEN_MODE: valid (현행 K·H mean, action16), all (K·H mean, action50),
# full (K mean 만, T 전체 보존 → COAST p24 모방용)
CAPTURE_TOKEN_MODE=${CAPTURE_TOKEN_MODE:-valid}
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}; port=$((BASE_PORT + idx))
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
    "cd ${REPO} && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
       --profile ${PROFILE} --host '*' --port ${port} --device cuda --feature-slice valid \
       --capture-token-mode ${CAPTURE_TOKEN_MODE} \
       > ${LOG}/fs_gpu${gpu}.log 2>&1"
  echo "[server] gpu=${gpu} port=${port} 기동 (token_mode=${CAPTURE_TOKEN_MODE})"
done

# ── 2. 모든 서버 ready 대기 (ZMQ ping) ──
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}; port=$((BASE_PORT + idx))
  ok=0
  for i in $(seq 1 50); do
    if ping_server "${port}"; then ok=1; break; fi
    if docker exec groot bash -lc "grep -qiE 'Error|Traceback' ${LOG}/fs_gpu${gpu}.log 2>/dev/null"; then
      echo "[server] gpu=${gpu} ERROR"; docker exec groot bash -lc "tail -5 ${LOG}/fs_gpu${gpu}.log"; exit 1; fi
    sleep 10
  done
  [ "${ok}" = 1 ] && echo "[server] gpu=${gpu} ready(port ${port})" || { echo "[server] gpu=${gpu} TIMEOUT"; exit 1; }
done

# ── 3. worker (background): round-robin task shard + ep 분배 수집 ──
# WORKERS_PER_SERVER>1 이면 같은 server(port)에 여러 worker 가 ZMQ 다중접속해
# server idle 을 fill (GPU util ↑). ep 는 sub_idx 별로 짝수/홀수/... 분배.
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
    mkdir -p "${tdir_host}"
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
          --n-episodes 1 --seed ${seed} >> ${LOG}/collect_gpu${gpu}_w${sub_idx}.log 2>&1
    done
    echo "[worker gpu=${gpu} w${sub_idx}] task${t} ${task} done"
  done
}

cleanup() {
  echo "[cleanup] feature_server 정리 (groot 컨테이너)"
  docker exec groot bash -lc "pkill -f feature_server.py" 2>/dev/null
  sleep 5
  local left
  left=$(docker exec groot bash -lc 'pgrep -fc feature_server.py' 2>/dev/null)
  if [ "${left:-0}" -gt 0 ]; then
    echo "[cleanup] WARN ${left} server still alive — SIGKILL"
    docker exec groot bash -lc "pkill -9 -f feature_server.py" 2>/dev/null
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8
}
trap cleanup EXIT INT TERM

for widx in $(seq 0 $((N_WORKERS - 1))); do worker "${widx}" & done
wait

echo "[done] 수집 완료 -> ${OUT_ROOT}"
find "${OUT_ROOT_HOST}" -name "*.pkl" | wc -l | xargs echo "pkl 개수:"
