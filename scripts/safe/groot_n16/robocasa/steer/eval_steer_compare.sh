#!/bin/bash
# COAST steer eval 비교: 10 task × 5 condition (baseline + 2 layer × 2 β) = 50 eval.
#
# 5 condition (label:layer:beta):
#   baseline (steering 없음에 해당, β=0 → M=I)
#   L0_b01, L0_b03 — balanced 방식 ℓ*=0 에서 β=0.1, 0.3
#   L4_b01, L4_b03 — coast 방식 ℓ*=4 에서 β=0.1, 0.3
#
# 6 GPU 병렬: task 를 round-robin 분배, 각 GPU 가 자기 task 의 5 condition 을 순차 실행.
# 각 condition 마다 해당 GPU 의 feature_server 를 (task-specific NPZ + steering-beta +
# steering-layer 로) 재기동한 뒤 robocasa eval client 실행.
#
# 출력: outputs/.../steer_eval/<RUN_TAG>/results.tsv

set -u

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
REPO=/temporal_vla
cd "${REPO_HOST}"

GPUS=(${GPUS:-4 5 6 4 5 6})
N_EP=${N_EP:-20}
N_ENVS=${N_ENVS:-2}
N_ACT=${N_ACT:-8}
MAXS=${MAXS:-720}
BASE_PORT=${BASE_PORT:-5701}
PROFILE=${PROFILE:-${REPO}/configs/checkpoints/groot__robocasa365_ckpt120000.yaml}
RUN_TAG=${RUN_TAG:-steer_eval_compare}
# Project-wide standard eval seed = collection seed base (do-dong-park 표준, task_sets.sh).
# 같은 (env, seed) → 같은 episode 시리즈 → 동료 수집 episode 와 매칭.
EVAL_SEED=${EVAL_SEED:-100000}

# RUN_DIR: 데이터/산출물 base (RUN_ID 디렉토리). 기본 = Phase 1 (K·H pooled, action16).
RUN_DIR=${RUN_DIR:-outputs/eval/robocasa/groot_n16/target_atomic_moderate10_multilayer_100ep}
NPZ_ROOT=${REPO}/${RUN_DIR}/conceptor_steering
RES_DIR_HOST=${RUN_DIR}/steer_eval/${RUN_TAG}
RES_DIR=${REPO}/${RUN_DIR}/steer_eval/${RUN_TAG}
mkdir -p "${RES_DIR_HOST}"
LOG=/tmp/steer_eval_${RUN_TAG}
mkdir -p "${LOG}"
docker exec groot bash -lc "mkdir -p ${LOG}" 2>/dev/null
docker exec robocasa bash -lc "mkdir -p ${LOG}" 2>/dev/null

# Canonical task 순서 (fit 시점 NPZ 경로의 task_${id}_${name} 와 일치).
CANONICAL=(CloseToasterOvenDoor NavigateKitchen OpenCabinet OpenDrawer PickPlaceCounterToCabinet
           PickPlaceCounterToStove PickPlaceDrawerToCounter SlideDishwasherRack TurnOnMicrowave TurnOnSinkFaucet)
declare -A CANONICAL_ID
for i in "${!CANONICAL[@]}"; do CANONICAL_ID[${CANONICAL[$i]}]=$i; done

TASKS=("${CANONICAL[@]}")
# CONDS 형식: "label:layer_dir:layer_int:beta"
#   label     — TSV/log 식별자
#   layer_dir — NPZ 디렉토리명 (예: "layer4" Phase1, "layer4_valid16" Phase2)
#   layer_int — server --steering-layer 값 (DiT block idx)
#   beta      — steering 강도. 0.0=baseline (M=I)
CONDS=("baseline:layer0:0:0.0" "L0_b01:layer0:0:0.1" "L0_b03:layer0:0:0.3" "L4_b01:layer4:4:0.1" "L4_b03:layer4:4:0.3")
# 환경변수로 subset override 가능 (공백구분).
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then TASKS=( ${TASKS_OVERRIDE} ); fi
if [[ -n "${CONDS_OVERRIDE:-}" ]]; then CONDS=( ${CONDS_OVERRIDE} ); fi
NPROC=${#GPUS[@]}
RESULTS_HOST=${RES_DIR_HOST}/results.tsv
RESULTS=${RES_DIR}/results.tsv
echo -e "task\tlayer\tbeta\tcondition\tsuccess_rate\tn_ep\teval_seed" > "${RESULTS_HOST}"

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

kill_port_server() {  # $1=port
  docker exec groot bash -lc "pkill -f 'feature_server.py.*--port $1' 2>/dev/null; sleep 3" 2>/dev/null
}

run_one() {  # $1=gpu $2=port $3=tidx $4=tname $5=cond
  local gpu=$1 port=$2 tidx=$3 tname=$4 cond=$5
  local label layer_dir layer beta
  IFS=':' read -r label layer_dir layer beta <<<"${cond}"
  local npz="${NPZ_ROOT}/${layer_dir}/truncated_w19/task_${tidx}_${tname}/conceptors.npz"
  local fslog="${LOG}/fs_gpu${gpu}_${tname}_${label}.log"
  local evlog="${LOG}/ev_gpu${gpu}_${tname}_${label}.log"
  local env="robocasa_panda_omron/${tname}_PandaOmron_Env"

  kill_port_server "${port}"
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
    "cd ${REPO} && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
       --profile ${PROFILE} --host '*' --port ${port} --device cuda --feature-slice valid \
       --steering-npz ${npz} --steering-beta ${beta} --steering-layer ${layer} \
       > ${fslog} 2>&1"

  local ok=0
  for i in $(seq 1 60); do
    if ping_server "${port}"; then ok=1; break; fi
    if docker exec groot bash -lc "grep -qiE 'Error|Traceback' ${fslog} 2>/dev/null"; then
      echo "[gpu${gpu} ${tname} ${label}] SERVER_ERR"
      docker exec groot bash -lc "tail -3 ${fslog}"
      echo -e "${tname}\t${layer}\t${beta}\t${label}\tSERVER_FAIL\t0" >> "${RESULTS_HOST}"
      return
    fi
    sleep 5
  done
  if [ "${ok}" != 1 ]; then
    echo "[gpu${gpu} ${tname} ${label}] TIMEOUT"
    echo -e "${tname}\t${layer}\t${beta}\t${label}\tTIMEOUT\t0" >> "${RESULTS_HOST}"
    return
  fi
  sleep 5  # bind grace

  # per-condition video/csv 디렉토리. 기본 EVAL_RUN_ID 가 너무 generic → unique 하게.
  local ev_video="${RES_DIR}/videos/${tname}__${label}"
  docker exec robocasa bash -lc "mkdir -p ${ev_video}" 2>/dev/null
  docker exec -e POLICY_CLIENT_HOST=127.0.0.1 -e PORT=${port} \
    -e EVAL_SEED=${EVAL_SEED} -e VIDEO_DIR=${ev_video} \
    -e EVAL_DEBUG_LANG=${EVAL_DEBUG_LANG:-} robocasa bash -lc \
    "bash ${REPO}/scripts/eval/groot_robocasa.sh client ${env} ${N_EP} ${N_ENVS} ${N_ACT} ${MAXS} > ${evlog} 2>&1"

  local sr
  sr=$(docker exec robocasa bash -lc "grep -i 'success rate' ${evlog} | tail -1 | sed 's/.*success rate:[[:space:]]*//'" | tr -d ' \r')
  [ -z "${sr}" ] && sr="PARSE_FAIL"
  echo -e "${tname}\t${layer}\t${beta}\t${label}\t${sr}\t${N_EP}\t${EVAL_SEED}" >> "${RESULTS_HOST}"
  echo "[gpu${gpu}] ${tname} ${label} (L${layer} β${beta}) seed=${EVAL_SEED} SR=${sr}"
}

worker() {  # $1=idx
  local idx=$1 gpu=${GPUS[$idx]} port=$((BASE_PORT + idx))
  for t in "${!TASKS[@]}"; do
    [ $((t % NPROC)) -ne "${idx}" ] && continue
    local tname=${TASKS[$t]}
    local tid=${CANONICAL_ID[${tname}]}
    for cond in "${CONDS[@]}"; do
      run_one "${gpu}" "${port}" "${tid}" "${tname}" "${cond}"
    done
  done
  kill_port_server "${port}"
}

echo "[start] GPUs=${GPUS[*]} tasks=${#TASKS[@]} conds=${#CONDS[@]} N_ep=${N_EP} out=${RES_DIR}"
for idx in "${!GPUS[@]}"; do worker "${idx}" & done
wait

echo "[done] -> ${RESULTS_HOST}"
column -t -s $'\t' "${RESULTS_HOST}" 2>/dev/null || cat "${RESULTS_HOST}"
