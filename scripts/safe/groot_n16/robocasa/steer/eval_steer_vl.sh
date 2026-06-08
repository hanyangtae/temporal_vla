#!/bin/bash
# VL pathway conceptor steering eval: 10 task × 3 condition (baseline + 2 β) = 30 eval.
#
# 3 condition:
#   baseline — steering 없음 (β=0 → M=I)
#   VL_b01   — VL pathway conceptor, β=0.1
#   VL_b03   — VL pathway conceptor, β=0.3
#
# Type-matched vs unmatched 비교:
#   VL 우위 task (SlideDishwasherRack, CloseToasterOvenDoor): VL steer = matched → ΔSR>0 기대
#   DiT 우위 task (OpenCabinet, OpenDrawer, PnP tasks): VL steer = unmatched → control
#   neutral task: control
#
# CONDS 형식: "label:beta"  (VL pathway는 layer 선택 없음 — action_head.vlln 전용)
#
# NPZ 경로: ${NPZ_ROOT}/truncated_w${TRUNC_W}/task_${id}_${name}/conceptors.npz
# TRUNC_W: fit_conceptor_steering.py --pathway vl --agg-mode truncated 산출값과 맞춰야 함.
# 기본 18. fit 완료 후 실제 W 확인하여 override:
#   TRUNC_W=26 RUN_TAG=vl_w26 bash eval_steer_vl.sh
#
# 출력: outputs/.../steer_eval/<RUN_TAG>/results.tsv
#
# 전제:
#   1) fit_conceptor_steering.py --pathway vl --per-task 실행 완료
#   2) groot/robocasa 컨테이너 기동 중

set -u

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
REPO=/temporal_vla
cd "${REPO_HOST}"

GPUS=(${GPUS:-0 1 2 3 4 5})
N_EP=${N_EP:-20}
N_ENVS=${N_ENVS:-2}
N_ACT=${N_ACT:-8}
MAXS=${MAXS:-720}
BASE_PORT=${BASE_PORT:-5801}
PROFILE=${PROFILE:-${REPO}/configs/checkpoints/groot__robocasa365_ckpt120000.yaml}
RUN_TAG=${RUN_TAG:-steer_eval_vl}

# VL fit 산출물이 있는 run (raw_rollouts + analysis 포함).
RUN_DIR=${RUN_DIR:-outputs/eval/robocasa/groot_n16/target_atomic_moderate10_pathway_pertoken_100ep}

# truncated_w 값 — fit 완료 후 metadata.json 의 max_len 값과 맞춰야 함.
TRUNC_W=${TRUNC_W:-18}

NPZ_ROOT=${REPO}/${RUN_DIR}/analysis/conceptor_steering_vl
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
# CONDS 형식: "label:beta"
#   label — TSV/log 식별자
#   beta  — steering 강도. 0.0=baseline (M=I)
CONDS=("baseline:0.0" "VL_b01:0.1" "VL_b03:0.3")
# 환경변수로 subset override 가능 (공백구분).
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then TASKS=( ${TASKS_OVERRIDE} ); fi
if [[ -n "${CONDS_OVERRIDE:-}" ]]; then CONDS=( ${CONDS_OVERRIDE} ); fi
NPROC=${#GPUS[@]}
RESULTS_HOST=${RES_DIR_HOST}/results.tsv
RESULTS=${RES_DIR}/results.tsv
echo -e "task\tpathway\tbeta\tcondition\tsuccess_rate\tn_ep" > "${RESULTS_HOST}"

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
  local label beta
  IFS=':' read -r label beta <<<"${cond}"
  local npz="${NPZ_ROOT}/truncated_w${TRUNC_W}/task_${tidx}_${tname}/conceptors.npz"
  local fslog="${LOG}/fs_gpu${gpu}_${tname}_${label}.log"
  local evlog="${LOG}/ev_gpu${gpu}_${tname}_${label}.log"
  local env="robocasa_panda_omron/${tname}_PandaOmron_Env"

  kill_port_server "${port}"
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
    "cd ${REPO} && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
       --profile ${PROFILE} --host '*' --port ${port} --device cuda --feature-slice valid \
       --steering-pathway vl --steering-npz ${npz} --steering-beta ${beta} \
       > ${fslog} 2>&1"

  local ok=0
  for i in $(seq 1 60); do
    if ping_server "${port}"; then ok=1; break; fi
    if docker exec groot bash -lc "grep -qiE 'Error|Traceback' ${fslog} 2>/dev/null"; then
      echo "[gpu${gpu} ${tname} ${label}] SERVER_ERR"
      docker exec groot bash -lc "tail -3 ${fslog}"
      echo -e "${tname}\tvl\t${beta}\t${label}\tSERVER_FAIL\t0" >> "${RESULTS_HOST}"
      return
    fi
    sleep 5
  done
  if [ "${ok}" != 1 ]; then
    echo "[gpu${gpu} ${tname} ${label}] TIMEOUT"
    echo -e "${tname}\tvl\t${beta}\t${label}\tTIMEOUT\t0" >> "${RESULTS_HOST}"
    return
  fi
  sleep 5  # bind grace

  docker exec -e POLICY_CLIENT_HOST=127.0.0.1 -e PORT=${port} robocasa bash -lc \
    "bash ${REPO}/scripts/eval/groot_robocasa.sh client ${env} ${N_EP} ${N_ENVS} ${N_ACT} ${MAXS} > ${evlog} 2>&1"

  local sr
  sr=$(docker exec robocasa bash -lc "grep -i 'success rate' ${evlog} | tail -1 | sed 's/.*success rate:[[:space:]]*//'")
  sr=$(echo "${sr}" | tr -d ' \r')
  [ -z "${sr}" ] && sr="PARSE_FAIL"
  echo -e "${tname}\tvl\t${beta}\t${label}\t${sr}\t${N_EP}" >> "${RESULTS_HOST}"
  echo "[gpu${gpu}] ${tname} ${label} (VL β${beta}) SR=${sr}"
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

echo "[start] GPUs=${GPUS[*]} tasks=${#TASKS[@]} conds=${#CONDS[@]} N_ep=${N_EP} TRUNC_W=${TRUNC_W}"
echo "        NPZ_ROOT=${NPZ_ROOT}"
echo "        out=${RES_DIR}"
for idx in "${!GPUS[@]}"; do worker "${idx}" & done
wait

echo "[done] -> ${RESULTS_HOST}"
column -t -s $'\t' "${RESULTS_HOST}" 2>/dev/null || cat "${RESULTS_HOST}"
