#!/bin/bash
# VL+DiT pathway conceptor steering 통합 SR eval (type-matched vs unmatched).
#
# 같은 pathway_pertoken run·같은 eval 파라미터로 baseline 1회를 VL·DiT 가 공유 →
# ΔSR 을 cross-run 교란 없이 직접 비교. (검토 결정 Q2)
#
# CONDS 형식: "label:pathway:layer:beta:npz_subdir:trunc_w"
#   label      — TSV/log 식별자
#   pathway    — '' (baseline) | 'vl' | 'dit'
#   layer      — dit pathway 의 DiT block idx (vl/baseline 은 공백)
#   beta       — steering 강도. 0.0=baseline (steering hook 미등록)
#   npz_subdir — analysis/ 아래 conceptor 디렉토리 (예: conceptor_steering_vl,
#                conceptor_steering_dit/layer4_valid16). baseline 은 공백.
#   trunc_w    — truncated_w<trunc_w> 폴더 선택. baseline 은 공백.
#
# NPZ 경로: ${RUN_DIR}/analysis/<npz_subdir>/truncated_w<trunc_w>/task_<id>_<name>/conceptors.npz
#
# 기본 CONDS: baseline + VL{0.1,0.3} + DiT{0.1,0.3} (task 당 5 eval, baseline 공유).
# pathway별 W 가 다를 수 있으므로 VL_TRUNC_W / DIT_TRUNC_W 를 fit metadata 와 맞춰 지정.
#
# 전제:
#   1) fit_conceptor_steering.py --pathway vl --per-task (conceptor_steering_vl)
#   2) fit_conceptor_steering.py --select-layer --force-layer 4 (conceptor_steering_dit/layer4_valid16)
#   3) groot/robocasa 컨테이너 기동 중
#
# 출력: outputs/.../steer_eval/<RUN_TAG>/results.tsv

set -u

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
REPO=/temporal_vla
cd "${REPO_HOST}"

# 평가 표준 (CLAUDE.md): GPU 양보 default, 고정 EVAL_SEED, per-episode logging.
GPUS=(${GPUS:-4 5 6 4 5 6})
N_EP=${N_EP:-20}
N_ENVS=${N_ENVS:-2}
N_ACT=${N_ACT:-8}
MAXS=${MAXS:-720}
EVAL_SEED=${EVAL_SEED:-100000}  # 평가 표준: 동료 collection seed (CLAUDE.md)
BASE_PORT=${BASE_PORT:-5801}
PROFILE=${PROFILE:-${REPO}/configs/checkpoints/groot__robocasa365_ckpt120000.yaml}
RUN_TAG=${RUN_TAG:-steer_eval_pathways}

RUN_DIR=${RUN_DIR:-outputs/eval/robocasa/groot_n16/target_atomic_moderate10_pathway_pertoken_100ep}

# pathway별 truncated_w — fit metadata.json 의 max_len 과 맞춰야 함.
VL_TRUNC_W=${VL_TRUNC_W:-18}
DIT_TRUNC_W=${DIT_TRUNC_W:-18}
# DiT steering layer (fit 시 --force-layer 와 일치).
DIT_LAYER=${DIT_LAYER:-4}
DIT_NPZ_SUBDIR=${DIT_NPZ_SUBDIR:-conceptor_steering_dit/layer${DIT_LAYER}_valid16}

ANALYSIS_ROOT=${REPO}/${RUN_DIR}/analysis
RES_DIR_HOST=${RUN_DIR}/steer_eval/${RUN_TAG}
RES_DIR=${REPO}/${RUN_DIR}/steer_eval/${RUN_TAG}
mkdir -p "${RES_DIR_HOST}"
LOG=/tmp/steer_eval_${RUN_TAG}
mkdir -p "${LOG}"
docker exec groot bash -lc "mkdir -p ${LOG}" 2>/dev/null
docker exec robocasa bash -lc "mkdir -p ${LOG}" 2>/dev/null

# Canonical task 순서 (fit NPZ 경로의 task_<id>_<name> 와 일치).
CANONICAL=(CloseToasterOvenDoor NavigateKitchen OpenCabinet OpenDrawer PickPlaceCounterToCabinet
           PickPlaceCounterToStove PickPlaceDrawerToCounter SlideDishwasherRack TurnOnMicrowave TurnOnSinkFaucet)
declare -A CANONICAL_ID
for i in "${!CANONICAL[@]}"; do CANONICAL_ID[${CANONICAL[$i]}]=$i; done

TASKS=("${CANONICAL[@]}")
# 기본 CONDS: baseline 공유 + VL·DiT 각 β{0.1,0.3}.
CONDS=(
  "baseline::::"
  "VL_b01:vl::0.1:conceptor_steering_vl:${VL_TRUNC_W}"
  "VL_b03:vl::0.3:conceptor_steering_vl:${VL_TRUNC_W}"
  "DiT_b01:dit:${DIT_LAYER}:0.1:${DIT_NPZ_SUBDIR}:${DIT_TRUNC_W}"
  "DiT_b03:dit:${DIT_LAYER}:0.3:${DIT_NPZ_SUBDIR}:${DIT_TRUNC_W}"
)
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then TASKS=( ${TASKS_OVERRIDE} ); fi
if [[ -n "${CONDS_OVERRIDE:-}" ]]; then CONDS=( ${CONDS_OVERRIDE} ); fi
NPROC=${#GPUS[@]}
RESULTS_HOST=${RES_DIR_HOST}/results.tsv
echo -e "task\tpathway\tlayer\tbeta\tcondition\tsuccess_rate\tn_ep\teval_seed" > "${RESULTS_HOST}"

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
  local label pathway layer beta npz_subdir trunc_w
  IFS=':' read -r label pathway layer beta npz_subdir trunc_w <<<"${cond}"
  local fslog="${LOG}/fs_gpu${gpu}_${tname}_${label}.log"
  local evlog="${LOG}/ev_gpu${gpu}_${tname}_${label}.log"
  local env="robocasa_panda_omron/${tname}_PandaOmron_Env"

  # steering 인자 구성. baseline(pathway 공백 또는 beta=0) → hook 미등록.
  local steer_args=""
  if [[ -n "${pathway}" && "${beta}" != "0.0" ]]; then
    local npz="${ANALYSIS_ROOT}/${npz_subdir}/truncated_w${trunc_w}/task_${tidx}_${tname}/conceptors.npz"
    # TRUNC_W 가드: NPZ 없으면 즉시 기록 후 skip (하드코딩 W 불일치 방지).
    if ! docker exec groot bash -lc "test -f ${npz}"; then
      echo "[gpu${gpu} ${tname} ${label}] NPZ_MISSING: ${npz}"
      echo -e "${tname}\t${pathway}\t${layer}\t${beta}\t${label}\tNPZ_MISSING\t0\t${EVAL_SEED}" >> "${RESULTS_HOST}"
      return
    fi
    steer_args="--steering-pathway ${pathway} --steering-npz ${npz} --steering-beta ${beta}"
    [[ "${pathway}" == "dit" ]] && steer_args="${steer_args} --steering-layer ${layer}"
  fi

  kill_port_server "${port}"
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
    "cd ${REPO} && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
       --profile ${PROFILE} --host '*' --port ${port} --device cuda --feature-slice valid \
       ${steer_args} > ${fslog} 2>&1"

  local ok=0
  for i in $(seq 1 60); do
    if ping_server "${port}"; then ok=1; break; fi
    if docker exec groot bash -lc "grep -qiE 'Error|Traceback' ${fslog} 2>/dev/null"; then
      echo "[gpu${gpu} ${tname} ${label}] SERVER_ERR"
      docker exec groot bash -lc "tail -3 ${fslog}"
      echo -e "${tname}\t${pathway}\t${layer}\t${beta}\t${label}\tSERVER_FAIL\t0\t${EVAL_SEED}" >> "${RESULTS_HOST}"
      return
    fi
    sleep 5
  done
  if [ "${ok}" != 1 ]; then
    echo "[gpu${gpu} ${tname} ${label}] TIMEOUT"
    echo -e "${tname}\t${pathway}\t${layer}\t${beta}\t${label}\tTIMEOUT\t0\t${EVAL_SEED}" >> "${RESULTS_HOST}"
    return
  fi
  sleep 5  # bind grace

  # per-episode logging: VIDEO_DIR 에 per_episode.tsv(success+instruction) 출력 (평가 표준).
  local ev_video="${RES_DIR}/videos/${tname}__${label}"
  docker exec robocasa bash -lc "mkdir -p ${ev_video}" 2>/dev/null
  docker exec -e POLICY_CLIENT_HOST=127.0.0.1 -e PORT=${port} \
    -e EVAL_SEED=${EVAL_SEED} -e VIDEO_DIR=${ev_video} robocasa bash -lc \
    "bash ${REPO}/scripts/eval/groot_robocasa.sh client ${env} ${N_EP} ${N_ENVS} ${N_ACT} ${MAXS} > ${evlog} 2>&1"

  local sr
  sr=$(docker exec robocasa bash -lc "grep -i 'success rate' ${evlog} | tail -1 | sed 's/.*success rate:[[:space:]]*//'")
  sr=$(echo "${sr}" | tr -d ' \r')
  [ -z "${sr}" ] && sr="PARSE_FAIL"
  echo -e "${tname}\t${pathway}\t${layer}\t${beta}\t${label}\t${sr}\t${N_EP}\t${EVAL_SEED}" >> "${RESULTS_HOST}"
  echo "[gpu${gpu}] ${tname} ${label} (${pathway:-base} β${beta}) seed=${EVAL_SEED} SR=${sr}"
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

echo "[start] GPUs=${GPUS[*]} tasks=${#TASKS[@]} conds=${#CONDS[@]} N_ep=${N_EP}"
echo "        VL_TRUNC_W=${VL_TRUNC_W} DIT_TRUNC_W=${DIT_TRUNC_W} DIT_LAYER=${DIT_LAYER}"
echo "        ANALYSIS_ROOT=${ANALYSIS_ROOT}"
echo "        out=${RES_DIR}"
for idx in "${!GPUS[@]}"; do worker "${idx}" & done
wait

echo "[done] -> ${RESULTS_HOST}"
column -t -s $'\t' "${RESULTS_HOST}" 2>/dev/null || cat "${RESULTS_HOST}"
