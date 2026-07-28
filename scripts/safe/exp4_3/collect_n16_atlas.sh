#!/bin/bash
# exp4-3 분리도 지도: GR00T N1.6 full-token 활성 수집 + 에피소드당 승준 직송.
#
# serve  = groot 컨테이너 feature_server.py, get_action_with_multilayer_features,
#          --capture-token-mode full (K mean·T[51] 보존, 32 DiT block, [L,T,D] per record).
# collect= robocasa 컨테이너, WORKTREE collect_rollout.py (--label-phases 배선) + phase 라벨.
#          컨테이너는 메인 트리를 /temporal_vla 로 마운트 → gr00t/robocasa 는 메인 트리 것(serve와 동일).
#          WORKTREE 스크립트만 실행해 phase 배선 사용(격리: 메인 트리 편집 없음).
# ship   = 에피소드마다 pkl+csv+mp4 를 승준 HDD 아카이브로 rsync → 크기 검증 → 로컬 pkl 삭제.
#          (로컬 30GB 여유 보호 · 토큰 pool 금지라 full pkl 원본을 승준에 보존.)
#
# 멱등: 원격에 task{tid}--ep{ep}--succ*.pkl 이 이미 있으면 그 ep skip.
# 사용: GPUS="4 5 6" EPS=30 bash collect_n16_atlas.sh   (호스트에서, docker 권한)
set -u

REPO_HOST=/home/dongkyu/pkt_ws/temporal_vla
REPO=/temporal_vla
WT=${REPO}/.claude/worktrees/exp4-3-atlas
WT_HOST=${REPO_HOST}/.claude/worktrees/exp4-3-atlas

GPUS=(${GPUS:-4 5 6})
WORKERS_PER_SERVER=${WORKERS_PER_SERVER:-2}
EPS=${EPS:-30}
SEED_START=${SEED_START:-100000}
BASE_PORT=${BASE_PORT:-8640}          # 격리 대역 8640~8659
PROFILE=${REPO}/configs/checkpoints/groot__robocasa365_ckpt120000.yaml
ENV_SOURCE=robocasa365
NACT=${NACT:-5}                       # get_action 당 실행 action 수(steering 표준=5)

# task:canonical_id (collect_multilayer_parallel.sh seen18 index). env=robocasa_panda_omron/<task>_PandaOmron_Env
TASKS=(${TASKS_OVERRIDE:-OpenDrawer:6 PickPlaceCounterToCabinet:8 OpenStandMixerHead:7})

STAGE_HOST=${WT_HOST}/outputs/eval/robocasa/groot_n16/exp4_3/collect
STAGE=${WT}/outputs/eval/robocasa/groot_n16/exp4_3/collect
LOG=${STAGE_HOST}/logs
mkdir -p "${LOG}"

REMOTE=kimseungjun@166.104.146.37; RPORT=11112
RARCH=datasets/temporal_vla_outputs/eval/robocasa/groot_n16/exp4_3/collect
RSH="ssh -p ${RPORT}"

PYP="${REPO}/src/policies/Isaac-GR00T:${REPO}/src/benchmarks/robocasa:${REPO}/src/benchmarks/robosuite:${REPO}"
NPROC=${#GPUS[@]}
N_WORKERS=$((NPROC * WORKERS_PER_SERVER))
echo "[setup] GPUs=${GPUS[*]} workers=${N_WORKERS} eps/task=${EPS} tasks=${#TASKS[@]} nact=${NACT}"
echo "[setup] stage=${STAGE_HOST}  remote=${REMOTE}:~/${RARCH}"

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

# ── 1. serve 기동 (GPU당 1개, full 모드, DiT 전용) ──
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}; port=$((BASE_PORT + idx))
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
    "cd ${REPO} && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
       --profile ${PROFILE} --host '*' --port ${port} --device cuda --feature-slice valid \
       --capture-token-mode full > ${STAGE}/logs/fs_gpu${gpu}_p${port}.log 2>&1" 2>/dev/null
  echo "[server] gpu=${gpu} port=${port} full-mode 기동"
done

# ── 2. ready 대기 (ZMQ ping) ──
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}; port=$((BASE_PORT + idx)); ok=0
  for i in $(seq 1 40); do
    if ping_server "${port}"; then ok=1; break; fi
    if docker exec groot bash -lc "grep -qiE 'Error|Traceback' ${STAGE}/logs/fs_gpu${gpu}_p${port}.log 2>/dev/null"; then
      echo "[server] gpu=${gpu} ERROR"; docker exec groot bash -lc "tail -8 ${STAGE}/logs/fs_gpu${gpu}_p${port}.log"; exit 1; fi
    sleep 8
  done
  [ "${ok}" = 1 ] && echo "[server] gpu=${gpu} ready(port ${port})" || { echo "[server] gpu=${gpu} TIMEOUT"; exit 1; }
done

# ── ship: 에피소드 산출물 → 승준 → 검증 → 로컬 pkl 삭제 ──
ship_ep() {  # $1=stem_noext(host) $2=task
  local base=$1 task=$2 name; name=$(basename "${base}")
  ${RSH} ${REMOTE} "mkdir -p ~/${RARCH}/${task}" 2>/dev/null
  rsync -a -e "${RSH}" "${base}.pkl" "${base}.csv" "${base}.mp4" "${REMOTE}:~/${RARCH}/${task}/" 2>/dev/null
  local lsz rsz
  lsz=$(stat -c%s "${base}.pkl" 2>/dev/null)
  rsz=$(${RSH} ${REMOTE} "stat -c%s ~/${RARCH}/${task}/${name}.pkl" 2>/dev/null)
  if [ -n "${lsz}" ] && [ "${lsz}" = "${rsz}" ]; then
    rm -f "${base}.pkl"; echo "[ship] ${task}/${name} OK(${lsz}B) 로컬pkl삭제"
  else
    echo "[ship] ${task}/${name} MISMATCH local=${lsz} remote=${rsz} — 로컬 유지"
  fi
}

# ── 3. worker: task(server 1:1) × ep(subworker parity) ──
worker() {
  local widx=$1
  local server_idx=$((widx / WORKERS_PER_SERVER))
  local sub_idx=$((widx % WORKERS_PER_SERVER))
  local port=$((BASE_PORT + server_idx))
  for ti in "${!TASKS[@]}"; do
    [ $((ti % NPROC)) -ne "${server_idx}" ] && continue
    local task tid; IFS=: read -r task tid <<< "${TASKS[$ti]}"
    local env="robocasa_panda_omron/${task}_PandaOmron_Env"
    local tdir_host="${STAGE_HOST}/${task}" tdir="${STAGE}/${task}"
    mkdir -p "${tdir_host}"
    for ep in $(seq 0 $((EPS - 1))); do
      [ $((ep % WORKERS_PER_SERVER)) -ne "${sub_idx}" ] && continue
      local seed=$((SEED_START + ep))
      if ${RSH} ${REMOTE} "ls ~/${RARCH}/${task}/task${tid}--ep${ep}--succ*.pkl" >/dev/null 2>&1; then
        echo "[skip] ${task} ep${ep} 원격에 이미 있음"; continue; fi
      docker exec -e MUJOCO_GL=egl -e ROBOCASA_ENV_SOURCE=${ENV_SOURCE} -e PYTHONPATH="${PYP}" robocasa \
        python ${WT}/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py \
          --policy-client-host 127.0.0.1 --policy-client-port ${port} \
          --feature-endpoint get_action_with_multilayer_features \
          --env-name "${env}" --robocasa-env-source ${ENV_SOURCE} \
          --output-dir "${tdir}" --task-id ${tid} --episode-start-idx ${ep} \
          --n-episodes 1 --seed ${seed} --n_action_steps ${NACT} \
          --label-phases --proximity-phases >> ${LOG}/collect_w${widx}.log 2>&1
      local produced; produced=$(ls ${tdir_host}/task${tid}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
      if [ -n "${produced}" ]; then ship_ep "${produced%.pkl}" "${task}";
      else echo "[worker w${widx}] ${task} ep${ep}: pkl 미생성(로그 확인)"; fi
    done
  done
  echo "[worker w${widx}] done (server=${server_idx} sub=${sub_idx})"
}

cleanup() {
  echo "[cleanup] 내 serve 정리 (포트 ${BASE_PORT}~$((BASE_PORT+NPROC-1)))"
  for idx in "${!GPUS[@]}"; do
    local port=$((BASE_PORT + idx))
    docker exec groot bash -lc "pkill -f 'feature_server.py.*--port ${port}'" 2>/dev/null
  done
  sleep 4
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '5,8p'
}
trap cleanup EXIT INT TERM

for widx in $(seq 0 $((N_WORKERS - 1))); do worker "${widx}" & done
wait
echo "[done] N1.6 수집 완료 → 승준 ~/${RARCH}"
${RSH} ${REMOTE} "find ~/${RARCH} -name '*.pkl' | wc -l" | xargs echo "[done] 원격 pkl 총:"
