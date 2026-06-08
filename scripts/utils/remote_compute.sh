#!/bin/bash
# 원격 compute 노드 워크플로우 헬퍼 (단일 출처).
#
# 동기: 대용량 rollout(raw_rollouts)이 원격 노드에 쌓여 있고, 분석/conceptor fit 같은
# 순수 CPU·numpy 작업은 데이터가 있는 원격에서 돌리고 결과(NPZ/plot/JSON, 소용량)만
# 회수하는 편이 34GB 재전송보다 싸다. 코드 동기화는 git 브랜치로 한다(scp 금지).
#
# 워크플로우:
#   1) 로컬에서 브랜치 작업 → push.        remote_compute.sh sync-code [branch]
#   2) 원격이 그 브랜치로 checkout (위 명령이 한 번에 처리).
#   3) 원격에서 분석/fit 실행.             remote_compute.sh run <cmd...>   (긴 작업은 run-bg)
#   4) 결과만 회수.                         remote_compute.sh pull-results <relpath...>
#   - 로컬에 쌓은 데이터를 원격으로 보낼 때: push-data <relpath...>
#
# 설정 (env 로 override, 기본값은 현재 원격 노드):
#   REMOTE_USER=kimseungjun  REMOTE_HOST=166.104.146.37  REMOTE_PORT=11112
#   REMOTE_REPO=~/workspace/temporal_vla
#
# 원격 env 제약 (2026-06-05 기준):
#   - base python3: numpy 1.21.5 + matplotlib 3.5.1, 단 **torch·scipy 없음**.
#   - rollout pkl 은 torch 텐서를 담고 있어 unpickle 에 torch 필요 → 분석/fit 은
#     **${REMOTE_PYTHON} (=~/anaconda3/bin/python, torch+numpy+matplotlib 보유)** 으로 돌린다.
#   - scipy 는 어느 python 에도 없음 → scipy 의존 코드는 원격 실행 불가.
#   - 무거운 numpy 는 OMP/OPENBLAS_NUM_THREADS 로 cap(공유 노드 예의).
#   run/run-bg 안에서 python 호출은 ${REMOTE_PYTHON} 변수를 export 해 두니 그걸 쓴다.
#
# 사용 예:
#   bash scripts/utils/remote_compute.sh sync-code feat/vl-pathway-steering
#   bash scripts/utils/remote_compute.sh run python3 scripts/safe/.../vl_dit_lda_analysis.py --task-dir ...
#   bash scripts/utils/remote_compute.sh run-bg phaseAB bash /tmp/phase_ab.sh
#   bash scripts/utils/remote_compute.sh tail phaseAB
#   bash scripts/utils/remote_compute.sh pull-results outputs/eval/robocasa/groot_n16/<run>/analysis

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-kimseungjun}"
REMOTE_HOST="${REMOTE_HOST:-166.104.146.37}"
REMOTE_PORT="${REMOTE_PORT:-11112}"
REMOTE_REPO="${REMOTE_REPO:-~/workspace/temporal_vla}"
REMOTE_THREADS="${REMOTE_THREADS:-8}"
# rollout pkl 이 torch 텐서라 unpickle 에 torch 필요 → anaconda python (torch 보유) 기본.
REMOTE_PYTHON="${REMOTE_PYTHON:-~/anaconda3/bin/python}"

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH=(ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}")
TARGET="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_LOG_DIR="/tmp/remote_compute_logs"

usage() {
  sed -n '2,32p' "${BASH_SOURCE[0]}"
  exit "${1:-0}"
}

# 원격에서 thread cap + REMOTE_PYTHON export + repo cd 후 명령 실행.
_remote_exec() {  # $@ = command
  local caps="export OMP_NUM_THREADS=${REMOTE_THREADS} OPENBLAS_NUM_THREADS=${REMOTE_THREADS} MKL_NUM_THREADS=${REMOTE_THREADS} NUMEXPR_NUM_THREADS=${REMOTE_THREADS} REMOTE_PYTHON=${REMOTE_PYTHON}"
  "${SSH[@]}" "cd ${REMOTE_REPO} && ${caps} && $*"
}

cmd_sync_code() {  # [branch]
  local branch="${1:-$(git -C "${LOCAL_REPO}" branch --show-current)}"
  echo "[sync-code] 로컬 push: ${branch} -> origin"
  git -C "${LOCAL_REPO}" push -u origin "${branch}"
  echo "[sync-code] 원격 checkout: ${branch}"
  _remote_exec "git fetch origin && git checkout ${branch} && git pull --ff-only origin ${branch}"
  echo "[sync-code] 원격 HEAD:"
  _remote_exec "git log --oneline -1"
}

cmd_push_data() {  # <relpath...>
  [ "$#" -ge 1 ] || { echo "push-data <relpath...> 필요"; exit 2; }
  for rel in "$@"; do
    echo "[push-data] ${rel} -> ${TARGET}:${REMOTE_REPO}/${rel}"
    _remote_exec "mkdir -p \"\$(dirname ${REMOTE_REPO}/${rel})\""
    rsync -a --info=progress2 -e "ssh -p ${REMOTE_PORT}" \
      "${LOCAL_REPO}/${rel}" "${TARGET}:${REMOTE_REPO}/$(dirname "${rel}")/"
  done
}

cmd_pull_results() {  # <relpath...>
  [ "$#" -ge 1 ] || { echo "pull-results <relpath...> 필요"; exit 2; }
  for rel in "$@"; do
    echo "[pull-results] ${TARGET}:${REMOTE_REPO}/${rel} -> ${rel}"
    mkdir -p "${LOCAL_REPO}/$(dirname "${rel}")"
    rsync -a --info=progress2 -e "ssh -p ${REMOTE_PORT}" \
      "${TARGET}:${REMOTE_REPO}/${rel}" "${LOCAL_REPO}/$(dirname "${rel}")/"
  done
}

cmd_run() {  # <command...>  (foreground)
  [ "$#" -ge 1 ] || { echo "run <command...> 필요"; exit 2; }
  _remote_exec "$*"
}

cmd_run_bg() {  # <logname> <command...>
  [ "$#" -ge 2 ] || { echo "run-bg <logname> <command...> 필요"; exit 2; }
  local name="$1"; shift
  local log="${REMOTE_LOG_DIR}/${name}.log"
  _remote_exec "mkdir -p ${REMOTE_LOG_DIR} && nohup bash -lc 'export OMP_NUM_THREADS=${REMOTE_THREADS} OPENBLAS_NUM_THREADS=${REMOTE_THREADS} MKL_NUM_THREADS=${REMOTE_THREADS} REMOTE_PYTHON=${REMOTE_PYTHON}; cd ${REMOTE_REPO}; $*' > ${log} 2>&1 & echo \"[run-bg] pid=\$! log=${log}\""
}

cmd_tail() {  # <logname>
  [ "$#" -ge 1 ] || { echo "tail <logname> 필요"; exit 2; }
  _remote_exec "tail -n ${TAIL_N:-40} ${REMOTE_LOG_DIR}/$1.log"
}

cmd_shell() { exec "${SSH[@]}" -t "cd ${REMOTE_REPO} && exec bash -l"; }

case "${1:-}" in
  sync-code)    shift; cmd_sync_code "$@" ;;
  push-data)    shift; cmd_push_data "$@" ;;
  pull-results) shift; cmd_pull_results "$@" ;;
  run)          shift; cmd_run "$@" ;;
  run-bg)       shift; cmd_run_bg "$@" ;;
  tail)         shift; cmd_tail "$@" ;;
  shell)        cmd_shell ;;
  -h|--help|"") usage 0 ;;
  *) echo "unknown subcommand: $1"; usage 2 ;;
esac
