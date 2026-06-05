#!/bin/bash
# T-full perT raw_rollouts 를 한양대 kimseungjun 서버로 rsync archive 후 로컬 삭제.
#
# 전제: ssh key 가 등록돼 있어야 함 (host: 166.104.146.37, port 11112, user kimseungjun).
#   사용자가 한 번만 실행: ssh-copy-id -p 11112 kimseungjun@166.104.146.37
#
# 검증 후 자동 삭제. NPZ + JSON + 06 doc 은 보존 (raw_rollouts/ 만 이동).
set -uo pipefail

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_HOST}"

RUN_ID=${RUN_ID:-target_atomic_moderate10_multilayer_perT_100ep}
LOCAL=outputs/eval/robocasa/groot_n16/${RUN_ID}/raw_rollouts
REMOTE_HOST=kimseungjun@166.104.146.37
REMOTE_PORT=11112
REMOTE=/home/kimseungjun/workspace/temporal_vla/outputs/eval/robocasa/groot_n16/${RUN_ID}/raw_rollouts

echo "[archive] local: ${LOCAL}"
echo "[archive] remote: ${REMOTE_HOST}:${REMOTE} (port ${REMOTE_PORT})"

# Step 0: 키 인증 확인
echo "[step 0] ssh key auth check"
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 -p ${REMOTE_PORT} ${REMOTE_HOST} "echo OK" >/dev/null 2>&1; then
  echo "[ERR] ssh key 미등록. 사용자가 먼저: ssh-copy-id -p ${REMOTE_PORT} ${REMOTE_HOST}"
  exit 1
fi

# Step 1: local pkl 개수 + sha 매니페스트
echo "[step 1] local manifest"
LOCAL_N=$(find "${LOCAL}" -name "*.pkl" | wc -l)
echo "  local pkl: ${LOCAL_N}"
LOCAL_SIZE=$(du -sb "${LOCAL}" | cut -f1)
echo "  local size: $(numfmt --to=iec ${LOCAL_SIZE})"

# Step 2: remote dir 생성
echo "[step 2] mkdir remote parent"
ssh -o BatchMode=yes -p ${REMOTE_PORT} ${REMOTE_HOST} "mkdir -p $(dirname ${REMOTE})" || { echo "[ERR] remote mkdir"; exit 1; }

# Step 3: rsync (재개 가능, atomic per-file)
echo "[step 3] rsync (partial+progress, 재개 가능)"
rsync -av --partial --info=progress2 -e "ssh -p ${REMOTE_PORT} -o BatchMode=yes" \
  "${LOCAL}/" "${REMOTE_HOST}:${REMOTE}/" || { echo "[ERR] rsync 실패"; exit 1; }

# Step 4: 검증 (pkl 개수 + total size 일치)
echo "[step 4] verify"
REMOTE_N=$(ssh -o BatchMode=yes -p ${REMOTE_PORT} ${REMOTE_HOST} "find ${REMOTE} -name '*.pkl' | wc -l")
REMOTE_SIZE=$(ssh -o BatchMode=yes -p ${REMOTE_PORT} ${REMOTE_HOST} "du -sb ${REMOTE} | cut -f1")
echo "  remote pkl: ${REMOTE_N} (local ${LOCAL_N})"
echo "  remote size: $(numfmt --to=iec ${REMOTE_SIZE}) (local $(numfmt --to=iec ${LOCAL_SIZE}))"
if [ "${REMOTE_N}" != "${LOCAL_N}" ]; then
  echo "[ERR] pkl 개수 불일치 — 삭제 안 함, 사용자 확인 필요"
  exit 1
fi
if [ "${REMOTE_SIZE}" != "${LOCAL_SIZE}" ]; then
  echo "[ERR] size 불일치 — 삭제 안 함, 사용자 확인 필요"
  exit 1
fi
echo "[step 4] OK 검증 통과"

# Step 5: 로컬 raw_rollouts 삭제 (NPZ/JSON/doc 보존)
echo "[step 5] local raw_rollouts 삭제"
rm -rf "${LOCAL}"
echo "[done] archive 완료 + 로컬 삭제. analysis 산출물(NPZ/JSON/doc) 보존됨."
df -h /home/dongkyu | tail -1
