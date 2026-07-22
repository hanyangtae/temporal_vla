#!/usr/bin/env bash
# patchceil 재실행 러너 — 지정 ep 구간을 결정적 재수집 (경량 캡처 CAP 1층 pooled).
#
# v2.1 역할 (plan pq3-wise-mist(파일명 유지), 구 pass A 는 승준 zst 실존 확인으로 삭제):
#   ① Phase 4 결정론 스모크 — 소수 ep (EP0..EP1) 재실행 후 succ + actions 를 승준
#      zst 기록과 대조 (check_passA.py 는 ho_base 대조용 — fit 대조는 스모크 스크립트).
#   ② 필요 시 임의 ep 구간 재생 (레시피 = fit 수집·heldout 공통: seed=scenario,
#      inference_seed=ep*1000, NAS=5, MAXEP=720, profile ckpt120000).
#
# 사용 (GPU 배정은 사용자 게이트 — exp3(구 pq3) 점유 확인 후):
#   EP0=0 EP1=4 GPUS_L="4 5 6" PORTS_L="8490 8491 8492" \
#     bash scripts/safe/groot_n15/robocasa/steer/patchceil/rerun_episodes.sh \
#     ppcc_bread_s300033 ppcc_bread_s400020
# resume-safe: 이미 pkl 있는 ep 는 skip. 에피소드당 fresh 프로세스(scene 오염 함정).
set -uo pipefail

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="${CAP:-15}"            # 경량 1층 pooled — actions/판정 확보 목적 (donor 아님)
EP0="${EP0:?EP0 필요 (예: 0)}"; EP1="${EP1:?EP1 필요 (예: 4)}"; STRIDE=1000; NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
CELL_INDEX=5
INSTR="Pick the bread from the counter and place it in the cabinet."
declare -A SEEDS=([ppcc_bread_s300033]=300033 [ppcc_bread_s400020]=400020)

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:?"GPUS_L 필요 (예: \"4 5 6\") — 발사 전 nvidia-smi/exp3 확인(사용자 게이트)"})
PORTS=(${PORTS_L:?"PORTS_L 필요 (예: \"8490 8491 8492\")"})
NW=${#PORTS[@]}
[ ${#GPUS[@]} -eq $NW ] || { echo "GPUS_L/PORTS_L 길이 불일치"; exit 2; }
CELLS=("$@"); [ ${#CELLS[@]} -gt 0 ] || CELLS=(ppcc_bread_s300033 ppcc_bread_s400020)

start_serves() {
  for i in $(seq 0 $((NW-1))); do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda --collect \
         --groot-dit-capture-layers ${CAP} > /tmp/patchceil_passA_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[passA] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || exit 11
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/patchceil_passA_${port}.log"; then
      echo "[passA] ABORT ${port}"; docker exec lerobot bash -lc "tail -20 /tmp/patchceil_passA_${port}.log"; exit 11
    fi
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }
trap kill_serves EXIT

start_serves
for CELL in "${CELLS[@]}"; do
  SEED="${SEEDS[$CELL]:?unknown cell ${CELL}}"
  OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/patchceil/${CELL}/passA"
  OUT_CONT="/temporal_vla/outputs/eval/robocasa/groot_n15/patchceil/${CELL}/passA"
  LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
  echo "[passA] $(date '+%F %T') cell=${CELL} seed=${SEED} ep${EP0}-${EP1}"
  run_w() {
    local wid=$1 port=$2
    for ep in $(seq $EP0 $EP1); do
      [ $((ep % NW)) -eq "$wid" ] || continue
      if ls "${OUT_HOST}/raw_rollouts/${TASK}/${CELL}/task${CELL_INDEX}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
      docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
        python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
        --output-dir "${OUT_CONT}/raw_rollouts" --cell-id "$CELL" --cell-index "$CELL_INDEX" \
        --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
        --seed "$SEED" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
        --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 \
        --wait-ready --proximity-phases 2>&1 \
        | grep -E "^wrote|Error|Traceback" || true
    done
  }
  for wid in $(seq 0 $((NW-1))); do
    run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/passA_w${wid}.log" 2>&1 &
  done
  wait
  n_done=$(ls "${OUT_HOST}/raw_rollouts/${TASK}/${CELL}"/task${CELL_INDEX}--ep*--succ*.pkl 2>/dev/null | wc -l)
  echo "[passA] ${CELL} done pkl=${n_done}/$((EP1-EP0+1))"
done
kill_serves; trap - EXIT
echo "[passA] $(date '+%F %T') ALL DONE — 다음: python3 scripts/safe/groot_n15/robocasa/steer/patchceil/check_passA.py"
