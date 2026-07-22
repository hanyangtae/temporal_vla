#!/usr/bin/env bash
# patchceil pass B: donor/placebo/sham 대상 full-token 재수집 (passB_manifest.tsv 기준).
#
# CAP 7층(0,2,4,8,10,12,15) × all_token_full — primary L15 + exploratory early/mid 를
# 한 번에 확보 (재수집 반복 회피). 용량 ~1GB/ep × 16 ≈ 16GB (NPZ 추출 후 정리 검토).
# 레시피 = fit 수집과 동일 (seed=scenario, inference_seed=ep×1000, NAS=5, MAXEP=720,
# fresh process/ep). 재수집분 succ·actions 는 A1 결정론 게이트로 승준 zst 와 대조.
#
# 사용 (GPU 1 사용자 승인 2026-07-16, 포트 8495/8496 유휴 확인):
#   GPUS_L="1 1" PORTS_L="8495 8496" \
#     bash scripts/safe/groot_n15/robocasa/steer/patchceil/collect_passB.sh
# resume-safe: 이미 pkl 있는 ep skip.
set -uo pipefail

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
STRIDE=1000; NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
CELL_INDEX=5
INSTR="Pick the bread from the counter and place it in the cabinet."
declare -A SEEDS=([ppcc_bread_s300033]=300033 [ppcc_bread_s400020]=400020)

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:?"GPUS_L 필요 (예: \"1 1\")"})
PORTS=(${PORTS_L:?"PORTS_L 필요 (예: \"8495 8496\")"})
NW=${#PORTS[@]}
CELLS=(ppcc_bread_s300033 ppcc_bread_s400020)
GROOT_OUT="outputs/eval/robocasa/groot_n15/patchceil"

start_serves() {
  for i in $(seq 0 $((NW-1))); do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda --collect \
         --groot-dit-capture-layers ${CAP} --groot-dit-token-pool all_token_full \
         > /tmp/patchceil_passB_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[passB] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || exit 11
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/patchceil_passB_${port}.log"; then
      echo "[passB] ABORT ${port}"; docker exec lerobot bash -lc "tail -20 /tmp/patchceil_passB_${port}.log"; exit 11
    fi
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }
trap kill_serves EXIT

start_serves
for CELL in "${CELLS[@]}"; do
  SEED="${SEEDS[$CELL]}"
  MANIFEST="${REPO_ROOT}/${GROOT_OUT}/${CELL}/passB_manifest.tsv"
  OUT_HOST="${REPO_ROOT}/${GROOT_OUT}/${CELL}/passB"
  OUT_CONT="/temporal_vla/${GROOT_OUT}/${CELL}/passB"
  LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
  mapfile -t EPS < <(awk -F'\t' 'NR>1{print $2}' "$MANIFEST")
  echo "[passB] $(date '+%F %T') cell=${CELL} eps=${EPS[*]}"
  run_w() {
    local wid=$1 port=$2
    local n=0
    for ep in "${EPS[@]}"; do
      n=$((n+1)); [ $((n % NW)) -eq "$wid" ] || continue
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
    run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/passB_w${wid}.log" 2>&1 &
  done
  wait
  n_done=$(ls "${OUT_HOST}/raw_rollouts/${TASK}/${CELL}"/task${CELL_INDEX}--ep*--succ*.pkl 2>/dev/null | wc -l)
  echo "[passB] ${CELL} done pkl=${n_done}/${#EPS[@]}"
done
kill_serves; trap - EXIT
echo "[passB] $(date '+%F %T') ALL_DONE"
