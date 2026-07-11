#!/usr/bin/env bash
# pq2 단일-arm cell 러너 (P2 선택 rollout + P3 held-out 공용).
# heldout_round_cell.sh 파생 — 차이: ①arm 1개/호출 ②EPLIST(비연속 episode 집합, P2 select-half)
# ③출력 루트 steer_eval_pq2/ (구 라운드 rename 충돌 회피) ④serve steer-preflight 로그를
# arm 스펙과 대조해 불일치 시 rollout 전 abort ⑤신명명 tag 그대로 사용.
#
# env (필수): CELL_ID TASK ENVN CELL_INDEX SEED INSTR ARM_TAG STEER_MODE
#   STEER_MODE: base | perm | gated | pos (pos=perm 기계식+C_success) | null (perm 기계식+null NPZ)
# env (steer arm): NPZ_DIR(perm/pos/null=global conceptor dir, gated=phase base dir)
#   STEER_LAYERS STEER_BETA [STEER_ALPHA](미지정=loader 가 NPZ meta selected_alpha 자동)
# env (공통): GPUS_L PORTS_L [EPLIST="0 3 7 ..."] 또는 [EP0 EP1] [OUT_TIER=p3|p2]
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
STRIDE=1000; NAS=5; MAXEP=720
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${SEED:?}" "${INSTR:?}" "${ARM_TAG:?}" "${STEER_MODE:?}"
STEER_BETA="${STEER_BETA:-0.3}"
STEER_KEY="C_steer"; [ "$STEER_MODE" = pos ] && STEER_KEY="C_success"
STEER_ALPHA_FLAG=""; [ -n "${STEER_ALPHA:-}" ] && STEER_ALPHA_FLAG="--steering-alpha ${STEER_ALPHA}"
if [ -n "${EPLIST:-}" ]; then EPS=($EPLIST); else EPS=($(seq "${EP0:?}" "${EP1:?}")); fi
OUT_TIER="${OUT_TIER:-p3}"
SEROOT="steer_eval_pq2/${OUT_TIER}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/${SEROOT}/${CELL_ID}"
OUT_CONT="/temporal_vla/outputs/eval/robocasa/groot_n15/${SEROOT}/${CELL_ID}"
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}

steer_flags() {
  case "$STEER_MODE" in
    base) echo "" ;;
    gated) echo "--steering-phase-npz-base ${NPZ_DIR:?} --steering-layers ${STEER_LAYERS:?} --steering-beta ${STEER_BETA} ${STEER_ALPHA_FLAG} --steering-key ${STEER_KEY}" ;;
    perm|pos|null) echo "--steering-npz-dir ${NPZ_DIR:?} --steering-layers ${STEER_LAYERS:?} --steering-beta ${STEER_BETA} ${STEER_ALPHA_FLAG} --steering-key ${STEER_KEY}" ;;
    *) echo "unknown STEER_MODE=$STEER_MODE" >&2; exit 2 ;;
  esac
}

start_serves() {
  local flags; flags=$(steer_flags)
  for i in $(seq 0 $((NW-1))); do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda --collect --capture-vl \
         --groot-dit-capture-layers ${CAP} ${flags} > /tmp/pq2_${CELL_ID}_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    [ $ok = 1 ] || { echo "[${CELL_ID}/${ARM_TAG}] serve ${port} TIMEOUT"; exit 11; }
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/pq2_${CELL_ID}_${port}.log"; then
      echo "[${CELL_ID}/${ARM_TAG}] ABORT ${port}"; docker exec lerobot bash -lc "tail -20 /tmp/pq2_${CELL_ID}_${port}.log"; exit 11
    fi
    # preflight 대조: steer arm 은 serve 가 로드한 npz·key·beta 가 스펙과 일치해야 rollout 시작
    if [ "$STEER_MODE" != base ]; then
      pf=$(docker exec lerobot bash -lc "grep '\[steer-preflight\]' /tmp/pq2_${CELL_ID}_${port}.log" || true)
      [ -n "$pf" ] || { echo "[${CELL_ID}/${ARM_TAG}] preflight 라인 없음 (steering 미적용?) ABORT"; exit 12; }
      echo "$pf" | grep -q "key=alpha.*_${STEER_KEY} " || { echo "[${CELL_ID}/${ARM_TAG}] preflight key 불일치: $pf"; exit 12; }
      echo "$pf" | grep -q "beta=${STEER_BETA}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight beta 불일치: $pf"; exit 12; }
      echo "$pf" | grep -q "npz=${NPZ_DIR}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight npz 경로 불일치: $pf"; exit 12; }
      if [ -n "${STEER_ALPHA:-}" ]; then
        echo "$pf" | grep -q "key=alpha${STEER_ALPHA}_" || { echo "[${CELL_ID}/${ARM_TAG}] preflight alpha 불일치: $pf"; exit 12; }
      fi
      # gated 는 실제 phase NPZ 로드가 최소 1건 있어야 함 — 'global' 만 로드되면 gating 이
      # 전혀 발화하지 않는 무-스티어링 arm 이 됨 (2026-07-11 무효 arm 사고 실증)
      if [ "$STEER_MODE" = gated ]; then
        echo "$pf" | grep -qE 'npz=[^ ]*/(reach-to-object|grasp|transport|place|insert-settle)/' \
          || { echo "[${CELL_ID}/${ARM_TAG}] preflight: gated 인데 phase NPZ 로드 0건 ABORT"; exit 12; }
      fi
      echo "$pf" | sed "s/^/[preflight ${port}] /" >> "${LOGDIR}/${ARM_TAG}_preflight.log"
    fi
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }

GATED_FLAG=""; [ "$STEER_MODE" = gated ] && GATED_FLAG="--gated-steering"
PROX_FLAG="--proximity-phases"

run_w() {
  local wid=$1 port=$2 k=0
  for ep in "${EPS[@]}"; do
    k=$((k+1)); [ $(( (k-1) % NW )) -eq "$wid" ] || continue
    if ls "${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "${OUT_CONT}/${ARM_TAG}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "$SEED" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready $PROX_FLAG $GATED_FLAG 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
}

echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') mode=${STEER_MODE} eps=${#EPS[@]} tier=${OUT_TIER}"
# arm 스펙 기록 (preflight·집계 대조용 단일 출처)
mkdir -p "${OUT_HOST}/${ARM_TAG}"
cat > "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json" <<EOF
{"cell":"${CELL_ID}","arm_tag":"${ARM_TAG}","mode":"${STEER_MODE}","npz_dir":"${NPZ_DIR:-}",
 "layers":"${STEER_LAYERS:-}","alpha":"${STEER_ALPHA:-meta}","beta":"${STEER_BETA}",
 "key":"${STEER_KEY}","eps":"${EPS[*]}","tier":"${OUT_TIER}","seed":"${SEED}"}
EOF
start_serves
for wid in $(seq 0 $((NW-1))); do run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/${ARM_TAG}_w${wid}.log" 2>&1 & done
wait
kill_serves
d="${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}"
s=$(ls "$d"/*succ1.pkl 2>/dev/null | wc -l); f=$(ls "$d"/*succ0.pkl 2>/dev/null | wc -l)
echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') DONE ${s}/$((s+f)) (기대 ${#EPS[@]})"
[ $((s+f)) -ge ${#EPS[@]} ]
