#!/usr/bin/env bash
# pq3 단일-arm cell 러너의 원격 워커 변형 (w2/.50, w48 공용) — serve 를 호스트 conda 로.
# pq3_cell_runner.sh 와 계약 동일 (manifest 주입/캡처 OFF/preflight 6종/ARM_SPEC), 차이는
# serve 기동·로그·health 가 호스트 측이라는 것뿐 (pq2_cell_runner_w2.sh 파생).
# w48 은 MACHINE_TAG·HF_HOME_OVERRIDE env 로 이 파일을 그대로 사용.
#
# env: pq3_cell_runner.sh 와 동일 + [HPY(serve python)] [HF_HOME_OVERRIDE] [MACHINE_TAG]
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
NAS=5; MAXEP=720
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${INSTR:?}" "${ARM_TAG:?}" "${STEER_MODE:?}" "${MANIFEST:?}"
STEER_BETA="${STEER_BETA:-0.1}"
STEER_KEY="C_steer"
STEER_TOKEN_SELECT="${STEER_TOKEN_SELECT:-all}"
STEER_DENOISE="${STEER_DENOISE:-per_step}"
STEER_ALPHA_FLAG=""; [ -n "${STEER_ALPHA:-}" ] && STEER_ALPHA_FLAG="--steering-alpha ${STEER_ALPHA}"
OUT_TIER="${OUT_TIER:-e1}"
EXPECT_N="${EXPECT_N:-30}"
SEROOT="steer_eval_pq3/${OUT_TIER}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
HPY="${HPY:-$HOME/miniconda3/envs/lerobot_050_groot/bin/python}"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/${SEROOT}/${CELL_ID}"
OUT_CONT="/temporal_vla/outputs/eval/robocasa/groot_n15/${SEROOT}/${CELL_ID}"
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:-2 2}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}

declare -A ENVSEED NOISESEED
MEPS=()
while IFS=$'\t' read -r ep envs noise _rest; do
  case "$ep" in ''|\#*|episode_idx|ep_idx|s_idx) continue ;; esac
  MEPS+=("$ep"); ENVSEED[$ep]=$envs; NOISESEED[$ep]=$noise
done < "$MANIFEST"
[ "${#MEPS[@]}" -eq "$EXPECT_N" ] || { echo "[${CELL_ID}/${ARM_TAG}] manifest 행수 ${#MEPS[@]} != EXPECT_N ${EXPECT_N} ABORT"; exit 13; }
if [ -n "${EPLIST:-}" ]; then EPS=($EPLIST); else EPS=("${MEPS[@]}"); fi
for ep in "${EPS[@]}"; do
  [ -n "${ENVSEED[$ep]:-}" ] || { echo "[${CELL_ID}/${ARM_TAG}] EPLIST ep=${ep} 가 manifest 에 없음 ABORT"; exit 13; }
done
MANIFEST_SHA=$(sha256sum "$MANIFEST" | cut -c1-12)

steer_flags() {
  case "$STEER_MODE" in
    base) echo "" ;;
    gated) echo "--steering-phase-npz-base ${NPZ_DIR:?} --steering-layers ${STEER_LAYERS:?} --steering-beta ${STEER_BETA} ${STEER_ALPHA_FLAG} --steering-key ${STEER_KEY} --steering-token-select ${STEER_TOKEN_SELECT} --steering-denoise ${STEER_DENOISE}" ;;
    perm|null) echo "--steering-npz-dir ${NPZ_DIR:?} --steering-layers ${STEER_LAYERS:?} --steering-beta ${STEER_BETA} ${STEER_ALPHA_FLAG} --steering-key ${STEER_KEY} --steering-token-select ${STEER_TOKEN_SELECT} --steering-denoise ${STEER_DENOISE}" ;;
    *) echo "unknown STEER_MODE=$STEER_MODE" >&2; exit 2 ;;
  esac
}

if [[ "$TASK" == *Drawer* ]]; then
  PHASE_RE='(reach-to-handle|grasp-handle|pull|push-back|disengage|wrong-grasp|open-done)'
else
  PHASE_RE='(reach-to-object|grasp|transport|place|insert-settle|wrong-grasp)'
fi

start_serves() {
  local flags i; flags=$(steer_flags)
  for i in $(seq 0 $((NW-1))); do
    ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        PYTHONPATH="$REPO_ROOT/lerobot/src" ${HF_HOME_OVERRIDE:+HF_HOME=$HF_HOME_OVERRIDE} \
        "$HPY" scripts/serve/lerobot.py --profile ${PROFILE} \
        --host '*' --port "${PORTS[$i]}" --device cuda ${flags} \
        > "/tmp/pq3w_${CELL_ID}_${PORTS[$i]}.log" 2>&1 < /dev/null & )
  done
  local port ok pf reg
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }; sleep 5
    done
    [ $ok = 1 ] || { echo "[${CELL_ID}/${ARM_TAG}] serve ${port} TIMEOUT"; exit 11; }
    if grep -qiE 'Traceback|FAILED|FileNotFound' "/tmp/pq3w_${CELL_ID}_${port}.log"; then
      echo "[${CELL_ID}/${ARM_TAG}] ABORT ${port}"; tail -20 "/tmp/pq3w_${CELL_ID}_${port}.log"; exit 11
    fi
    pf=$(grep '\[steer-preflight\]' "/tmp/pq3w_${CELL_ID}_${port}.log" || true)
    # module logger INFO 는 serve 로그 파일에 안 남음 — print 기반 [steer-registered] 대조
    reg=$(grep '\[steer-registered\]' "/tmp/pq3w_${CELL_ID}_${port}.log" || true)
    if [ "$STEER_MODE" = base ]; then
      [ -z "$pf$reg" ] || { echo "[${CELL_ID}/${ARM_TAG}] base 인데 steering 로드됨 ABORT"; exit 12; }
      continue
    fi
    [ -n "$pf" ] || { echo "[${CELL_ID}/${ARM_TAG}] preflight 라인 없음 ABORT"; exit 12; }
    echo "$pf" | grep -q "npz=${NPZ_DIR}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight npz 불일치"; exit 12; }
    if [ "$STEER_DENOISE" = per_step ]; then
      for k in 0 1 2 3; do
        echo "$pf" | grep -q "step=${k} key=step${k}_alpha" \
          || { echo "[${CELL_ID}/${ARM_TAG}] preflight step${k} key 없음"; exit 12; }
      done
      echo "$pf" | grep -q "denoise=per_step" || { echo "[${CELL_ID}/${ARM_TAG}] preflight denoise 불일치"; exit 12; }
    else
      echo "$pf" | grep -q "key=alpha.*_${STEER_KEY} " || { echo "[${CELL_ID}/${ARM_TAG}] preflight key 불일치"; exit 12; }
    fi
    if [ -n "${STEER_ALPHA:-}" ]; then
      echo "$pf" | grep -q "alpha${STEER_ALPHA}_" || { echo "[${CELL_ID}/${ARM_TAG}] preflight alpha 불일치"; exit 12; }
    fi
    echo "$pf" | grep -q "beta=${STEER_BETA}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight beta 불일치"; exit 12; }
    echo "$reg" | grep -q "token_select=${STEER_TOKEN_SELECT}" || { echo "[${CELL_ID}/${ARM_TAG}] token_select 불일치"; exit 12; }
    echo "$reg" | grep -q "denoise=${STEER_DENOISE}" || { echo "[${CELL_ID}/${ARM_TAG}] denoise 등록 불일치"; exit 12; }
    if [ -n "${NPZ_SHAS:-}" ]; then
      for sha in ${NPZ_SHAS}; do
        echo "$pf" | grep -q "sha=${sha}" || { echo "[${CELL_ID}/${ARM_TAG}] Gate D NPZ sha=${sha} 미로드 ABORT"; exit 12; }
      done
    fi
    if [ "$STEER_MODE" = gated ]; then
      echo "$pf" | grep -qE "npz=[^ ]*/${PHASE_RE}/" \
        || { echo "[${CELL_ID}/${ARM_TAG}] gated phase NPZ 로드 0건 ABORT"; exit 12; }
    fi
    { echo "$pf"; echo "$reg"; } | sed "s/^/[preflight ${port}] /" >> "${LOGDIR}/${ARM_TAG}_preflight.log"
  done
}
kill_serves() { local port pid; for port in "${PORTS[@]}"; do
    for pid in $(pgrep -f "lerobot.py.*--port ${port}"); do kill "$pid" 2>/dev/null || true; done
  done; sleep 5; }

GATED_FLAG=""; [ "$STEER_MODE" = gated ] && GATED_FLAG="--gated-steering"
PROX_FLAG="--proximity-phases"

run_w() {
  local wid=$1 port=$2 k=0
  for ep in "${EPS[@]}"; do
    k=$((k+1)); [ $(( (k-1) % NW )) -eq "$wid" ] || continue
    if ls "${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.json >/dev/null 2>&1; then continue; fi
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "${OUT_CONT}/${ARM_TAG}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "${ENVSEED[$ep]}" --inference-seed "${NOISESEED[$ep]}" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --no-features $PROX_FLAG $GATED_FLAG 2>&1 | grep -E "^wrote|Error|Traceback" || true
  done
}

echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') ${MACHINE_TAG:-w2} mode=${STEER_MODE} eps=${#EPS[@]}/${EXPECT_N} tier=${OUT_TIER} manifest=${MANIFEST_SHA}"
mkdir -p "${OUT_HOST}/${ARM_TAG}"
SEEDPAIRS=$(for ep in "${MEPS[@]}"; do printf '%s:%s:%s ' "$ep" "${ENVSEED[$ep]}" "${NOISESEED[$ep]}"; done)
cat > "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json" <<EOF
{"cell":"${CELL_ID}","arm_tag":"${ARM_TAG}","mode":"${STEER_MODE}","npz_dir":"${NPZ_DIR:-}",
 "layers":"${STEER_LAYERS:-}","alpha":"${STEER_ALPHA:-meta_per_step}","beta":"${STEER_BETA}",
 "key":"${STEER_KEY}","token_select":"${STEER_TOKEN_SELECT}","denoise":"${STEER_DENOISE}",
 "expect_n":${EXPECT_N},"tier":"${OUT_TIER}","manifest":"${MANIFEST}","manifest_sha":"${MANIFEST_SHA}",
 "npz_shas":"${NPZ_SHAS:-}","machine":"${MACHINE_TAG:-worker2-a100}",
 "seed_pairs":"${SEEDPAIRS% }"}
EOF
start_serves
for wid in $(seq 0 $((NW-1))); do run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/${ARM_TAG}_w${wid}.log" 2>&1 & done
wait
kill_serves
d="${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}"
s=$(ls "$d"/*succ1.json 2>/dev/null | wc -l); f=$(ls "$d"/*succ0.json 2>/dev/null | wc -l)
echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') DONE ${s}/$((s+f)) (기대 ${#EPS[@]})"
[ $((s+f)) -ge ${#EPS[@]} ]
