#!/usr/bin/env bash
# pq3 단일-arm cell 러너 (pq2_cell_runner.sh 파생 — scene-diverse eval 전용).
# 차이: ① episode i → (env_seed_i, noise_seed_i) 를 eval_manifest 에서 주입 (pq2 의
# 고정 SEED 와 결별) ② serve 캡처 OFF + 클라이언트 --no-features (eval activation
# 미저장 — 판정 사이드카 json) ③ steer arm 은 --steering-token-select all
# --steering-denoise per_step 기본 주입 ④ preflight 6종: npz sha·step별 α key·β·
# token_select·denoise·expected-count(+선택 Gate D hash) ⑤ gated phase 정규식 task별.
#
# env (필수): CELL_ID TASK ENVN CELL_INDEX INSTR ARM_TAG STEER_MODE MANIFEST
#   STEER_MODE: base | perm | gated | null (null=perm 기계식+위약 NPZ)
#   MANIFEST: eval_manifest.tsv 또는 sweep_manifest.tsv (episode_idx/env_seed/noise_seed[/...])
# env (steer arm): NPZ_DIR STEER_LAYERS STEER_BETA [STEER_ALPHA]
#   [STEER_TOKEN_SELECT=all] [STEER_DENOISE=per_step] [NPZ_SHAS="sha12 ..."(Gate D 동결 대조)]
# env (공통): GPUS_L PORTS_L [EPLIST="0 3 ..."(재개용 부분집합)] [EXPECT_N=30] [OUT_TIER=e1]
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
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/${SEROOT}/${CELL_ID}"
OUT_CONT="/temporal_vla/outputs/eval/robocasa/groot_n15/${SEROOT}/${CELL_ID}"
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}

# ── manifest 로드: episode_idx → env_seed / noise_seed ─────────────────────────
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

# gated phase NPZ 정규식 — task 별 라벨 집합 (drawer = DrawerPhaseLabeler 7-phase)
if [[ "$TASK" == *Drawer* ]]; then
  PHASE_RE='(reach-to-handle|grasp-handle|pull|push-back|disengage|wrong-grasp|open-done)'
else
  PHASE_RE='(reach-to-object|grasp|transport|place|insert-settle|wrong-grasp)'
fi

start_serves() {
  local flags; flags=$(steer_flags)
  for i in $(seq 0 $((NW-1))); do
    # 캡처 플래그 없음 (eval activation 미저장) — pq2 러너와의 핵심 차이
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda ${flags} \
         > /tmp/pq3_${CELL_ID}_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    [ $ok = 1 ] || { echo "[${CELL_ID}/${ARM_TAG}] serve ${port} TIMEOUT"; exit 11; }
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/pq3_${CELL_ID}_${port}.log"; then
      echo "[${CELL_ID}/${ARM_TAG}] ABORT ${port}"; docker exec lerobot bash -lc "tail -20 /tmp/pq3_${CELL_ID}_${port}.log"; exit 11
    fi
    pf=$(docker exec lerobot bash -lc "grep '\[steer-preflight\]' /tmp/pq3_${CELL_ID}_${port}.log" || true)
    # module logger INFO 는 serve 로그 파일에 안 남음(uvicorn 로거만 출력, 2026-07-15 실증)
    # — 등록 대조는 print 기반 [steer-registered] 라인 사용
    reg=$(docker exec lerobot bash -lc "grep '\[steer-registered\]' /tmp/pq3_${CELL_ID}_${port}.log" || true)
    if [ "$STEER_MODE" = base ]; then
      # 배선 위약: base serve 에 steering 이 붙어 있으면 즉시 abort
      [ -z "$pf$reg" ] || { echo "[${CELL_ID}/${ARM_TAG}] base 인데 steering 로드됨 ABORT: $pf$reg"; exit 12; }
      continue
    fi
    # ── preflight 6종 (계획서 v9 §E) ───────────────────────────────────────
    [ -n "$pf" ] || { echo "[${CELL_ID}/${ARM_TAG}] preflight 라인 없음 (steering 미적용?) ABORT"; exit 12; }
    # ① npz 경로
    echo "$pf" | grep -q "npz=${NPZ_DIR}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight npz 경로 불일치"; exit 12; }
    # ② α key — per_step 은 step0..K-1 라인 전부, 지정 α 면 그 값
    if [ "$STEER_DENOISE" = per_step ]; then
      for k in 0 1 2 3; do
        echo "$pf" | grep -q "step=${k} key=step${k}_alpha" \
          || { echo "[${CELL_ID}/${ARM_TAG}] preflight step${k} key 없음 (per_step 4개 미로드)"; exit 12; }
      done
      echo "$pf" | grep -q "denoise=per_step" || { echo "[${CELL_ID}/${ARM_TAG}] preflight denoise 불일치"; exit 12; }
    else
      echo "$pf" | grep -q "key=alpha.*_${STEER_KEY} " || { echo "[${CELL_ID}/${ARM_TAG}] preflight key 불일치"; exit 12; }
    fi
    if [ -n "${STEER_ALPHA:-}" ]; then
      echo "$pf" | grep -q "alpha${STEER_ALPHA}_" || { echo "[${CELL_ID}/${ARM_TAG}] preflight alpha 불일치"; exit 12; }
    fi
    # ③ β
    echo "$pf" | grep -q "beta=${STEER_BETA}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight beta 불일치"; exit 12; }
    # ④⑤ token_select·denoise (등록 로그)
    echo "$reg" | grep -q "token_select=${STEER_TOKEN_SELECT}" || { echo "[${CELL_ID}/${ARM_TAG}] token_select 불일치: $reg"; exit 12; }
    echo "$reg" | grep -q "denoise=${STEER_DENOISE}" || { echo "[${CELL_ID}/${ARM_TAG}] denoise 등록 불일치: $reg"; exit 12; }
    # ⑥ Gate D NPZ sha 동결 대조 (NPZ_SHAS 지정 시 전부 존재해야 함)
    if [ -n "${NPZ_SHAS:-}" ]; then
      for sha in ${NPZ_SHAS}; do
        echo "$pf" | grep -q "sha=${sha}" || { echo "[${CELL_ID}/${ARM_TAG}] Gate D NPZ sha=${sha} 미로드 ABORT"; exit 12; }
      done
    fi
    if [ "$STEER_MODE" = gated ]; then
      echo "$pf" | grep -qE "npz=[^ ]*/${PHASE_RE}/" \
        || { echo "[${CELL_ID}/${ARM_TAG}] preflight: gated 인데 phase NPZ 로드 0건 ABORT"; exit 12; }
    fi
    { echo "$pf"; echo "$reg"; } | sed "s/^/[preflight ${port}] /" >> "${LOGDIR}/${ARM_TAG}_preflight.log"
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }

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
      --no-features $PROX_FLAG $GATED_FLAG 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
}

echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') mode=${STEER_MODE} eps=${#EPS[@]}/${EXPECT_N} tier=${OUT_TIER} manifest=${MANIFEST_SHA}"
mkdir -p "${OUT_HOST}/${ARM_TAG}"
SEEDPAIRS=$(for ep in "${MEPS[@]}"; do printf '%s:%s:%s ' "$ep" "${ENVSEED[$ep]}" "${NOISESEED[$ep]}"; done)
cat > "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json" <<EOF
{"cell":"${CELL_ID}","arm_tag":"${ARM_TAG}","mode":"${STEER_MODE}","npz_dir":"${NPZ_DIR:-}",
 "layers":"${STEER_LAYERS:-}","alpha":"${STEER_ALPHA:-meta_per_step}","beta":"${STEER_BETA}",
 "key":"${STEER_KEY}","token_select":"${STEER_TOKEN_SELECT}","denoise":"${STEER_DENOISE}",
 "expect_n":${EXPECT_N},"tier":"${OUT_TIER}","manifest":"${MANIFEST}","manifest_sha":"${MANIFEST_SHA}",
 "npz_shas":"${NPZ_SHAS:-}","machine":"${MACHINE_TAG:-local}",
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
