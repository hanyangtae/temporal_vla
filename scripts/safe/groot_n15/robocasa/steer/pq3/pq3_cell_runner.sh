#!/usr/bin/env bash
# pq3 단일-arm cell 러너 (pq2_cell_runner.sh 파생 — scene-diverse eval 전용).
# 차이: ① episode i → (env_seed_i, noise_seed_i) 를 eval_manifest 에서 주입 ② 캡처 OFF
# (클라이언트 --no-features = skip_features chunk 추론 — /act 큐 팝 경로 금지, Gate2 치명#1)
# ③ steer arm --steering-token-select all --steering-denoise per_step 기본 ④ preflight:
# boot_id(포트의 남의 서버 오인 방지)·npz sha(Gate D 필수)·step별 α key(파일별 4개)·β·
# token_select·denoise·health 지문·expected-count ⑤ gated phase 집합 정확 일치
# ⑥ ARM_SPEC 불변(spec_sha) + 사이드카 run_tag (재라벨링 오염 방지, Gate2 치명#4).
#
# env (필수): CELL_ID TASK ENVN CELL_INDEX INSTR ARM_TAG STEER_MODE MANIFEST
#   STEER_MODE: base | perm | gated | null
# env (steer arm): NPZ_DIR STEER_LAYERS STEER_BETA NPZ_SHAS="sha12 ..."(Gate D 동결, 필수)
#   [STEER_ALPHA] [STEER_TOKEN_SELECT=all] [STEER_DENOISE=per_step]
#   [GATED_PHASES="ph1,ph2,..."(gated 기대 phase 집합 — 지정 시 정확 일치 강제)]
# env (공통): GPUS_L PORTS_L [EPLIST(재개용)] [EXPECT_N=30] [OUT_TIER=e1] [MACHINE_TAG]
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
NAS=5; MAXEP=720; CHUNK_LEN=16
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${INSTR:?}" "${ARM_TAG:?}" "${STEER_MODE:?}" "${MANIFEST:?}"
STEER_BETA="${STEER_BETA:-0.1}"
STEER_KEY="C_steer"
STEER_TOKEN_SELECT="${STEER_TOKEN_SELECT:-all}"
STEER_DENOISE="${STEER_DENOISE:-per_step}"
STEER_ALPHA_FLAG=""; [ -n "${STEER_ALPHA:-}" ] && STEER_ALPHA_FLAG="--steering-alpha ${STEER_ALPHA}"
if [ "$STEER_MODE" != base ]; then
  : "${NPZ_SHAS:?steer arm 은 NPZ_SHAS(Gate D 동결 sha12 목록) 필수 — 생략 금지 (Gate2 높음#9)}"
fi
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
# eval 동결 대조: eval_reserved.json 이 곁에 있으면 기록된 manifest sha16 과 일치해야 함
RESERVED="$(dirname "$MANIFEST")/eval_reserved.json"
if [ -f "$RESERVED" ]; then
  want16=$(grep -o '"eval_manifest_sha": *"[0-9a-f]*"' "$RESERVED" | grep -o '[0-9a-f]\{16\}')
  have16=$(sha256sum "$MANIFEST" | cut -c1-16)
  [ "$want16" = "$have16" ] || { echo "[${CELL_ID}/${ARM_TAG}] eval manifest 가 동결본과 다름 (${have16}!=${want16}) ABORT"; exit 13; }
fi

steer_flags() {
  local extra=""
  [ -n "${GATED_PHASES:-}" ] && extra="--steering-phases ${GATED_PHASES}"
  case "$STEER_MODE" in
    base) echo "" ;;
    gated) echo "--steering-phase-npz-base ${NPZ_DIR:?} --steering-layers ${STEER_LAYERS:?} --steering-beta ${STEER_BETA} ${STEER_ALPHA_FLAG} --steering-key ${STEER_KEY} --steering-token-select ${STEER_TOKEN_SELECT} --steering-denoise ${STEER_DENOISE} ${extra}" ;;
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

kill_port() { # port — 해당 포트의 기존 serve 제거 (남의 서버 오인 방지, Gate2 치명#3)
  docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${1}' || true" 2>/dev/null || true
}
kill_serves() { for port in "${PORTS[@]}"; do kill_port "$port"; done; sleep 5; }
trap kill_serves EXIT

start_serves() {
  local flags; flags=$(steer_flags)
  for i in $(seq 0 $((NW-1))); do
    kill_port "${PORTS[$i]}"
  done
  sleep 3
  for i in $(seq 0 $((NW-1))); do
    docker exec lerobot bash -lc "rm -f /tmp/pq3_${CELL_ID}_${PORTS[$i]}.log"
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
    LOG="/tmp/pq3_${CELL_ID}_${port}.log"
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound|Address already in use|Errno 98' ${LOG}"; then
      echo "[${CELL_ID}/${ARM_TAG}] ABORT ${port}"; docker exec lerobot bash -lc "tail -20 ${LOG}"; exit 11
    fi
    # ── boot id: fresh 로그의 [serve-boot] 와 /health boot_id 일치 = 우리 프로세스 ──
    boot_log=$(docker exec lerobot bash -lc "grep -o '\[serve-boot\] id=[0-9a-f]*' ${LOG}" | tail -1 | cut -d= -f2 || true)
    health=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health")
    boot_http=$(echo "$health" | grep -o '"boot_id":"[0-9a-f]*"' | cut -d'"' -f4 || true)
    [ -n "$boot_log" ] && [ "$boot_log" = "$boot_http" ] \
      || { echo "[${CELL_ID}/${ARM_TAG}] boot_id 불일치 (log=${boot_log:-none} http=${boot_http:-none}) — 포트에 다른 서버 ABORT"; exit 12; }
    pf=$(docker exec lerobot bash -lc "grep '\[steer-preflight\]' ${LOG}" || true)
    reg=$(docker exec lerobot bash -lc "grep '\[steer-registered\]' ${LOG}" || true)
    if [ "$STEER_MODE" = base ]; then
      # 배선 위약: base serve 에 steering 이 붙어 있으면 즉시 abort (로그+health 이중)
      [ -z "$pf$reg" ] || { echo "[${CELL_ID}/${ARM_TAG}] base 인데 steering 로드됨 ABORT"; exit 12; }
      echo "$health" | grep -q '"steering":null' \
        || { echo "[${CELL_ID}/${ARM_TAG}] base 인데 health steering 지문 존재 ABORT: $health"; exit 12; }
      continue
    fi
    # ── preflight (Gate2 반영판) ─────────────────────────────────────────────
    [ -n "$pf" ] || { echo "[${CELL_ID}/${ARM_TAG}] preflight 라인 없음 (steering 미적용?) ABORT"; exit 12; }
    echo "$pf" | grep -q "npz=${NPZ_DIR}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight npz 경로 불일치"; exit 12; }
    if [ "$STEER_DENOISE" = per_step ]; then
      # 로드된 npz "파일별로" step0..3 이 정확히 1회씩 있어야 함 (부분 로드 무음 방지)
      bad=$(echo "$pf" | awk '{npz=""; st="";
          for(i=1;i<=NF;i++){ if($i ~ /^npz=/) npz=$i; if($i ~ /^step=/) st=$i }
          if (npz != "" && st != "") cnt[npz]++ }
        END{ for(k in cnt) if (cnt[k] != 4) print k "=" cnt[k] }')
      [ -z "$bad" ] || { echo "[${CELL_ID}/${ARM_TAG}] per-step 키 불완전: $bad ABORT"; exit 12; }
      echo "$pf" | grep -q "denoise=per_step" || { echo "[${CELL_ID}/${ARM_TAG}] preflight denoise 불일치"; exit 12; }
    else
      echo "$pf" | grep -q "key=alpha.*_${STEER_KEY} " || { echo "[${CELL_ID}/${ARM_TAG}] preflight key 불일치"; exit 12; }
    fi
    if [ -n "${STEER_ALPHA:-}" ]; then
      echo "$pf" | grep -q "alpha${STEER_ALPHA}_" || { echo "[${CELL_ID}/${ARM_TAG}] preflight alpha 불일치"; exit 12; }
    fi
    echo "$pf" | grep -q "beta=${STEER_BETA}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight beta 불일치"; exit 12; }
    echo "$reg" | grep -q "token_select=${STEER_TOKEN_SELECT}" || { echo "[${CELL_ID}/${ARM_TAG}] token_select 불일치: $reg"; exit 12; }
    echo "$reg" | grep -q "denoise=${STEER_DENOISE}" || { echo "[${CELL_ID}/${ARM_TAG}] denoise 등록 불일치: $reg"; exit 12; }
    # Gate D NPZ sha 동결 대조 (전 sha 가 preflight 와 health 지문 양쪽에 존재해야 함)
    for sha in ${NPZ_SHAS}; do
      echo "$pf" | grep -q "sha=${sha}" || { echo "[${CELL_ID}/${ARM_TAG}] Gate D NPZ sha=${sha} preflight 미로드 ABORT"; exit 12; }
      echo "$health" | grep -q "${sha}" || { echo "[${CELL_ID}/${ARM_TAG}] health 지문에 sha=${sha} 없음 ABORT"; exit 12; }
    done
    echo "$health" | grep -q "\"denoise\":\"${STEER_DENOISE}\"" || { echo "[${CELL_ID}/${ARM_TAG}] health denoise 불일치 ABORT"; exit 12; }
    if [ "$STEER_MODE" = gated ]; then
      echo "$pf" | grep -qE "npz=[^ ]*/${PHASE_RE}/" \
        || { echo "[${CELL_ID}/${ARM_TAG}] preflight: gated 인데 phase NPZ 로드 0건 ABORT"; exit 12; }
      if [ -n "${GATED_PHASES:-}" ]; then
        want=$(echo "$GATED_PHASES" | tr ',' '\n' | sort | paste -sd,)
        got=$(echo "$reg" | grep -o 'phases=[^ ]*' | head -1 | cut -d= -f2 | tr ',' '\n' | sort | paste -sd,)
        [ "$want" = "$got" ] || { echo "[${CELL_ID}/${ARM_TAG}] gated phase 집합 불일치 (want=${want} got=${got}) ABORT"; exit 12; }
      fi
    fi
    { echo "$pf"; echo "$reg"; } | sed "s/^/[preflight ${port}] /" >> "${LOGDIR}/${ARM_TAG}_preflight.log"
  done
}

GATED_FLAG=""; [ "$STEER_MODE" = gated ] && GATED_FLAG="--gated-steering"
PROX_FLAG="--proximity-phases"

run_w() {
  local wid=$1 port=$2 k=0
  for ep in "${EPS[@]}"; do
    k=$((k+1)); [ $(( (k-1) % NW )) -eq "$wid" ] || continue
    local existing
    existing=$(ls "${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.json 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
      # 재개 안전: 기존 사이드카가 현 spec 의 run_tag 와 일치할 때만 skip (재라벨링 방지)
      if grep -q "\"run_tag\": \"${RUN_TAG}\"" "$existing"; then continue; fi
      echo "[${CELL_ID}/${ARM_TAG}] ep${ep} 기존 사이드카 run_tag 불일치 — 오염 의심 ABORT" >&2
      exit 14
    fi
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "${OUT_CONT}/${ARM_TAG}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "${ENVSEED[$ep]}" --inference-seed "${NOISESEED[$ep]}" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --no-features --expect-chunk-len "$CHUNK_LEN" --run-tag "$RUN_TAG" $PROX_FLAG $GATED_FLAG 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
}

echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') mode=${STEER_MODE} eps=${#EPS[@]}/${EXPECT_N} tier=${OUT_TIER} manifest=${MANIFEST_SHA}"
mkdir -p "${OUT_HOST}/${ARM_TAG}"
SEEDPAIRS=$(for ep in "${MEPS[@]}"; do printf '%s:%s:%s ' "$ep" "${ENVSEED[$ep]}" "${NOISESEED[$ep]}"; done)
# ARM_SPEC 불변 코어 (machine 등 휘발 필드 제외) → spec_sha → run_tag
CORE_SPEC="{\"cell\":\"${CELL_ID}\",\"arm_tag\":\"${ARM_TAG}\",\"mode\":\"${STEER_MODE}\",\"npz_dir\":\"${NPZ_DIR:-}\",\"layers\":\"${STEER_LAYERS:-}\",\"alpha\":\"${STEER_ALPHA:-meta_per_step}\",\"beta\":\"${STEER_BETA}\",\"key\":\"${STEER_KEY}\",\"token_select\":\"${STEER_TOKEN_SELECT}\",\"denoise\":\"${STEER_DENOISE}\",\"expect_n\":${EXPECT_N},\"tier\":\"${OUT_TIER}\",\"manifest_sha\":\"${MANIFEST_SHA}\",\"npz_shas\":\"${NPZ_SHAS:-}\",\"gated_phases\":\"${GATED_PHASES:-}\",\"seed_pairs\":\"${SEEDPAIRS% }\"}"
SPEC_SHA=$(printf '%s' "$CORE_SPEC" | sha256sum | cut -c1-12)
RUN_TAG="${ARM_TAG}:${SPEC_SHA}"
if [ -f "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json" ]; then
  old_sha=$(grep -o '"spec_sha": *"[0-9a-f]*"' "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json" | grep -o '[0-9a-f]\{12\}' || true)
  if [ -n "$old_sha" ] && [ "$old_sha" != "$SPEC_SHA" ]; then
    echo "[${CELL_ID}/${ARM_TAG}] ARM_SPEC 불변 위반: 기존 spec_sha=${old_sha} != ${SPEC_SHA} — 디렉토리 정리 후 재실행하라 ABORT"
    exit 14
  fi
fi
printf '{"spec_sha":"%s","machine":"%s","manifest":"%s","core":%s}\n' \
  "$SPEC_SHA" "${MACHINE_TAG:-local}" "$MANIFEST" "$CORE_SPEC" > "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json"
start_serves
for wid in $(seq 0 $((NW-1))); do run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/${ARM_TAG}_w${wid}.log" 2>&1 & done
wait
kill_serves
trap - EXIT
d="${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}"
s=$(ls "$d"/*succ1.json 2>/dev/null | wc -l); f=$(ls "$d"/*succ0.json 2>/dev/null | wc -l)
echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') DONE ${s}/$((s+f)) (기대 ${#EPS[@]})"
[ $((s+f)) -ge ${#EPS[@]} ]
