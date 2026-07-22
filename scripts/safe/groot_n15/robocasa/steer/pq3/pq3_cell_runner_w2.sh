#!/usr/bin/env bash
# pq3 단일-arm cell 러너의 원격 워커 변형 (w2/.50 기본, w48 은 MACHINE_TAG·HF_HOME_OVERRIDE
# env 로 동일 사용) — serve 를 호스트 conda 로. 계약·preflight 는 pq3_cell_runner.sh 와 동일
# (boot_id·health 지문·ARM_SPEC 불변·run_tag 포함), serve 기동·로그·health 만 호스트 측.
#
# env: pq3_cell_runner.sh 와 동일 + [HPY(serve python)] [HF_HOME_OVERRIDE] [MACHINE_TAG]
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
  : "${NPZ_SHAS:?steer arm 은 NPZ_SHAS(Gate D 동결 sha12 목록) 필수 — 생략 금지}"
fi
if [ "$STEER_MODE" = gated ]; then
  : "${GATED_PHASES:?gated arm 은 GATED_PHASES(성립 게이트 report 의 phase 집합) 필수}"
fi
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
RESERVED="$(dirname "$MANIFEST")/eval_reserved.json"
if [ "$(basename "$MANIFEST")" = "eval_manifest.tsv" ]; then
  [ -f "$RESERVED" ] || { echo "[${CELL_ID}/${ARM_TAG}] eval_reserved.json 부재 — freeze 미완 ABORT"; exit 13; }
  want16=$(grep -o '"eval_manifest_sha": *"[0-9a-f]*"' "$RESERVED" | grep -o '[0-9a-f]\{16\}')
  have16=$(sha256sum "$MANIFEST" | cut -c1-16)
  [ "$want16" = "$have16" ] || { echo "[${CELL_ID}/${ARM_TAG}] eval manifest 가 동결본과 다름 ABORT"; exit 13; }
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

if [[ "$TASK" == *Drawer* ]]; then
  PHASE_RE='(reach-to-handle|grasp-handle|pull|push-back|disengage|wrong-grasp|open-done)'
else
  PHASE_RE='(reach-to-object|grasp|transport|place|insert-settle|wrong-grasp)'
fi

kill_port() { local pid; for pid in $(pgrep -f "lerobot.py.*--port ${1}"); do kill "$pid" 2>/dev/null || true; done; }
kill_serves() { for port in "${PORTS[@]}"; do kill_port "$port"; done; sleep 5; }
trap kill_serves EXIT

start_serves() {
  local flags i; flags=$(steer_flags)
  for i in $(seq 0 $((NW-1))); do kill_port "${PORTS[$i]}"; done
  sleep 3
  for i in $(seq 0 $((NW-1))); do
    rm -f "/tmp/pq3w_${CELL_ID}_${PORTS[$i]}.log"
    ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        PYTHONPATH="$REPO_ROOT/lerobot/src" ${HF_HOME_OVERRIDE:+HF_HOME=$HF_HOME_OVERRIDE} \
        "$HPY" scripts/serve/lerobot.py --profile ${PROFILE} \
        --host '*' --port "${PORTS[$i]}" --device cuda ${flags} \
        > "/tmp/pq3w_${CELL_ID}_${PORTS[$i]}.log" 2>&1 < /dev/null & )
  done
  local port ok pf reg health boot_log boot_http bad want got
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }; sleep 5
    done
    [ $ok = 1 ] || { echo "[${CELL_ID}/${ARM_TAG}] serve ${port} TIMEOUT"; exit 11; }
    LOG="/tmp/pq3w_${CELL_ID}_${port}.log"
    if grep -qiE 'Traceback|FAILED|FileNotFound|Address already in use|Errno 98' "$LOG"; then
      echo "[${CELL_ID}/${ARM_TAG}] ABORT ${port}"; tail -20 "$LOG"; exit 11
    fi
    boot_log=$(grep -o '\[serve-boot\] id=[0-9a-f]*' "$LOG" | tail -1 | cut -d= -f2 || true)
    health=$(curl -s -m 3 "http://127.0.0.1:${port}/health")
    boot_http=$(echo "$health" | grep -o '"boot_id":"[0-9a-f]*"' | cut -d'"' -f4 || true)
    [ -n "$boot_log" ] && [ "$boot_log" = "$boot_http" ] \
      || { echo "[${CELL_ID}/${ARM_TAG}] boot_id 불일치 — 포트에 다른 서버 ABORT"; exit 12; }
    pf=$(grep '\[steer-preflight\]' "$LOG" || true)
    reg=$(grep '\[steer-registered\]' "$LOG" || true)
    if [ "$STEER_MODE" = base ]; then
      [ -z "$pf$reg" ] || { echo "[${CELL_ID}/${ARM_TAG}] base 인데 steering 로드됨 ABORT"; exit 12; }
      echo "$health" | grep -q '"steering":null' \
        || { echo "[${CELL_ID}/${ARM_TAG}] base 인데 health steering 지문 존재 ABORT"; exit 12; }
      continue
    fi
    [ -n "$pf" ] || { echo "[${CELL_ID}/${ARM_TAG}] preflight 라인 없음 ABORT"; exit 12; }
    echo "$pf" | grep -q "npz=${NPZ_DIR}" || { echo "[${CELL_ID}/${ARM_TAG}] preflight npz 불일치"; exit 12; }
    if [ "$STEER_DENOISE" = per_step ]; then
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
    echo "$reg" | grep -q "token_select=${STEER_TOKEN_SELECT}" || { echo "[${CELL_ID}/${ARM_TAG}] token_select 불일치"; exit 12; }
    echo "$reg" | grep -q "denoise=${STEER_DENOISE}" || { echo "[${CELL_ID}/${ARM_TAG}] denoise 등록 불일치"; exit 12; }
    loaded_set=$(echo "$health" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(" ".join(sorted(set((d.get("steering") or {}).get("npz_shas") or []))))' 2>/dev/null)
    want_set=$(echo ${NPZ_SHAS} | tr ' ' '\n' | sort -u | paste -sd' ')
    [ -n "$loaded_set" ] && [ "$loaded_set" = "$want_set" ] \
      || { echo "[${CELL_ID}/${ARM_TAG}] NPZ sha 집합 불일치 (loaded='${loaded_set}' want='${want_set}') ABORT"; exit 12; }
    for sha in ${NPZ_SHAS}; do
      echo "$pf" | grep -q "sha=${sha}" || { echo "[${CELL_ID}/${ARM_TAG}] Gate D NPZ sha=${sha} 미로드 ABORT"; exit 12; }
    done
    echo "$health" | grep -q "\"denoise\":\"${STEER_DENOISE}\"" || { echo "[${CELL_ID}/${ARM_TAG}] health denoise 불일치 ABORT"; exit 12; }
    if [ "$STEER_MODE" = gated ]; then
      echo "$pf" | grep -qE "npz=[^ ]*/${PHASE_RE}/" \
        || { echo "[${CELL_ID}/${ARM_TAG}] gated phase NPZ 로드 0건 ABORT"; exit 12; }
      if [ -n "${GATED_PHASES:-}" ]; then
        want=$(echo "$GATED_PHASES" | tr ',' '\n' | sort | paste -sd,)
        got=$(echo "$reg" | grep -o 'phases=[^ ]*' | head -1 | cut -d= -f2 | tr ',' '\n' | sort | paste -sd,)
        [ "$want" = "$got" ] || { echo "[${CELL_ID}/${ARM_TAG}] gated phase 집합 불일치 ABORT"; exit 12; }
      fi
    fi
    norms=$(grep '\[steer-norms\]' "$LOG" || true)
    { echo "$pf"; echo "$reg"; echo "$norms"; } | sed "s/^/[preflight ${port}] /" >> "${LOGDIR}/${ARM_TAG}_preflight.log"
    mkdir -p "${OUT_HOST}/${ARM_TAG}"
    echo "$norms" > "${OUT_HOST}/${ARM_TAG}/steer_norms_${port}.log"
    echo "$health" > "${OUT_HOST}/${ARM_TAG}/serve_fingerprint_${port}.json"
  done
}

GATED_FLAG=""; [ "$STEER_MODE" = gated ] && GATED_FLAG="--gated-steering"
PROX_FLAG="--proximity-phases"

run_w() {
  local wid=$1 port=$2 k=0 rc out existing fail=0
  for ep in "${EPS[@]}"; do
    k=$((k+1)); [ $(( (k-1) % NW )) -eq "$wid" ] || continue
    existing=$(ls "${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.json 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
      if grep -q "\"run_tag\": \"${RUN_TAG}\"" "$existing"; then continue; fi
      echo "[${CELL_ID}/${ARM_TAG}] ep${ep} 기존 사이드카 run_tag 불일치 — 오염 의심 ABORT" >&2
      exit 14
    fi
    rc=0
    out=$(docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "${OUT_CONT}/${ARM_TAG}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "${ENVSEED[$ep]}" --inference-seed "${NOISESEED[$ep]}" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --no-features --expect-chunk-len "$CHUNK_LEN" --run-tag "$RUN_TAG" $PROX_FLAG $GATED_FLAG 2>&1) || rc=$?
    echo "$out" | grep -E "^wrote|Error|Traceback" || true
    if [ "$rc" -ne 0 ]; then
      echo "[${CELL_ID}/${ARM_TAG}] ep${ep} collector rc=${rc}"; fail=1
    fi
  done
  return $fail
}

echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') ${MACHINE_TAG:-w2} mode=${STEER_MODE} eps=${#EPS[@]}/${EXPECT_N} tier=${OUT_TIER} manifest=${MANIFEST_SHA}"
mkdir -p "${OUT_HOST}/${ARM_TAG}"
SEEDPAIRS=$(for ep in "${MEPS[@]}"; do printf '%s:%s:%s ' "$ep" "${ENVSEED[$ep]}" "${NOISESEED[$ep]}"; done)
CORE_SPEC="{\"cell\":\"${CELL_ID}\",\"arm_tag\":\"${ARM_TAG}\",\"mode\":\"${STEER_MODE}\",\"npz_dir\":\"${NPZ_DIR:-}\",\"layers\":\"${STEER_LAYERS:-}\",\"alpha\":\"${STEER_ALPHA:-meta_per_step}\",\"beta\":\"${STEER_BETA}\",\"key\":\"${STEER_KEY}\",\"token_select\":\"${STEER_TOKEN_SELECT}\",\"denoise\":\"${STEER_DENOISE}\",\"expect_n\":${EXPECT_N},\"tier\":\"${OUT_TIER}\",\"manifest_sha\":\"${MANIFEST_SHA}\",\"npz_shas\":\"${NPZ_SHAS:-}\",\"gated_phases\":\"${GATED_PHASES:-}\",\"seed_pairs\":\"${SEEDPAIRS% }\"}"
SPEC_SHA=$(printf '%s' "$CORE_SPEC" | sha256sum | cut -c1-12)
RUN_TAG="${ARM_TAG}:${SPEC_SHA}"
if [ -f "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json" ]; then
  old_sha=$(grep -o '"spec_sha": *"[0-9a-f]*"' "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json" | grep -o '[0-9a-f]\{12\}' || true)
  if [ -n "$old_sha" ] && [ "$old_sha" != "$SPEC_SHA" ]; then
    echo "[${CELL_ID}/${ARM_TAG}] ARM_SPEC 불변 위반: ${old_sha} != ${SPEC_SHA} ABORT"
    exit 14
  fi
fi
printf '{"spec_sha":"%s","machine":"%s","manifest":"%s","core":%s}\n' \
  "$SPEC_SHA" "${MACHINE_TAG:-worker2-a100}" "$MANIFEST" "$CORE_SPEC" > "${OUT_HOST}/${ARM_TAG}/ARM_SPEC.json"
start_serves
PIDS=()
for wid in $(seq 0 $((NW-1))); do
  run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/${ARM_TAG}_w${wid}.log" 2>&1 & PIDS+=($!)
done
WFAIL=0
for p in "${PIDS[@]}"; do wait "$p" || WFAIL=1; done
kill_serves
trap - EXIT
d="${OUT_HOST}/${ARM_TAG}/raw_rollouts/${TASK}/${CELL_ID}"
MISS=0
for ep in "${EPS[@]}"; do
  cnt=$(ls "$d"/task${CELL_INDEX}--ep${ep}--succ*.json 2>/dev/null | wc -l)
  if [ "$cnt" -ne 1 ]; then
    echo "[${CELL_ID}/${ARM_TAG}] ep${ep} 사이드카 ${cnt}개 (기대 1)"; MISS=1; continue
  fi
  grep -q "\"run_tag\": \"${RUN_TAG}\"" "$d"/task${CELL_INDEX}--ep${ep}--succ*.json \
    || { echo "[${CELL_ID}/${ARM_TAG}] ep${ep} run_tag 불일치"; MISS=1; }
done
s=$(ls "$d"/*succ1.json 2>/dev/null | wc -l); f=$(ls "$d"/*succ0.json 2>/dev/null | wc -l)
echo "[${CELL_ID}/${ARM_TAG}] $(date '+%F %T') DONE ${s}/$((s+f)) (기대 ${#EPS[@]}, wfail=${WFAIL} miss=${MISS})"
[ "$WFAIL" -eq 0 ] && [ "$MISS" -eq 0 ]
