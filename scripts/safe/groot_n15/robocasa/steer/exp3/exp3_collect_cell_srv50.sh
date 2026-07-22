#!/usr/bin/env bash
# exp3(구 pq3) fit 수집 러너 — 원격 워커(srv50(구 w2/worker2)/.50·srv48(구 w48)) 변형. exp3_collect_cell.sh 와 계약 동일하되
# serve 를 호스트 conda 로 기동(exp3_cell_runner_srv50.sh 방식). SHIP 은 워커→승준 직송
# (fit30 라운드: A100 pubkey 를 승준 authorized_keys 에 등록, 사용자 승인 2026-07-17).
#
# env (필수): CELL_ID TASK ENVN CELL_INDEX INSTR MANIFEST GPUS_L PORTS_L
# env (선택): EPLIST SHIP=0|1 SJ SJ_PORT SJ_ROOT [HPY] [HF_HOME_OVERRIDE] [MACHINE_TAG]
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
NAS=5; MAXEP=720
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${INSTR:?}" "${MANIFEST:?}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
FITROOT="outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts"  # 워커 원격 dir — 원격은 구명(pq) 유지
OUT_HOST="${REPO_ROOT}/${FITROOT}"
OUT_CONT="/temporal_vla/${FITROOT}"
LOGDIR="${OUT_HOST}/${TASK}/${CELL_ID}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}
SHIP="${SHIP:-0}"
SJ="${SJ:-kimseungjun@166.104.146.37}"; SJ_PORT="${SJ_PORT:-11112}"
SJ_ROOT="${SJ_ROOT:-~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts}"  # 승준 원격 — 원격은 구명(pq) 유지
HPY="${HPY:-$HOME/miniconda3/envs/lerobot_050_groot/bin/python}"

declare -A ENVSEED NOISESEED
MEPS=()
while IFS=$'\t' read -r ep envs noise _rest; do
  case "$ep" in ''|\#*|s_idx|episode_idx) continue ;; esac
  MEPS+=("$ep"); ENVSEED[$ep]=$envs; NOISESEED[$ep]=$noise
done < "$MANIFEST"
if [ -n "${EPLIST:-}" ]; then EPS=($EPLIST); else EPS=($(printf '%s\n' "${MEPS[@]}" | head -15)); fi
for ep in "${EPS[@]}"; do
  [ -n "${ENVSEED[$ep]:-}" ] || { echo "[collect-srv50 ${CELL_ID}] ep=${ep} 가 collect_plan 에 없음 ABORT"; exit 13; }
done

kill_port() { pkill -f "serve/lerobot.py.*--port ${1}" 2>/dev/null || true; }
kill_serves() { for port in "${PORTS[@]}"; do kill_port "$port"; done; sleep 5; }
trap kill_serves EXIT

start_serves() {
  for i in $(seq 0 $((NW-1))); do kill_port "${PORTS[$i]}"; done
  sleep 3
  for i in $(seq 0 $((NW-1))); do
    local LOG="/tmp/exp3c_${CELL_ID}_${PORTS[$i]}.log"; rm -f "$LOG"
    ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        PYTHONPATH="$REPO_ROOT/lerobot/src" ${HF_HOME_OVERRIDE:+HF_HOME=$HF_HOME_OVERRIDE} \
        ${HF_HOME_OVERRIDE:+HF_HUB_OFFLINE=1} \
        "$HPY" scripts/serve/lerobot.py --profile ${PROFILE} \
          --host '*' --port "${PORTS[$i]}" --device cuda --collect --capture-vl \
          --groot-dit-capture-layers ${CAP} --groot-dit-token-pool all_token_full \
          --groot-vl-capture-point post_vl_sa_full \
        > "$LOG" 2>&1 < /dev/null & )
  done
  for port in "${PORTS[@]}"; do
    local ok=0 LOG="/tmp/exp3c_${CELL_ID}_${port}.log"
    for _ in $(seq 1 180); do
      curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
      sleep 5
    done
    [ $ok = 1 ] || { echo "[collect-srv50 ${CELL_ID}] serve ${port} TIMEOUT"; exit 11; }
    grep -qiE 'Traceback|FAILED|FileNotFound|Address already in use|Errno 98' "$LOG" \
      && { echo "[collect-srv50 ${CELL_ID}] serve ABORT ${port}"; tail -20 "$LOG"; exit 11; }
    boot_log=$(grep -o '\[serve-boot\] id=[0-9a-f]*' "$LOG" | tail -1 | cut -d= -f2 || true)
    health=$(curl -s -m 3 "http://127.0.0.1:${port}/health")
    boot_http=$(echo "$health" | grep -o '"boot_id":"[0-9a-f]*"' | cut -d'"' -f4 || true)
    [ -n "$boot_log" ] && [ "$boot_log" = "$boot_http" ] \
      || { echo "[collect-srv50 ${CELL_ID}] boot_id 불일치 ABORT"; exit 12; }
    echo "$health" | grep -q '"capture_token_mode":"all_token_full"' \
      || { echo "[collect-srv50 ${CELL_ID}] capture_token_mode != all_token_full ABORT"; exit 12; }
  done
}

ship_ep() { # stem — 승준 직송 3중 검증(exp3_collect_cell.sh 동일 표준)
  local stem=$1 d="${OUT_HOST}/${TASK}/${CELL_ID}" pkl lsize lsha rsize todo
  pkl="${d}/${stem}.pkl"
  [ -e "$pkl" ] || return 0
  if [ -L "$pkl" ]; then echo "[ship] ${stem}.pkl 심링크 — 직송 금지"; return 1; fi
  lsize=$(stat -c%s "$pkl"); lsha=$(sha256sum "$pkl" | cut -d' ' -f1)
  if [ "$lsize" -lt 5000000 ]; then
    echo "[ship] ${stem}.pkl size=${lsize}B < 5MB 상식 위반, 중단"; return 1
  fi
  ssh -p "$SJ_PORT" "$SJ" "mkdir -p ${SJ_ROOT}/${TASK}/${CELL_ID}" || return 1
  rsync -c -e "ssh -p ${SJ_PORT}" "$pkl" "${SJ}:${SJ_ROOT}/${TASK}/${CELL_ID}/" || return 1
  rsize=$(ssh -p "$SJ_PORT" "$SJ" "find ${SJ_ROOT}/${TASK}/${CELL_ID} -maxdepth 1 -type f -name '${stem}.pkl' -printf '%s'" 2>/dev/null || true)
  [ "$rsize" = "$lsize" ] || { echo "[ship] ${stem}.pkl 원격 size 불일치 (${rsize:-none}!=${lsize})"; return 1; }
  todo=$(rsync -c -n -e "ssh -p ${SJ_PORT}" "$pkl" "${SJ}:${SJ_ROOT}/${TASK}/${CELL_ID}/" | grep -c "$(basename "$pkl")" || true)
  [ "$todo" -eq 0 ] || { echo "[ship] ${stem}.pkl 체크섬 불일치"; return 1; }
  printf '%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "${stem}.pkl" "$lsize" "$lsha" >> "${d}/SHIPPED.tsv"
  rm -f "$pkl"
  echo "[ship] ${stem}.pkl -> 승준 (verified, local removed)"
}

run_w() {
  local wid=$1 port=$2 k=0 rc out fail=0
  for ep in "${EPS[@]}"; do
    k=$((k+1)); [ $(( (k-1) % NW )) -eq "$wid" ] || continue
    local stem_glob="${OUT_HOST}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"
    if ! ls "${stem_glob}"*.csv >/dev/null 2>&1; then
      rc=0
      out=$(docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
        python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
        --output-dir "${OUT_CONT}" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
        --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
        --seed "${ENVSEED[$ep]}" --inference-seed "${NOISESEED[$ep]}" --n-action-steps "$NAS" \
        --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
        --proximity-phases 2>&1) || rc=$?
      echo "$out" | grep -E "^wrote|Error|Traceback" || true
      if [ "$rc" -ne 0 ]; then echo "[collect-srv50 ${CELL_ID}] ep${ep} rc=${rc}"; fail=1; continue; fi
    fi
    if [ "$SHIP" = 1 ]; then
      for st in "${stem_glob}"1 "${stem_glob}"0; do
        [ -e "${st}.pkl" ] && { ship_ep "$(basename "$st" .pkl)" || { echo "[collect-srv50 ${CELL_ID}] ep${ep} 직송 실패"; fail=1; }; }
      done
    fi
  done
  return $fail
}

echo "[collect-srv50 ${CELL_ID}] $(date '+%F %T') ${MACHINE_TAG:-srv50} eps=${#EPS[@]} ship=${SHIP} manifest=$(sha256sum "$MANIFEST" | cut -c1-12)"
start_serves
PIDS=()
for wid in $(seq 0 $((NW-1))); do
  run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/collect_srv50_w${wid}.log" 2>&1 & PIDS+=($!)
done
WFAIL=0
for p in "${PIDS[@]}"; do wait "$p" || WFAIL=1; done
kill_serves
trap - EXIT
d="${OUT_HOST}/${TASK}/${CELL_ID}"
MISS=0
for ep in "${EPS[@]}"; do
  cnt=$(ls "$d"/task${CELL_INDEX}--ep${ep}--succ*.csv 2>/dev/null | wc -l)
  [ "$cnt" -eq 1 ] || { echo "[collect-srv50 ${CELL_ID}] ep${ep} csv 스템 ${cnt}개 (기대 1)"; MISS=1; continue; }
  if [ "$SHIP" = 1 ]; then
    stem=$(basename "$(ls "$d"/task${CELL_INDEX}--ep${ep}--succ*.csv)" .csv)
    grep -q "	${stem}.pkl	" "$d/SHIPPED.tsv" 2>/dev/null \
      || { echo "[collect-srv50 ${CELL_ID}] ep${ep} SHIPPED 미등재"; MISS=1; }
    [ ! -e "$d/${stem}.pkl" ] || { echo "[collect-srv50 ${CELL_ID}] ep${ep} 로컬 pkl 잔존"; MISS=1; }
  fi
done
[ "$WFAIL" = 0 ] && [ "$MISS" = 0 ] \
  && echo "[collect-srv50 ${CELL_ID}] $(date '+%F %T') DONE stems=$(ls "$d"/task${CELL_INDEX}--ep*--succ*.csv 2>/dev/null | wc -l) (기대 ${#EPS[@]}, wfail=${WFAIL} miss=${MISS})" \
  || { echo "[collect-srv50 ${CELL_ID}] FAIL wfail=${WFAIL} miss=${MISS}"; exit 1; }
