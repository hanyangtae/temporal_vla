#!/usr/bin/env bash
# pq3 fit 수집 러너 — full-token capture ON, scene-diverse (episode i → seed_i).
# pq2_cell_runner.sh 의 수집 계열 파생. 계획서 v9 §C + 파이프라인 규약:
#   ① collect_plan.tsv 의 (s_idx, env_seed, noise_seed) 를 그대로 구동 (S 순서 준수,
#      backfill 은 EPLIST 로 부분집합 지정)
#   ② serve = --collect --groot-dit-token-pool all_token_full --capture-vl
#      --groot-vl-capture-point post_vl_sa_full --groot-dit-capture-layers 0,2,4,8,10,12,15
#   ③ natural reset 강제 — ep-meta replay 플래그 미사용 (OpenDrawer drawer_side 함정)
#   ④ SHIP=1 이면 에피소드 완료마다 승준 rsync(-c 체크섬) → 검증 후 로컬 pkl 삭제
#      (fit activation 은 승준에만 보관 — 디스크 98% 사고 재발 방지). 스템 csv·mp4 는
#      로컬 유지 (p0 게이트·freeze 스캔용).
#
# env (필수): CELL_ID TASK ENVN CELL_INDEX INSTR MANIFEST(collect_plan.tsv) GPUS_L PORTS_L
# env (선택): EPLIST="0 1 ... 14"(기본 S0..S14) SHIP=0|1 SJ=승준 ssh 대상 SJ_PORT SJ_ROOT
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
NAS=5; MAXEP=720
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${INSTR:?}" "${MANIFEST:?}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
FITROOT="outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts"
OUT_HOST="${REPO_ROOT}/${FITROOT}"
OUT_CONT="/temporal_vla/${FITROOT}"
LOGDIR="${OUT_HOST}/${TASK}/${CELL_ID}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}
SHIP="${SHIP:-0}"
SJ="${SJ:-kimseungjun@166.104.146.37}"; SJ_PORT="${SJ_PORT:-11112}"
SJ_ROOT="${SJ_ROOT:-~/workspace/temporal_vla/${FITROOT}}"

declare -A ENVSEED NOISESEED
MEPS=()
while IFS=$'\t' read -r ep envs noise _rest; do
  case "$ep" in ''|\#*|s_idx|episode_idx) continue ;; esac
  MEPS+=("$ep"); ENVSEED[$ep]=$envs; NOISESEED[$ep]=$noise
done < "$MANIFEST"
if [ -n "${EPLIST:-}" ]; then EPS=($EPLIST); else EPS=($(printf '%s\n' "${MEPS[@]}" | head -15)); fi
for ep in "${EPS[@]}"; do
  [ -n "${ENVSEED[$ep]:-}" ] || { echo "[collect ${CELL_ID}] ep=${ep} 가 collect_plan 에 없음 ABORT"; exit 13; }
done

kill_port() { docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${1}' || true" 2>/dev/null || true; }
kill_serves() { for port in "${PORTS[@]}"; do kill_port "$port"; done; sleep 5; }
trap kill_serves EXIT

start_serves() {
  # 포트의 기존 서버 오인 방지 (eval 러너와 동일 — Gate2 R3 높음#4)
  for i in $(seq 0 $((NW-1))); do kill_port "${PORTS[$i]}"; done
  sleep 3
  for i in $(seq 0 $((NW-1))); do
    docker exec lerobot bash -lc "rm -f /tmp/pq3c_${CELL_ID}_${PORTS[$i]}.log"
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda --collect --capture-vl \
         --groot-dit-capture-layers ${CAP} --groot-dit-token-pool all_token_full \
         --groot-vl-capture-point post_vl_sa_full \
         > /tmp/pq3c_${CELL_ID}_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    [ $ok = 1 ] || { echo "[collect ${CELL_ID}] serve ${port} TIMEOUT"; exit 11; }
    LOG="/tmp/pq3c_${CELL_ID}_${port}.log"
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound|Address already in use|Errno 98' ${LOG}"; then
      echo "[collect ${CELL_ID}] serve ABORT ${port}"; docker exec lerobot bash -lc "tail -20 ${LOG}"; exit 11
    fi
    boot_log=$(docker exec lerobot bash -lc "grep -o '\[serve-boot\] id=[0-9a-f]*' ${LOG}" | tail -1 | cut -d= -f2 || true)
    health=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health")
    boot_http=$(echo "$health" | grep -o '"boot_id":"[0-9a-f]*"' | cut -d'"' -f4 || true)
    [ -n "$boot_log" ] && [ "$boot_log" = "$boot_http" ] \
      || { echo "[collect ${CELL_ID}] boot_id 불일치 — 포트에 다른 서버 ABORT"; exit 12; }
    # full-token 캡처 preflight: health 의 capture_token_mode 확인 (구/신 혼입 방지)
    echo "$health" | grep -q '"capture_token_mode":"all_token_full"' \
      || { echo "[collect ${CELL_ID}] serve ${port} capture_token_mode != all_token_full ABORT"; exit 12; }
  done
}

ship_ep() { # stem — pkl 승준 직송 후 3중 검증(체크섬·실물 size·상식 크기), 통과 시에만 로컬 삭제
  # [유실 사건 표준 2026-07-16] 이름 세기 금지 · 실물(-type f) 기준 · 평균 크기 상식 체크.
  # 심링크는 애초에 만들지 않음(-L 불요, 실물 전송). SHIPPED.tsv ledger 에 size+sha 기록.
  local stem=$1 d="${OUT_HOST}/${TASK}/${CELL_ID}"
  local pkl="${d}/${stem}.pkl"
  [ -f "$pkl" ] || return 0
  if [ -L "$pkl" ]; then echo "[ship] ${stem}.pkl 이 심링크 — 직송 금지·조사 필요"; return 1; fi
  local lsize lsha
  lsize=$(stat -c %s "$pkl")
  # 상식 체크: full-token fit pkl 은 개당 수십 MB — 5MB 미만이면 껍데기/축소본 의심, 중단
  if [ "$lsize" -lt 5000000 ]; then
    echo "[ship] ${stem}.pkl size=${lsize}B < 5MB — full-token pkl 상식 위반, 중단·조사"; return 1
  fi
  lsha=$(sha256sum "$pkl" | cut -d' ' -f1)
  ssh -p "$SJ_PORT" "$SJ" "mkdir -p ${SJ_ROOT}/${TASK}/${CELL_ID}" || return 1
  rsync -c -e "ssh -p ${SJ_PORT}" "$pkl" "${SJ}:${SJ_ROOT}/${TASK}/${CELL_ID}/" || return 1
  # 검증 ①: 원격 실물 size 대조 (find -type f 기준 — 심링크·부재 모두 실패)
  local rsize
  rsize=$(ssh -p "$SJ_PORT" "$SJ" "find ${SJ_ROOT}/${TASK}/${CELL_ID} -maxdepth 1 -type f -name '${stem}.pkl' -printf '%s'" 2>/dev/null || true)
  [ "$rsize" = "$lsize" ] || { echo "[ship] ${stem}.pkl 원격 실물 size 불일치 (${rsize:-none}!=${lsize}) — 로컬 보존"; return 1; }
  # 검증 ②: 재-rsync dry-run 이 전송할 게 없어야 함 (내용 체크섬 일치)
  local todo
  todo=$(rsync -c -n -e "ssh -p ${SJ_PORT}" "$pkl" "${SJ}:${SJ_ROOT}/${TASK}/${CELL_ID}/" | grep -c "$(basename "$pkl")" || true)
  [ "$todo" -eq 0 ] || { echo "[ship] ${stem}.pkl 체크섬 불일치 — 로컬 보존"; return 1; }
  # ledger (Gate C 재검증·아카이브 감사 근거)
  printf '%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "${stem}.pkl" "$lsize" "$lsha" >> "${d}/SHIPPED.tsv"
  rm -f "$pkl"
  echo "[ship] ${stem}.pkl -> 승준 (size+sha verified, local removed)"
}

run_w() {
  local wid=$1 port=$2 k=0 rc out fail=0
  for ep in "${EPS[@]}"; do
    k=$((k+1)); [ $(( (k-1) % NW )) -eq "$wid" ] || continue
    local stem_glob="${OUT_HOST}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"
    if ! ls "${stem_glob}"*.csv >/dev/null 2>&1; then
      # collector rc 소거 금지 (Gate2 R2 높음#3)
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
      if [ "$rc" -ne 0 ]; then echo "[collect ${CELL_ID}] ep${ep} collector rc=${rc}"; fail=1; continue; fi
    fi
    if [ "$SHIP" = 1 ]; then
      for st in "${stem_glob}"1 "${stem_glob}"0; do
        if [ -e "${st}.pkl" ]; then
          ship_ep "$(basename "$st")" || { echo "[collect ${CELL_ID}] ep${ep} 직송 실패"; fail=1; }
        fi
      done
    fi
  done
  return $fail
}

echo "[collect ${CELL_ID}] $(date '+%F %T') eps=${#EPS[@]} ship=${SHIP} manifest=$(sha256sum "$MANIFEST" | cut -c1-12)"
start_serves
PIDS=()
for wid in $(seq 0 $((NW-1))); do
  run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/collect_w${wid}.log" 2>&1 & PIDS+=($!)
done
WFAIL=0
for p in "${PIDS[@]}"; do wait "$p" || WFAIL=1; done
kill_serves
trap - EXIT
d="${OUT_HOST}/${TASK}/${CELL_ID}"
# 완료 판정: 요청한 각 ep 의 csv 스템 존재 + (SHIP=1) ledger 등재·로컬 pkl 부재
MISS=0
for ep in "${EPS[@]}"; do
  cnt=$(ls "$d"/task${CELL_INDEX}--ep${ep}--succ*.csv 2>/dev/null | wc -l)
  [ "$cnt" -eq 1 ] || { echo "[collect ${CELL_ID}] ep${ep} csv 스템 ${cnt}개 (기대 1)"; MISS=1; continue; }
  if [ "$SHIP" = 1 ]; then
    stem=$(basename "$(ls "$d"/task${CELL_INDEX}--ep${ep}--succ*.csv)" .csv)
    grep -q "	${stem}.pkl	" "$d/SHIPPED.tsv" 2>/dev/null \
      || { echo "[collect ${CELL_ID}] ep${ep} SHIPPED ledger 미등재"; MISS=1; }
    [ ! -e "$d/${stem}.pkl" ] \
      || { echo "[collect ${CELL_ID}] ep${ep} 직송 후 로컬 pkl 잔존 (검증 실패 흔적)"; MISS=1; }
  fi
done
n=$(ls "$d"/task${CELL_INDEX}--ep*--succ*.csv 2>/dev/null | wc -l)
echo "[collect ${CELL_ID}] $(date '+%F %T') DONE stems=${n} (기대 ${#EPS[@]}, wfail=${WFAIL} miss=${MISS})"
[ "$WFAIL" -eq 0 ] && [ "$MISS" -eq 0 ]
