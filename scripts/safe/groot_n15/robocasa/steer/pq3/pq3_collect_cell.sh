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

start_serves() {
  for i in $(seq 0 $((NW-1))); do
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
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/pq3c_${CELL_ID}_${port}.log"; then
      echo "[collect ${CELL_ID}] serve ABORT ${port}"; docker exec lerobot bash -lc "tail -20 /tmp/pq3c_${CELL_ID}_${port}.log"; exit 11
    fi
    # full-token 캡처 preflight: health 의 capture_token_mode 확인 (구/신 혼입 방지)
    hm=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health" | grep -o '"capture_token_mode":"all_token_full"' || true)
    [ -n "$hm" ] || { echo "[collect ${CELL_ID}] serve ${port} capture_token_mode != all_token_full ABORT"; exit 12; }
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }

ship_ep() { # stem — pkl 승준 직송(-c 체크섬) 후 재검증·로컬 pkl 삭제
  local stem=$1 d="${OUT_HOST}/${TASK}/${CELL_ID}"
  local pkl="${d}/${stem}.pkl"
  [ -f "$pkl" ] || return 0
  ssh -p "$SJ_PORT" "$SJ" "mkdir -p ${SJ_ROOT}/${TASK}/${CELL_ID}" || return 1
  rsync -c -e "ssh -p ${SJ_PORT}" "$pkl" "${SJ}:${SJ_ROOT}/${TASK}/${CELL_ID}/" || return 1
  # 검증: 재-rsync dry-run 이 전송할 게 없어야 함 (체크섬 일치)
  local todo
  todo=$(rsync -c -n -e "ssh -p ${SJ_PORT}" "$pkl" "${SJ}:${SJ_ROOT}/${TASK}/${CELL_ID}/" | grep -c "$(basename "$pkl")" || true)
  if [ "$todo" -eq 0 ]; then rm -f "$pkl"; echo "[ship] ${stem}.pkl -> 승준 (verified, local removed)"
  else echo "[ship] ${stem}.pkl 체크섬 불일치 — 로컬 보존"; return 1; fi
}

run_w() {
  local wid=$1 port=$2 k=0
  for ep in "${EPS[@]}"; do
    k=$((k+1)); [ $(( (k-1) % NW )) -eq "$wid" ] || continue
    local stem_glob="${OUT_HOST}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"
    if ls "${stem_glob}"*.csv >/dev/null 2>&1; then continue; fi
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "${OUT_CONT}" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "${ENVSEED[$ep]}" --inference-seed "${NOISESEED[$ep]}" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --proximity-phases 2>&1 | grep -E "^wrote|Error|Traceback" || true
    if [ "$SHIP" = 1 ]; then
      for st in "${stem_glob}"1 "${stem_glob}"0; do
        [ -e "${st}.pkl" ] && ship_ep "$(basename "$st")"
      done
    fi
  done
}

echo "[collect ${CELL_ID}] $(date '+%F %T') eps=${#EPS[@]} ship=${SHIP} manifest=$(sha256sum "$MANIFEST" | cut -c1-12)"
start_serves
for wid in $(seq 0 $((NW-1))); do run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/collect_w${wid}.log" 2>&1 & done
wait
kill_serves
d="${OUT_HOST}/${TASK}/${CELL_ID}"
n=$(ls "$d"/task${CELL_INDEX}--ep*--succ*.csv 2>/dev/null | wc -l)
echo "[collect ${CELL_ID}] $(date '+%F %T') DONE stems=${n} (기대 ${#EPS[@]})"
[ "$n" -ge "${#EPS[@]}" ]
