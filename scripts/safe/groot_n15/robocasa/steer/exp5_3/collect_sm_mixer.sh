#!/bin/bash
# exp5-3 Phase 2 — mixer(OpenStandMixerHead) scene-matched base 수집 (srv50 GPU3, serve 6).
#
# 설계 = drawer/beer 와 동일: scenario_seed 20종 × inference_seed 8종 = 160판, 무개입(A0),
# capture ON(full-token [L,K=4,T=49,D=1536]) → 한 scene 안에 succ/fail 혼재 → scene 통제 대조.
# 수요: exp5-3 Phase 2(다른 task family 로 within-scene 결론 일반화) + exp5-1 SAE G1/G2 +
#       exp4-1 primary cell.
#
# seed 선별 근거:
#   · instruction 단일 확인 — 20/20 "Open the stand mixer head." (env reset 스캔, 2026-07-27)
#     → drawer(좌/우)·PPCC(물체) 와 달리 variant 선별 불필요.
#   · **feasibility 필터 적용** — mixer_feasibility.json 의 BLOCKED 5종
#     (100010·100022·100037·100071·100094) 제외. 기하학적으로 성공 불가한 seed 는
#     어떤 개입으로도 구제 불가라 분모를 오염시킨다.
#   → 사용 seed = 100000-100009 + 100011-100020 (feasible 20종)
#
# ⚠ mixer 결정론 주의: exp4-1 재현 검증에서 21판 중 5판이 fail↔succ 뒤집혔다(경계 민감).
#   base 표본 수집에는 무해하나, 이 데이터로 paired steering 을 하려면 뒤집힘율을 먼저 재야 한다.
#
# 산출물은 3분 경과분부터 승준 HDD 직송 후 로컬 pkl 삭제 (srv50 디스크 여유 ~181G 방어).
set -u
REPO="${REPO:-$HOME/pkt_ws/temporal_vla}"
cd "$REPO" || exit 1
PYP="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
CELL="exp41_mixer"
TASK="OpenStandMixerHead"
ENVN="robocasa_panda_omron/OpenStandMixerHead_PandaOmron_Env"
IDX=0
INSTR="Open the stand mixer head."
OUT_REL="outputs/eval/robocasa/groot_n15/scene_matched_mixer/$CELL"
OUT_ABS="$REPO/$OUT_REL"
PORTS=(8640 8641 8642 8643 8644 8645)
N_SEED=${N_SEED:-8}
LOG="$HOME/sm_mixer.log"
DONE_FLAG="$HOME/sm_mixer.DONE"
SJ="kimseungjun@166.104.146.37"; SJP=11112
RDEST="/home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/scene_matched_mixer"

SEEDS=(100000 100001 100002 100003 100004 100005 100006 100007 100008 100009 \
       100011 100012 100013 100014 100015 100016 100017 100018 100019 100020)

ROWS=()
scene=0
for S in "${SEEDS[@]}"; do
  for k in $(seq 0 $((N_SEED-1))); do
    ROWS+=("$S $((1000000*k)) $((scene*N_SEED+k))")
  done
  scene=$((scene+1))
done

{ echo "=== mixer scene-matched 수집 시작 $(date -u +%FT%T) ==="
  echo "총 ${#ROWS[@]}판 (scene ${#SEEDS[@]} × seed $N_SEED), 워커 ${#PORTS[@]}"; } | tee -a "$LOG"

run_one() {
  local S=$1 inf=$2 epidx=$3 port=$4
  local d="$OUT_ABS/raw_rollouts/$TASK/$CELL"
  # resume: pkl(미전송) 또는 csv(전송 후 pkl 만 삭제되고 남음)
  if ls "$d/task${IDX}--ep${epidx}--succ"*.pkl >/dev/null 2>&1 \
     || ls "$d/task${IDX}--ep${epidx}--succ"*.csv >/dev/null 2>&1; then return 0; fi
  docker exec -e MUJOCO_GL=egl \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e NUMEXPR_NUM_THREADS=2 \
    -e PYTHONPATH="$PYP" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:$port" --task "$TASK" --env-name "$ENVN" \
    --output-dir "/temporal_vla/$OUT_REL/raw_rollouts" --cell-id "$CELL" --cell-index "$IDX" \
    --canonical-instruction "$INSTR" --episode-start-idx "$epidx" --n-episodes 1 \
    --seed "$S" --inference-seed "$inf" --n-action-steps 5 --max-episode-steps 720 \
    --video-fps 20 --steps-per-render 2 --wait-ready --proximity-phases \
    > /dev/null 2>&1
  echo "[$([ $? = 0 ] && echo ok || echo FAIL:$?)] S$S inf$inf ep$epidx p$port $(date -u +%T)" >> "$LOG"
}

do_ship() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p $SJP "$SJ" "mkdir -p $RDEST" >/dev/null 2>&1 || return 1
  rsync -a --partial -e "ssh -o BatchMode=yes -p $SJP" --exclude '*.pkl' \
    "$OUT_ABS/" "$SJ:$RDEST/$CELL/" >/dev/null 2>&1
  local lst="$HOME/.sm_mixer_pkl"
  ( cd "$OUT_ABS" && find . -name '*.pkl' -mmin +3 ) > "$lst" 2>/dev/null
  local n; n=$(wc -l < "$lst")
  if [ "$n" -gt 0 ]; then
    rsync -a --partial --remove-source-files --files-from="$lst" \
      -e "ssh -o BatchMode=yes -p $SJP" "$OUT_ABS/" "$SJ:$RDEST/$CELL/" >/dev/null 2>&1
    echo "[ship] $(date -u +%T) pkl $n 건 전송+삭제 rc=$?" >> "$LOG"
  fi
}

ship_loop() {
  while [ ! -f "$DONE_FLAG" ]; do sleep 300; do_ship; done
  sleep 200; do_ship
  echo "[ship] 최종 전송 완료 $(date -u +%FT%T)" >> "$LOG"
}

worker() {
  local wi=$1 port=${PORTS[$1]}
  for ((i=wi; i<${#ROWS[@]}; i+=${#PORTS[@]})); do
    run_one ${ROWS[$i]} $port
  done
}

rm -f "$DONE_FLAG"
ship_loop &
SHIP_PID=$!
WPIDS=()
for w in "${!PORTS[@]}"; do worker $w & WPIDS+=($!); done
wait "${WPIDS[@]}"       # ★워커만 대기 (bare wait 은 serve/ship 까지 기다려 영구 정지)
touch "$DONE_FLAG"
wait "$SHIP_PID" 2>/dev/null
{ echo "=== 수집 완료 $(date -u +%FT%T) ==="
  echo "로컬 잔여 pkl: $(find "$OUT_ABS" -name '*.pkl' 2>/dev/null | wc -l)"
  echo "완료 판수(csv): $(find "$OUT_ABS" -name '*.csv' 2>/dev/null | wc -l)"
  echo "succ/fail: $(find "$OUT_ABS" -name '*.csv' | grep -c succ1)/$(find "$OUT_ABS" -name '*.csv' | grep -c succ0)"; } | tee -a "$LOG"
