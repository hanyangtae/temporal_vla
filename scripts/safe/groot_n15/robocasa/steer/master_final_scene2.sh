#!/usr/bin/env bash
# ★ 최종 scene-seed 매트릭스 v2 — bread 우선 2-phase, GPU 4/5/6 (0번 동료 양보).
# PhaseB: bread 3신규 수집 → bread fits(ps×3, xb×3) → bread 4cell 체인(base,ps,xb,perm/gated×15/30/60,+L4/L812)
#         → bread 중간 집계+Notion.
# PhaseA: apple 3신규 수집 → apple fits(ps,xa)+grand(gx, 8cell) → apple 4cell 체인(base,ps,xa,gx)
#         + bread gx arm → 최종 집계+Notion+rsync.
set -uo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
LOG=outputs/eval/robocasa/groot_n15/pipeline_final.log
exec >>"$LOG" 2>&1
echo "=========== FINAL2 START $(date '+%F %T') ==========="
PY=~/miniconda3/envs/libero_bench/bin/python
FITPY=scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py
R=$(ls scripts/safe/groot_n15/robocasa/steer/heldout*_round_cell.sh)
SRC6=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts
AN=outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
CROOT=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
FIT_GROUPS="global,reach-to-object,grasp,transport,place,insert-settle"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
# GPU/port lanes: GPU4=8470,8471 / GPU5=8474,8475 / GPU6=8472,8473
CELLS=(
  "ppcc_bread|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|100084|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s300028|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|300028|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s300033|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|300033|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s400020|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|400020|Pick the bread from the counter and place it in the cabinet."
  "ppcs_apple|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100074|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100050|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100050|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100084|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100084|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100104|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100104|Pick the apple from the plate and place it in the pan."
)
BREADS="ppcc_bread ppcc_bread_s300028 ppcc_bread_s300033 ppcc_bread_s400020"
APPLES="ppcs_apple ppcs_apple_s100050 ppcs_apple_s100084 ppcs_apple_s100104"
row_of() { for c in "${CELLS[@]}"; do [[ "$c" == "$1|"* ]] && { echo "$c"; return; }; done; }

start_serve() { docker exec -d -e CUDA_VISIBLE_DEVICES="$1" lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
     --host '*' --port ${2} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
     > /tmp/final_${2}.log 2>&1 < /dev/null &"; }
wait_health() { for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${1}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && return 0; sleep 5; done; return 1; }
kill_all_serves() { for p in 8470 8471 8472 8473 8474 8475 8476 8477; do
    docker exec lerobot bash -lc "pkill -9 -f 'lerobot.py.*${p}' || true" 2>/dev/null || true; done; sleep 3; }
collect_cell() {  # row portA portB
  IFS='|' read -r cell task env ci seed instr <<<"$1"
  local pA=$2 pB=$3
  cw() { local wid=$1 port=$2
    for ep in $(seq 0 59); do
      [ $((ep % 2)) -eq "$wid" ] || continue
      ls "$SRC6/${task}/${cell}/task${ci}--ep${ep}--succ"*.pkl >/dev/null 2>&1 && continue
      docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
        python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        --vla-server "http://127.0.0.1:${port}" --task "$task" --env-name "$env" \
        --output-dir "/temporal_vla/${SRC6}" --cell-id "$cell" --cell-index "$ci" \
        --canonical-instruction "$instr" --episode-start-idx "$ep" --n-episodes 1 \
        --seed "$seed" --inference-seed "$((ep * 1000))" --n-action-steps 5 \
        --max-episode-steps 720 --video-fps 20 --steps-per-render 2 --wait-ready \
        --proximity-phases 2>&1 | grep -E "^wrote|Traceback" || true
    done; }
  cw 0 "$pA" & cw 1 "$pB" & wait
  echo "[collect] $cell: $(ls $SRC6/${task}/${cell}/*.pkl 2>/dev/null | wc -l)판"
}
fit_dir() {  # rundir cellpath outdir
  [ -f "$3/fit_summary.json" ] && return 0
  mkdir -p "$3"
  OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 $PY $FITPY --run-dir "$1" --cell "$2" \
    --groups "$FIT_GROUPS" --carve-window 0 --out-dir "$3" > "$3.fitlog" 2>&1 || true
  grep -m1 '\[done\]' "$3.fitlog" >/dev/null || { echo "[FIT FAILED] $3"; tail -5 "$3.fitlog"; exit 2; }
}
ps_fits() {  # cell들에 per-seed 15/30/60 fit
  for cell in "$@"; do
    IFS='|' read -r c task env ci seed instr <<<"$(row_of $cell)"
    for N in 15 30 60; do
      SUBD=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_ps${N}/$task/$c
      mkdir -p "$SUBD"
      for ep in $(seq 0 $((N - 1))); do
        src=$(ls $(pwd)/$SRC6/$task/$c/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
        [ -n "$src" ] && ln -sf "$src" "$SUBD/"
      done
      fit_dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_ps${N} "$task/$c" "$AN/final_ps${N}/$c"
    done
  done
}
pool_fit() {  # name cells...
  local name=$1; shift; local cells=("$@")
  for N in 15 30 60; do
    D=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_${name}${N}/Pool/all
    mkdir -p "$D"
    local i=0
    while [ $i -lt $N ]; do
      local cl=${cells[$((i % ${#cells[@]}))]}
      local ep=$((i / ${#cells[@]}))
      IFS='|' read -r c task env ci seed instr <<<"$(row_of $cl)"
      src=$(ls $(pwd)/$SRC6/$task/$c/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
      [ -n "$src" ] && ln -sf "$src" "$D/${c}--ep${ep}.pkl"
      i=$((i + 1))
    done
    fit_dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_${name}${N} "Pool/all" "$AN/final_${name}${N}/all"
    for c2 in $BREADS $APPLES; do ln -sfn all "$AN/final_${name}${N}/$c2"; done
  done
}
chain() {  # cell gpu pA pB armsets...  (armset = "SUF|ARMS|NPZSUB|LAYERS")
  local cell=$1 gpu=$2 pA=$3 pB=$4; shift 4
  IFS='|' read -r c task env ci seed instr <<<"$(row_of $cell)"
  mkdir -p outputs/eval/robocasa/groot_n15/steer_eval/${c}
  for spec in "$@"; do
    IFS='|' read -r suf arms npzsub lyr <<<"$spec"
    CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
      GPUS_L="$gpu $gpu" PORTS_L="$pA $pB" EP0=60 EP1=119 PROX=1 \
      SUF=$suf ARMS="$arms" NPZ_ROOT=${CROOT}/$npzsub STEER_LAYERS="${lyr:-4,8,12}" \
      bash "$R" >> outputs/eval/robocasa/groot_n15/steer_eval/${c}/final_chain.log 2>&1
  done
  echo "[chain $c] DONE $(date '+%T')"
}

# ============ PHASE B: BREAD ============
if [ ! -f "$SRC6/.S1B_DONE" ]; then
  echo "[B-S1] bread 신규 3 cell 수집 $(date '+%T')"
  start_serve 4 8470; start_serve 4 8471; start_serve 5 8474; start_serve 5 8475; start_serve 6 8472; start_serve 6 8473
  for p in 8470 8471 8474 8475 8472 8473; do wait_health $p || { echo "[B-S1] serve $p TIMEOUT"; exit 11; }; done
  collect_cell "$(row_of ppcc_bread_s300028)" 8470 8471 &
  collect_cell "$(row_of ppcc_bread_s300033)" 8474 8475 &
  collect_cell "$(row_of ppcc_bread_s400020)" 8472 8473 &
  wait
  kill_all_serves
  touch "$SRC6/.S1B_DONE"
fi
echo "[B-S2] bread fits $(date '+%T')"
ps_fits $BREADS
pool_fit xb $BREADS
echo "[B-S3] bread 체인 $(date '+%T')"
BREAD_SETS=(
  "ps15|base perm gated|final_ps15|"
  "ps30|perm gated|final_ps30|"
  "ps60|perm gated|final_ps60|"
  "xb15|perm gated|final_xb15|"
  "xb30|perm gated|final_xb30|"
  "xb60|perm gated|final_xb60|"
  "L4|gated|final_ps60|4"
  "L812|gated|final_ps60|8,12"
)
( chain ppcc_bread 4 8470 8471 "${BREAD_SETS[@]}" ) &
( chain ppcc_bread_s400020 7 8476 8477 "${BREAD_SETS[@]}" ) &
( chain ppcc_bread_s300028 5 8474 8475 "${BREAD_SETS[@]}" ) &
( chain ppcc_bread_s300033 6 8472 8473 "${BREAD_SETS[@]}" ) &
wait
echo "[B] bread 완료 → 중간 집계 $(date '+%F %T')"
set -a; . ./.env; set +a
BREAD_ONLY=1 $PY scripts/safe/groot_n15/robocasa/steer/aggregate_final_scene.py || echo "[B] 집계 스크립트 실패(계속)"
touch outputs/eval/robocasa/groot_n15/FINAL2_BREAD_DONE

# ============ PHASE A: APPLE + GRAND ============
if [ ! -f "$SRC6/.S1A_DONE" ]; then
  echo "[A-S1] apple 신규 3 cell 수집 $(date '+%T')"
  start_serve 4 8470; start_serve 4 8471; start_serve 5 8474; start_serve 5 8475; start_serve 6 8472; start_serve 6 8473
  for p in 8470 8471 8474 8475 8472 8473; do wait_health $p || { echo "[A-S1] serve $p TIMEOUT"; exit 11; }; done
  collect_cell "$(row_of ppcs_apple_s100050)" 8470 8471 &
  collect_cell "$(row_of ppcs_apple_s100084)" 8474 8475 &
  collect_cell "$(row_of ppcs_apple_s100104)" 8472 8473 &
  wait
  kill_all_serves
  touch "$SRC6/.S1A_DONE"
fi
echo "[A-S2] apple/grand fits $(date '+%T')"
ps_fits $APPLES
pool_fit xa $APPLES
pool_fit gx $BREADS $APPLES
echo "[A-S3] apple 체인 + bread gx $(date '+%T')"
APPLE_SETS=(
  "ps15|base perm gated|final_ps15|"
  "ps30|perm gated|final_ps30|"
  "ps60|perm gated|final_ps60|"
  "xa15|perm gated|final_xa15|"
  "xa30|perm gated|final_xa30|"
  "xa60|perm gated|final_xa60|"
  "gx15|perm gated|final_gx15|"
  "gx30|perm gated|final_gx30|"
  "gx60|perm gated|final_gx60|"
)
GX_SETS=( "gx15|perm gated|final_gx15|" "gx30|perm gated|final_gx30|" "gx60|perm gated|final_gx60|" )
( chain ppcs_apple 4 8470 8471 "${APPLE_SETS[@]}"; chain ppcc_bread 4 8470 8471 "${GX_SETS[@]}" ) &
( chain ppcs_apple_s100050 7 8476 8477 "${APPLE_SETS[@]}"; chain ppcc_bread_s400020 7 8476 8477 "${GX_SETS[@]}" ) &
( chain ppcs_apple_s100084 5 8474 8475 "${APPLE_SETS[@]}"; chain ppcc_bread_s300028 5 8474 8475 "${GX_SETS[@]}" ) &
( chain ppcs_apple_s100104 6 8472 8473 "${APPLE_SETS[@]}"; chain ppcc_bread_s300033 6 8472 8473 "${GX_SETS[@]}" ) &
wait
echo "[A] 완료 → 최종 집계 $(date '+%F %T')"
$PY scripts/safe/groot_n15/robocasa/steer/aggregate_final_scene.py || echo "[A] 집계 실패"
echo "[A] rsync"
rsync -az -e "ssh -o BatchMode=yes -p 11112" \
  outputs/eval/robocasa/groot_n15/steer_eval outputs/eval/robocasa/groot_n15/phase_event_6p \
  kimseungjun@166.104.146.37:workspace/temporal_vla/outputs/eval/robocasa/groot_n15/ && echo "[A] rsync ok"
touch outputs/eval/robocasa/groot_n15/FINAL2_ALL_DONE
echo "=========== FINAL2 COMPLETE $(date '+%F %T') ==========="
