#!/bin/bash
# 로컬 lane runner: 큐에서 작업을 꺼내 heldout_round_cell.sh(arm) / 수집 루프(collect) 실행.
# usage: lane_runner_local.sh <lane_id> <gpu> <portA> <portB>
set -u
source "$(dirname "${BASH_SOURCE[0]}")/queue_lib.sh"
LANE=$1 GPU=$2 PA=$3 PB=$4
HELDOUT=scripts/safe/groot_n15/robocasa/steer/heldout_round_cell.sh
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
CROOT=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
cd "$REPO"
echo "[$LANE] start gpu=$GPU ports=$PA,$PB $(date -u '+%FT%T')"

serve_up() { docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
     --host '*' --port ${1} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
     > /tmp/queue_${1}.log 2>&1 < /dev/null &"; }
serve_health() { local i; for i in $(seq 1 120); do
    docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${1}/health 2>/dev/null" | grep -q '"status":"ok"' && return 0
    sleep 5; done; return 1; }
serve_down() { local p; for p in "$@"; do docker exec lerobot bash -lc "pkill -9 -f 'lerobot.py.*${p}' || true" 2>/dev/null || true; done; sleep 3; }

run_collect() { # cell task env ci seed instr → rc
  local cell=$1 task=$2 env=$3 ci=$4 seed=$5 instr=$6
  serve_up "$PA"; serve_up "$PB"
  serve_health "$PA" || { serve_down "$PA" "$PB"; return 11; }
  serve_health "$PB" || { serve_down "$PA" "$PB"; return 11; }
  cw() { local wid=$1 port=$2 ep
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
  cw 0 "$PA" & cw 1 "$PB" & wait
  serve_down "$PA" "$PB"
  return 0
}

while true; do
  line=$(pop_job local "$LANE") || {
    if [ ! -s "$QROOT/queue.tsv" ] && [ -z "$(ls "$QROOT"/running 2>/dev/null)" ]; then
      echo "[$LANE] queue 소진 — 종료 $(date -u '+%FT%T')"; break
    fi
    sleep 180; continue; }
  IFS='|' read -r jt cell suf arms npzsub layers cons retry <<<"$line"
  IFS='|' read -r c task env ci seed instr <<<"$(row_of "$cell")"
  echo "[$LANE] JOB $jt $cell $arms$suf (retry=$retry) $(date -u '+%FT%T')"

  if [ "$jt" = collect ]; then
    run_collect "$c" "$task" "$env" "$ci" "$seed" "$instr"
    cnt=$(fit_count "$c")
    mark_machine "$REPO/$SRC6/$task/$c" "local-gpu$GPU" "$cnt"
    if [ "$cnt" -ge 60 ]; then finish_job "$line" "$LANE" "local-gpu$GPU" "$cnt" "$cnt"
    else echo "[$LANE] collect 부족($cnt/60) → 재큐"; requeue_job "$line" "$LANE" local-only; fi
    continue
  fi

  tag=$(tag_of "$arms" "$suf")
  before=$(pkl_count "$cell" "$tag")
  CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
    GPUS_L="$GPU $GPU" PORTS_L="$PA $PB" EP0=60 EP1=119 PROX=1 SUF="$suf" ARMS="$arms" \
    env NPZ_ROOT="$CROOT/${npzsub:-final_ps60}" ${layers:+STEER_LAYERS="$layers"} \
    bash "$HELDOUT" >> "$SE/$c/final_chain.log" 2>&1
  rc=$?
  after=$(pkl_count "$cell" "$tag")
  mark_machine "$REPO/$SE/$c/$tag" "local-gpu$GPU" "$((after - before))"
  if [ "$after" -ge 60 ]; then
    finish_job "$line" "$LANE" "local-gpu$GPU" "$((after - before))" "$after"
  else
    echo "[$LANE] $tag ${after}/60 (rc=$rc) → 재큐"
    requeue_job "$line" "$LANE" local-only
    sleep 60
  fi
done
