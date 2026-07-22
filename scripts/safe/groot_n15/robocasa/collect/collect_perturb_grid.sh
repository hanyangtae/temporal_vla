#!/usr/bin/env bash
# exp4-2 P0 Track P 러너 — 3 phase:
#   PHASE=baseline : ppcc_bread baseline 재수집 (--no-features, ep 0..N_BASE-1) → 앵커 원천
#   PHASE=grid     : build_perturb_grid.py 산출 grid.tsv 실행 (--no-features, 실패율 측정)
#   PHASE=capture  : CONFIGS="c1_s100 g1_x010 ..." 채택 config 만 캡처 ON 재실행 (fit용)
# 사용 (worktree 에서, 빈 GPU 확인 후):
#   PHASE=baseline GPUS_L="0 0" PORTS_L="8480 8481" bash collect_perturb_grid.sh
#   PHASE=grid     GPUS_L="0 0" PORTS_L="8480 8481" bash collect_perturb_grid.sh
#   PHASE=capture  CONFIGS="..." GPUS_L="0 0" PORTS_L="8480 8481" bash collect_perturb_grid.sh
# detached 권장: setsid nohup bash ... > <P0>/logs/orch_$PHASE.log 2>&1 < /dev/null &
set -uo pipefail

PHASE="${PHASE:?baseline|grid|capture}"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
TASK=PickPlaceCounterToCabinet
ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
CELL_INDEX=5; CELL_ID=ppcc_bread
INSTR="Pick the bread from the counter and place it in the cabinet."
NAS=5; MAXEP=720; SEED=100084; STRIDE=1000
N_BASE="${N_BASE:-18}"
CAP="0,2,4,8,10,12,15"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WT_HOST="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
MAIN_HOST="$(cd -- "${WT_HOST}/../../.." && pwd)"
WT_CONT="/temporal_vla/.claude/worktrees/exp4-2-induced-failures"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
P0="${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp42_induced/p0"
LOGDIR="${P0}/logs"; mkdir -p "$LOGDIR"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}

to_cont() { echo "${1/#${MAIN_HOST}//temporal_vla}"; }

serve_extra=""
collect_extra="--no-features --expect-chunk-len 16"
if [ "$PHASE" = "capture" ]; then
  serve_extra="--collect --capture-vl --groot-vl-capture-point post_vl_sa_full --groot-dit-capture-layers ${CAP}"
  collect_extra=""
fi

start_serves() {
  for i in $(seq 0 $((NW - 1))); do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd ${WT_CONT} && setsid nohup python scripts/serve/exp42_serve.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda ${serve_extra} \
         > /tmp/exp42_p0_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0
    for _ in $(seq 1 150); do
      st=$(curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[p0-${PHASE}] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || { docker exec lerobot bash -lc "tail -30 /tmp/exp42_p0_${port}.log" || true; exit 11; }
  done
  # preflight: 컨테이너가 어느 트리의 robosuite/collector 를 쓰는지 기록 (worktree submodule 부재 함정)
  docker exec -e PYTHONPATH="$PYPATH" robocasa python -c \
    "import robosuite, robocasa; print('[preflight] robosuite=', robosuite.__file__); print('[preflight] robocasa=', robocasa.__file__)" \
    | tee "${LOGDIR}/preflight_${PHASE}.log"
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/exp42_serve.py.*--port ${port}' || true" 2>/dev/null || true; done; }
trap kill_serves EXIT

run_ep() {  # port out_host ep inf extra...
  local port=$1 out_host=$2 ep=$3 inf=$4; shift 4
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
    python "${WT_CONT}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py" \
    --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
    --output-dir "$(to_cont "$out_host")/raw_rollouts" --cell-id "$CELL_ID" \
    --cell-index "$CELL_INDEX" --canonical-instruction "$INSTR" \
    --episode-start-idx "$ep" --n-episodes 1 --seed "$SEED" --inference-seed "$inf" \
    --n-action-steps "$NAS" --max-episode-steps "$MAXEP" \
    --video-fps 20 --steps-per-render 2 --wait-ready "$@" 2>&1 \
    | grep -E "^wrote|Error|Traceback|ABORT" || true
}

done_mark() { ls "${1}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${2}--succ"* >/dev/null 2>&1; }

worker_baseline() {  # wid port
  local wid=$1 port=$2
  for idx in $(seq 0 $((N_BASE - 1))); do
    [ $((idx % NW)) -eq "$wid" ] || continue
    local out="${P0}/baseline"
    done_mark "$out" "$idx" && { echo "[w${wid}] skip ep${idx}"; continue; }
    echo "[w${wid}] baseline ep${idx}"
    run_ep "$port" "$out" "$idx" $((idx * STRIDE)) $collect_extra
  done
}

worker_grid() {  # wid port
  local wid=$1 port=$2 i=0
  while IFS=$'\t' read -r mode config ep_idx inf spec tag; do
    [ "$mode" = "mode" ] && continue
    i=$((i + 1)); [ $((i % NW)) -eq "$wid" ] || continue
    if [ "$PHASE" = "capture" ]; then
      case " ${CONFIGS:?채택 config 목록 필요} " in *" ${config} "*) ;; *) continue ;; esac
      [[ "$tag" == *_sham_* ]] && continue   # 캡처 재실행은 실섭동만
      local out="${P0}/capture/${config}"
    else
      local out="${P0}/grid/${config}"
    fi
    done_mark "$out" "$ep_idx" && continue
    echo "[w${wid}] ${PHASE} ${tag}"
    run_ep "$port" "$out" "$ep_idx" "$inf" --perturb-spec "$(to_cont "$spec")" \
      --run-tag "$tag" $collect_extra
  done < "${P0}/grid.tsv"
}

start_serves
echo "[p0-${PHASE}] $(date '+%F %T') start (NW=${NW})"
for wid in $(seq 0 $((NW - 1))); do
  if [ "$PHASE" = "baseline" ]; then
    worker_baseline "$wid" "${PORTS[$wid]}" > "${LOGDIR}/${PHASE}_w${wid}.log" 2>&1 &
  else
    [ -f "${P0}/grid.tsv" ] || { echo "ABORT: grid.tsv 없음 — build_perturb_grid.py 먼저"; exit 12; }
    worker_grid "$wid" "${PORTS[$wid]}" > "${LOGDIR}/${PHASE}_w${wid}.log" 2>&1 &
  fi
done
wait
kill_serves; trap - EXIT
echo "[p0-${PHASE}] $(date '+%F %T') DONE"
touch "${LOGDIR}/P0_${PHASE}_DONE"
