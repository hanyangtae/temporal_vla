#!/usr/bin/env bash
# exp4-2/exp5-2 P0 Track P 러너 — 3 phase:
#   PHASE=baseline : cell baseline 재수집 (--no-features, ep 0..N_BASE-1) → 앵커 원천
#   PHASE=grid     : build_perturb_grid.py 산출 grid.tsv 실행 (--no-features, 실패율 측정)
#   PHASE=capture  : CONFIGS="c1_s100 g1_x010 ..." 채택 config 만 캡처 ON 재실행 (fit용)
# 사용 (worktree 에서, 빈 GPU 확인 후):
#   PHASE=baseline GPUS_L="0 0" PORTS_L="8480 8481" bash collect_perturb_grid.sh          # ppcc(기본)
#   PHASE=grid     GPUS_L="0 0" PORTS_L="8480 8481" bash collect_perturb_grid.sh
#   PHASE=capture  CONFIGS="..." GPUS_L="0 0" PORTS_L="8480 8481" bash collect_perturb_grid.sh
#   CELL=drawer_left PHASE=baseline CAPTURE=1 GPUS_L="0 0" PORTS_L="8480 8481" bash ...   # exp5-2
#   CELL=mixer MIXER_SEED=<feasibility 통과 seed> MIXER_CELL_INDEX=<n> PHASE=baseline ...
# detached 권장: setsid nohup bash ... > <P0>/logs/orch_$PHASE.log 2>&1 < /dev/null &
set -uo pipefail

PHASE="${PHASE:?baseline|grid|capture}"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
# ── cell 정의 (docs/steering/29 §0 표) — 기본값 = exp4-2 ppcc_bread (회귀 방지) ────────
# build_perturb_grid.py 의 CELL_DEFAULTS 와 같은 cell 이름을 쓴다 (--cell 인자).
CELL="${CELL:-ppcc_bread}"
case "$CELL" in
  ppcc_bread)
    TASK=PickPlaceCounterToCabinet
    ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
    CELL_INDEX=5; SEED=100084
    INSTR="Pick the bread from the counter and place it in the cabinet."
    ;;
  drawer_left)
    # ⚠️ OpenDrawer 는 seed 에 따라 좌/우 variant 가 바뀐다 (robot 배치 rng — kitchen_drawer.py
    #    _place_robot → drawer_side, ep_meta lang "Open the {side} drawer."). seed 100001 이
    #    정말 left 인지는 **GPU/컨테이너 세션에서 reset 1회로 확인** 필요:
    #      analyze/mixer_scene_feasibility.py --task OpenDrawer --fixture drawer --seeds 100001
    #      (결과 JSON 의 ep_lang 필드) — 정책 없이 판정 가능.
    #    불일치 시 collector 의 canonical-instruction 검증이 즉시 ABORT 시킨다(무해한 실패).
    TASK=OpenDrawer
    ENVN="robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
    CELL_INDEX="${DRAWER_CELL_INDEX:-8}"; SEED="${DRAWER_SEED:-100001}"
    INSTR="Open the left drawer."
    ;;
  mixer)
    # seed/cell_index 는 feasibility 스캔(analyze/mixer_scene_feasibility.py)으로 확정 —
    # 기하 불가 seed 실존(100010 BLOCKED, memory scene-feasibility-filter). instruction 은
    # 1종 고정 (robocasa kitchen_stand_mixer.py OpenStandMixerHead.get_ep_meta).
    TASK=OpenStandMixerHead
    ENVN="robocasa_panda_omron/OpenStandMixerHead_PandaOmron_Env"
    CELL_INDEX="${MIXER_CELL_INDEX:?mixer cell_index 필요 (docs/steering/27)}"
    SEED="${MIXER_SEED:?mixer seed 필요 — feasibility 스캔 통과 seed}"
    INSTR="Open the stand mixer head."
    ;;
  *) echo "ABORT: unknown CELL=$CELL (ppcc_bread|drawer_left|mixer)"; exit 10 ;;
esac
CELL_ID="$CELL"
NAS=5; MAXEP="${MAXEP:-720}"; STRIDE=1000
N_BASE="${N_BASE:-18}"
CAP="0,2,4,8,10,12,15"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WT_HOST="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
MAIN_HOST="$(cd -- "${WT_HOST}/../../.." && pwd)"
# 컨테이너 경로는 호스트 worktree 경로에서 유도 (worktree 이름 하드코딩 금지 — exp5-2 는
# .claude/worktrees/exp5-2-perturb-steering).
WT_CONT="${WT_HOST/#${MAIN_HOST}//temporal_vla}"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
# 출력 루트: ppcc 는 exp4-2 기존 경로 유지, 신규 cell 은 exp52_induced/<cell>. P0_DIR 로 override.
if [ "$CELL" = "ppcc_bread" ]; then
  P0="${P0_DIR:-${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp42_induced/p0}"
else
  P0="${P0_DIR:-${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp52_induced/${CELL}}"
fi
LOGDIR="${P0}/logs"; mkdir -p "$LOGDIR"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}

to_cont() { echo "${1/#${MAIN_HOST}//temporal_vla}"; }

serve_extra=""
collect_extra="--no-features --expect-chunk-len 16"
if [ "$PHASE" = "capture" ]; then
  # fit 호환 캡처 규약 (24b §2.1): action_token_mean + vlln_mean (post_vl_sa_full 은 donor 전용)
  serve_extra="--collect --capture-vl --groot-dit-capture-layers ${CAP}"
  collect_extra=""
fi
# PHASE=baseline + CAPTURE=1: clean baseline 을 capture ON 으로 (섭동 없는 activation 확보 —
# perturbed 짝의 state-matched clean, 같은 머신 결정론). 출력은 baseline_cap 로 분리.
if [ "$PHASE" = "baseline" ] && [ "${CAPTURE:-0}" = "1" ]; then
  serve_extra="--collect --capture-vl --groot-dit-capture-layers ${CAP}"
  collect_extra=""
  BASELINE_OUT="${P0}/baseline_cap"
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
    --video-fps 20 --steps-per-render 2 --wait-ready --proximity-phases "$@" 2>&1 \
    | grep -E "^wrote|Error|Traceback|ABORT" || true
}
# --proximity-phases: passB(자연실패, bridge 대조측)와 동일한 6p 라벨 체계 유지

done_mark() { ls "${1}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${2}--succ"* >/dev/null 2>&1; }

worker_baseline() {  # wid port
  local wid=$1 port=$2
  for idx in $(seq 0 $((N_BASE - 1))); do
    [ $((idx % NW)) -eq "$wid" ] || continue
    local out="${BASELINE_OUT:-${P0}/baseline}"
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
      # grid 톱업: CONFIGS 지정 시 해당 config 만 (미지정 = 전체)
      if [ -n "${CONFIGS:-}" ]; then
        case " ${CONFIGS} " in *" ${config} "*) ;; *) continue ;; esac
      fi
      local out="${P0}/grid/${config}"
      # sham 은 출력 분리 — 같은 ep 의 real 행과 스템 충돌(P0 실측: real ep10 1판 skip) 방지
      [[ "$tag" == *_sham_* ]] && out="${P0}/grid/${config}__sham"
    fi
    done_mark "$out" "$ep_idx" && continue
    echo "[w${wid}] ${PHASE} ${tag}"
    run_ep "$port" "$out" "$ep_idx" "$inf" --perturb-spec "$(to_cont "$spec")" \
      --run-tag "$tag" $collect_extra
  done < "${P0}/grid.tsv"
}

start_serves
echo "[p0-${PHASE}] $(date '+%F %T') start (CELL=${CELL} seed=${SEED} cell_index=${CELL_INDEX} NW=${NW})"
echo "[p0-${PHASE}] P0=${P0}  WT_CONT=${WT_CONT}"
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
DONE_TAG="${PHASE}$([ "${CAPTURE:-0}" = "1" ] && echo _cap)"
echo "[p0-${DONE_TAG}] $(date '+%F %T') DONE"
touch "${LOGDIR}/P0_${DONE_TAG}_DONE"
