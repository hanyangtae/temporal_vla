#!/usr/bin/env bash
# exp4-1 oracle rescue 러너 — t0_manifest 행 단위로 arm rollout 수행 (24a §7).
# heldout_round_cell{,_host}.sh 의 serve/health/worker 골격을 t0-manifest iterate 로 개조.
#
# 실행 위치 = episode 를 수집한 머신 (결정론 머신-로컬):
#   eval-풀 bread/drawer → srv48, beer → srv50 (SERVE_MODE=host, GPU 1개 × serve 6)
#   fit-풀 전부 → kanu (SERVE_MODE=docker, 빈 GPU × serve 2)
#
# env:
#   CELL_ID ARM(A0|A|setM|setM_pl) T0_MANIFEST NPZ_ROOT OUT_ROOT
#   SERVE_MODE=host|docker  GPUS_L="2 2 2 2 2 2"  PORTS_L="8480 ... 8485"
#   POOL=eval|fit|all(기본 eval)  ROW_FILTER(옵션: episode_idx 콤마목록 — sentinel/스모크용)
#   BETA_A=0.1  BETA_SETM=1.0  STEER_EXTRA(옵션)
# arm 별 NPZ base: $NPZ_ROOT/$CELL_ID/{A,setM,setM_pl}/steer/dit_L*/conceptors.npz
# fit-풀 setM/setM_pl 은 per-target LOO base($NPZ_ROOT/$CELL_ID/setM_loo/ep$E)로 ep당 serve 재기동.
set -uo pipefail
: "${CELL_ID:?}" "${ARM:?}" "${T0_MANIFEST:?}" "${OUT_ROOT:?}"
SERVE_MODE="${SERVE_MODE:-host}"
POOL="${POOL:-eval}"
NPZ_ROOT="${NPZ_ROOT:-}"
BETA_A="${BETA_A:-0.1}"; BETA_SETM="${BETA_SETM:-1.0}"
STRIDE_NAS=5; MAXEP=720
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../../../.." && pwd)"
# 상대경로 입력 정규화 — OUT_CONT 의 /temporal_vla 매핑이 절대경로 전제 (srv48 사고 교훈)
case "$OUT_ROOT" in /*) ;; *) OUT_ROOT="$REPO_ROOT/$OUT_ROOT" ;; esac
case "$T0_MANIFEST" in /*) ;; *) T0_MANIFEST="$REPO_ROOT/$T0_MANIFEST" ;; esac
[ -z "$NPZ_ROOT" ] || case "$NPZ_ROOT" in /*) ;; *) NPZ_ROOT="$REPO_ROOT/$NPZ_ROOT" ;; esac
source "$HERE/cells.env"; cell_params "$CELL_ID"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}
OUT_HOST="${OUT_ROOT}/${CELL_ID}/${ARM}"
OUT_CONT="/temporal_vla${OUT_HOST#"$REPO_ROOT"}"
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"

# ---- t0 manifest 행 로드 (cell·pool 필터, steering arm 은 t0_record=NA 스킵) ----------
# 열: cell pool episode_idx scenario_seed inference_seed ... t0_env_step t0_record note
mapfile -t ROWS < <(awk -F'\t' -v cell="$CELL_ID" -v pool="$POOL" -v arm="$ARM" '
  NR==1 { for (i=1;i<=NF;i++) h[$i]=i; next }
  $h["cell"]!=cell { next }
  pool!="all" && $h["pool"]!=pool { next }
  arm!="A0" && ($h["t0_record"]=="NA" || $h["t0_record"]=="") { next }
  { print $h["pool"] "\t" $h["episode_idx"] "\t" $h["scenario_seed"] "\t" \
        $h["inference_seed"] "\t" $h["t0_record"] }' "$T0_MANIFEST")
if [ -n "${ROW_FILTER:-}" ]; then
  mapfile -t ROWS < <(printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v f=",$ROW_FILTER," 'index(f, ","$2",")')
fi
[ "${#ROWS[@]}" -gt 0 ] || { echo "[$CELL_ID/$ARM] 대상 행 0개 (pool=$POOL)"; exit 0; }
echo "[$CELL_ID/$ARM] rows=${#ROWS[@]} pool=$POOL mode=$SERVE_MODE gpus=${GPUS[*]} ports=${PORTS[*]}"

# ---- serve ---------------------------------------------------------------------------
serve_steer_flags() {  # $1 = NPZ base 디렉토리 (steer/dit_L*/conceptors.npz 포함)
  local base="$1" op="$2" beta="$3" lyr
  lyr=$(basename "$(ls -d "$base"/steer/dit_L* 2>/dev/null | head -1)" | sed 's/dit_L//')
  [ -n "$lyr" ] || { echo "[$CELL_ID/$ARM] NPZ 없음: $base/steer/dit_L*" >&2; return 1; }
  echo "--steering-phase-npz-base $base --steering-phases steer --steering-layers $lyr --steering-op $op --steering-beta $beta"
}
start_serves() {  # $@ = 추가 serve 플래그
  local i
  for i in $(seq 0 $((NW-1))); do
    if [ "$SERVE_MODE" = host ]; then
      ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
          PYTHONPATH="$REPO_ROOT/lerobot/src" "$HOME/miniconda3/envs/lerobot_050_groot/bin/python" \
          scripts/serve/lerobot.py --profile $PROFILE --host '*' --port "${PORTS[$i]}" \
          --device cuda "$@" > "/tmp/exp41_${CELL_ID}_${ARM}_${PORTS[$i]}.log" 2>&1 < /dev/null & )
    else
      docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -c \
        "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile $PROFILE \
         --host '*' --port ${PORTS[$i]} --device cuda $* \
         > /tmp/exp41_${CELL_ID}_${ARM}_${PORTS[$i]}.log 2>&1"
    fi
  done
  local port ok log
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
      sleep 5
    done
    log="/tmp/exp41_${CELL_ID}_${ARM}_${port}.log"
    [ "$SERVE_MODE" = host ] || log="docker:$log"
    echo "[$CELL_ID/$ARM] serve $port health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || { echo "[$CELL_ID/$ARM] ABORT serve $port ($log)"; exit 11; }
  done
}
kill_serves() {
  local port pid
  for port in "${PORTS[@]}"; do
    if [ "$SERVE_MODE" = host ]; then
      for pid in $(pgrep -f "lerobot.py.*--port ${port}"); do kill "$pid" 2>/dev/null || true; done
    else
      docker exec lerobot bash -c "pkill -f 'lerobot.py.*--port ${port}'" 2>/dev/null || true
    fi
  done
  sleep 5
}

# ---- collector -----------------------------------------------------------------------
run_row() {  # port pool ep scen inf k
  local port=$1 pool=$2 ep=$3 scen=$4 inf=$5 k=$6 latch=""
  if ls "${OUT_HOST}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.json >/dev/null 2>&1; then
    return 0  # idempotent resume
  fi
  [ "$ARM" != "A0" ] && latch="--steer-from-record $k"
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
    --output-dir "${OUT_CONT}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
    --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
    --seed "$scen" --inference-seed "$inf" --n-action-steps "$STRIDE_NAS" \
    --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
    --no-features --proximity-phases $latch ${STEER_EXTRA:-} 2>&1 \
    | grep -E "wrote|Error|Traceback" || true
}

if [ "$POOL" = "all" ] && { [ "$ARM" = "setM" ] || [ "$ARM" = "setM_pl" ]; }; then
  # setM 계열은 fit-풀이 LOO NPZ 라 풀 혼합 실행 금지 — eval/fit 별도 호출
  echo "[$CELL_ID/$ARM] POOL=all 금지 (LOO 분리) — POOL=eval 과 POOL=fit 으로 나눠 실행"; exit 2
fi
if [ "$ARM" = "A0" ] || [ "$ARM" = "A" ] || [ "$POOL" = "eval" ]; then
  # 공유 NPZ 1개 — serve 일괄 기동 후 worker striping
  FLAGS=""
  case "$ARM" in
    A0) FLAGS="" ;;
    A) FLAGS="$(serve_steer_flags "$NPZ_ROOT/$CELL_ID/A" conceptor "$BETA_A") --steering-denoise ${DENOISE_A:-per_step}" || exit 12 ;;
    setM) FLAGS=$(serve_steer_flags "$NPZ_ROOT/$CELL_ID/setM" setpoint "$BETA_SETM") || exit 12 ;;
    setM_pl) FLAGS=$(serve_steer_flags "$NPZ_ROOT/$CELL_ID/setM_pl" setpoint "$BETA_SETM") || exit 12 ;;
    *) echo "unknown ARM: $ARM"; exit 2 ;;
  esac
  start_serves $FLAGS
  run_worker() {
    local wid=$1 i
    for i in "${!ROWS[@]}"; do
      [ $((i % NW)) -eq "$wid" ] || continue
      IFS=$'\t' read -r pool ep scen inf k <<< "${ROWS[$i]}"
      run_row "${PORTS[$wid]}" "$pool" "$ep" "$scen" "$inf" "$k"
    done
  }
  for wid in $(seq 0 $((NW-1))); do run_worker "$wid" > "${LOGDIR}/w${wid}.log" 2>&1 & done
  wait
  kill_serves
else
  # fit-풀 setM/setM_pl — per-target LOO NPZ 로 ep 당 serve 재기동 (worker 0 단독)
  LOO_DIR=$([ "$ARM" = setM ] && echo setM_loo || echo setM_pl_loo)
  for i in "${!ROWS[@]}"; do
    IFS=$'\t' read -r pool ep scen inf k <<< "${ROWS[$i]}"
    FLAGS=$(serve_steer_flags "$NPZ_ROOT/$CELL_ID/$LOO_DIR/ep$ep" setpoint "$BETA_SETM") || exit 12
    start_serves $FLAGS
    run_row "${PORTS[0]}" "$pool" "$ep" "$scen" "$inf" "$k" >> "${LOGDIR}/loo.log" 2>&1
    kill_serves
  done
fi

# ---- 결과 요약 ------------------------------------------------------------------------
d="${OUT_HOST}/raw_rollouts/${TASK}/${CELL_ID}"
s=$(ls "$d"/*succ1.json 2>/dev/null | wc -l); f=$(ls "$d"/*succ0.json 2>/dev/null | wc -l)
echo -e "[$CELL_ID/$ARM] done\trescued=${s}\tfail=${f}\ttotal=$((s+f))/${#ROWS[@]}"
touch "${LOGDIR}/EXP41_${CELL_ID}_${ARM}_${POOL}_DONE"
