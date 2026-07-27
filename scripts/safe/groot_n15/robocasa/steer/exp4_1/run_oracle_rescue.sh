#!/usr/bin/env bash
# exp4-1 oracle rescue 러너 — t0_manifest 행 단위로 arm rollout 수행 (24a §7).
# heldout_round_cell{,_host}.sh 의 serve/health/worker 골격을 t0-manifest iterate 로 개조.
#
# 실행 위치 = episode 를 수집한 머신 (결정론 머신-로컬):
#   eval-풀 bread/drawer → srv48, beer → srv50 (SERVE_MODE=host, GPU 1개 × serve 6)
#   fit-풀 전부 → kanu (SERVE_MODE=docker, 빈 GPU × serve 2)
#
# env:
#   CELL_ID ARM T0_MANIFEST NPZ_ROOT OUT_ROOT
#   arm 순서(2026-07-23 사용자 확정): A0 · noise_resample ·
#     setM_permanent · setM_permanent_placebo · setM_gated · setM_gated_placebo ·
#     setM_future_only · setM_future_only_placebo (setM 계열 마지막) ·
#     conceptor_permanent · conceptor_gated (맨 뒤)
#   (gated arm = client 가 t0 이후 현재 phase 를 POST — phase 별 NPZ 디렉토리 필요.
#    noise_resample = steering 없이 t0 이후 denoise seed offset — 방향 없는 개입 대조)
#   SERVE_MODE=host|docker  GPUS_L="2 2 2 2 2 2"  PORTS_L="8480 ... 8485"
#   POOL=eval|fit|all(기본 eval)  ROW_FILTER(옵션: episode_idx 콤마목록 — sentinel/스모크용)
#   BETA_CONCEPTOR=0.1  BETA_SETM=1.0  STEER_EXTRA(옵션)
# arm 별 NPZ base: $NPZ_ROOT/$CELL_ID/{conceptor,setM,setM_pl}/steer/dit_L*/conceptors.npz
# fit-풀 setM/setM_pl 은 per-target LOO base($NPZ_ROOT/$CELL_ID/setM_loo/ep$E)로 ep당 serve 재기동.
set -uo pipefail
: "${CELL_ID:?}" "${ARM:?}" "${T0_MANIFEST:?}" "${OUT_ROOT:?}"
SERVE_MODE="${SERVE_MODE:-host}"
POOL="${POOL:-eval}"
NPZ_ROOT="${NPZ_ROOT:-}"
BETA_CONCEPTOR="${BETA_CONCEPTOR:-0.1}"; BETA_SETM="${BETA_SETM:-1.0}"
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
serve_steer_flags() {  # $1 = NPZ base (하위 = phase 디렉토리들: permanent='steer' 1개, gated=phase명들)
  local base="$1" op="$2" beta="$3" lyr serve_base phases ph
  phases=""
  for ph in "$base"/*/; do
    [ -d "$ph" ] && ls "$ph"dit_L* >/dev/null 2>&1 && phases="${phases:+$phases,}$(basename "$ph")"
  done
  [ -n "$phases" ] || { echo "[$CELL_ID/$ARM] NPZ phase 디렉토리 없음: $base" >&2; return 1; }
  lyr=$(basename "$(ls -d "$base/${phases%%,*}"/dit_L* 2>/dev/null | head -1)" | sed 's/dit_L//')
  # docker 모드 serve 는 컨테이너 안 — repo mount(/temporal_vla) 경로로 번역 (Gate2 P1-1)
  serve_base="$base"
  [ "$SERVE_MODE" = host ] || serve_base="/temporal_vla${base#"$REPO_ROOT"}"
  echo "--steering-phase-npz-base $serve_base --steering-phases $phases --steering-layers $lyr --steering-op $op --steering-beta $beta"
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
  local port=$1 pool=$2 ep=$3 scen=$4 inf=$5 k=$6 latch="" rc
  # pool 별 하위 디렉토리 — fit/eval 의 episode_idx 충돌로 resume 이 타 pool 행을
  # 건너뛰는 ITT 무음 탈락 방지 (Gate2 P1-2)
  if ls "${OUT_HOST}/${pool}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.json >/dev/null 2>&1; then
    return 0  # idempotent resume
  fi
  case "$ARM" in
    A0) latch="" ;;
    noise_resample) latch="--reseed-from-record $k" ;;
    # 모든 gated 계열은 phase 조건부 — *_gated_future_only·*_gated_future_only_placebo 포함.
    # (구 `*_gated|*_gated_placebo` 는 _future_only 접미사를 놓쳐 permanent latch 로 빠지고
    #  want="steer" → 미등록 phase → serve identity 무음화 되는 버그였음, 2026-07-24 수정.)
    *_gated|*_gated_*) latch="--steer-from-record $k --steer-phase-mode current" ;;
    *) latch="--steer-from-record $k"
       [ "$pool" = fit ] && latch="$latch --steer-phase-name ep${ep}" ;;
  esac
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
    --output-dir "${OUT_CONT}/${pool}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
    --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
    --seed "$scen" --inference-seed "$inf" --n-action-steps "$STRIDE_NAS" \
    --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
    --no-features --proximity-phases $latch ${STEER_EXTRA:-} 2>&1 \
    | grep -E "wrote|Error|Traceback"
  rc=${PIPESTATUS[0]}  # grep 무매치는 무시, collector 실패는 전파 (Gate2 P1-3)
  [ "$rc" -eq 0 ] || echo "[collector FAIL rc=$rc] pool=$pool ep=$ep" >&2
  return "$rc"
}

# fit-풀 arm 정책 (2026-07-23): LOO 연산자가 있는 setM_permanent/future_only 계열 + 방향 없는
# noise_resample 만. gated·conceptor 는 LOO 미산출 → **eval-풀 전용**(in-sample 평가 금지).
# A0 는 fit30 수집 자체가 무개입 실패분이라 재실행 불요 (sentinel 48/48 로 결정론 확인 완료).
if [ "$POOL" = "fit" ]; then
  case "$ARM" in
    setM_permanent|setM_permanent_placebo|setM_future_only|setM_future_only_placebo|noise_resample) ;;
    *) echo "[$CELL_ID/$ARM] fit-풀 미허용 arm (LOO 없음 → in-sample). eval-풀 전용"; exit 2 ;;
  esac
fi
if [ "$POOL" = "all" ]; then echo "[$CELL_ID/$ARM] POOL=all 금지 (풀별 오염 구조 상이)"; exit 2; fi
FLAGS=""
case "$ARM" in
  A0|noise_resample) FLAGS="" ;;
  conceptor_permanent|conceptor_gated)
    # exp3 배포와 동일 정렬: pooled fit → 전 토큰 적용 (token-select all)
    FLAGS="$(serve_steer_flags "$NPZ_ROOT/$CELL_ID/$ARM" conceptor "$BETA_CONCEPTOR") --steering-denoise ${DENOISE_CONCEPTOR:-per_step} --steering-token-select all" || exit 12 ;;
  conceptor_permanent_future_only|conceptor_gated_future_only)
    # future_only = 같은 conceptor M(base NPZ 재사용, 재fit 불요) 을 future 세그먼트
    # 토큰([1:T-horizon]) 에만 적용 (token-select future). setM future_only 와 정렬.
    _cb="${ARM%_future_only}"
    FLAGS="$(serve_steer_flags "$NPZ_ROOT/$CELL_ID/$_cb" conceptor "$BETA_CONCEPTOR") --steering-denoise ${DENOISE_CONCEPTOR:-per_step} --steering-token-select future" || exit 12 ;;
  setM_*)
    _NB="$NPZ_ROOT/$CELL_ID/$ARM"; [ "$POOL" = fit ] && _NB="${_NB}_loo"
    FLAGS="$(serve_steer_flags "$_NB" setpoint_seg "$BETA_SETM") --steering-token-select all" || exit 12 ;;
  *) echo "unknown ARM: $ARM"; exit 2 ;;
esac
start_serves $FLAGS
run_worker() {
  local wid=$1 i nfail=0
  for i in "${!ROWS[@]}"; do
    [ $((i % NW)) -eq "$wid" ] || continue
    IFS=$'\t' read -r pool ep scen inf k <<< "${ROWS[$i]}"
    run_row "${PORTS[$wid]}" "$pool" "$ep" "$scen" "$inf" "$k" || nfail=$((nfail+1))
  done
  echo "$nfail" > "${LOGDIR}/w${wid}.nfail"
}
for wid in $(seq 0 $((NW-1))); do run_worker "$wid" > "${LOGDIR}/w${wid}.log" 2>&1 & done
wait
kill_serves

# ---- 결과 요약 + 완주 게이트 — 선택된 ROWS 의 행별 산출 존재로 판정 (Gate2 확인라운드
# P1: 디렉토리 전체 카운트는 ROW_FILTER/부분 재실행에서 false INCOMPLETE·fake DONE 양쪽 가능)
s=0; f=0; miss=0
for i in "${!ROWS[@]}"; do
  IFS=$'\t' read -r pool ep scen inf k <<< "${ROWS[$i]}"
  j=$(ls "${OUT_HOST}/${pool}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.json 2>/dev/null | head -1)
  if [ -z "$j" ]; then miss=$((miss+1)); continue; fi
  case "$j" in *succ1.json) s=$((s+1)) ;; *) f=$((f+1)) ;; esac
done
echo -e "[$CELL_ID/$ARM] rescued=${s}\tfail=${f}\tmiss=${miss}\t/${#ROWS[@]}"
if [ "$miss" -ne 0 ]; then
  echo "[$CELL_ID/$ARM] INCOMPLETE — 결측 ${miss}행 (worker nfail: $(cat "${LOGDIR}"/w*.nfail 2>/dev/null | tr '\n' ' '))"
  exit 13
fi
touch "${LOGDIR}/EXP41_${CELL_ID}_${ARM}_${POOL}_DONE"
echo "[$CELL_ID/$ARM] done"
