#!/usr/bin/env bash
# exp5-4 Phase A 러너 (srv50) — drawer_right, 신규 seed manifest 기반.
#
# 단계는 --stage 로 분리한다 (Codex Gate1 리뷰: A 끝나면 메인 세션이 선택 manifest 를
# 봉인한 뒤에만 B 를 돌린다 — 러너가 A→B 를 자동 연결하면 봉인이 성립하지 않는다).
#   probe    : 20 scene × 후보 8 → t=0 활성만 (rollout 없음). 끝나면 PROBE.DONE 후 정지.
#   rollout  : 160판 본 rollout (--no-features, 판정 json+mp4 만).
#   sanity   : capture ON 2판 (probe 활성 ↔ rollout record0 대조용).
#   smoke    : 결정성 smoke (smoke_probe.py ① ② ③ + 동일 rollout 2회 ④).
#
# 사용:
#   SRV_GPU="2 2 2 2 2 2" bash scripts/safe/groot_n15/robocasa/steer/exp5_4/probe_collect.sh --stage probe
#   ... --stage rollout / --stage sanity / --stage smoke
# env: SRV_GPU(필수, 포트당 GPU id 목록) PORTS HPY MANIFEST MACHINE_TAG CONTAINER
set -uo pipefail

STAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:?}"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
case "$STAGE" in probe|rollout|sanity|smoke) ;; *) echo "--stage {probe|rollout|sanity|smoke} 필요"; exit 2 ;; esac

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
TASK=OpenDrawer
ENVN="robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
CELL_ID=pq3_drawer_right; CELL_INDEX=7
INSTR="Open the right drawer."
NAS=5; MAXEP=720
CONTAINER="${CONTAINER:-robocasa}"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
HPY="${HPY:-$HOME/miniconda3/envs/lerobot_050_groot/bin/python}"
PORTS=(${PORTS:-8620 8621 8622 8623 8624 8625})
GPUS=(${SRV_GPU:?SRV_GPU 미지정 — 발사 시 포트 수만큼 GPU id 를 지정하라 (예: SRV_GPU=\"2 2 2 2 2 2\")})
NW=${#PORTS[@]}
[ "${#GPUS[@]}" = "$NW" ] || { echo "[ABORT] SRV_GPU 개수(${#GPUS[@]}) != PORTS 개수($NW)"; exit 2; }

REL_BASE="outputs/eval/robocasa/groot_n15/exp5_4"
OUT_BASE="${REPO_ROOT}/${REL_BASE}"
RAW_REL="${REL_BASE}/A_new_seed/raw_rollouts"
RAW_HOST="${REPO_ROOT}/${RAW_REL}"
SAN_REL="${REL_BASE}/sanity/raw_rollouts"
SMOKE_REL="${REL_BASE}/smoke/raw_rollouts"
MANIFEST="${MANIFEST:-${OUT_BASE}/seed_manifest.tsv}"
LOG="${OUT_BASE}/probe_collect_${STAGE}.log"
mkdir -p "$OUT_BASE" "$RAW_HOST"

[ -f "$MANIFEST" ] || { echo "[ABORT] manifest 없음: $MANIFEST (make_seed_manifest.py 먼저)"; exit 2; }
MSHA="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
# probe 산출은 seed manifest sha 로 네임스페이스 — manifest 가 바뀌면 옛 npz 를 절대
# 재사용하지 않는다 (Gate2 P2). 같은 manifest 안에서는 seeds 대조까지 이중 확인.
PROBE_REL_SUB="probe/m${MSHA:0:8}"
PROBE_DIR="${OUT_BASE}/${PROBE_REL_SUB}"
mkdir -p "$PROBE_DIR"
{
  echo "${MACHINE_TAG:-srv50} — exp5-4 Phase A ($(date -u +%FT%T)) stage=$STAGE"
  echo "ports=${PORTS[*]} gpus=${GPUS[*]} manifest_sha256=${MSHA}"
} >> "${OUT_BASE}/MACHINE.txt"
echo "=== stage=$STAGE 시작 $(date -u +%FT%T) manifest_sha=${MSHA:0:12} ===" | tee -a "$LOG"

# ---- manifest 로드 --------------------------------------------------------
# 행: scene_idx scene cand_idx base_seed episode_idx probe_order rollout_order
ROWS=()
while IFS=$'\t' read -r s_idx scene c_idx base ep p_ord r_ord; do
  case "$s_idx" in ''|scene_idx|\#*) continue ;; esac
  ROWS+=("$s_idx $scene $c_idx $base $ep $p_ord $r_ord")
done < "$MANIFEST"
[ "${#ROWS[@]}" -gt 0 ] || { echo "[ABORT] manifest 행 0"; exit 2; }
SCENES=($(printf '%s\n' "${ROWS[@]}" | awk '{print $2}' | awk '!seen[$0]++'))

# ---- serve (plain — steering 플래그 없음) ---------------------------------
kill_port() { pkill -f "serve/lerobot.py.*--port ${1}" 2>/dev/null || true; }
kill_serves() { for p in "${PORTS[@]}"; do kill_port "$p"; done; sleep 5; }
trap kill_serves EXIT

serve_up() {
  for p in "${PORTS[@]}"; do kill_port "$p"; done
  sleep 3
  for i in $(seq 0 $((NW-1))); do
    local SLOG="/tmp/exp54_${PORTS[$i]}.log"; rm -f "$SLOG"
    ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        PYTHONPATH="$REPO_ROOT/lerobot/src" \
        ${HF_HOME_OVERRIDE:+HF_HOME=$HF_HOME_OVERRIDE} \
        ${HF_HOME_OVERRIDE:+HF_HUB_OFFLINE=1} \
        "$HPY" scripts/serve/lerobot.py --profile ${PROFILE} \
          --host '*' --port "${PORTS[$i]}" --device cuda \
          --groot-dit-token-pool all_token_full --groot-dit-capture-layers ${CAP} \
        > "$SLOG" 2>&1 < /dev/null & )
  done
  for p in "${PORTS[@]}"; do
    local ok=0 SLOG="/tmp/exp54_${p}.log"
    for _ in $(seq 1 180); do
      curl -s -m 3 "http://127.0.0.1:${p}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
      sleep 5
    done
    [ $ok = 1 ] || { echo "[ABORT] serve $p TIMEOUT" | tee -a "$LOG"; tail -20 "$SLOG"; exit 11; }
    grep -qiE 'Traceback|Address already in use|Errno 98' "$SLOG" \
      && { echo "[ABORT] serve $p 부팅 오류" | tee -a "$LOG"; tail -20 "$SLOG"; exit 11; }
    curl -s -m 3 "http://127.0.0.1:${p}/health" | grep -q '"capture_token_mode":"all_token_full"' \
      || { echo "[ABORT] serve $p capture_token_mode != all_token_full" | tee -a "$LOG"; exit 12; }
  done
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$LOG"
}

# ---- 단위 작업 ------------------------------------------------------------
probe_one() {  # $1=scene $2=port
  local S=$1 port=$2 rc=0
  local out_host="${PROBE_DIR}/scene${S}.npz"
  # probe 호출 순서 = manifest probe_order
  local seeds
  seeds=$(printf '%s\n' "${ROWS[@]}" | awk -v s="$S" '$2==s {print $6"\t"$4}' | sort -n | cut -f2 | paste -sd, -)
  [ -n "$seeds" ] || { echo "[ABORT] scene$S seed 목록 비어 있음" >> "$LOG"; return 1; }
  if [ -f "$out_host" ]; then
    # resume: 파일 존재만으로 재사용하지 않는다 — npz 의 seeds 가 현재 manifest 행과
    # 정확히 같을 때만 유효 (PROBE_DIR 이 manifest sha 로 네임스페이스돼 있어도 이중 방어).
    docker exec -e PYTHONPATH="$PYPATH" "$CONTAINER" python -c \
"import sys,numpy as np
z=np.load(sys.argv[1]); got=[int(v) for v in z['seeds']]
want=[int(v) for v in sys.argv[2].split(',')]
sys.exit(0 if got==want else 9)" \
      "/temporal_vla/${REL_BASE}/${PROBE_REL_SUB}/scene${S}.npz" "$seeds" >/dev/null 2>&1
    rc=$?
    if [ "$rc" = 0 ]; then echo "[skip] probe scene$S (npz seeds 일치)" >> "$LOG"; return 0; fi
    echo "[ABORT] probe scene$S 기존 npz 의 seeds 가 manifest 와 불일치 (rc=$rc) — 수동 확인 필요" | tee -a "$LOG"
    return 1
  fi
  docker exec -e MUJOCO_GL=egl \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    -e PYTHONPATH="$PYPATH" "$CONTAINER" \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
    --output-dir "/temporal_vla/${RAW_REL}" \
    --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" --canonical-instruction "$INSTR" \
    --seed "$S" --n-action-steps "$NAS" --max-episode-steps "$MAXEP" --wait-ready \
    --probe-seeds "$seeds" --probe-out "/temporal_vla/${REL_BASE}/${PROBE_REL_SUB}/scene${S}.npz" \
    >> "${OUT_BASE}/probe_scene${S}.log" 2>&1
  rc=$?
  echo "[$([ "$rc" = 0 ] && echo ok || echo FAIL:$rc)] probe S$S p$port $(date -u +%T)" >> "$LOG"
  [ "$rc" = 0 ] || return "$rc"
  [ -f "$out_host" ] || { echo "[FAIL] probe S$S rc=0 인데 npz 없음" | tee -a "$LOG"; return 6; }
  return 0
}

rollout_one() {  # $1=scene $2=base_seed $3=epidx $4=port $5=rel_out $6=extra_flags...
  local S=$1 base=$2 epidx=$3 port=$4 rel=$5 rc=0; shift 5
  local d="${REPO_ROOT}/${rel}/${TASK}/${CELL_ID}"
  if ls "$d/task${CELL_INDEX}--ep${epidx}--succ"*.json >/dev/null 2>&1 \
     || ls "$d/task${CELL_INDEX}--ep${epidx}--succ"*.csv >/dev/null 2>&1; then
    echo "[skip] ep$epidx (마커 존재)" >> "$LOG"; return 0
  fi
  docker exec -e MUJOCO_GL=egl \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    -e PYTHONPATH="$PYPATH" "$CONTAINER" \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
    --output-dir "/temporal_vla/${rel}" \
    --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" --canonical-instruction "$INSTR" \
    --episode-start-idx "$epidx" --n-episodes 1 \
    --seed "$S" --inference-seed "$base" --n-action-steps "$NAS" --max-episode-steps "$MAXEP" \
    --video-fps 20 --steps-per-render 2 --wait-ready --proximity-phases "$@" \
    >> "${OUT_BASE}/rollout_ep${epidx}.log" 2>&1
  rc=$?
  echo "[$([ "$rc" = 0 ] && echo ok || echo FAIL:$rc)] rollout S$S base$base ep$epidx p$port $(date -u +%T)" >> "$LOG"
  return "$rc"   # ★collector 종료코드 보존 — 워커가 이걸로 실패를 집계한다
}

# ---- stage 실행 -----------------------------------------------------------
serve_up

case "$STAGE" in
probe)
  WPIDS=()
  for ((w=0; w<NW; w++)); do
    (
      f=0
      for ((i=w; i<${#SCENES[@]}; i+=NW)); do probe_one "${SCENES[$i]}" "${PORTS[$w]}" || f=1; done
      exit $f
    ) & WPIDS+=($!)
  done
  WFAIL=0; for p in "${WPIDS[@]}"; do wait "$p" || WFAIL=1; done
  N_NPZ=$(ls "$PROBE_DIR"/scene*.npz 2>/dev/null | wc -l)
  echo "probe npz ${N_NPZ}/${#SCENES[@]} wfail=${WFAIL}" | tee -a "$LOG"
  if [ "$N_NPZ" = "${#SCENES[@]}" ] && [ "$WFAIL" = 0 ]; then
    touch "${OUT_BASE}/PROBE.DONE"
    echo "probe_dir=${PROBE_DIR}" >> "${OUT_BASE}/MACHINE.txt"
    echo "=== PROBE.DONE — make_selection_manifest.py 로 봉인 후 --stage rollout ===" | tee -a "$LOG"
  else
    echo "[FAIL] probe 미완 (npz ${N_NPZ}/${#SCENES[@]}, wfail=${WFAIL}) — DONE 미기록" | tee -a "$LOG"
    kill_serves; trap - EXIT; exit 5
  fi
  ;;

rollout)
  [ -f "${OUT_BASE}/PROBE.DONE" ] || { echo "[ABORT] PROBE.DONE 없음 — probe 먼저" | tee -a "$LOG"; exit 3; }
  # ★선택 manifest 봉인 강제 (Gate2 P1): probe 결과로 순위를 확정·해시 봉인한 뒤에만
  # 본 rollout 을 돈다. 봉인 없이 돌면 "굴린 뒤 순위를 정했다" 는 사후선택 비판을
  # 그대로 되받는다 (계획서 §3-1).
  SEL="${OUT_BASE}/selection_manifest.tsv"
  SEL_SHA="${OUT_BASE}/selection_manifest.sha256"
  [ -f "$SEL" ] && [ -f "$SEL_SHA" ] || {
    echo "[ABORT] 봉인된 선택 manifest 없음: $SEL (+.sha256) — make_selection_manifest.py 먼저" | tee -a "$LOG"; exit 4; }
  SEL_NOW=$(sha256sum "$SEL" | cut -d' ' -f1)
  SEL_REC=$(tr -d ' \t\n' < "$SEL_SHA" | grep -oE '[0-9a-f]{64}' | head -1)
  [ -n "$SEL_REC" ] && [ "$SEL_NOW" = "$SEL_REC" ] || {
    echo "[ABORT] selection_manifest sha 불일치: 파일=${SEL_NOW:0:12} 봉인=${SEL_REC:0:12}" | tee -a "$LOG"; exit 4; }
  echo "selection_manifest sha256=${SEL_NOW:0:12} 검증 통과 (rows=$(($(wc -l < "$SEL")-1)))" | tee -a "$LOG"
  echo "selection_manifest_sha256=${SEL_NOW}" >> "${OUT_BASE}/MACHINE.txt"

  # rollout_order 로 정렬된 실행 순서
  ORDERED=()
  while IFS= read -r r; do ORDERED+=("$r"); done < <(printf '%s\n' "${ROWS[@]}" | sort -k7,7n)
  WPIDS=()
  for ((w=0; w<NW; w++)); do
    (
      f=0
      for ((i=w; i<${#ORDERED[@]}; i+=NW)); do
        set -- ${ORDERED[$i]}   # s_idx scene c_idx base ep p_ord r_ord
        rollout_one "$2" "$4" "$5" "${PORTS[$w]}" "$RAW_REL" --no-features || f=1
      done
      exit $f
    ) & WPIDS+=($!)
  done
  WFAIL=0; for p in "${WPIDS[@]}"; do wait "$p" || WFAIL=1; done
  N_JSON=$(find "$RAW_HOST" -name 'task*--ep*--succ*.json' | wc -l)
  echo "rollout json ${N_JSON}/${#ROWS[@]} wfail=${WFAIL}" | tee -a "$LOG"
  echo "succ 분포: $(find "$RAW_HOST" -name '*.json' | grep -oE 'succ[01]' | sort | uniq -c | tr '\n' ' ')" | tee -a "$LOG"
  if [ "$N_JSON" = "${#ROWS[@]}" ] && [ "$WFAIL" = 0 ]; then
    touch "${OUT_BASE}/ROLLOUT.DONE"
  else
    echo "[FAIL] rollout 미완 (json ${N_JSON}/${#ROWS[@]}, wfail=${WFAIL}) — DONE 미기록" | tee -a "$LOG"
    kill_serves; trap - EXIT; exit 5
  fi
  ;;

sanity)
  # capture ON 2판 — probe 활성(record0) 과 bit 대조용 (check_probe_identity.py).
  # 대상 = 첫 scene 의 cand0, 마지막 scene 의 cand(k-1)
  FIRST=$(printf '%s\n' "${ROWS[@]}" | awk '$1==0 && $3==0')
  LAST_S=$(printf '%s\n' "${ROWS[@]}" | awk '{print $1}' | sort -n | tail -1)
  LAST_C=$(printf '%s\n' "${ROWS[@]}" | awk -v s="$LAST_S" '$1==s {print $3}' | sort -n | tail -1)
  LAST=$(printf '%s\n' "${ROWS[@]}" | awk -v s="$LAST_S" -v c="$LAST_C" '$1==s && $3==c')
  SFAIL=0
  for r in "$FIRST" "$LAST"; do
    set -- $r
    rollout_one "$2" "$4" "$5" "${PORTS[0]}" "$SAN_REL" || SFAIL=1
  done
  N_PKL=$(find "${REPO_ROOT}/${SAN_REL}" -name '*.pkl' | wc -l)
  echo "sanity pkl: ${N_PKL} (기대 2) sfail=${SFAIL}" | tee -a "$LOG"
  echo "다음: check_probe_identity.py 로 probe npz(${PROBE_DIR}) ↔ sanity pkl record0 대조" | tee -a "$LOG"
  if [ "$N_PKL" -lt 2 ] || [ "$SFAIL" != 0 ]; then
    echo "[FAIL] sanity 미완 — DONE 미기록" | tee -a "$LOG"; kill_serves; trap - EXIT; exit 5
  fi
  ;;

smoke)
  # ①②③ = 서버 결정성 (컨테이너 안에서 env obs 를 만들어 HTTP 직접 호출)
  set -- $(printf '%s\n' "${ROWS[@]}" | awk '$1==0 && $3==0')
  SM_SCENE=$2; SM_SEED=$4; MFAIL=0
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" "$CONTAINER" \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/steer/exp5_4/smoke_probe.py \
    --task "$TASK" --env-name "$ENVN" --seed "$SM_SCENE" --inference-seed "$SM_SEED" \
    --n-action-steps "$NAS" --max-episode-steps "$MAXEP" \
    --servers "$(printf 'http://127.0.0.1:%s\n' "${PORTS[@]}" | paste -sd, -)" \
    2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "[FAIL] smoke_probe.py rc=${PIPESTATUS[0]}" | tee -a "$LOG"; MFAIL=1; }
  # ④ = 같은 (scene, base_seed) end-to-end rollout 2회 (success flip 여부)
  for rep in 1 2; do
    rollout_one "$SM_SCENE" "$SM_SEED" "90${rep}" "${PORTS[0]}" "${SMOKE_REL}_rep${rep}" --no-features || MFAIL=1
  done
  SUCCS=()
  for rep in 1 2; do
    f=$(ls "${REPO_ROOT}/${SMOKE_REL}_rep${rep}/${TASK}/${CELL_ID}"/task*--ep90${rep}--succ*.json 2>/dev/null | head -1)
    echo "④ rep${rep}: $(basename "${f:-none}")" | tee -a "$LOG"
    SUCCS+=("$(basename "${f:-none}" | grep -oE 'succ[01]' || echo none)")
  done
  if [ "${SUCCS[0]}" = none ] || [ "${SUCCS[0]}" != "${SUCCS[1]}" ]; then
    echo "[FAIL] ④ 동일 (scene,seed) rollout 결과 불일치: ${SUCCS[*]}" | tee -a "$LOG"; MFAIL=1
  else
    echo "④ 결정성 OK (${SUCCS[0]} 일치)" | tee -a "$LOG"
  fi
  [ "$MFAIL" = 0 ] || { kill_serves; trap - EXIT; exit 5; }
  ;;
esac

kill_serves
trap - EXIT
echo "=== stage=$STAGE 완료 $(date -u +%FT%T) ===" | tee -a "$LOG"
touch "${OUT_BASE}/STAGE_${STAGE}.DONE"
