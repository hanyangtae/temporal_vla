#!/usr/bin/env bash
# v5 첫 셀 게이트 — handoff_20260902_grid_recollect_v5 §0.1 / §0.6-2.
#
# 같은 좌표(첫 채택 셀)를 4번 돌려 success·traj.csv(추론당 action 행) bit 재현을 대조한다.
#   A  수집 경로: collect_grid.sh (serve --collect --capture-vl, 캡처 ON, ep_meta 메모리)
#   B  A 와 동일 경로 fresh serve+collector 재실행           → 수집 경로 자기재현
#   C  eval 경로: run_online_gated_eval.sh 관례 (serve 캡처 hook 만·--collect 없음,
#      collector --no-features --cell-id --episode-start-idx ep --expect-chunk-len 16,
#      **ep_meta JSON 로드**(--ep-meta-dir + --ep-meta-load-env-name))  → 수집=eval 판정
#   D  C 에서 ep_meta JSON 로드만 제거(reset(seed) 재캡처)   → C 불일치 시 원인 국소화
# 판정은 scripts/collect/compare_cell_runs.py 가 낸다 (A=B=C 가 아니면 본수집 금지·보고).
#
# 좌표는 3축(docs/04 §3.1.1): collect_grid.sh 의 cells_todo.tsv 열 =
#   key instr si k seed ni inf env task text  (k = jitter_reset_idx, legacy plan 이면 빈 칸).
# k 는 계획의 채택 값 그대로다 — 평탄 si 에서 유도하지 않는다.
#
# 필수 env: PLAN_JSON GATE_ROOT INSTR GPU  (+ srv: SERVE_MODE=host SERVE_PY SERVE_PYTHONPATH)
# 선택: PORT_BASE(8700) PY_HOST(python3)
set -uo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PLAN_JSON:?}" "${GATE_ROOT:?}" "${INSTR:?}" "${GPU:?}"
PORT_BASE="${PORT_BASE:-8700}"
SERVE_MODE="${SERVE_MODE:-docker}"
SERVE_PY="${SERVE_PY:-python}"
SERVE_PYTHONPATH="${SERVE_PYTHONPATH:-}"
PY_HOST="${PY_HOST:-python3}"
GATE_ROOT="$(realpath -m -- "$GATE_ROOT")"; mkdir -p "$GATE_ROOT"
PLAN_JSON_ABS="$(realpath -m -- "$PLAN_JSON")"
RUNNER="${REPO_ROOT}/scripts/safe/groot_n15/robocasa/collect/collect_grid.sh"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
log() { echo "[gate] $(date '+%F %T') $*"; }
GIT_COMMON="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "${REPO_ROOT}/.git")"
MOUNT_ROOT="${MOUNT_ROOT:-$(dirname -- "$GIT_COMMON")}"   # 컨테이너 /temporal_vla = 메인 워크트리
to_container() { case "$1" in "${MOUNT_ROOT}"/*) printf '/temporal_vla/%s\n' "${1#"${MOUNT_ROOT}"/}";; *) printf '%s\n' "$1";; esac; }

export PLAN_JSON="$PLAN_JSON_ABS" INSTRUCTIONS="$INSTR" GPUS="$GPU" SERVES_PER_GPU=1 \
  NOISE_LIMIT=1 MAX_CELLS=1 DONE_LIST=/dev/null SERVE_MODE SERVE_PY SERVE_PYTHONPATH PY_HOST

run_collect_path() {  # tag port
  local tag="$1" port="$2"
  log "== $tag: 수집 경로 (collect_grid.sh, port $port)"
  GRID_ROOT="${GATE_ROOT}/${tag}" PORT_BASE="$port" bash "$RUNNER" > "${GATE_ROOT}/${tag}.log" 2>&1
  local rc=$?
  log "$tag rc=$rc meta=$(find "${GATE_ROOT}/${tag}" -name meta.json | wc -l)"
  return $rc
}

run_collect_path A "$PORT_BASE" || { log "ABORT: A 실패"; exit 20; }
run_collect_path B "$((PORT_BASE + 1))" || { log "ABORT: B 실패"; exit 21; }

# ── 셀 정보 (A 의 todo 표) ──
# 탭 대신 US(0x1f)로 읽는다 — bash read 는 연속 탭을 하나로 접어 k 빈 칸이면 열이 밀린다.
IFS=$'\037' read -r key instr si k seed ni inf env task text \
  < <(head -n 1 "${GATE_ROOT}/A/logs/cells_todo.tsv" | tr '\t' '\037')
slug="${instr//\//_}"
# episode_idx 는 3축 좌표를 평탄화해 유일성만 확보 (k 없으면 0 자리).
ep=$(( (si * 100 + ${k:-0}) * 100 + ni ))
log "셀 ${key} seed=${seed} inf=${inf} si=${si} k=${k:-<none>} ni=${ni} env=${env} task=${task}"

# ── plan 스칼라 ──
eval "$("$PY_HOST" - "$REPO_ROOT" "$PLAN_JSON_ABS" <<'PY'
import shlex, sys
sys.path.insert(0, sys.argv[1])
from src.collect.plan import CollectionPlan
p = CollectionPlan.load(sys.argv[2])
print(f"PLAN_CKPT={shlex.quote(p.ckpt)}"); print(f"CAPTURE_LAYERS={','.join(map(str,p.capture_layers))}"); print(f"TOKEN_MODE={p.token_mode}")
PY
)"
PROFILE="configs/checkpoints/${PLAN_CKPT}.yaml"

# ── eval 경로 serve (run_online_gated_eval.sh serve_flags_for 의 캡처 부분 = detector 없이) ──
EPORT=$((PORT_BASE + 2))
EFLAGS="--groot-dit-capture-layers ${CAPTURE_LAYERS} --groot-dit-token-pool ${TOKEN_MODE}"
if [ "$SERVE_MODE" = host ]; then
  ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" \
      PYTHONPATH="${SERVE_PYTHONPATH}${SERVE_PYTHONPATH:+:}${REPO_ROOT}" \
      "$SERVE_PY" "${REPO_ROOT}/scripts/serve/lerobot.py" --profile "${REPO_ROOT}/${PROFILE}" \
      --host '*' --port "$EPORT" --device cuda $EFLAGS > "${GATE_ROOT}/serve_eval_${EPORT}.log" 2>&1 < /dev/null & )
  hc() { curl -s -m 3 "http://127.0.0.1:${EPORT}/health" 2>/dev/null; }
  ks() { pkill -f "serve/lerobot.py.*--port ${EPORT}" 2>/dev/null || true; }
else
  docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e MKL_NUM_THREADS=4 lerobot bash -lc \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} --host '*' --port ${EPORT} --device cuda ${EFLAGS} > /tmp/gate_serve_${EPORT}.log 2>&1 < /dev/null &"
  hc() { docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${EPORT}/health 2>/dev/null"; }
  ks() { docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${EPORT}' || true" 2>/dev/null || true; }
fi
trap ks EXIT INT TERM
ok=0; for _ in $(seq 1 150); do hc | grep -q '"status":"ok"' && { ok=1; break; }; sleep 5; done
[ $ok = 1 ] || { log "ABORT: eval serve ${EPORT} 기동 실패"; exit 22; }
log "eval serve ${EPORT} health=ok"

# ── ep_meta JSON: A 가 export 한 것을 eval 관례 위치(<dir>/<task>/<slug>/)로 ──
EPM="${GATE_ROOT}/ep_meta_eval"; mkdir -p "${EPM}/${task}/${slug}"
cp "${GATE_ROOT}/A/ep_meta/${task}/"*.json "${EPM}/${task}/${slug}/"
log "ep_meta JSON: $(ls "${EPM}/${task}/${slug}/")"

run_eval_path() {  # tag load_epmeta(0|1)
  local tag="$1" load="$2"
  log "== $tag: eval 경로 (--no-features, ep_meta JSON 로드=$load)"
  local out="${GATE_ROOT}/${tag}"; mkdir -p "$out"
  local args=(
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py
    --vla-server "http://127.0.0.1:${EPORT}" --task "$task" --env-name "$env"
    --output-dir "$(to_container "$out")/raw_rollouts" --cell-id "$slug" --cell-index 0
    --canonical-instruction "$text"
    --episode-start-idx "$ep" --n-episodes 1
    --seed "$seed" --inference-seed "$inf"
    --n-action-steps 5 --max-episode-steps 720
    --video-fps 20 --steps-per-render 2 --wait-ready
    --no-features --expect-chunk-len 16
    --grid-root "$(to_container "$out")" --plan-json "$(to_container "$PLAN_JSON_ABS")"
    --scene-idx "$si" --noise-idx "$ni" --grid-instruction "$instr" --arm-dir "gate_${tag}"
    --run-tag "v5_gate_${tag}"
  )
  [ -n "$k" ] && args+=(--jitter-reset-idx "$k")   # legacy 2축 plan 이면 생략
  [ "$load" = 1 ] && args+=(--ep-meta-dir "$(to_container "$EPM")" --ep-meta-load-env-name "$env")
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa "${args[@]}" > "${GATE_ROOT}/${tag}.log" 2>&1
  local rc=$?
  log "$tag rc=$rc meta=$(find "$out" -name meta.json | wc -l)"
  return $rc
}
run_eval_path C 1 || log "WARN: C 실패 (로그 ${GATE_ROOT}/C.log)"
run_eval_path D 0 || log "WARN: D 실패 (로그 ${GATE_ROOT}/D.log)"
ks; trap - EXIT

"$PY_HOST" "${REPO_ROOT}/scripts/collect/compare_cell_runs.py" "${GATE_ROOT}/A" "${GATE_ROOT}/B" "${GATE_ROOT}/C" "${GATE_ROOT}/D" \
  | tee "${GATE_ROOT}/VERDICT.txt"
log "DONE"
