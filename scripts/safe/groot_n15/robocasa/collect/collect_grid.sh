#!/usr/bin/env bash
# 좌표 기반 수집 러너 (docs/04 §3.1 그리드 · docs/steering/38 §3).
#
# collection_plan.json 의 셀(instruction × scene × noise) 중 이 머신이 맡은 것만 골라
# http_feature_collect.py 를 좌표 인자와 함께 호출한다. 경로 조립·좌표 해석은 하지 않는다 —
# src/collect/plan.py 가 단일 출처이고, 이 스크립트는 계획에서 읽은 값을 그대로 넘기기만 한다.
#
# 발사 (장시간 run 은 반드시 setsid nohup — agent 백그라운드 job 은 harness 가 중간에 kill 한다):
#
#   PLAN_JSON=outputs/collect/n15_grid_v1/collection_plan.json \
#   GRID_ROOT=/home/dongkyu/pkt_ws/temporal_vla/outputs/collect/grid_staging \
#   INSTRUCTIONS='OpenDrawer/left,OpenDrawer/right' \
#   GPUS=4,5 SERVES_PER_GPU=1 PORT_BASE=8600 NOISE_LIMIT=10 \
#   setsid nohup bash scripts/safe/groot_n15/robocasa/collect/collect_grid.sh \
#     > outputs/collect/logs/grid_orch.log 2>&1 < /dev/null &
#
# 완료 판정은 로그가 아니라 grid 디렉토리의 meta.json 존재로 한다.
#
# ── 계획 스키마 가정 ────────────────────────────────────────────────────────────
#   extra["env_names"][instruction]      → RoboCasa env id (필수)
#         예: "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
#   extra["task_names"][instruction]     → --task 값 (선택). 없으면 env id 에서 유도
#         ("robocasa_panda_omron/OpenDrawer_PandaOmron_Env" → "OpenDrawer")
#   extra["instruction_text"][instruction] → --canonical-instruction 으로 넘길 실제 언어 문자열
#         (선택). 없으면 instruction 키 자체를 쓴다 — 키가 이미 문장이면 그대로 맞는다.
#   ckpt                                 → configs/checkpoints/<ckpt>.yaml (serve --profile)
#   capture_layers                       → --groot-dit-capture-layers
# 하드코딩 금지: 위 값은 전부 PLAN_JSON 에서 읽는다.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
: "${PLAN_JSON:?PLAN_JSON (collection_plan.json 경로) 필요}"
: "${GRID_ROOT:?GRID_ROOT (로컬 staging 루트) 필요}"
: "${INSTRUCTIONS:?INSTRUCTIONS (쉼표구분 instruction 키) 필요}"
: "${GPUS:?GPUS (쉼표구분) 필요}"
SERVES_PER_GPU="${SERVES_PER_GPU:-1}"
PORT_BASE="${PORT_BASE:-8600}"
NOISE_LIMIT="${NOISE_LIMIT:-10}"
DONE_LIST="${DONE_LIST:-${GRID_ROOT}/shipped_cells.txt}"
DRY_RUN="${DRY_RUN:-0}"
PY_HOST="${PY_HOST:-python3}"

# 컨테이너에서 본 경로. repo 는 /temporal_vla 로 마운트된다.
to_container() {  # 호스트 절대경로 → 컨테이너 경로 (repo 밖이면 그대로)
  case "$1" in
    "${REPO_ROOT}"/*) printf '/temporal_vla/%s\n' "${1#"${REPO_ROOT}"/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}
GRID_ROOT="$(realpath -m -- "$GRID_ROOT")"
PLAN_JSON_ABS="$(realpath -m -- "$PLAN_JSON")"
GRID_ROOT_CONT="${GRID_ROOT_CONT:-$(to_container "$GRID_ROOT")}"
PLAN_JSON_CONT="${PLAN_JSON_CONT:-$(to_container "$PLAN_JSON_ABS")}"

LOGDIR="${LOGDIR:-${GRID_ROOT}/logs}"
mkdir -p "$LOGDIR" "$GRID_ROOT"

PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

log() { echo "[grid] $(date '+%F %T') $*"; }

# ── 1. 계획에서 스칼라 읽기 ────────────────────────────────────────────────────
eval "$("$PY_HOST" - "$REPO_ROOT" "$PLAN_JSON_ABS" <<'PY'
import shlex, sys
sys.path.insert(0, sys.argv[1])
from src.collect.plan import CollectionPlan

plan = CollectionPlan.load(sys.argv[2])
out = {
    "PLAN_ID": plan.plan_id,
    "PLAN_CKPT": plan.ckpt,
    "CAPTURE_LAYERS": ",".join(str(x) for x in plan.capture_layers),
    "TOKEN_MODE": plan.token_mode,
    "PLAN_N_CELLS": str(plan.n_cells),
}
for k, v in out.items():
    print(f"{k}={shlex.quote(v)}")
PY
)"
[ -n "${PLAN_ID:-}" ] || { log "ABORT: 계획 로드 실패 ($PLAN_JSON_ABS)"; exit 10; }
PROFILE="configs/checkpoints/${PLAN_CKPT}.yaml"
[ -f "${REPO_ROOT}/${PROFILE}" ] || { log "ABORT: 프로파일 없음 ${PROFILE} (plan.ckpt=${PLAN_CKPT})"; exit 11; }
log "plan_id=${PLAN_ID} cells=${PLAN_N_CELLS} ckpt=${PLAN_CKPT} layers=${CAPTURE_LAYERS} token_mode=${TOKEN_MODE}"

# ── 2. 셀 목록 (INSTRUCTIONS · NOISE_LIMIT 필터) ──────────────────────────────
CELLS_TSV="${LOGDIR}/cells_planned.tsv"
"$PY_HOST" - "$REPO_ROOT" "$PLAN_JSON_ABS" "$INSTRUCTIONS" "$NOISE_LIMIT" > "$CELLS_TSV" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from src.collect.plan import CollectionPlan

plan = CollectionPlan.load(sys.argv[2])
wanted = [s for s in sys.argv[3].split(",") if s.strip()]
wanted = [s.strip() for s in wanted]
limit = int(sys.argv[4])
unknown = [w for w in wanted if w not in plan.instructions]
if unknown:
    sys.exit(f"계획에 없는 instruction: {unknown} (있는 것: {list(plan.instructions)})")
extra = plan.extra or {}
env_names = extra.get("env_names", {})
task_names = extra.get("task_names", {})
instr_text = extra.get("instruction_text", {})
missing_env = [w for w in wanted if w not in env_names]
if missing_env:
    sys.exit(f"extra['env_names'] 에 없는 instruction: {missing_env} — 계획을 고칠 것")


def derive_task(env: str) -> str:
    leaf = env.split("/")[-1]
    return leaf.split("_PandaOmron")[0].split("_Env")[0]


adopted = set((extra.get("adopted_cells") or []))
for cell in plan.cells():
    if cell.instruction not in wanted or cell.noise_idx >= limit:
        continue
    if adopted and cell.key not in adopted:   # v3 지터 plan: 채택 셀만 (dummy 제외)
        continue
    env = env_names[cell.instruction]
    print("\t".join([
        cell.key, cell.instruction, str(cell.scene_idx), str(cell.env_seed),
        str(cell.noise_idx), str(cell.inference_seed), env,
        task_names.get(cell.instruction) or derive_task(env),
        instr_text.get(cell.instruction, cell.instruction),
    ]))
PY
rc=$?
[ $rc -eq 0 ] || { log "ABORT: 셀 목록 생성 실패 (rc=$rc)"; cat "$CELLS_TSV"; exit 12; }
N_PLANNED=$(wc -l < "$CELLS_TSV")
log "계획 셀(필터 후) ${N_PLANNED} 개 → ${CELLS_TSV}"

# ── 3. 결손만 남기기 (DONE_LIST + 로컬 meta.json) ─────────────────────────────
TODO_TSV="${LOGDIR}/cells_todo.tsv"
: > "$TODO_TSV"
n_done_list=0; n_done_local=0
while IFS=$'\t' read -r key instr si seed ni inf env task text; do
  [ -n "$key" ] || continue
  if [ -f "$DONE_LIST" ] && grep -Fxq -- "$key" "$DONE_LIST"; then
    n_done_list=$((n_done_list + 1)); continue
  fi
  # machine 층은 serve 가 정한다 → 어느 머신이든 있으면 수집됨으로 본다.
  if compgen -G "${GRID_ROOT}/${PLAN_ID}/*/${instr}/s${si}/n${ni}/*/meta.json" > /dev/null 2>&1; then
    n_done_local=$((n_done_local + 1)); continue
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$key" "$instr" "$si" "$seed" "$ni" "$inf" "$env" "$task" "$text" >> "$TODO_TSV"
done < "$CELLS_TSV"
N_TODO=$(wc -l < "$TODO_TSV")
log "결손 ${N_TODO} / 계획 ${N_PLANNED} (완료: DONE_LIST ${n_done_list} + 로컬 meta.json ${n_done_local})"
[ "$N_TODO" -gt 0 ] || { log "할 일 없음 — 종료"; exit 0; }

# ── 4. serve 기동 (GPU × SERVES_PER_GPU) ──────────────────────────────────────
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
PORTS=(); PORT_GPUS=()
p=$PORT_BASE
for g in "${GPU_ARR[@]}"; do
  for _ in $(seq 1 "$SERVES_PER_GPU"); do
    PORTS+=("$p"); PORT_GPUS+=("$g"); p=$((p + 1))
  done
done
NWORKERS=${#PORTS[@]}
log "serve ${NWORKERS} 개 — gpus=${GPUS} serves_per_gpu=${SERVES_PER_GPU} ports=${PORTS[*]}"

# SERVE_MODE=docker(기본): docker `lerobot` 컨테이너에서 serve (kanu·pdk_external).
# SERVE_MODE=host: 호스트 conda 로 serve (srv48·srv50 — lerobot 컨테이너 없음,
#   memory a100-srv50-parallel-eval 관례). SERVE_PY=파이썬 경로,
#   SERVE_PYTHONPATH=lerobot 패키지 경로(예 ${REPO_ROOT}/lerobot/src) 지정.
SERVE_MODE="${SERVE_MODE:-docker}"
SERVE_PY="${SERVE_PY:-python}"
SERVE_PYTHONPATH="${SERVE_PYTHONPATH:-}"

cleanup() {
  log "cleanup: serve 정리"
  for port in "${PORTS[@]}"; do
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry] pkill serve --port ${port} (${SERVE_MODE})"
    elif [ "$SERVE_MODE" = "host" ]; then
      pkill -f "serve/lerobot.py.*--port ${port}" 2>/dev/null || true
    else
      docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_serve() {  # gpu port
  local gpu="$1" port="$2"
  if [ "$SERVE_MODE" = "host" ]; then
    local hcmd=("$SERVE_PY" "${REPO_ROOT}/scripts/serve/lerobot.py" --profile "${REPO_ROOT}/${PROFILE}"
      --host '*' --port "$port" --device cuda --collect --capture-vl
      --groot-dit-token-pool "$TOKEN_MODE" --groot-dit-capture-layers "$CAPTURE_LAYERS")
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry] host serve gpu=${gpu} port=${port}: ${hcmd[*]}"
    else
      # exp2 srv50_collect_cell.sh 의 검증된 블록과 동일: cd REPO_ROOT + env 주입
      ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" \
          PYTHONPATH="${SERVE_PYTHONPATH}${SERVE_PYTHONPATH:+:}${REPO_ROOT}" \
          "${hcmd[@]}" > "${LOGDIR}/serve_${port}.log" 2>&1 < /dev/null & )
    fi
    return
  fi
  local cmd="cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
    --host '*' --port ${port} --device cuda --collect --capture-vl \
    --groot-dit-token-pool ${TOKEN_MODE} --groot-dit-capture-layers ${CAPTURE_LAYERS} \
    > /tmp/grid_serve_${port}.log 2>&1 < /dev/null &"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry] docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} lerobot bash -lc \"${cmd}\""
  else
    docker exec -d -e CUDA_VISIBLE_DEVICES="$gpu" -e OMP_NUM_THREADS="${SERVE_OMP_THREADS:-4}" -e OPENBLAS_NUM_THREADS="${SERVE_OMP_THREADS:-4}" -e MKL_NUM_THREADS="${SERVE_OMP_THREADS:-4}" lerobot bash -lc "$cmd"
  fi
}
health_curl() {  # port
  if [ "$SERVE_MODE" = "host" ]; then
    curl -s -m 3 "http://127.0.0.1:${1}/health" 2>/dev/null
  else
    docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${1}/health 2>/dev/null"
  fi
}
# ★ 순차 기동: 로드 중 VRAM 피크(~12GB)가 안정 상태(~6GB)의 2배라, 동시에 띄우면
# 피크가 겹쳐 OOM (kanu A4000 2-serve 실측: 11.96+3.76GB 로 16GB 초과 사망).
# 한 serve 가 health ok 로 피크를 지난 뒤에 다음을 띄운다.
for i in "${!PORTS[@]}"; do
  start_serve "${PORT_GPUS[$i]}" "${PORTS[$i]}"
  if [ "$DRY_RUN" = "1" ]; then echo "[dry] health poll ${PORTS[$i]}"; continue; fi
  ok=0
  for _ in $(seq 1 150); do
    st=$(health_curl "${PORTS[$i]}" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && { ok=1; break; }
    sleep 5
  done
  log "serve ${PORTS[$i]} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  [ $ok = 1 ] || { log "ABORT: serve ${PORTS[$i]} 기동 실패"; exit 13; }
done

# ── 5. 워커 fan-out (셀 인덱스 mod 워커수) ────────────────────────────────────
run_worker() {  # wid port
  local wid="$1" port="$2" idx=-1
  while IFS=$'\t' read -r key instr si seed ni inf env task text; do
    idx=$((idx + 1))
    [ $((idx % NWORKERS)) -eq "$wid" ] || continue
    # 재확인 — 다른 워커/이전 run 이 이미 채웠을 수 있다 (판정은 meta.json).
    if compgen -G "${GRID_ROOT}/${PLAN_ID}/*/${instr}/s${si}/n${ni}/*/meta.json" > /dev/null 2>&1; then
      echo "[w${wid}] skip(done) ${key}"; continue
    fi
    # ★ backpressure: 이관(shipper)이 못 따라가 staging 이 cap 을 넘으면 수집을
    # 멈추고 기다린다 — 08-18 실측: 이관 병목으로 staging 134GB 폭주 → 디스크
    # 포화 → 수집 전체 사망. cap 은 STAGING_WAIT_GB(기본 25GB).
    while :; do
      local st_kb; st_kb=$(du -sk "$GRID_ROOT" 2>/dev/null | awk '{print $1}')
      [ "$((st_kb / 1048576))" -lt "${STAGING_WAIT_GB:-25}" ] && break
      echo "[w${wid}] staging $((st_kb / 1048576))GB ≥ ${STAGING_WAIT_GB:-25}GB — 이관 대기 120s"
      sleep 120
    done
    echo "[w${wid}] $(date '+%F %T') ${key} seed=${seed} inf=${inf}"
    local args=(
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py
      --vla-server "http://127.0.0.1:${port}"
      --task "$task" --env-name "$env"
      # 워커별 분리 필수: 모든 셀이 task0--ep0 라 공유 _work 면 .groot_video_tmp 가
      # 겹쳐 한 워커의 cleanup 이 다른 워커의 tmp video 를 지운다 (스모크 실측 실패).
      --output-dir "${GRID_ROOT_CONT}/_work/w${wid}"
      --grid-root "$GRID_ROOT_CONT" --plan-json "$PLAN_JSON_CONT"
      --scene-idx "$si" --noise-idx "$ni" --grid-instruction "$instr"
      --canonical-instruction "$text"
      --episode-start-idx 0 --n-episodes 1
      --seed "$seed" --inference-seed "$inf"
      --n-action-steps 5 --max-episode-steps 720
      --video-fps 20 --steps-per-render 2 --wait-ready
    )
    if [ "${JITTER_MODE:-0}" = "1" ]; then
      # v3 지터 plan: si=scene*100+k 평탄 좌표 → 실제 reset_idx=si%100.
      # ep_meta 는 GRID_ROOT/ep_meta 에 export (base seed 키) — 아카이브 동봉 대상.
      args+=(--jitter-reset-idx "$((si % 100))" --ep-meta-dir "${GRID_ROOT_CONT}/ep_meta")
    fi
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry] docker exec robocasa ${args[*]}"
    else
      docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa "${args[@]}" 2>&1 \
        | grep -E "^wrote|Error|Traceback|좌표" || true
    fi
  done < "$TODO_TSV"
  echo "[w${wid}] DONE"
}

for i in "${!PORTS[@]}"; do
  run_worker "$i" "${PORTS[$i]}" > "${LOGDIR}/worker_w${i}.log" 2>&1 &
done
wait

# ── 6. 결과 요약 (meta.json 기준) ─────────────────────────────────────────────
n_have=0
while IFS=$'\t' read -r key instr si seed ni inf env task text; do
  compgen -G "${GRID_ROOT}/${PLAN_ID}/*/${instr}/s${si}/n${ni}/*/meta.json" > /dev/null 2>&1 \
    && n_have=$((n_have + 1))
done < "$TODO_TSV"
log "수집 완료 ${n_have} / 목표 ${N_TODO}"
touch "${LOGDIR}/GRID_ROUND_DONE"
log "DONE"
