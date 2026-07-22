# exp4-2 smoke 공용 헬퍼 — `source` 해서 사용 (단독 실행 아님).
# serve = worktree exp42_serve.py (lerobot.py 심링크 — pkill 격리, patchceil 선례).
# collector = worktree http_feature_collect.py (Track P 수정본). 데이터는 main-tree outputs.
set -uo pipefail

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
TASK=PickPlaceCounterToCabinet
ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
CELL_INDEX=5; CELL_ID=ppcc_bread
INSTR="Pick the bread from the counter and place it in the cabinet."
NAS=5; MAXEP=720; SEED=100084

_SMOKE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WT_HOST="$(cd -- "${_SMOKE_DIR}/../../../../../../.." && pwd)"          # worktree root
MAIN_HOST="$(cd -- "${WT_HOST}/../../.." && pwd)"                       # main repo root
WT_CONT="/temporal_vla/.claude/worktrees/exp4-2-induced-failures"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
SMOKE_ROOT="${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp42_induced/smoke"
JUDGE="${_SMOKE_DIR}/smoke_judge.py"
JUDGE_CONT="${WT_CONT}/scripts/safe/groot_n15/robocasa/steer/induced/smoke/smoke_judge.py"

to_cont() { echo "${1/#${MAIN_HOST}//temporal_vla}"; }

start_serve() {  # gpu port extra...
  local gpu=$1 port=$2; shift 2
  docker exec -d -e CUDA_VISIBLE_DEVICES="$gpu" lerobot bash -lc \
    "cd ${WT_CONT} && setsid nohup python scripts/serve/exp42_serve.py --profile ${PROFILE} \
       --host '*' --port ${port} --device cuda $* > /tmp/exp42_${port}.log 2>&1 < /dev/null &"
}

wait_health() {  # port
  local port=$1 ok=0 st
  for _ in $(seq 1 150); do
    st=$(curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && { ok=1; break; }
    sleep 5
  done
  if [ $ok != 1 ]; then
    echo "ABORT: serve ${port} not ready"
    docker exec lerobot bash -lc "tail -30 /tmp/exp42_${port}.log" || true
    exit 11
  fi
}

kill_serve() { docker exec lerobot bash -lc "pkill -f 'serve/exp42_serve.py.*--port ${1}' || true" 2>/dev/null || true; }

run_ep() {  # port out_dir_host ep_idx inf_seed extra...
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

ep_file() { ls "${1}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${3:-0}--succ"*."${2}" 2>/dev/null | head -1; }

judge()   { python3 "$JUDGE" "$@"; }                                  # host: csv/json
judge_c() { docker exec lerobot python "$JUDGE_CONT" "$@"; }          # 컨테이너: pkl

# 판정 집계
N_PASS=0; N_FAIL=0; N_SOFT=0
check() {  # label cmd...
  local label=$1; shift
  if "$@"; then N_PASS=$((N_PASS+1)); else echo "[FAIL] ${label}"; N_FAIL=$((N_FAIL+1)); fi
}
check_soft() {  # label cmd...  (실패해도 게이트는 통과 — 보고만)
  local label=$1; shift
  if "$@"; then N_PASS=$((N_PASS+1)); else echo "[SOFT-FAIL] ${label} (판정 강등 — 보고서 확인)"; N_SOFT=$((N_SOFT+1)); fi
}
summary() {  # smoke_name
  echo "== ${1}: PASS=${N_PASS} SOFT=${N_SOFT} FAIL=${N_FAIL} =="
  [ "$N_FAIL" -eq 0 ] || exit 1
}
