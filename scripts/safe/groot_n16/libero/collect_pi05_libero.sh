#!/usr/bin/env bash
# pi0.5 × LIBERO SAFE 수집 (phase 라벨 동봉). serve(lerobot_safe) + client(libero_bench)
# 를 한 스크립트로 묶고, 끝나면 serve kill + COLLECT_DONE 센티넬을 쓴다.
# setsid 로 detach 해서 띄우는 것을 전제로 한다(세션·harness 무관 생존).
#
# 사용:
#   setsid nohup bash scripts/safe/groot_n16/libero/collect_pi05_libero.sh \
#     > outputs/eval/libero/pi05/run.log 2>&1 < /dev/null &
#   echo $! > outputs/eval/libero/pi05/run.pid
# env override: GPU, PORT, TASK_SUITE, NUM_TRIALS, MAX_TASKS, RUN_ID, CAP_LAYERS, PROFILE, OUT
set -uo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
source ~/miniconda3/etc/profile.d/conda.sh
set -a; . ./.env; set +a   # HF_TOKEN (gated paligemma)

PROFILE="${PROFILE:-configs/checkpoints/lerobot_pi05__libero_exec5.yaml}"
GPU="${GPU:-5}"; PORT="${PORT:-8411}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-15}"
MAX_TASKS="${MAX_TASKS:-}"        # 빈값=전체 task
TASK_NAMES="${TASK_NAMES:-}"      # 콤마구분 부분일치(예: KITCHEN_SCENE4,KITCHEN_SCENE6). 빈값=전체
RUN_ID="${RUN_ID:-scout15}"
CAP_LAYERS="${CAP_LAYERS:-0,5,11,17}"
MAX_STEPS="${MAX_STEPS:-}"        # 빈값=suite 기본 horizon
SUCCESS_DEADLINE="${SUCCESS_DEADLINE:-}"  # 빈값이고 MAX_STEPS 있으면 = MAX_STEPS (검증 통과용)
EAGER="${EAGER:-0}"              # 1=TORCHDYNAMO_DISABLE(컴파일 끔; 진단/멀티레이어 캡처용)
NO_CAP="${NO_CAP:-0}"           # 1=기본 캡처(--pi05-expert-capture-layers 생략)
OUT="${OUT:-outputs/eval/libero/pi05}"
mkdir -p "$OUT"
[ "$EAGER" = 1 ] && { export TORCHDYNAMO_DISABLE=1; echo "[collect] EAGER mode: TORCHDYNAMO_DISABLE=1 (compile off)"; }
SERVE_LOG="$OUT/serve_${RUN_ID}.log"
DONE="$OUT/COLLECT_DONE_${RUN_ID}"
rm -f "$DONE"

SERVE_PID=""
cleanup() {
  echo "[collect] cleanup: kill serve pid=${SERVE_PID}"
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
  sleep 2
  pkill -f "lerobot.py.*--port ${PORT}" 2>/dev/null
  sleep 1
  echo "[collect] post-cleanup GPU:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed -n "$((GPU+1))p"
}
trap cleanup EXIT

echo "[collect] $(date '+%F %T') start: profile=$PROFILE gpu=$GPU port=$PORT suite=$TASK_SUITE trials=$NUM_TRIALS max_tasks=${MAX_TASKS:-all} run_id=$RUN_ID cap=$CAP_LAYERS"

# ── serve 기동 (lerobot_safe, --collect) ──────────────────────────────────────
CAP_ARG="--pi05-expert-capture-layers $CAP_LAYERS"
[ "$NO_CAP" = 1 ] && { CAP_ARG=""; echo "[collect] default capture (no --pi05-expert-capture-layers)"; }
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=scripts/utils \
  HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  conda run --no-capture-output -n lerobot_safe python scripts/serve/lerobot.py \
  --profile "$PROFILE" --device cuda --port "$PORT" --collect \
  $CAP_ARG > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
echo "[collect] serve pid=$SERVE_PID; waiting /health (cold load ~2-3min)…"

ready=0
for i in $(seq 1 120); do          # 최대 ~10분
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then ready=1; echo "[collect] serve /health OK after ~$((i*5))s"; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "[collect] ERROR serve died on startup; tail serve log:"; tail -n 30 "$SERVE_LOG"; exit 1; fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "[collect] ERROR serve not ready (timeout); tail serve log:"; tail -n 30 "$SERVE_LOG"; exit 1; }

# ── client 수집 (libero_bench, --safe-collect) ────────────────────────────────
MAXT_ARG=""; [ -n "$MAX_TASKS" ] && MAXT_ARG="--max-tasks $MAX_TASKS"
MAXS_ARG=""; [ -n "$MAX_STEPS" ] && MAXS_ARG="--max-steps $MAX_STEPS"
# MAX_STEPS 지정 시 success-deadline 도 그에 맞춰야 함(libero.py: deadline>max_steps 면 ValueError).
[ -n "$MAX_STEPS" ] && [ -z "$SUCCESS_DEADLINE" ] && SUCCESS_DEADLINE="$MAX_STEPS"
DL_ARG=""; [ -n "$SUCCESS_DEADLINE" ] && DL_ARG="--success-deadline $SUCCESS_DEADLINE"
TN_ARG=""; [ -n "$TASK_NAMES" ] && TN_ARG="--task-names $TASK_NAMES"
REPO=/home/dongkyu/pkt_ws/temporal_vla
echo "[collect] client start $(date '+%T')"
PYTHONPATH="$REPO:$REPO/src/benchmarks/LIBERO:$REPO/src:$REPO/scripts/utils:$REPO/src/policies/openvla-oft" \
  conda run --no-capture-output -n libero_bench python scripts/eval/libero.py \
  --task-suite "$TASK_SUITE" --server-url "http://localhost:${PORT}" \
  --num-trials "$NUM_TRIALS" $MAXT_ARG $MAXS_ARG $DL_ARG $TN_ARG \
  --video-dir "$OUT/videos_${RUN_ID}" \
  --safe-collect --safe-output-dir "$OUT" --safe-run-id "$RUN_ID"
RC=$?
NPKL=$(find "$OUT/${RUN_ID}/raw_rollouts" -name '*.pkl' 2>/dev/null | wc -l)
echo "[collect] client done rc=$RC pkl_count=$NPKL $(date '+%T')"

# ── self-verify: 최신 pkl 에 phase 라벨이 latent 와 정렬되어 들어갔나 ──────────
NEWPKL=$(find "$OUT/${RUN_ID}/raw_rollouts" -name '*.pkl' 2>/dev/null | head -1)
if [ -n "$NEWPKL" ]; then
  conda run --no-capture-output -n libero_bench python - "$NEWPKL" <<'PYEOF'
import sys, pickle, numpy as np
d = pickle.load(open(sys.argv[1], "rb"))
print("[verify] pkl:", sys.argv[1])
print("[verify] keys:", sorted(d.keys()))
print("[verify] episode_success =", d.get("episode_success"), "| phase_scheme =", d.get("phase_scheme"))
print("[verify] event_order:", d.get("event_order"))
print("[verify] event_steps:", d.get("event_steps"))
pt, fp, hs = d.get("phase_timeline"), d.get("feature_phases"), d.get("hidden_states")
print("[verify] phase_timeline len:", None if pt is None else len(pt))
print("[verify] phases seen (ordered-unique):", None if not pt else list(dict.fromkeys(pt)))
print("[verify] feature_phases (label->count):",
      None if not fp else {p: fp.count(p) for p in dict.fromkeys(fp)})
print("[verify] n_latents:", None if hs is None else len(hs),
      "| hs[0].shape:", None if not hs else np.asarray(hs[0]).shape)
print("[verify] ALIGNED(n_latents==len(feature_phases)):",
      (hs is not None and fp is not None and len(hs) == len(fp)))
PYEOF
fi

echo "$RC" > "$DONE"
echo "[collect] $(date '+%F %T') COLLECT_DONE -> $DONE (rc=$RC, pkl=$NPKL)"
