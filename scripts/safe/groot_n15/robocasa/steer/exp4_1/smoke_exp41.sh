#!/usr/bin/env bash
# exp4-1 smoke 1-4 (24a §8) — kanu, 결정적 실패 episode 1개(bread fit-풀 ep10)로:
#   S1 K=0        → phase_gated_flags 전부 True
#   S2 K=10^9     → 전부 False + A0 대비 사이드카 동일(steps·n_inf·env_step_phases·succ)
#   S3 K=20       → first_true==20, sum==n_inf−20
#   S4 affine 항등 → β=0 serve 가 A0 와 사이드카 동일 (r̂=0 벡터는 로더 단위노름 계약상
#                    표현 불가 — off-phase no-op(S2)와 β=0 산술 항등이 두 identity 경로 커버)
# 각 config = fresh serve(GPU 1개, port 1개) + fresh collector process. 산출물은 smoke 디렉토리.
set -uo pipefail
GPU="${GPU:-4}"; PORT="${PORT:-8600}"
CELL_ID=pq3_ppcc_bread; EP=10; SCEN=400411; INF=1010000
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../../../.." && pwd)"
source "$HERE/cells.env"; cell_params "$CELL_ID"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
NPZ_BASE_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/exp4_1/npz/${CELL_ID}/setM"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/exp4_1/smoke"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/exp4_1/smoke
mkdir -p "$OUT_HOST"
LYR=$(basename "$(ls -d "$NPZ_BASE_HOST"/steer/dit_L* | head -1)" | sed 's/dit_L//')
NPZ_BASE_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/exp4_1/npz/${CELL_ID}/setM

start_serve() {  # $@=steering flags
  docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" lerobot bash -c \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile $PROFILE \
     --host '*' --port $PORT --device cuda $* > /tmp/exp41_smoke_$PORT.log 2>&1"
  local ok=0
  for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    sleep 5
  done
  [ $ok = 1 ] || { echo "[smoke] serve TIMEOUT"; docker exec lerobot tail -20 /tmp/exp41_smoke_$PORT.log; exit 11; }
}
kill_serve() { docker exec lerobot bash -c "pkill -f 'lerobot.py.*--port $PORT'" 2>/dev/null || true; sleep 5; }

run_ep() {  # tag extra_client...
  local tag="$1"; shift
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
    --vla-server "http://127.0.0.1:$PORT" --task "$TASK" --env-name "$ENVN" \
    --output-dir "$OUT_CONT/$tag/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
    --canonical-instruction "$INSTR" --episode-start-idx "$EP" --n-episodes 1 \
    --seed "$SCEN" --inference-seed "$INF" --n-action-steps 5 --max-episode-steps 720 \
    --video-fps 20 --steps-per-render 2 --wait-ready --no-features --proximity-phases "$@" \
    2>&1 | grep -E "wrote|Error|Traceback" || true
}

STEER_FLAGS="--steering-phase-npz-base $NPZ_BASE_CONT --steering-phases steer --steering-layers $LYR --steering-op setpoint --steering-beta 1.0"

echo "== [smoke] A0 (무개입 기준) =="
start_serve; run_ep A0; kill_serve
echo "== [smoke] S1 K=0 =="
start_serve $STEER_FLAGS; run_ep S1_k0 --steer-from-record 0
echo "== [smoke] S3 K=20 (serve 재사용) =="
run_ep S3_k20 --steer-from-record 20
echo "== [smoke] S2 K=10^9 =="
run_ep S2_koff --steer-from-record 1000000000
kill_serve
echo "== [smoke] S4 β=0 =="
start_serve ${STEER_FLAGS/--steering-beta 1.0/--steering-beta 0.0}; run_ep S4_beta0 --steer-from-record 0
kill_serve

python3 - "$OUT_HOST" <<'PYEOF'
import json, sys, glob
root = sys.argv[1]
def side(tag):
    p = glob.glob(f"{root}/{tag}/raw_rollouts/*/*/task*--succ*.json")
    assert p, f"{tag} 사이드카 없음"
    return json.load(open(p[0]))
a0, s1, s3, s2, s4 = (side(t) for t in ("A0","S1_k0","S3_k20","S2_koff","S4_beta0"))
ok = True
def chk(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok &= cond
f1 = s1["phase_gated_flags"]; chk("S1 전부 True", all(f1) and len(f1) == s1["n_inferences"], f"n={len(f1)}")
f3 = s3["phase_gated_flags"]
chk("S3 first_true==20", (True in f3) and f3.index(True) == 20)
chk("S3 sum==n-20", sum(f3) == s3["n_inferences"] - 20)
f2 = s2["phase_gated_flags"]; chk("S2 전부 False", not any(f2))
for k in ("steps","n_inferences","episode_success","first_success_step","env_step_phases"):
    chk(f"S2≡A0 {k}", s2.get(k) == a0.get(k))
    chk(f"S4≡A0 {k}", s4.get(k) == a0.get(k))
print("SMOKE", "ALL PASS" if ok else "FAILURES ABOVE")
raise SystemExit(0 if ok else 1)
PYEOF
