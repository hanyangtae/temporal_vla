#!/usr/bin/env bash
# exp4-2 B3 donor 수집 — OpenDrawer(pq3_drawer_left cell) 성공 full-token L15 donor.
# 24b B3: 타 task donor 를 ppcc 에 주입 — 창 이후 closed-loop 에서 서랍식 행동 잔존 관전.
# 8ep 수집(full-token, ~600MB/ep) → succ 에서 4개 추출.
# 사용: GPU=<idx> PORT=8493 bash collect_b3_donors.sh
set -uo pipefail

GPU="${GPU:?빈 GPU}"
PORT="${PORT:-8493}"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
TASK=OpenDrawer
ENVN="robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
CELL_ID=b3_drawer_left; CELL_INDEX=8; SEED=100001
INSTR="Open the left drawer."
NAS=5; MAXEP=720; CAP="0,2,4,8,10,12,15"; N=8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WT_HOST="$(cd -- "${SCRIPT_DIR}/../../../../../.." && pwd)"
MAIN_HOST="$(cd -- "${WT_HOST}/../../.." && pwd)"
WT_CONT="/temporal_vla/.claude/worktrees/exp4-2-induced-failures"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
B3="${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp42_induced/b3_donors"
mkdir -p "$B3"

to_cont() { echo "${1/#${MAIN_HOST}//temporal_vla}"; }

docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" lerobot bash -lc \
  "cd ${WT_CONT} && setsid nohup python scripts/serve/exp42_serve.py --profile ${PROFILE} \
     --host '*' --port ${PORT} --device cuda --collect \
     --groot-dit-capture-layers ${CAP} --groot-dit-token-pool all_token_full \
     > /tmp/exp42_b3_${PORT}.log 2>&1 < /dev/null &"
ok=0
for _ in $(seq 1 150); do
  curl -s -m 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
  sleep 5
done
[ $ok = 1 ] || { echo ABORT-serve; docker exec lerobot bash -lc "tail -20 /tmp/exp42_b3_${PORT}.log"; exit 11; }

for ep in $(seq 0 $((N - 1))); do
  if ! ls "$B3/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then
    echo "[b3] ep$ep"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python "${WT_CONT}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py" \
      --vla-server "http://127.0.0.1:${PORT}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "$(to_cont "$B3")/raw_rollouts" --cell-id "$CELL_ID" \
      --cell-index "$CELL_INDEX" --canonical-instruction "$INSTR" \
      --episode-start-idx "$ep" --n-episodes 1 --seed "$SEED" \
      --inference-seed "$((ep * 1000))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --proximity-phases 2>&1 | grep -E "^wrote|Error|Traceback" || true
  fi
done
docker exec lerobot bash -lc "pkill -f 'serve/exp42_serve.py.*--port ${PORT}' || true"

# succ 4개 → L15 donor 추출
n_donor=0
for pkl in "$B3/raw_rollouts/${TASK}/${CELL_ID}"/task${CELL_INDEX}--ep*--succ1.pkl; do
  [ -f "$pkl" ] || continue
  [ $n_donor -ge 4 ] && break
  ep=$(basename "$pkl" | sed -E 's/.*--ep([0-9]+)--.*/\1/')
  if [ ! -f "$B3/donor_ep${ep}_L15.npz" ]; then
    docker exec lerobot python \
      "${WT_CONT}/scripts/safe/groot_n15/robocasa/steer/patchceil/extract_donor_npz.py" \
      --pkl "$(to_cont "$pkl")" --out "$(to_cont "$B3")/donor_ep${ep}_L15.npz" \
      --layers 15 --cap "$CAP"
  fi
  n_donor=$((n_donor + 1))
done
echo "[b3] DONE: succ=$(ls "$B3/raw_rollouts/${TASK}/${CELL_ID}"/*succ1.pkl 2>/dev/null | wc -l)/$N donors=$(ls "$B3"/donor_*.npz 2>/dev/null | wc -l)"
