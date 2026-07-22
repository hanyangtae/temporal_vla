#!/usr/bin/env bash
# exp4-1 mixer(OpenStandMixerHead) C0 수집 — kanu (신규 cell exp41_mixer).
#   PHASE=fit  : feasible seed 앞 30개, 캡처 ON(full-token, exp3 fit 규약) — ep 0..29
#                episode 완료 즉시 승준 HDD 직송 + 로컬 pkl 삭제 (디스크 압박·수집 직송 관례)
#   PHASE=eval : 다음 30개, 캡처 OFF(--no-features) — ep 30..59
# feasibility 필터: mixer_feasibility.json 의 feasible=true seed 만 (공유문서 §5).
# env: GPU(기본 4) PORTS="8600 8601" PHASE=fit|eval
set -uo pipefail
GPU="${GPU:-4}"; PORTS=(${PORTS:-8600 8601}); NW=${#PORTS[@]}
PHASE="${PHASE:?fit|eval}"
CELL_ID=exp41_mixer; TASK=OpenStandMixerHead
ENVN=robocasa_panda_omron/OpenStandMixerHead_PandaOmron_Env
CELL_INDEX=0; INSTR="Open the stand mixer head."
NAS=5; MAXEP=720; STRIDE=1000; CAP="0,2,4,8,10,12,15"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../../../.." && pwd)"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
FEAS="$REPO_ROOT/outputs/eval/robocasa/groot_n15/exp4_1/mixer_feasibility.json"
OUT_HOST="$REPO_ROOT/outputs/eval/robocasa/groot_n15/exp4_1/mixer_c0/$PHASE"
OUT_CONT="/temporal_vla/outputs/eval/robocasa/groot_n15/exp4_1/mixer_c0/$PHASE"
SHIP_DEST="kimseungjun@166.104.146.37"
SHIP_PORT=11112
SHIP_ROOT="datasets/temporal_vla_outputs/eval/robocasa/groot_n15/exp41_mixer/$PHASE"
LOGDIR="$OUT_HOST/logs"; mkdir -p "$LOGDIR"

mapfile -t SEEDS < <(python3 -c "
import json
rows = json.load(open('$FEAS'))
feas = [r['seed'] for r in rows if r['feasible']]
lo, hi = (0, 30) if '$PHASE' == 'fit' else (30, 60)
print('\n'.join(str(s) for s in feas[lo:hi]))")
EP0=$([ "$PHASE" = fit ] && echo 0 || echo 30)
echo "[$CELL_ID/$PHASE] seeds=${#SEEDS[@]} ep0=$EP0 gpu=$GPU ports=${PORTS[*]}"

if [ "$PHASE" = fit ]; then
  SERVE_EXTRA="--collect --capture-vl --groot-dit-capture-layers $CAP --groot-dit-token-pool all_token_full"
  CLIENT_EXTRA=""
else
  SERVE_EXTRA=""
  CLIENT_EXTRA="--no-features"
fi

for i in $(seq 0 $((NW-1))); do
  docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" lerobot bash -c \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile $PROFILE \
     --host '*' --port ${PORTS[$i]} --device cuda $SERVE_EXTRA \
     > /tmp/exp41_mixer_${PORTS[$i]}.log 2>&1"
done
for port in "${PORTS[@]}"; do
  ok=0; for _ in $(seq 1 150); do
    curl -s -m 3 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    sleep 5
  done
  echo "[$CELL_ID/$PHASE] serve $port health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  [ $ok = 1 ] || exit 11
done

ship_ep() {  # ep 완료분 승준 직송 (+ 검증 후 로컬 pkl 삭제 — json/mp4 는 주석용 보존)
  local d="$OUT_HOST/raw_rollouts/$TASK/$CELL_ID" f
  ssh -p $SHIP_PORT -o BatchMode=yes "$SHIP_DEST" "mkdir -p ~/$SHIP_ROOT" 2>/dev/null || true
  for f in "$d"/task${CELL_INDEX}--ep$1--succ*.*; do
    [ -e "$f" ] || continue
    rsync -a -e "ssh -p $SHIP_PORT" "$f" "$SHIP_DEST:~/$SHIP_ROOT/" || { echo "[ship FAIL] $f"; return 1; }
  done
  for f in "$d"/task${CELL_INDEX}--ep$1--succ*.pkl; do
    [ -e "$f" ] || continue
    rsz=$(ssh -p $SHIP_PORT -o BatchMode=yes "$SHIP_DEST" "stat -c%s ~/$SHIP_ROOT/$(basename "$f")" 2>/dev/null)
    lsz=$(stat -c%s "$f")
    [ "$rsz" = "$lsz" ] && rm "$f" || echo "[ship 검증 실패 — pkl 보존] $f ($lsz vs $rsz)"
  done
}

run_w() {
  local wid=$1 port=$2 idx ep seed
  for idx in "${!SEEDS[@]}"; do
    [ $((idx % NW)) -eq "$wid" ] || continue
    ep=$((EP0 + idx)); seed="${SEEDS[$idx]}"
    if ls "$OUT_HOST/raw_rollouts/$TASK/$CELL_ID/task${CELL_INDEX}--ep${ep}--succ"*.json >/dev/null 2>&1; then
      continue
    fi
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:$port" --task "$TASK" --env-name "$ENVN" \
      --output-dir "$OUT_CONT/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "$seed" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --proximity-phases $CLIENT_EXTRA 2>&1 | grep -E "wrote|Error|Traceback" || true
    [ "$PHASE" = fit ] && ship_ep "$ep"
  done
}
for wid in $(seq 0 $((NW-1))); do run_w "$wid" "${PORTS[$wid]}" > "$LOGDIR/w${wid}.log" 2>&1 & done
wait
for port in "${PORTS[@]}"; do
  docker exec lerobot bash -c "pkill -f 'lerobot.py.*--port $port'" 2>/dev/null || true
done
d="$OUT_HOST/raw_rollouts/$TASK/$CELL_ID"
s=$(ls "$d"/*succ1.json 2>/dev/null | wc -l); f=$(ls "$d"/*succ0.json 2>/dev/null | wc -l)
echo "[$CELL_ID/$PHASE] done succ=$s fail=$f total=$((s+f))/${#SEEDS[@]}"
touch "$LOGDIR/MIXER_C0_${PHASE}_DONE"
