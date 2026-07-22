#!/usr/bin/env bash
# P2 pool(cross_scene/grand) 선택 rollout — granularity 선택은 conceptor 공유 단위와 일치(Gate1).
# pool conceptor 1개를 소속 scene 들의 select-half 부분집합(scene당 앞 K판, 결정적)에 평가.
# 한 arm = serve 1회 기동으로 전 scene 순회 (steering 은 scene 무관 — collector 인자만 변경).
# usage: SCOPE=cross_scene_bread NPZ_SUB=cross_scene_bread_fit30 SCENES="ppcc_bread ..." \
#        K=8 GPU=5 PA=8474 PB=8475 SINGLE_L=4 bash p2_select_pool.sh
set -uo pipefail
: "${SCOPE:?}" "${NPZ_SUB:?}" "${SCENES:?}" "${K:?}" "${GPU:?}" "${PA:?}" "${PB:?}" "${SINGLE_L:?}"
MYREPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
cd "$MYREPO"
source scripts/safe/groot_n15/robocasa/steer/queue/queue_lib.sh  # REPO 덮임 주의 — MYREPO 만 사용
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
NPZ_DIR="/temporal_vla/outputs/eval/robocasa/groot_n15/exp2_analysis/conceptors/_pools/${NPZ_SUB}/global"
OUT_HOST="$MYREPO/outputs/eval/robocasa/groot_n15/steer_eval_exp2/p2_pool/${SCOPE}"
OUT_CONT="/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval_exp2/p2_pool/${SCOPE}"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
mkdir -p "$OUT_HOST/logs"

eps_of() { python3 -c "import json; print(' '.join(map(str, sorted(json.load(open('outputs/eval/robocasa/groot_n15/exp2_manifests/$1/split.json'))['select_half'])[:$K])))"; }

serve_up() { docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
     --host '*' --port ${1} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
     --steering-npz-dir ${NPZ_DIR} --steering-layers ${2} --steering-beta ${3} --steering-key C_steer \
     > /tmp/exp2pool_${SCOPE}_${1}.log 2>&1 < /dev/null &"; }
serve_health() { local i; for i in $(seq 1 150); do
    docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${1}/health 2>/dev/null" | grep -q '"status":"ok"' && return 0
    sleep 5; done; return 1; }
serve_down() { local p; for p in "$PA" "$PB"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${p}' || true" 2>/dev/null || true; done; sleep 5; }

run_arm() { # tag layers beta
  local tag=$1 layers=$2 beta=$3
  serve_up "$PA" "$layers" "$beta"; serve_up "$PB" "$layers" "$beta"
  serve_health "$PA" || { echo "[$SCOPE/$tag] serve TIMEOUT"; serve_down; return 11; }
  serve_health "$PB" || { echo "[$SCOPE/$tag] serve TIMEOUT"; serve_down; return 11; }
  local pf
  pf=$(docker exec lerobot bash -lc "grep '\[steer-preflight\]' /tmp/exp2pool_${SCOPE}_${PA}.log" || true)
  [ -n "$pf" ] || { echo "[$SCOPE/$tag] preflight 없음 ABORT"; serve_down; return 12; }
  echo "$pf" | grep -q "beta=${beta}" || { echo "[$SCOPE/$tag] preflight beta 불일치"; serve_down; return 12; }
  echo "$pf" >> "$OUT_HOST/logs/${tag}_preflight.log"
  local scene
  for scene in $SCENES; do
    IFS='|' read -r c task env ci seed instr <<<"$(row_of "$scene")"
    local eps; eps=$(eps_of "$scene")
    local w=0 port ep
    for port in "$PA" "$PB"; do
      ( k=0; for ep in $eps; do
          k=$((k+1)); [ $(( (k-1) % 2 )) -eq "$w" ] || continue
          ls "$OUT_HOST/$tag/raw_rollouts/$task/$c/task${ci}--ep${ep}--succ"*.pkl >/dev/null 2>&1 && continue
          docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
            python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
            --vla-server "http://127.0.0.1:${port}" --task "$task" --env-name "$env" \
            --output-dir "$OUT_CONT/$tag/raw_rollouts" --cell-id "$c" --cell-index "$ci" \
            --canonical-instruction "$instr" --episode-start-idx "$ep" --n-episodes 1 \
            --seed "$seed" --inference-seed "$((ep * 1000))" --n-action-steps 5 \
            --max-episode-steps 720 --video-fps 20 --steps-per-render 2 --wait-ready \
            --proximity-phases 2>&1 | grep -E "^wrote|Traceback" || true
        done ) > "$OUT_HOST/logs/${tag}_${scene}_w${w}.log" 2>&1 &
      w=$((w+1))
    done
    wait
  done
  serve_down
  echo "[$SCOPE/$tag] $(date '+%F %T') arm 완료"
}

echo "[$SCOPE] pool P2 시작: single=L$SINGLE_L, K=$K, scenes=($SCENES)"
run_arm "p2_single_L${SINGLE_L}_b01" "$SINGLE_L" 0.1
run_arm "p2_single_L${SINGLE_L}_b03" "$SINGLE_L" 0.3
run_arm "p2_multi_4_8_12_b01" "4,8,12" 0.1
run_arm "p2_multi_4_8_12_b03" "4,8,12" 0.3
echo "[$SCOPE] pool P2 select rollout 완료"
