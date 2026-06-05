#!/bin/bash
# COAST conceptor steering eval 오케스트레이션 (호스트에서 docker exec).
# config 별로 groot 컨테이너의 feature_server 를 (해당 NPZ/β로) 재기동하고,
# robocasa 컨테이너에서 ZMQ eval client 를 돌려 SR 을 기록한다.
#
# baseline = β=0 (M=I, steering 무효). steered = β>0.
# 같은 eval 파라미터로 돌리므로 baseline 대비 ΔSR 은 내부적으로 일관(collection
# horizon 과 무관). 결과: /tmp/steer_eval_results.tsv
set -u

REPO=/temporal_vla
NPZ_ROOT="$REPO/outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/conceptor_steering"
TASK_DIR="task_9_PickPlaceCounterToStove"
ENV_NAME="robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env"
NEP=${NEP:-20}; NENV=${NENV:-2}; NACT=${NACT:-8}; MAXS=${MAXS:-720}
PORT=5557
RESULTS=/tmp/steer_eval_results.tsv

echo -e "config\tbeta\tmode\tsuccess_rate\tresults" > "$RESULTS"

wait_server_ready() {
  local log=$1
  for i in $(seq 1 40); do
    if docker exec groot bash -lc "grep -qi '\[steering\]' $log 2>/dev/null"; then return 0; fi
    if docker exec groot bash -lc "grep -qiE 'Error|Traceback' $log 2>/dev/null"; then
      echo "  !! SERVER ERROR (see $log)"; docker exec groot bash -lc "tail -5 $log"; return 1
    fi
    sleep 10
  done
  echo "  !! SERVER TIMEOUT"; return 1
}

run_config() {
  local name=$1 mode=$2 beta=$3
  local npz="$NPZ_ROOT/$mode/$TASK_DIR/conceptors.npz"
  local fslog="/tmp/fs_${name}.log"
  local evlog="/tmp/eval_${name}.log"
  echo "==================== [$name] mode=$mode beta=$beta ===================="

  docker exec groot bash -lc "pkill -f feature_server.py 2>/dev/null; sleep 4" 2>/dev/null
  docker exec -d -e CUDA_VISIBLE_DEVICES=0 -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
    "cd $REPO && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
       --profile configs/checkpoints/groot__robocasa365_ckpt120000.yaml \
       --host '*' --port $PORT --device cuda --feature-slice valid \
       --steering-npz $npz --steering-beta $beta > $fslog 2>&1"

  if ! wait_server_ready "$fslog"; then
    echo -e "${name}\t${beta}\t${mode}\tSERVER_FAIL\t-" >> "$RESULTS"; return
  fi
  sleep 10  # server.run() bind grace

  docker exec -e POLICY_CLIENT_HOST=127.0.0.1 -e PORT=$PORT robocasa bash -lc \
    "bash $REPO/scripts/eval/groot_robocasa.sh client $ENV_NAME $NEP $NENV $NACT $MAXS > $evlog 2>&1"

  local sr res
  sr=$(docker exec robocasa bash -lc "grep -i 'success rate' $evlog | tail -1 | sed 's/.*success rate:[[:space:]]*//'")
  res=$(docker exec robocasa bash -lc "grep -i 'results:' $evlog | tail -1")
  echo -e "${name}\t${beta}\t${mode}\t${sr}\t${res}" >> "$RESULTS"
  echo "  [$name] SR=${sr}"
}

run_config baseline      coast          0.0
run_config coast_b03     coast          0.3
run_config epmean_b03    episode_mean   0.3
run_config trunc18_b03   truncated_w18  0.3

docker exec groot bash -lc "pkill -f feature_server.py 2>/dev/null" 2>/dev/null
echo "==================== ALL DONE ===================="
cat "$RESULTS"
