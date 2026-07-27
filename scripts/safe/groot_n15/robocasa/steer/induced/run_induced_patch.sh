#!/usr/bin/env bash
# exp4-2 Track I P0 러너 — arm_plan.tsv 를 캡처 ON patch serve 로 실행 (run_patch_rollouts 개조).
# 유도실패 rollout 이 fit 의 fail 클래스이므로 **캡처 ON**(--patch-allow-collect) 이 표준.
# PATHWAY=dit|vl 로 serve 구성 분리 (plan 의 pathway 열 필터).
# 사용: PATHWAY=dit GPUS_L="4 4" PORTS_L="8474 8475" PLAN=<arm_plan.tsv> bash run_induced_patch.sh
set -uo pipefail

PATHWAY="${PATHWAY:?dit|vl}"
PLAN="${PLAN:?arm_plan.tsv 경로}"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
TASK=PickPlaceCounterToCabinet
ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
CELL_INDEX=5; CELL_ID=ppcc_bread
INSTR="Pick the bread from the counter and place it in the cabinet."
NAS=5; MAXEP=720; SEED=100084
CAP="0,2,4,8,10,12,15"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WT_HOST="$(cd -- "${SCRIPT_DIR}/../../../../../.." && pwd)"
MAIN_HOST="$(cd -- "${WT_HOST}/../../.." && pwd)"
WT_CONT="/temporal_vla/.claude/worktrees/exp4-2-induced-failures"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
P0="${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp42_induced/p0"
LOGDIR="${P0}/logs"; mkdir -p "$LOGDIR"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}

to_cont() { echo "${1/#${MAIN_HOST}//temporal_vla}"; }

if [ "$PATHWAY" = "dit" ]; then
  PATCH_ARGS="--patch-layers 15 --patch-token-select all"
else
  PATCH_ARGS="--patch-pathway vl"
fi
# 캡처 kind 는 fit 호환 규약 (24b §2.1): action_token_mean + vlln_mean.
# post_vl_sa_full 은 B1 donor 추출 전용 (record당 3.4MB — 본 수집에 쓰면 디스크 폭증).
SERVE_EXTRA="${PATCH_ARGS} --patch-allow-collect --collect --capture-vl \
  --groot-dit-capture-layers ${CAP}"

start_serves() {
  for i in $(seq 0 $((NW - 1))); do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd ${WT_CONT} && setsid nohup python scripts/serve/exp42_serve.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda ${SERVE_EXTRA} \
         > /tmp/exp42_patch_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0
    for _ in $(seq 1 150); do
      st=$(curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[patch-${PATHWAY}] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || { docker exec lerobot bash -lc "tail -30 /tmp/exp42_patch_${port}.log" || true; exit 11; }
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/exp42_serve.py.*--port ${port}' || true" 2>/dev/null || true; done; }
trap kill_serves EXIT

run_worker() {  # wid port
  local wid=$1 port=$2
  while IFS=$'\t' read -r variant pathway ep_idx inf npz t0 d0 plen tag; do
    tag="${tag//$'\r'/}"
    [ "$variant" = "variant" ] && continue
    [ "$pathway" = "$PATHWAY" ] || continue
    [ $((ep_idx % NW)) -eq "$wid" ] || continue
    local out_host="${P0}/trackI/${variant}"
    mkdir -p "$out_host"
    if ls "${out_host}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep_idx}--succ"*.pkl >/dev/null 2>&1; then
      continue
    fi
    r=$(curl -s -m 30 -X POST "http://127.0.0.1:${port}/patch_arm" -H 'Content-Type: application/json' \
      -d "{\"npz\":\"$(to_cont "$npz")\",\"start_record\":${t0},\"donor_start\":${d0},\"patch_len\":${plen},\"tag\":\"${tag}\"}") || true
    echo "$r" | grep -q '"ok":true' || { echo "[${tag}] ARM_FAIL: $r"; continue; }
    echo "[w${wid}] ${tag}"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python "${WT_CONT}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py" \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "$(to_cont "$out_host")/raw_rollouts" --cell-id "$CELL_ID" \
      --cell-index "$CELL_INDEX" --canonical-instruction "$INSTR" \
      --episode-start-idx "$ep_idx" --n-episodes 1 --seed "$SEED" --inference-seed "$inf" \
      --n-action-steps "$NAS" --max-episode-steps "$MAXEP" \
      --video-fps 20 --steps-per-render 2 --wait-ready --proximity-phases 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
    curl -s -m 10 "http://127.0.0.1:${port}/patch_status" > "${out_host}/status_ep${ep_idx}.json" || true
  done < "$PLAN"
}

start_serves
echo "[patch-${PATHWAY}] $(date '+%F %T') plan=${PLAN}"
for wid in $(seq 0 $((NW - 1))); do
  run_worker "$wid" "${PORTS[$wid]}" > "${LOGDIR}/trackI_${PATHWAY}_w${wid}.log" 2>&1 &
done
wait
kill_serves; trap - EXIT
echo "[patch-${PATHWAY}] $(date '+%F %T') DONE"
touch "${LOGDIR}/TRACKI_${PATHWAY}_DONE"
