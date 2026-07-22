#!/usr/bin/env bash
# patchceil rollout 러너 — arm_plan.tsv 를 worktree patch serve 로 실행.
#
# STAGE=anchors : A2/A3 (캡처 ON serve 1개 — emitted actions 를 pkl 로 남겨 수치 대조)
# STAGE=main    : nopatch/donor/placebo/shuffle (캡처 OFF --no-features, serve N개)
#
# serve = worktree(exp/patching-ceiling) lerobot.py — exp3(구 pq3) 트리 무접촉. 클라이언트는
# main tree http_feature_collect.py 그대로 (patch 트리거는 serve record 카운터).
# 러너 절차(rollout 1개): /patch_arm(또는 disarm) → collector 1ep → /patch_status 저장.
#
# 사용 (GPU1 사용자 승인):
#   STAGE=anchors GPUS_L="1" PORTS_L="8497" bash run_patch_rollouts.sh
#   STAGE=main GPUS_L="1 1" PORTS_L="8497 8498" bash run_patch_rollouts.sh
set -uo pipefail

STAGE="${STAGE:?anchors|main}"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
CELL_INDEX=5
INSTR="Pick the bread from the counter and place it in the cabinet."
declare -A SEEDS=([ppcc_bread_s300033]=300033 [ppcc_bread_s400020]=400020)

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
GROOT="outputs/eval/robocasa/groot_n15/patchceil"
WORKTREE_CONT="/temporal_vla/.claude/worktrees/patching-ceiling"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:?}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}
CELLS=(ppcc_bread_s300033 ppcc_bread_s400020)

if [ "$STAGE" = "anchors" ]; then
  SERVE_EXTRA="--collect --groot-dit-capture-layers 15 --patch-allow-collect"
  ARM_FILTER='^anchor_'
  COLLECT_EXTRA=""
else
  SERVE_EXTRA=""
  ARM_FILTER='^(nopatch|donor|placebo|shuffle)$'
  COLLECT_EXTRA="--no-features --expect-chunk-len 16"
fi

start_serves() {
  for i in $(seq 0 $((NW-1))); do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd ${WORKTREE_CONT} && setsid nohup python scripts/serve/patchceil_serve.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda \
         --patch-layers 15 --patch-token-select all ${SERVE_EXTRA} \
         > /tmp/patchceil_${STAGE}_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[${STAGE}] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || exit 11
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/patchceil_${STAGE}_${port}.log"; then
      echo "[${STAGE}] ABORT ${port}"; docker exec lerobot bash -lc "tail -20 /tmp/patchceil_${STAGE}_${port}.log"; exit 11
    fi
    curl -s -m 3 "http://127.0.0.1:${port}/health" | grep -o '"patch":{[^}]*"layers":\[15\][^}]*' >/dev/null \
      || { echo "[${STAGE}] ABORT ${port}: /health 에 patch 지문 없음 (worktree serve 아님?)"; exit 12; }
  done
}
# patchceil_serve.py = lerobot.py 심링크 — cmdline 을 분리해 exp3 등 외부의
# 'serve/lerobot.py' 광범위 pkill 에 면역 (2026-07-16 17:23 serve 피격 사고 재발 방지)
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/patchceil_serve.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }
trap kill_serves EXIT

run_worker() {  # wid port plan_cell
  local wid=$1 port=$2 plan_cell=$3
  local plan="${REPO_ROOT}/${GROOT}/${plan_cell}/arm_plan.tsv"
  while IFS=$'\t' read -r cell target_ep arm donor_ep npz t0 donor_start patch_len inf_seed tag; do
    tag="${tag//$'\r'/}"   # python csv 기본 \r\n 개행 방어 (JSON 안 CR → 422)
    [ "$cell" = "cell" ] && continue
    echo "$arm" | grep -qE "$ARM_FILTER" || continue
    # worker 배정은 target_ep 기준 — 행 순번 modulo 는 arm 주기(4)와 공명해 한 arm 이
    # 한 worker 에 몰린다 (nopatch 재실행 시 병렬화 0 되는 함정)
    [ $((target_ep % NW)) -eq "$wid" ] || continue
    local out_host="${REPO_ROOT}/${GROOT}/${plan_cell}/rollouts/${arm}"
    local out_cont="/temporal_vla/${GROOT}/${plan_cell}/rollouts/${arm}"
    mkdir -p "$out_host"
    # resume-safe: 판정 산출물 존재 시 skip (main=json sidecar, anchors=pkl)
    if ls "${out_host}/raw_rollouts/${TASK}/${cell}/task${CELL_INDEX}--ep${target_ep}--succ"* >/dev/null 2>&1; then continue; fi
    # 1) arm / disarm
    if [ "$arm" = "nopatch" ]; then
      r=$(curl -s -m 10 -X POST "http://127.0.0.1:${port}/patch_disarm") || true
      echo "$r" | grep -q '"ok":true' || { echo "[${tag}] DISARM_FAIL: $r"; continue; }
    else
      r=$(curl -s -m 30 -X POST "http://127.0.0.1:${port}/patch_arm" -H 'Content-Type: application/json' \
        -d "{\"npz\":\"${npz}\",\"start_record\":${t0},\"donor_start\":${donor_start},\"patch_len\":${patch_len},\"tag\":\"${tag}\"}") || true
      echo "$r" | grep -q '"ok":true' || { echo "[${tag}] ARM_FAIL: $r"; continue; }
    fi
    # 2) rollout (에피소드당 fresh 프로세스, scenario 는 row 의 cell 기준)
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "${out_cont}/raw_rollouts" --cell-id "$cell" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$target_ep" --n-episodes 1 \
      --seed "${SEEDS[$cell]}" --inference-seed "$inf_seed" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 \
      --wait-ready --proximity-phases $COLLECT_EXTRA 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
    # 3) 발화 창 검증용 status 사이드카
    curl -s -m 10 "http://127.0.0.1:${port}/patch_status" > "${out_host}/status_ep${target_ep}.json" || true
  done < "$plan"
}

start_serves
for CELL in "${CELLS[@]}"; do
  echo "[${STAGE}] $(date '+%F %T') plan=${CELL}"
  LOGDIR="${REPO_ROOT}/${GROOT}/${CELL}/rollouts/logs"; mkdir -p "$LOGDIR"
  for wid in $(seq 0 $((NW-1))); do
    run_worker "$wid" "${PORTS[$wid]}" "$CELL" > "${LOGDIR}/${STAGE}_w${wid}.log" 2>&1 &
  done
  wait
done
kill_serves; trap - EXIT
echo "[${STAGE}] $(date '+%F %T') STAGE_DONE"
