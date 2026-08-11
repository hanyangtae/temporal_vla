#!/bin/bash
# 논문 프로토콜 풀사이즈 재현 (Stage 1c) — 3 seed × alpha top-3 × 4 arm.
#
# 사용: bash run_full_repro.sh "<gpu 목록>"   예) bash run_full_repro.sh "0 1 2 3"
#
# GPU 하나당 워커 1개가 FIFO 큐에서 lane 을 꺼내 순차 실행한다(flock 로 배타).
# 이미 완료된 lane(DONE sentinel)은 건너뛰므로 중단 후 재실행해도 안전.
set -uo pipefail

GPUS=${1:?"gpu list, e.g. \"0 1 2 3\""}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="$REPO_ROOT/scripts/rl2_vla/stage1_simpler/run_arm.sh"
EXP="$REPO_ROOT/RL2-VLA/experiments"
QUEUE="$EXP/full_repro_queue.txt"
LOCK="$EXP/full_repro_queue.lock"
mkdir -p "$EXP"

# lane 정의: "<arm> <seed> [alpha]"  — seed42 alpha 0.1 은 Stage 1b 에서 완료됨
cat > "$QUEUE" <<'EOF'
always 0
always 7
adaptive 0 0.1
adaptive 7 0.2
adaptive 42 0.15
adaptive 42 0.05
adaptive 0 0.15
adaptive 0 0.2
adaptive 7 0.25
adaptive 7 0.3
rephrase 0
rephrase 7
vanilla 42
vanilla 0
vanilla 7
EOF

pop_job() {  # 큐 맨 위 한 줄을 배타적으로 꺼낸다
    flock "$LOCK" bash -c '
        q="$1"; [ -s "$q" ] || exit 1
        head -1 "$q"; tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
    ' _ "$QUEUE"
}

worker() {
    local gpu=$1
    while true; do
        local job; job=$(pop_job) || { echo "[gpu$gpu] 큐 소진"; return 0; }
        [ -z "$job" ] && return 0
        read -r arm seed alpha <<< "$job"
        local lane="$arm"; [ -n "${alpha:-}" ] && lane="${arm}_a${alpha}"
        local dir="$EXP/stage1b_OOD_seed${seed}/${lane}"
        if [ -f "$dir/DONE" ]; then echo "[gpu$gpu] skip $lane seed$seed (완료됨)"; continue; fi
        echo "[gpu$gpu] START $lane seed$seed $(date +%H:%M)"
        bash "$RUNNER" "$arm" "$gpu" OOD "$seed" 50 ${alpha:-} \
            > "$EXP/full_${arm}_s${seed}${alpha:+_a$alpha}.log" 2>&1
        echo "[gpu$gpu] END   $lane seed$seed rc=$? $(date +%H:%M)"
    done
}

for g in $GPUS; do worker "$g" & done
wait
touch "$EXP/FULL_REPRO_DONE"
echo "[run_full_repro] 전체 완료 $(date +%F\ %H:%M)"
