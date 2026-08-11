#!/bin/bash
# 재학습 SAFE(ours)로 adaptive arm 재평가 — 3 seed × 우리 α top-3 (9 lane).
# 사용: bash rerun_adaptive_ours.sh "<gpu 목록>"
set -uo pipefail

GPUS=${1:?"gpu list"}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="$REPO_ROOT/scripts/rl2_vla/stage1_simpler/run_arm.sh"
EXP="$REPO_ROOT/RL2-VLA/experiments"
QUEUE="$EXP/adaptive_ours_queue.txt"
LOCK="$EXP/adaptive_ours_queue.lock"

export SAFE_DIR_OVERRIDE="$REPO_ROOT/RL2-VLA/third_party/SAFE/logs/open_pizero-bridge-lstm-ours_cpTrue/20260807/123421"
export LANE_TAG="ours"

# rl2_cp_alphas_combined_ours.json 의 seed별 top-3
cat > "$QUEUE" <<'EOF'
42 0.2
0 0.2
7 0.2
42 0.3
0 0.15
7 0.25
42 0.35
0 0.25
7 0.3
EOF

pop_job() {
    flock "$LOCK" bash -c '
        q="$1"; [ -s "$q" ] || exit 1
        head -1 "$q"; tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
    ' _ "$QUEUE"
}

worker() {
    local gpu=$1
    while true; do
        local job; job=$(pop_job) || { echo "[gpu$gpu] 큐 소진"; return 0; }
        read -r seed alpha <<< "$job"
        local dir="$EXP/stage1b_OOD_seed${seed}/adaptive_a${alpha}_ours"
        [ -f "$dir/DONE" ] && { echo "[gpu$gpu] skip s$seed a$alpha"; continue; }
        echo "[gpu$gpu] START adaptive_ours s$seed a$alpha $(date +%H:%M)"
        bash "$RUNNER" adaptive "$gpu" OOD "$seed" 50 "$alpha" \
            > "$EXP/ours_adaptive_s${seed}_a${alpha}.log" 2>&1
        echo "[gpu$gpu] END   adaptive_ours s$seed a$alpha rc=$? $(date +%H:%M)"
    done
}

for g in $GPUS; do worker "$g" & done
wait
touch "$EXP/ADAPTIVE_OURS_DONE"
echo "[rerun_adaptive_ours] 전체 완료 $(date +%F\ %H:%M)"
