#!/usr/bin/env bash
# Cross-instruction action-phase conceptor 체인 (확정판 라운드 종료 후 자동 실행).
#   fit: 3 instruction 데이터 pooled, 총 30/60/90 rollout(각 cell 10/20/30, ep 앞에서부터)
#   eval: 각 cell ep60-119 gated arm만 (baseline은 확정판 것 재사용), SUF=cr{30,60,90}
# GPU 예산: cell당 GPU 1개 × serve 2 = 총 6 serve (사용자 상한).
set -uo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
P='steer/held'"out_round"
while pgrep -f "$P" >/dev/null; do sleep 120; done
echo "[cross] $(date '+%F %T') 확정판 종료 감지 — cross fit 시작"

SRC=outputs/eval/robocasa/groot_n15/phase_event_strict/raw_rollouts_fit60
BREAD_SRC=outputs/eval/robocasa/groot_n15/phase_event_strict/raw_rollouts   # bread fit60은 원본 트리
FITPY=scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py
AN=outputs/eval/robocasa/groot_n15/phase_event_strict/analysis

for N in 30 60 90; do
  PER=$((N / 3))
  D=outputs/eval/robocasa/groot_n15/phase_event_strict/raw_rollouts_cross${N}/Cross/all
  mkdir -p "$D"
  for spec in "ppcc_bread PickPlaceCounterToCabinet 5 $BREAD_SRC" \
              "ppcs_apple PickPlaceCounterToStove 1 $SRC" \
              "ppcc_potato PickPlaceCounterToCabinet 4 $SRC"; do
    set -- $spec; cell=$1; T=$2; ci=$3; root=$4
    for ep in $(seq 0 $((PER - 1))); do
      src=$(ls $(pwd)/$root/$T/$cell/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
      [ -n "$src" ] && ln -sf "$src" "$D/"
    done
  done
  echo "[cross] fit${N}: $(ls $D/*.pkl 2>/dev/null | wc -l)판 (succ=$(ls $D/*succ1.pkl 2>/dev/null|wc -l))"
  OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 ~/miniconda3/envs/libero_bench/bin/python \
    "$FITPY" --run-dir outputs/eval/robocasa/groot_n15/phase_event_strict/raw_rollouts_cross${N} \
    --cell Cross/all --groups "global,reach-to-object,pre-grasp,transport,pre-place,insert-settle" \
    --carve-window 3 --out-dir "$AN/conceptor_steering_n15_cross${N}/all" 2>&1 | grep -E '^\[cell|\[done'
  # 런처 BASE=NPZ_ROOT/CELL_ID 계약 맞춤: cell명 심링크
  for cell in ppcc_bread ppcs_apple ppcc_potato; do
    ln -sfn "$(pwd)/$AN/conceptor_steering_n15_cross${N}/all" "$AN/conceptor_steering_n15_cross${N}/$cell"
  done
done

R=$(ls scripts/safe/groot_n15/robocasa/steer/heldout*_round_cell.sh)
run_cell_sizes() {  # cell task envn ci seed instr gpu portA portB
  local cell=$1 task=$2 envn=$3 ci=$4 seed=$5 instr=$6 gpu=$7 pA=$8 pB=$9
  for N in 30 60 90; do
    echo "[cross] $(date '+%F %T') $cell gated cr${N}"
    CELL_ID=$cell TASK=$task ENVN=$envn CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
      GPUS_L="$gpu $gpu" PORTS_L="$pA $pB" EP0=60 EP1=119 \
      SUF=cr${N} ARMS="gated" \
      NPZ_ROOT=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_strict/analysis/conceptor_steering_n15_cross${N} \
      bash "$R" >> outputs/eval/robocasa/groot_n15/steer_eval/${cell}/cross_rounds.log 2>&1
  done
}
run_cell_sizes ppcc_bread PickPlaceCounterToCabinet robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env 5 100084 "Pick the bread from the counter and place it in the cabinet." 0 8480 8481 &
run_cell_sizes ppcs_apple PickPlaceCounterToStove robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env 1 100074 "Pick the apple from the plate and place it in the pan." 4 8470 8471 &
run_cell_sizes ppcc_potato PickPlaceCounterToCabinet robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env 4 200019 "Pick the potato from the counter and place it in the cabinet." 6 8472 8473 &
wait
echo "[cross] $(date '+%F %T') CROSS_ALL_DONE"
