# α×β sweep (bread 100084, global perm): FINAL2 완료 후 자동 실행
cd /home/dongkyu/pkt_ws/temporal_vla
while [ ! -f outputs/eval/robocasa/groot_n15/FINAL2_ALL_DONE ]; do sleep 300; done
PYB=~/miniconda3/envs/libero_bench/bin/python
AN=outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
# α=3 재료 fit (α 0.3/1 은 기존 NPZ에 있음)
$PYB scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py \
  --run-dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts \
  --cell PickPlaceCounterToCabinet/ppcc_bread --groups global --alphas 3 \
  --carve-window 0 --out-dir "$AN/final_a3/ppcc_bread" > "$AN/final_a3.fitlog" 2>&1
R=$(ls scripts/safe/groot_n15/robocasa/steer/heldout*_round_cell.sh)
run_arm() {  # alpha beta npzsub
  CELL_ID=ppcc_bread TASK=PickPlaceCounterToCabinet \
  ENVN=robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env \
  CELL_INDEX=5 SEED=100084 INSTR="Pick the bread from the counter and place it in the cabinet." \
  GPUS_L="4 4" PORTS_L="8470 8471" EP0=60 EP1=119 PROX=1 \
  SUF="a${1/./}b${2/./}" ARMS="perm" STEER_ALPHA=$1 STEER_BETA=$2 \
  NPZ_ROOT=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/$3 \
  bash "$R" >> outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/ab_sweep.log 2>&1
}
for B in 0.1 0.3 0.5; do
  run_arm 0.3 $B final_ps60   # α0.3 (β0.3은 기존과 중복이나 SUF 달라 재확인용 — skip하려면 기존 결과 사용)
  run_arm 1   $B final_ps60   # 선택 α=1
  run_arm 3   $B final_a3     # α=3
done
echo "AB_SWEEP_DONE $(date '+%F %T')"
