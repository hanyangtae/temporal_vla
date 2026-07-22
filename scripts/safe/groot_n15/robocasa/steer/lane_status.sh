#!/usr/bin/env bash
# GPU 차선 진행률 한눈에 (read-only). 사용: bash scripts/safe/groot_n15/robocasa/steer/lane_status.sh
# 또는 자동 갱신: watch -n 30 bash scripts/safe/groot_n15/robocasa/steer/lane_status.sh
R=/home/dongkyu/pkt_ws/temporal_vla
E=$R/outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread
B=raw_rollouts/PickPlaceCounterToCabinet/ppcc_bread
echo "=== $(date '+%T') ==="

echo "--- [GPU0,1] layer ablation (각 30) ---"
for t in mlg_all7 mlg_L4_8 mlg_L8_12 mlg_L8only mlg_L12only; do
  d="$E/steered_${t}/$B"
  n=$(ls $d/*.pkl 2>/dev/null | wc -l)
  if [ -f "$E/sr_result_${t}.tsv" ]; then echo "  $t: 완료 → $(tail -1 $E/sr_result_${t}.tsv | cut -f5)"
  elif [ "$n" -gt 0 ]; then echo "  $t: ${n}/30 진행중"
  else echo "  $t: 대기"; fi
done

echo "--- [GPU2,3] 확증 n=90 (ep30-89 추가) ---"
for c in baseline steered_ml_global; do
  n=$(ls $E/$c/$B/*.pkl 2>/dev/null | wc -l)
  echo "  $c: ${n}/90"
done
[ -f "$E/sr_result_confirm90.tsv" ] && { echo "  완료:"; cat "$E/sr_result_confirm90.tsv" | tail -2 | sed 's/^/    /'; }

echo "--- [GPU4,5] strict steer 큐 (각 30) ---"
for t in st_pregrasp_ml st_reach_ml st_global_ml st_pregrasp_L4 st_reach_L4; do
  d="$E/steered_${t}/$B"
  n=$(ls $d/*.pkl 2>/dev/null | wc -l)
  if [ -f "$E/sr_result_${t}.tsv" ]; then echo "  $t: 완료 → $(tail -1 $E/sr_result_${t}.tsv | cut -f5)"
  elif [ "$n" -gt 0 ]; then echo "  $t: ${n}/30 진행중"
  else echo "  $t: 대기"; fi
done

echo "--- sentinel/GPU ---"
for s in ABLATION_QUEUE_DONE CONFIRM90_DONE STRICT_QUEUE_DONE; do
  [ -f "$E/logs/$s" ] && echo "  $s ✓"
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F, '{printf "  GPU%s:%s", $1, $2} END{print ""}'
