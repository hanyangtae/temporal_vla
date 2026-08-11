#!/bin/bash
# A-S2: apple per-seed(ps15/30/60 ×4) + xa pool + gx(8-cell) pool fit → NPZ srv50(구 w2/worker2) push → NPZ_READY.
# master_final_scene2.sh 의 fit_dir/ps_fits/pool_fit 로직 복제 (fit 무결성 게이트 포함).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/queue_lib.sh"
cd "$REPO"
# libero_bench env 소실(2026-07-08)로 lerobot 컨테이너 python 사용 (torch·numpy 보장, /temporal_vla=repo)
PY="docker exec -e OMP_NUM_THREADS=12 -e OPENBLAS_NUM_THREADS=12 -w /temporal_vla lerobot python"
FITPY=scripts/fit/fit_phase_conceptor.py
FIT_GROUPS="global,reach-to-object,grasp,transport,place,insert-settle"

fit_dir() {  # rundir cellpath outdir
  [ -f "$3/fit_summary.json" ] && return 0
  mkdir -p "$3"
  $PY $FITPY --run-dir "$1" --cell "$2" \
    --groups "$FIT_GROUPS" --carve-window 0 --out-dir "$3" > "$3.fitlog" 2>&1 || true
  grep -m1 '\[done\]' "$3.fitlog" >/dev/null || { echo "[FIT FAILED] $3"; tail -5 "$3.fitlog"; exit 2; }
}
ps_fits() {
  local cell c task env ci seed instr N ep src SUBD
  for cell in "$@"; do
    IFS='|' read -r c task env ci seed instr <<<"$(row_of $cell)"
    for N in 15 30 60; do
      SUBD=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_ps${N}/$task/$c
      mkdir -p "$SUBD"
      for ep in $(seq 0 $((N - 1))); do
        src=$(ls $(pwd)/$SRC6/$task/$c/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
        [ -n "$src" ] && ln -srf "$src" "$SUBD/"
      done
      fit_dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_ps${N} "$task/$c" "$AN/final_ps${N}/$c"
    done
  done
}
pool_fit() {
  local name=$1; shift; local cells=("$@")
  local N D i cl ep c task env ci seed instr src c2
  for N in 15 30 60; do
    D=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_${name}${N}/Pool/all
    mkdir -p "$D"
    i=0
    while [ $i -lt $N ]; do
      cl=${cells[$((i % ${#cells[@]}))]}
      ep=$((i / ${#cells[@]}))
      IFS='|' read -r c task env ci seed instr <<<"$(row_of $cl)"
      src=$(ls $(pwd)/$SRC6/$task/$c/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
      [ -n "$src" ] && ln -srf "$src" "$D/${c}--ep${ep}.pkl"
      i=$((i + 1))
    done
    fit_dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_${name}${N} "Pool/all" "$AN/final_${name}${N}/all"
    for c2 in ${BREADS} ${APPLES}; do ln -sfn all "$AN/final_${name}${N}/$c2"; done
  done
}

echo "[A-S2] fits START $(date -u '+%FT%T')"
ps_fits $APPLES
pool_fit xa $APPLES
pool_fit gx $BREADS $APPLES
echo "[A-S2] fits DONE $(date -u '+%FT%T')"

echo "[A-S2] NPZ push → srv50"
rsync -aR outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_ps15 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_ps30 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_ps60 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_xa15 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_xa30 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_xa60 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_xb15 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_xb30 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_xb60 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_gx15 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_gx30 \
          outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_gx60 \
  "$SRV50:pkt_ws/temporal_vla/" || { echo "[A-S2] NPZ push 실패"; exit 3; }
touch "$QROOT/NPZ_READY"
echo "[A-S2] NPZ_READY $(date -u '+%FT%T')"
