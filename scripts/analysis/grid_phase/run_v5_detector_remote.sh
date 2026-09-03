#!/usr/bin/env bash
# v5 detector 체인 (승준 원격, CPU) — action phase 파이프(segA + ae_v5_k8 번들) 완료를 기다렸다가
#   1) segA → segA_ck8 (phase_code 를 k8 cluster 라벨로 교체한 사본, rewrite_shard_clusters)
#   2) failure_detector_sim --truncate-train phase-gt (= cluster-k8 dwell cap) pertask lstm
#      · v4r 관례: layer12·denoise -1·seg all·α{0.05,0.1,0.2}·seed0·scene split 3/1/1
#      · eval 대상 셀(v5_eval_cells.tsv)은 train/calib 에서 제외 (in-sample 방지, scene 은 유지)
#   3) 산출 = $REPO/outputs/analysis/grid_phase/detector_v5/cluster-k8/detector_pertask_lstm_<slug>.pt
# 완주 마커: V5_DETECTOR_DONE. 멱등(산출 있으면 skip).
set -euo pipefail

REPO="${REPO:-$HOME/workspace/temporal_vla_safeablate}"
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
STORE="${STORE:-$HOME/datasets/temporal_vla_store/groot/n15}"
OUT="${OUT:-$STORE/analysis/grid_phase_v5}"
AP_LOG="${AP_LOG:-$HOME/workspace/logs/v5_actionphase.log}"
BUNDLE="$OUT/ae_v5_k8/ae_bundle_v5_k8.npz"
SLUGS="${SLUGS:-DishwasherRack_out,OpenDrawer_left,OvenRack_out,PPCC_bread,PPCC_candle,PPCC_jug,PPCC_marshmallow}"
EXCL="$REPO/configs/collect/n15_grid_v5_scenario/v5_eval_cells.tsv"
DET_OUT="$REPO/outputs/analysis/grid_phase/detector_v5/cluster-k8"
WAIT_MAX_S="${WAIT_MAX_S:-43200}"

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16

ts() { date +%F' '%T; }

# ── 0. action phase 완료 대기 (마커 + 번들 + 라벨 10개) ─────────────────────────
t0=$(date +%s)
while :; do
  n_lab=$(ls "$OUT"/ae_v5_k8/labels_*_k8.npz 2>/dev/null | wc -l)
  if grep -q "V5_ACTIONPHASE_DONE" "$AP_LOG" 2>/dev/null && [[ -s "$BUNDLE" ]] && [[ "$n_lab" -ge 10 ]]; then
    echo "[det] action phase 완료 확인 ($(ts)) labels=$n_lab"; break
  fi
  if (( $(date +%s) - t0 > WAIT_MAX_S )); then
    echo "[det] ERROR: action phase 대기 초과 (labels=$n_lab, bundle=$([[ -s "$BUNDLE" ]] && echo y || echo n))" >&2
    exit 13
  fi
  sleep 120
done

# ── 1. segA → segA_ck8 ───────────────────────────────────────────────────────
IFS=',' read -r -a SLUG_ARR <<< "$SLUGS"
n_have=0
for s in "${SLUG_ARR[@]}"; do [[ -s "$OUT/segA_ck8/$s.npz" ]] && n_have=$((n_have+1)); done
if [[ "$n_have" -eq "${#SLUG_ARR[@]}" ]]; then
  echo "[det] skip rewrite — segA_ck8 ${n_have}/${#SLUG_ARR[@]} 존재"
else
  echo "[det] rewrite_shard_clusters 시작 ($(ts))"
  "$PY" "$REPO/scripts/analysis/grid_phase/rewrite_shard_clusters.py" \
    --shard-dir "$OUT/segA" --labels-dir "$OUT/ae_v5_k8" --out-dir "$OUT/segA_ck8" \
    --k 8 --shards "$SLUGS"
fi
for s in "${SLUG_ARR[@]}"; do
  [[ -s "$OUT/segA_ck8/$s.npz" ]] || { echo "[det] ERROR: segA_ck8/$s.npz 없음" >&2; exit 13; }
done

# ── 2. detector 학습 ─────────────────────────────────────────────────────────
n_pt=$(ls "$DET_OUT"/detector_pertask_lstm_*.pt 2>/dev/null | wc -l)
if [[ "$n_pt" -eq "${#SLUG_ARR[@]}" && -s "$DET_OUT/sim_summary.tsv" ]]; then
  echo "[det] skip sim — detector ${n_pt}개 존재"
else
  mkdir -p "$DET_OUT"
  echo "[det] failure_detector_sim 시작 ($(ts))"
  "$PY" "$REPO/scripts/analysis/grid_phase/failure_detector_sim.py" \
    --shard-dir "$OUT/segA_ck8" --out "$DET_OUT" --shards "$SLUGS" \
    --arm pertask --models lstm --alphas 0.05,0.1,0.2 \
    --truncate-train phase-gt --train-scenes 3 --calib-scenes 1 --test-scenes 1 \
    --seed 0 --threads 16 --exclude-cells-tsv "$EXCL" --quiet
fi

# ── 3. 감사 ──────────────────────────────────────────────────────────────────
n_pt=$(ls "$DET_OUT"/detector_pertask_lstm_*.pt 2>/dev/null | wc -l)
if [[ "$n_pt" -ne "${#SLUG_ARR[@]}" ]]; then
  echo "[det] ERROR: detector ${n_pt}/${#SLUG_ARR[@]}" >&2; exit 13
fi
"$PY" - "$DET_OUT" <<'PYEOF'
import sys, json, torch
from pathlib import Path
d = Path(sys.argv[1])
cfg = json.loads((d / "sim_detail.json").read_text())["config"]
print("[audit] exclude:", json.dumps(cfg.get("exclude_cells"), ensure_ascii=False))
for p in sorted(d.glob("detector_pertask_lstm_*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    bands = ck["cp_bands"]; task = next(iter(bands)); alphas = sorted(bands[task])
    print(f"[audit] {p.name}: task={task} alphas={alphas} feature={ck['feature']}")
    assert "0.10" in alphas, "α=0.1 밴드 없음"
PYEOF
echo "[det] V5_DETECTOR_DONE $(date -Is) → $DET_OUT"
