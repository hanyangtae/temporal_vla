#!/bin/bash
# CPU archive host only; source data and collection jobs are read-only.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
STORE="$HOME/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6"
GRID="$HOME/datasets/temporal_vla_store/groot/n15/grid"
CFG=configs/experiments/v6_ck8_topup_20260909
BUILD=outputs/analysis/grid_phase/v6_ck8_topup_build_20260909
for cell in PPCC_bread_s4 PPCC_marshmallow_s3 PPCC_marshmallow_s4; do
  mkdir -p "$BUILD/$cell/released"
  test ! -e "$BUILD/$cell/released/ready.json"
  echo '{"detectors": [], "operators": [], "failed": null, "complete": false}' > "$BUILD/$cell/released/ready.json"
done
for cell in PPCC_bread_s4 PPCC_marshmallow_s3 PPCC_marshmallow_s4; do
  slug="${cell%_s*}"
  scene="${cell##*_s}"
  "$PY" scripts/analysis/grid_phase/prepare_ck8_raw_scene.py \
    --grid-root "$GRID" --index-tsv "$CFG/$cell/source_index.tsv" \
    --bundle "$STORE/ae_k8/ae_bundle_k8.npz" --slug "$slug" \
    --out "$BUILD/$cell/prepared/${slug}__s${scene}.npz" --wait-for-archive
  "$PY" scripts/analysis/grid_phase/build_v6_ck8_dwell.py \
    --store "$STORE" --targets "$CFG/$cell/targets.tsv" \
    --episodes "$CFG/$cell/episodes.tsv" --index-tsv "$CFG/$cell/source_index.tsv" \
    --prepared-dir "$BUILD/$cell/prepared" --out "$BUILD/$cell" --min-calib-succ 9
done
