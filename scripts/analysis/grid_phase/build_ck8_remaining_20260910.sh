#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
STORE="$HOME/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6"
GRID="$HOME/datasets/temporal_vla_store/groot/n15/grid"
CFG=configs/experiments/v6_ck8_remaining_20260910
BUILD=outputs/analysis/grid_phase/v6_ck8_remaining_build_20260910
for group in srv50_drawer srv48_dish; do
  mkdir -p "$BUILD/$group/released"
  test ! -e "$BUILD/$group/released/ready.json"
  echo '{"detectors": [], "operators": [], "failed": null, "complete": false}' > "$BUILD/$group/released/ready.json"
done
for group in srv50_drawer srv48_dish; do
  for source in "$CFG/$group/"*__s*_source.tsv; do
    stem="$(basename "$source" _source.tsv)"
    slug="${stem%__s*}"
    "$PY" scripts/analysis/grid_phase/prepare_ck8_raw_scene.py \
      --grid-root "$GRID" --index-tsv "$source" \
      --bundle "$STORE/ae_k8/ae_bundle_k8.npz" --slug "$slug" \
      --out "$BUILD/$group/prepared/$stem.npz"
  done
  "$PY" scripts/analysis/grid_phase/build_v6_ck8_dwell.py \
    --store "$STORE" --targets "$CFG/$group/targets.tsv" \
    --episodes "$CFG/$group/episodes.tsv" --index-tsv "$CFG/$group/source_index.tsv" \
    --prepared-dir "$BUILD/$group/prepared" --out "$BUILD/$group" --min-calib-succ 3
done
