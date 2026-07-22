#!/usr/bin/env bash
# patchceil donor NPZ 일괄 추출 — pass B pkl → donors/ep{N}_L15.npz (+donor 는 _shuf.npz).
# lerobot 컨테이너에서 python 실행 (pkl 이 torch 텐서 포함). placebo/sham(실패)도 추출
# (--allow-fail). CAP 은 pass B 수집값(0,2,4,8,10,12,15), 추출 layer 는 primary L15.
# 사용: bash scripts/safe/groot_n15/robocasa/steer/patchceil/extract_donors.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
GROOT="outputs/eval/robocasa/groot_n15/patchceil"
TASK=PickPlaceCounterToCabinet
LAYERS="${LAYERS:-15}"
CAP="0,2,4,8,10,12,15"

for CELL in ppcc_bread_s300033 ppcc_bread_s400020; do
  MANIFEST="${REPO_ROOT}/${GROOT}/${CELL}/passB_manifest.tsv"
  mkdir -p "${REPO_ROOT}/${GROOT}/${CELL}/donors"
  while IFS=$'\t' read -r cell ep succ role _seed; do
    [ "$cell" = "cell" ] && continue
    src_host=$(ls "${REPO_ROOT}/${GROOT}/${CELL}/passB/raw_rollouts/${TASK}/${CELL}/task5--ep${ep}--succ"*.pkl 2>/dev/null | head -1)
    [ -n "$src_host" ] || { echo "MISSING pkl: ${CELL} ep${ep}"; exit 1; }
    src_cont="/temporal_vla/${src_host#${REPO_ROOT}/}"
    out_cont="/temporal_vla/${GROOT}/${CELL}/donors/ep${ep}_L15.npz"
    if [ ! -f "${REPO_ROOT}/${GROOT}/${CELL}/donors/ep${ep}_L15.npz" ]; then
      docker exec lerobot python /temporal_vla/scripts/safe/groot_n15/robocasa/steer/patchceil/extract_donor_npz.py \
        --pkl "$src_cont" --out "$out_cont" --layers "$LAYERS" --cap "$CAP" \
        $([ "$succ" = "0" ] && echo --allow-fail)
    fi
    if [ "$role" = "donor" ] && [ ! -f "${REPO_ROOT}/${GROOT}/${CELL}/donors/ep${ep}_L15_shuf.npz" ]; then
      docker exec lerobot python /temporal_vla/scripts/safe/groot_n15/robocasa/steer/patchceil/make_shuffle_npz.py \
        "$out_cont" "/temporal_vla/${GROOT}/${CELL}/donors/ep${ep}_L15_shuf.npz"
    fi
  done < "$MANIFEST"
  ls "${REPO_ROOT}/${GROOT}/${CELL}/donors" | wc -l | xargs echo "[extract_donors] ${CELL} npz:"
done
echo "EXTRACT_DONORS_DONE"
