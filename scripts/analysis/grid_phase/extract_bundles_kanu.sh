#!/bin/bash
# kanu 수집분 grid rollout 219판 → replay 번들 일괄 추출 (**원격 노드에서 실행**).
#
# 동기: kanu 영상은 하단 caption burn-in 으로 장면이 가려져 복원 불가 →
# 저장 action 을 env 에 다시 먹여(replay) 클린 영상을 만든다. replay 에 필요한 것은
# rollout.pkl(607MB) 통째가 아니라 action·seed·env_name 뿐이므로 원격에서 번들만 뽑고
# 그것만 회수한다 (docs/04: pkl 통째 전송 금지).
#
# 사용 (원격):
#   bash scripts/analysis/grid_phase/extract_bundles_kanu.sh [OUT_DIR]
# 기본 OUT_DIR=/tmp/replay_bundles_kanu
#
# 산출:
#   <OUT>/<stem>.bundle.pkl        stem = <cell>__<task>__<variant>__<sN>__<nM>
#   <OUT>/manifest.tsv             stem \t <cell>/kanu/<...>/base/video.mp4 (교체 대상 상대경로)
# 공유 노드 예의: 순차 실행 + BLAS thread cap.
set -euo pipefail

GRID="${GRID_ROOT:-$HOME/datasets/temporal_vla_store/groot/n15/grid}"
OUT="${1:-/tmp/replay_bundles_kanu}"
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

mkdir -p "$OUT"
MAN="$OUT/manifest.tsv"
: > "$MAN"

n=0
for cell in 979d4833a7db b8054b5e7258; do
  while IFS= read -r pkl; do
    rel="${pkl#"$GRID"/}"                      # <cell>/kanu/<task>/.../base/rollout.pkl
    key="${rel%/base/rollout.pkl}"             # <cell>/kanu/<task>/<variant>/sN/nM
    stem="$(echo "${key/\/kanu\//\/}" | sed 's#/#__#g')"   # 예: b8054b5e7258__OpenDrawer__left__s2__n0
    bundle="$OUT/$stem.bundle.pkl"
    if [[ ! -f "$bundle" ]]; then
      "$PY" "$REPO/scripts/analysis/grid_phase/extract_replay_bundle.py" \
        --pkl "$pkl" --out "$bundle" >/dev/null
    fi
    printf '%s\t%s\n' "$stem" "${rel%/rollout.pkl}/video.mp4" >> "$MAN"
    n=$((n+1))
    [[ $((n % 20)) -eq 0 ]] && echo "[extract] $n ..."
  done < <(find "$GRID/$cell/kanu" -name rollout.pkl | sort)
done

echo "[extract] 총 $n 판 → $OUT"
du -sh "$OUT"
wc -l "$MAN"
