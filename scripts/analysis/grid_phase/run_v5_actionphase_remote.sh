#!/usr/bin/env bash
# grid v5 → action phase 산출물 (승준 원격, CPU 전용).
# handoff_20260903_actionphase.md §7 체크리스트 1~2단계의 실행판.
#
#   1) segA shard 추출 — instruction 별 **순차** 실행 + workers ≤3
#      (10종 일괄 workers 4 실행 OOM 실사고 08-20, extract_grid_matrix.py 주석 참조).
#      완료 shard(.npz 존재)는 멱등 skip — 재발사 시 이어서 돈다.
#   2) ae_cluster — AE(1536→16) 전 shard 공용 1개 + instruction 별 KMeans k8.
#      --export-bundle 필수 (없으면 encoder 미보존 — ae_cluster.py 주석의 실사고).
#
# 실행 (승준, ~/anaconda3/bin/python = torch+numpy CPU):
#   mkdir -p ~/workspace/logs
#   setsid nohup bash ~/workspace/temporal_vla/scripts/analysis/grid_phase/run_v5_actionphase_remote.sh \
#     > ~/workspace/logs/v5_actionphase.log 2>&1 < /dev/null &
#
# 완료 판정은 sentinel 문자열이 아니라 산출물로 한다:
#   $OUT/segA/*.npz 10개(감사 통과) + $OUT/ae_v5_k8/ae_bundle_v5_k8.npz
set -euo pipefail

PY="${PY:-$HOME/anaconda3/bin/python}"
REPO="${REPO:-$HOME/workspace/temporal_vla}"
STORE="${STORE:-$HOME/datasets/temporal_vla_store/groot/n15}"
OUT="${OUT:-$STORE/analysis/grid_phase_v5}"
INDEX="${INDEX:-$REPO/configs/collect/n15_grid_v5_scenario/index_rollouts_v5.tsv}"
GRID="${GRID:-$STORE/grid}"
WORKERS="${WORKERS:-3}"    # OOM 상한 — 3 초과 금지

# 공유 노드 CPU cap (메모리 규약 OMP ≤16)
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16

# index_rollouts_v5.tsv 의 grid_instruction 값과 정확히 일치해야 한다.
INSTRUCTIONS=(
  "CoffeeSetupMug" "DishwasherRack/out" "OpenDrawer/left" "OpenDrawer/right"
  "OvenRack/out" "PPCC/apple" "PPCC/bread" "PPCC/candle" "PPCC/jug" "PPCC/marshmallow"
)

slug_of() { local s="${1//\//_}"; echo "${s// /_}"; }

mkdir -p "$OUT/segA"
for instr in "${INSTRUCTIONS[@]}"; do
  slug="$(slug_of "$instr")"
  npz="$OUT/segA/${slug}.npz"
  if [[ -s "$npz" ]]; then
    echo "[wrap] skip $instr — $npz 존재"
    continue
  fi
  echo "[wrap] extract $instr → $slug ($(date +%F' '%T))"
  "$PY" "$REPO/scripts/analysis/grid_phase/extract_grid_matrix.py" \
    --grid-root "$GRID" --index-tsv "$INDEX" --out-dir "$OUT" \
    --instructions "$instr" --tier segA --workers "$WORKERS"
  # per-invocation summary 는 같은 파일을 덮어쓰므로 slug 별로 보존
  mv "$OUT/segA_summary.json" "$OUT/segA_summary_${slug}.json"
done

n_shard=$(ls "$OUT"/segA/*.npz 2>/dev/null | wc -l)
if [[ "$n_shard" -ne "${#INSTRUCTIONS[@]}" ]]; then
  echo "[wrap] ERROR: shard ${n_shard}/${#INSTRUCTIONS[@]} — 결손" >&2
  exit 13
fi

# 판수 감사 — 기대개수 대조(무음 탈락 방지): 각 shard eps == 125 (v5 = instruction 당 125판)
"$PY" - "$OUT/segA" <<'PYEOF'
import sys
import numpy as np
from pathlib import Path

seg = Path(sys.argv[1])
bad = []
for p in sorted(seg.glob("*.npz")):
    with np.load(p, allow_pickle=False) as z:
        ep = z["ep_id"]; succ = z["succ"]
        n_ep = len(np.unique(ep))
        n_rec = int(z["X"].shape[0])
        n_succ_ep = len({int(e) for e, s in zip(ep, succ) if s == 1})
    print(f"[audit] {p.stem}: eps={n_ep} rec={n_rec} succ_eps={n_succ_ep}", flush=True)
    if n_ep != 125:
        bad.append((p.stem, n_ep))
if bad:
    sys.exit(f"[audit] eps != 125: {bad}")
print("[audit] OK — 10 shard x 125 eps")
PYEOF

BUNDLE="$OUT/ae_v5_k8/ae_bundle_v5_k8.npz"
if [[ -s "$BUNDLE" ]]; then
  echo "[wrap] skip ae_cluster — $BUNDLE 존재"
else
  mkdir -p "$OUT/ae_v5_k8"
  echo "[wrap] ae_cluster 시작 ($(date +%F' '%T))"
  "$PY" "$REPO/scripts/analysis/grid_phase/ae_cluster.py" \
    --shard-dir "$OUT/segA" --mode all --dump-labels \
    --out-dir "$OUT/ae_v5_k8" --export-bundle "$BUNDLE"
fi

echo "[wrap] V5_ACTIONPHASE_DONE $(date -Is)"
