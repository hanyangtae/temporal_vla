#!/usr/bin/env bash
# v6 fail detector 증분 학습 (승준 원격, CPU) — 수집 완료 instruction부터 먼저 돌린다.
#
# 대기하지 않고 "준비된 것부터" 처리하는 루프다. 한 바퀴마다:
#   1) 셀 TSV(전체 파이프 제공)에서 instruction 목록을 읽고
#   2) 그 instruction의 segA shard(action phase 산출)가 있는지 보고
#   3) 아직 안 돌린 instruction이면 loko-cell arm 학습 → 셀별 ckpt + registry
#   4) 전부 처리했고 --once 면 종료, 아니면 POLL_S 후 다시
#
# 멱등: instruction 단위 완료 마커(<DET_OUT>/.done_<slug>)로 재실행 시 skip.
# 산출: $DET_OUT/loko/<slug>/s<i>/j<r>/detector_pertask_lstm_<slug>.pt
#       $DET_OUT/<slug>/{sim_summary.tsv, sim_detail.json, cell_registry.tsv}
set -euo pipefail

REPO="${REPO:-$HOME/workspace/temporal_vla_safeablate}"
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
STORE="${STORE:-$HOME/datasets/temporal_vla_store/groot/n15}"
# shard 소재지 — action phase 산출 규칙(2026-09-04): scene 단위 부분 shard 는
# segA_scene/<slug>__s<i>.npz, instruction 완주분은 segA/<slug>.npz. 둘 다 뒤진다.
SEG_DIRS="${SEG_DIRS:-$STORE/analysis/grid_phase_v6/segA_scene $STORE/analysis/grid_phase_v6/segA}"
CELLS="${CELLS:-$REPO/configs/collect/n15_grid_v6_scene_jitter/v6_loko_cells.tsv}"
DET_OUT="${DET_OUT:-$REPO/outputs/analysis/grid_phase/detector_v6}"
TRUNC="${TRUNC:-phase-gt}"          # 1차 = GT phase 절제 (ck8 절제판은 AE 후 증분)
ALPHAS="${ALPHAS:-0.1}"
MIN_POOL_FAIL="${MIN_POOL_FAIL:-3}"
MIN_CALIB_SUCC="${MIN_CALIB_SUCC:-9}"
CP_FOLDS="${CP_FOLDS:-0}"           # 0 = episode LOO
POLL_S="${POLL_S:-600}"
ONCE="${ONCE:-0}"
THREADS="${THREADS:-8}"             # 승준 CPU cap 8 (사용자 규약)

export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

ts() { date +%F' '%T; }
mkdir -p "$DET_OUT"

# 셀 TSV → slug 목록 (instruction 또는 slug 열, '/'·공백 → '_')
slugs_from_cells() {
  [[ -s "$CELLS" ]] || return 0
  "$PY" - "$CELLS" <<'PYEOF'
import csv, sys
seen, out = set(), []
with open(sys.argv[1], encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        v = r.get("slug") or r.get("instruction") or ""
        s = v.replace("/", "_").replace(" ", "_")
        if s and s not in seen:
            seen.add(s); out.append(s)
print("\n".join(out))
PYEOF
}

run_one() {   # <slug> <shard-stems(csv)> <shard-dir>
  local slug="$1" stems="$2" segdir="$3"
  local out="$DET_OUT/$slug"
  mkdir -p "$out"
  echo "[v6det] $slug 학습 시작 ($(ts)) dir=$(basename "$segdir") shards=$stems"
  local rc=0
  "$PY" "$REPO/scripts/analysis/grid_phase/failure_detector_sim.py" \
    --shard-dir "$segdir" --shards "$stems" --out "$out" \
    --arm loko-cell --loko-cells-tsv "$CELLS" \
    --models lstm --alphas "$ALPHAS" \
    --truncate-train "$TRUNC" \
    --min-pool-fail "$MIN_POOL_FAIL" --min-calib-succ "$MIN_CALIB_SUCC" \
    --cp-folds "$CP_FOLDS" --seed 0 --threads "$THREADS" --quiet || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "[v6det] ERROR: $slug sim 실패 rc=$rc — 완료 마커 쓰지 않음 ($(ts))" >&2
    return "$rc"
  fi
  # ckpt 는 sim 이 --out 아래 loko/<slug>/s*/j*/ 로 쓴다 → 공용 트리로 모은다
  if [[ -d "$out/loko" ]]; then
    mkdir -p "$DET_OUT/loko"
    cp -r "$out/loko/." "$DET_OUT/loko/"
  fi
  # ck8(cluster phase) shard 로 돌린 경우: 어느 AE 번들에서 나온 라벨인지 기록해 둔다.
  # action phase 는 shard 가 늘 때마다 임시 번들(ae_k8_partial)을 재생성하므로, 정식
  # 번들(ae_k8) 확정 후 재산출 대상을 식별하려면 학습 shard 목록의 지문이 필요하다.
  if [[ -n "${BUNDLE_MANIFEST:-}" && -s "${BUNDLE_MANIFEST}" ]]; then
    { echo "# AE 번들 provenance (ck8 절제/라벨 사용 시)"
      echo "manifest_path_basename: $(basename "$BUNDLE_MANIFEST")"
      echo "manifest_sha256: $(sha256sum "$BUNDLE_MANIFEST" | cut -d' ' -f1)"
      echo "captured_at: $(date -Is)"
      echo "--- shard 목록 ---"
      cat "$BUNDLE_MANIFEST"
    } > "$out/ae_bundle_provenance.txt"
  fi
  local n_ck
  n_ck=$(find "$DET_OUT/loko" -name "detector_pertask_lstm_${slug}*.pt" 2>/dev/null | wc -l || true)
  # 오케스트레이터가 잡을 실제 경로를 그대로 찍어 둔다(부분 shard면 파일명에 _s<i> 가 붙는다)
  find "$DET_OUT/loko" -name "detector_pertask_lstm_${slug}*.pt" 2>/dev/null | sed "s|^$DET_OUT/||" | sort || true
  local n_reg=0
  [[ -s "$out/cell_registry.tsv" ]] && n_reg=$(( $(wc -l < "$out/cell_registry.tsv") - 1 ))
  if [[ "$n_reg" -eq 0 ]]; then
    echo "[v6det] ERROR: $slug registry 0행 — 셀 매칭/게이트 확인 필요, 마커 안 씀 ($(ts))" >&2
    return 13
  fi
  echo "[v6det] $slug 완료 ($(ts)) — ckpt ${n_ck} / registry 행 ${n_reg}"
  date -Is > "$DET_OUT/.done_${slug}"
}

while :; do
  mapfile -t SLUGS < <(slugs_from_cells)
  if [[ "${#SLUGS[@]}" -eq 0 ]]; then
    echo "[v6det] 셀 TSV 없음/빈값: $CELLS ($(ts))"
  fi
  pending=0
  for slug in "${SLUGS[@]}"; do
    [[ -f "$DET_OUT/.done_${slug}" ]] && continue
    # shard 파일명은 action phase 산출 규칙을 따른다 — instruction 전체(<slug>.npz)일 수도,
    # 완료 scene 만 담은 부분 shard(<slug>_s0.npz 등)일 수도 있어 둘 다 받는다.
    seg_hit=""; stems=""
    for d in $SEG_DIRS; do
      [[ -d "$d" ]] || continue
      # .partial* 격리본(이관 미완 shard)은 절대 집지 않는다 — 판수가 모자란 채로
      # 학습하면 pool/게이트 수치가 조용히 틀어진다(2026-09-04 dish-R 46/50 사례).
      mapfile -t SH < <(find "$d" -maxdepth 1 -type f \( -name "${slug}.npz" -o -name "${slug}_*.npz" \) \
                        ! -name "*partial*" ! -name "*.tmp" 2>/dev/null | sort || true)
      [[ "${#SH[@]}" -eq 0 ]] && continue
      seg_hit="$d"
      for f in "${SH[@]}"; do
        b="$(basename "$f" .npz)"
        stems="${stems:+$stems,}$b"
      done
      break   # sim 은 shard-dir 하나만 받는다 — 먼저 찾은 디렉터리를 쓴다(scene 부분 shard 우선)
    done
    if [[ -z "$seg_hit" ]]; then
      echo "[v6det] shard 대기: $slug ($(ts))"
      pending=$((pending+1)); continue
    fi
    run_one "$slug" "$stems" "$seg_hit" || { echo "[v6det] ERROR: $slug 학습 실패 (rc=$?)" >&2; pending=$((pending+1)); }
  done
  if [[ "$ONCE" == "1" ]]; then
    echo "[v6det] once 모드 종료 — 미처리 ${pending} ($(ts))"; break
  fi
  if [[ "${#SLUGS[@]}" -gt 0 && "$pending" -eq 0 ]]; then
    echo "[v6det] V6_DETECTOR_ALL_DONE $(date -Is)"; break
  fi
  sleep "$POLL_S"
done
