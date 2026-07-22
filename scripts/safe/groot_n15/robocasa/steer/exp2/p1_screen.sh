#!/bin/bash
# exp2(구 pq2) P1: manifest 기반 conceptor refit 스크리닝 (rollout 0판, 순수 CPU).
# 각 manifest 에 대해 fit v2 를 7 capture layer × α grid(기본 5종) × 6그룹으로 실행.
# usage: p1_screen.sh <manifest.tsv>...
#   env: PYBIN(기본 python3; 승준은 ~/anaconda3/bin/python), OMP_NUM_THREADS(기본 4),
#        MAN_ROOT/OUT_ROOT(출력 경로 유도), JOBS(동시 fit 수, 기본 2 — 공유 노드 CPU 예의)
# 게이트: fit v2 rc!=0(빈 fit 포함) 은 [FAIL] 로 표면화, fit_summary.json 비어있으면 실패.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$REPO"
PYBIN=${PYBIN:-python3}
# 원격(승준) 실행 스크립트 — outputs 는 원격 구명(pq) 유지 (로컬은 symlink 로 호환)
MAN_ROOT=${MAN_ROOT:-outputs/eval/robocasa/groot_n15/pq2_manifests}
OUT_ROOT=${OUT_ROOT:-outputs/eval/robocasa/groot_n15/pq2_analysis/conceptors}
FITPY=scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py
# 주의: GROUPS 는 bash 특수 변수(대입 무시, gid 로 확장 — 실제 사고 2026-07-10) → FIT_GROUPS
FIT_GROUPS="global,reach-to-object,grasp,transport,place,insert-settle"
JOBS=${JOBS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4}

fit_one() { # manifest_path
  local mf=$1 rel stem out rc
  rel=${mf#"$MAN_ROOT"/}
  stem=${rel%.tsv}
  out="$OUT_ROOT/$stem"
  if [ -s "$out/fit_summary.json" ] && [ "$(cat "$out/fit_summary.json")" != "{}" ]; then
    echo "[skip] $stem (기존 fit)"; return 0
  fi
  mkdir -p "$(dirname "$out")"
  # --cell "pq2/..." 는 fit 산출물(NPZ 메타) 식별자 — 기존 fit 과의 일치 위해 구명 유지
  "$PYBIN" "$FITPY" --cell "pq2/$(basename "$stem")" --manifest "$mf" \
    --groups "$FIT_GROUPS" --out-dir "$out" > "$out.fitlog" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$out/fit_summary.json" ] || [ "$(cat "$out/fit_summary.json")" = "{}" ]; then
    echo "[FAIL rc=$rc] $stem — $(tail -n 2 "$out.fitlog" | head -1)"; return 1
  fi
  echo "[done] $stem ($(grep -c 'sel_alpha' "$out.fitlog") fits)"
}

i=0
for mf in "$@"; do
  [ -f "$mf" ] || { echo "[FAIL] manifest 없음: $mf"; continue; }
  fit_one "$mf" &
  i=$((i+1))
  [ $((i % JOBS)) -eq 0 ] && wait
done
wait

# 최종 판정은 산출물 기준 (bg 서브셸 rc 는 못 믿음 — [[long-run-detach-setsid]])
fails=0
for mf in "$@"; do
  stem=${mf#"$MAN_ROOT"/}; stem=${stem%.tsv}
  out="$OUT_ROOT/$stem"
  if [ ! -s "$out/fit_summary.json" ] || [ "$(cat "$out/fit_summary.json" 2>/dev/null)" = "{}" ]; then
    echo "[VERIFY-FAIL] $stem"; fails=$((fails+1))
  fi
done
echo "[p1_screen] 완료: $#개 요청, 실패 $fails"
[ "$fails" -eq 0 ]
