#!/usr/bin/env bash
# grid <라운드> → action phase 산출물 (승준 원격, CPU 전용).
# handoff_20260903_actionphase.md §7 체크리스트 1~2단계의 실행판.
#
#   1) segA shard 추출 — instruction 별 **순차** 실행 + workers ≤3
#      (일괄 workers 4 실행 OOM 실사고 08-20, extract_grid_matrix.py 주석 참조).
#      완료 shard(.npz 존재)는 멱등 skip — 재발사 시 이어서 돈다.
#   2) ae_cluster — AE(1536→16) 전 shard 공용 1개 + instruction 별 KMeans k8.
#      --export-bundle 필수 (없으면 encoder 미보존 — ae_cluster.py 주석의 실사고).
#
# **격자를 하드코딩하지 않는다** — instruction 목록도 instruction 당 기대 판수도
# `--index-tsv` 에서 읽는다. v5(10키×125판)에서 v6(12키×75판)로 갈아탈 때 러너를
# 고치다 격자 상수가 어긋나는 사고를 막기 위한 것이다. 인덱스가 곧 계약이다.
#
# 실행 (승준, ~/anaconda3/bin/python = torch+numpy CPU). TAG·INDEX 는 필수:
#   mkdir -p ~/workspace/logs
#   setsid nohup env TAG=v6 \
#     INDEX=~/workspace/temporal_vla/configs/collect/<plan>/index_rollouts_v6.tsv \
#     bash ~/workspace/temporal_vla/scripts/analysis/grid_phase/run_actionphase_remote.sh \
#     > ~/workspace/logs/actionphase_v6.log 2>&1 < /dev/null &
#
# 완료 판정은 sentinel 문자열이 아니라 산출물로 한다:
#   $OUT/segA/*.npz (인덱스의 instruction 전부, 감사 통과) + $OUT/ae_k8/ae_bundle_k8.npz
set -euo pipefail

TAG="${TAG:?TAG 를 지정할 것 (예: v5, v6) — 산출 디렉토리·번들 이름에 쓰인다}"
INDEX="${INDEX:?INDEX 를 지정할 것 — build_grid_index.py 산출 rollouts tsv 절대경로}"

PY="${PY:-$HOME/anaconda3/bin/python}"
REPO="${REPO:-$HOME/workspace/temporal_vla}"
STORE="${STORE:-$HOME/datasets/temporal_vla_store/groot/n15}"
OUT="${OUT:-$STORE/analysis/grid_phase_${TAG}}"
GRID="${GRID:-$STORE/grid}"
WORKERS="${WORKERS:-3}"    # OOM 상한 — 3 초과 금지
K="${K:-8}"

# 공유 노드 CPU cap (메모리 규약 OMP ≤16)
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16

[[ -s "$INDEX" ]] || { echo "[wrap] ERROR: INDEX 없음/빈 파일: $INDEX" >&2; exit 2; }

# QA 무효 셀 제외 — EXCLUDE_CELLS = 한 줄에 셀 rel_path 하나(`#` 주석·빈 줄 허용).
# 예: v6 QA 의 configs/collect/n15_grid_v6_scene_jitter/qa_invalid_cells.txt
#     (영상 정지 + VL hidden 상수화 5셀). 제외분을 뺀 인덱스를 만들어 **추출과 감사가
#     같은 모수**를 보게 한다 — 인덱스를 안 거르고 추출만 거르면 감사가 결손으로 오판한다.
mkdir -p "$OUT"
if [[ -n "${EXCLUDE_CELLS:-}" ]]; then
  [[ -s "$EXCLUDE_CELLS" ]] || { echo "[wrap] ERROR: EXCLUDE_CELLS 없음: $EXCLUDE_CELLS" >&2; exit 2; }
  filtered="$OUT/index_filtered.tsv"
  awk -F'\t' -v exf="$EXCLUDE_CELLS" '
    BEGIN {
      while ((getline line < exf) > 0) {
        sub(/\r$/, "", line); sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line)
        if (line == "" || line ~ /^#/) continue
        ex[line] = 1; nex++
      }
    }
    NR==1 { for (i=1;i<=NF;i++) c[$i]=i; print; next }
    {
      rp = ("rel_path" in c) ? $c["rel_path"] : ""
      # rel_path 는 plan_id 접두가 붙어 있고 제외 목록은 보통 machine/… 부터다 — 양쪽 다 맞춰 본다.
      short = rp; sub(/^[^\/]+\//, "", short)
      if ((rp in ex) || (short in ex)) { drop++; next }
      print
    }
    END { printf "[wrap] 제외 목록 %d행 → 인덱스에서 %d행 제거\n", nex, drop+0 > "/dev/stderr" }
  ' "$INDEX" > "$filtered"
  n_before=$(( $(wc -l < "$INDEX") - 1 )); n_after=$(( $(wc -l < "$filtered") - 1 ))
  if [[ "$n_after" -eq "$n_before" ]]; then
    echo "[wrap] ERROR: 제외 목록이 인덱스와 한 행도 안 맞았다 — 경로 형식 확인" >&2
    exit 2
  fi
  echo "[wrap] 무효 셀 제외: $n_before → $n_after 행 ($EXCLUDE_CELLS)"
  INDEX="$filtered"
fi

# 인덱스 → (instruction, 기대 판수). 열 위치는 헤더에서 찾는다(하드코딩 금지).
# base arm + has_pkl 인 행만 세어 추출기의 필터와 같은 모수를 본다.
INDEX_COUNTS="$(awk -F'\t' '
  NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
  {
    if ((("armsig" in c) && $c["armsig"] != "base")) next;
    if ((("has_pkl" in c) && ($c["has_pkl"]=="0" || $c["has_pkl"]=="false"))) next;
    n[$c["grid_instruction"]]++
  }
  END { for (k in n) printf "%s\t%d\n", k, n[k] }' "$INDEX" | sort)"
[[ -n "$INDEX_COUNTS" ]] || { echo "[wrap] ERROR: $INDEX 에서 instruction 을 못 읽었다" >&2; exit 2; }

mapfile -t ALL_INSTRUCTIONS < <(cut -f1 <<< "$INDEX_COUNTS")
echo "[wrap] TAG=$TAG index=$INDEX instruction=${#ALL_INSTRUCTIONS[@]}"
while IFS=$'\t' read -r _i _n; do echo "[wrap]   $_i: $_n 판(기대)"; done <<< "$INDEX_COUNTS"

INSTRUCTIONS=("${ALL_INSTRUCTIONS[@]}")
# 부분 실행: INSTR_CSV="A,B,..." 로 대상 축소. 이때 AE 는 건너뛴다 —
# AE 는 인덱스의 instruction 전체가 모인 뒤에만 (SKIP_AE=1 로도 강제 가능).
SKIP_AE="${SKIP_AE:-0}"
if [[ -n "${INSTR_CSV:-}" ]]; then
  IFS=',' read -r -a INSTRUCTIONS <<< "$INSTR_CSV"
  SKIP_AE=1
fi

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

# 요청분 shard 존재 확인 (부분 실행이면 요청 slug 만)
missing=0
for instr in "${INSTRUCTIONS[@]}"; do
  slug="$(slug_of "$instr")"
  if [[ ! -s "$OUT/segA/${slug}.npz" ]]; then
    echo "[wrap] ERROR: ${slug}.npz 없음 — 결손" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || exit 13

# 판수 감사 — shard 실판수 vs 인덱스 기대 판수 대조(무음 탈락 방지).
printf '%s\n' "$INDEX_COUNTS" > "$OUT/.expected_counts.tsv"
"$PY" - "$OUT/segA" "$OUT/.expected_counts.tsv" <<'PYEOF'
import sys
import numpy as np
from pathlib import Path

seg, exp_tsv = Path(sys.argv[1]), Path(sys.argv[2])


def slug_of(s):
    for ch in ("/", " ", "\\", "\t"):
        s = s.replace(ch, "_")
    return s


expected = {}
for line in exp_tsv.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    instr, n = line.split("\t")
    expected[slug_of(instr)] = int(n)

bad = []
for p in sorted(seg.glob("*.npz")):
    with np.load(p, allow_pickle=False) as z:
        ep, succ = z["ep_id"], z["succ"]
        n_ep = len(np.unique(ep))
        n_rec = int(z["X"].shape[0])
        n_succ_ep = len({int(e) for e, s in zip(ep, succ) if s == 1})
    want = expected.get(p.stem)
    print(f"[audit] {p.stem}: eps={n_ep}/{want} rec={n_rec} succ_eps={n_succ_ep}",
          flush=True)
    if want is None:
        bad.append((p.stem, "인덱스에 없는 shard"))
    elif n_ep != want:
        bad.append((p.stem, f"{n_ep} != {want}"))
if bad:
    sys.exit(f"[audit] 판수 불일치: {bad}")
print(f"[audit] OK — shard {len(list(seg.glob('*.npz')))}개 전부 기대 판수 일치")
PYEOF

# 번들 경로 = `<OUT>/ae_k<K>/ae_bundle_k<K>.npz`. OUT 이 이미 라운드를 담고 있으므로
# (기본 `analysis/grid_phase_<TAG>`) 안쪽 이름에 TAG 를 또 넣지 않는다 — 소비 측
# (연산자 fit) 기본 경로와 맞춘 규약이다.
BUNDLE="$OUT/ae_k${K}/ae_bundle_k${K}.npz"
n_all=$(ls "$OUT"/segA/*.npz 2>/dev/null | wc -l)
n_want="${#ALL_INSTRUCTIONS[@]}"
if [[ "$SKIP_AE" == "1" ]]; then
  echo "[wrap] SKIP_AE — ae_cluster 생략 (segA shard ${n_all}/${n_want})"
  echo "[wrap] ACTIONPHASE_${TAG}_PARTIAL_DONE $(date -Is)"
  exit 0
fi
if [[ "$n_all" -ne "$n_want" ]]; then
  echo "[wrap] ERROR: ae_cluster 는 인덱스의 instruction 전체(${n_want}) 필요 — 현재 ${n_all}" >&2
  exit 13
fi
if [[ -s "$BUNDLE" ]]; then
  echo "[wrap] skip ae_cluster — $BUNDLE 존재"
else
  mkdir -p "$(dirname "$BUNDLE")"
  echo "[wrap] ae_cluster 시작 ($(date +%F' '%T))"
  "$PY" "$REPO/scripts/analysis/grid_phase/ae_cluster.py" \
    --shard-dir "$OUT/segA" --mode all --dump-labels --k "$K" \
    --out-dir "$(dirname "$BUNDLE")" --export-bundle "$BUNDLE"
fi

echo "[wrap] ACTIONPHASE_${TAG}_DONE $(date -Is)"
