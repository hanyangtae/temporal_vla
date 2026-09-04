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
K="${K:-8}"

# ── CPU 예산 (docs/05 §1: 승준은 8코어 공유 — 내 프로세스 **합산 스레드 ≤ 8**) ─────
# 추출은 워커 프로세스가 WORKERS 개 뜨고 각자 BLAS 를 쓰므로 WORKERS × OMP_THREADS 가
# 실제 점유다. 기본 2×4=8. 이관·수집이 같이 돌면 WORKERS=1 로 낮출 것.
# (구 기본값은 workers 3 × OMP 16 = 48 스레드로 이 규약을 훨씬 넘겼다.)
WORKERS="${WORKERS:-2}"          # OOM 상한 3 이하 · CPU 예산상 보통 2
OMP_THREADS="${OMP_THREADS:-4}"
if (( WORKERS * OMP_THREADS > 8 )); then
  echo "[wrap] ERROR: WORKERS($WORKERS) × OMP_THREADS($OMP_THREADS) > 8 — 승준 CPU 규약 위반" >&2
  exit 2
fi
export OMP_NUM_THREADS="$OMP_THREADS" OPENBLAS_NUM_THREADS="$OMP_THREADS" \
       MKL_NUM_THREADS="$OMP_THREADS" NUMEXPR_NUM_THREADS="$OMP_THREADS"

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

# 인덱스 → 셀 단위 기대표 (instruction, scene, j, 판수). 열 위치는 헤더에서 찾는다.
# base arm + has_pkl 인 행만 세어 추출기의 필터와 같은 모수를 본다. **(instruction, scene, j)
# 단위**로 세는 이유: 소비 측(연산자 fit)의 pool 계산이 그 단위라, instruction 총계만
# 맞고 셀 분포가 어긋나는 결손을 놓치지 않으려는 것.
CELL_COUNTS="$(awk -F'\t' '
  NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
  {
    if ((("armsig" in c) && $c["armsig"] != "base")) next;
    if ((("has_pkl" in c) && ($c["has_pkl"]=="0" || $c["has_pkl"]=="false"))) next;
    j = ("jitter_idx" in c) ? $c["jitter_idx"] : (("jitter_reset_idx" in c) ? $c["jitter_reset_idx"] : "-1")
    n[$c["grid_instruction"] "\t" $c["scene_idx"] "\t" j]++
  }
  END { for (k in n) printf "%s\t%d\n", k, n[k] }' "$INDEX" | sort)"
INDEX_COUNTS="$(awk -F'\t' '{n[$1]+=$4} END {for (k in n) printf "%s\t%d\n", k, n[k]}' \
  <<< "$CELL_COUNTS" | sort)"
[[ -n "$INDEX_COUNTS" ]] || { echo "[wrap] ERROR: $INDEX 에서 instruction 을 못 읽었다" >&2; exit 2; }

mapfile -t ALL_INSTRUCTIONS < <(cut -f1 <<< "$INDEX_COUNTS")
echo "[wrap] TAG=$TAG index=$INDEX instruction=${#ALL_INSTRUCTIONS[@]}"
while IFS=$'\t' read -r _i _n; do echo "[wrap]   $_i: $_n 판(기대)"; done <<< "$INDEX_COUNTS"

INSTRUCTIONS=("${ALL_INSTRUCTIONS[@]}")
# 부분 실행: INSTR_CSV="A,B,..." 로 대상 축소.
SKIP_AE="${SKIP_AE:-0}"
if [[ -n "${INSTR_CSV:-}" ]]; then
  IFS=',' read -r -a INSTRUCTIONS <<< "$INSTR_CSV"
  [[ -n "${PARTIAL_AE:-}" ]] || SKIP_AE=1   # 부분 실행 기본은 AE 생략
fi

# PARTIAL_AE=1 : **수집 완주 전 임시 번들**. 그때까지 모인 shard 만으로 AE·KMeans 를 돌려
# 소비 세션이 먼저 붙어 볼 수 있게 한다. 정식 번들과 **섞이면 안 되므로** 산출을
# `ae_k<K>_partial/` 로 분리하고, 어느 shard 로 학습했는지 provenance(번들 안)와
# `PARTIAL_SHARDS.txt` 에 남긴다. 최종 번들(전 instruction)은 PARTIAL_AE 없이 돌려
# `ae_k<K>/` 에 만든다 — AE 는 전 shard 공용 1개 규약이라 임시본과 수치가 다르다.
AE_SUBDIR="ae_k${K}"
if [[ -n "${PARTIAL_AE:-}" ]]; then
  AE_SUBDIR="ae_k${K}_partial"
  SKIP_AE=0
fi

slug_of() { local s="${1//\//_}"; echo "${s// /_}"; }

# ── (instruction, scene) 단위 추출 ────────────────────────────────────────────
# 수집 완료 단위가 instruction 이 아니라 **(instruction, scene)** 인 경우(2026-09-04
# 사용자 지시: "s0 drawer-left, s1 oven-out-left 부터"). 산출은 instruction shard 와
# **다른 디렉토리** `segA_scene/<slug>__s<i>.npz` 에 둔다 — 같은 폴더에 두면 ae_cluster 가
# scene shard 를 별개 instruction 으로 잡아 KMeans 단위가 조용히 바뀐다.
# 나중에 그 instruction 의 scene 이 다 모이면 merge_scene_shards.py 로 합쳐
# `segA/<slug>.npz` 를 만든다(재추출 불필요).
if [[ -n "${INSTR_SCENES:-}" ]]; then
  mkdir -p "$OUT/segA_scene"
  IFS=',' read -r -a pairs <<< "$INSTR_SCENES"
  for pair in "${pairs[@]}"; do
    instr="${pair%:*}"; sc="${pair##*:}"
    slug="$(slug_of "$instr")"
    dst="$OUT/segA_scene/${slug}__s${sc}.npz"
    if [[ -s "$dst" ]]; then echo "[wrap] skip $instr s$sc — $dst 존재"; continue; fi
    sub_idx="$OUT/.index_${slug}__s${sc}.tsv"
    awk -F'\t' -v I="$instr" -v S="$sc" '
      NR==1 { for (i=1;i<=NF;i++) c[$i]=i; print; next }
      $c["grid_instruction"]==I && $c["scene_idx"]==S { print }' "$INDEX" > "$sub_idx"
    n_sub=$(( $(wc -l < "$sub_idx") - 1 ))
    if [[ "$n_sub" -le 0 ]]; then
      echo "[wrap] ERROR: 인덱스에 $instr s$sc 행이 없다" >&2; exit 2
    fi
    # **이관 미완 검사** — 인덱스 행(=계획된 판)과 pkl 보유 행이 다르면 shard 가 조용히
    # 짧아진다. 셀 감사는 기대치를 같은 인덱스에서 만들므로 이 결손을 못 잡는다
    # (실사고 2026-09-04: dish-R s1 meta 50 / pkl 46 → 46판 shard 가 "일치"로 통과).
    n_pkl=$(awk -F'\t' '
      NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
      { if (("has_pkl" in c) && ($c["has_pkl"]=="0" || $c["has_pkl"]=="false")) next; n++ }
      END { print n+0 }' "$sub_idx")
    if [[ "$n_pkl" -ne "$n_sub" ]]; then
      echo "[wrap] ERROR: $instr s$sc — 인덱스 $n_sub 행 중 pkl 보유 $n_pkl (이관 미완 $(( n_sub - n_pkl ))판)." >&2
      echo "[wrap]        이관 완료 후 재실행할 것. 짧은 shard 를 그대로 내려면 ALLOW_PARTIAL_PKL=1." >&2
      [[ -n "${ALLOW_PARTIAL_PKL:-}" ]] || exit 14
      echo "[wrap] ALLOW_PARTIAL_PKL — 결손 $(( n_sub - n_pkl ))판을 뺀 채 진행한다" >&2
    fi
    echo "[wrap] extract $instr s$sc ($n_sub 판) → ${slug}__s${sc} ($(date +%F' '%T))"
    tmp_out="$OUT/.tmp_${slug}__s${sc}"
    rm -rf "$tmp_out"; mkdir -p "$tmp_out"
    "$PY" "$REPO/scripts/analysis/grid_phase/extract_grid_matrix.py" \
      --grid-root "$GRID" --index-tsv "$sub_idx" --out-dir "$tmp_out" \
      --instructions "$instr" --tier segA --workers "$WORKERS"
    mv "$tmp_out/segA/${slug}.npz" "$dst"
    mv "$tmp_out/segA_summary.json" "$OUT/segA_summary_${slug}__s${sc}.json"
    rm -rf "$tmp_out"
    # scene shard 감사 — 판수·(scene, j) 분포를 부분 인덱스와 대조 (fail-loud)
    "$PY" - "$dst" "$sub_idx" "$OUT/audit_cells_scene.tsv" <<'PYEOF'
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

shard, idx_tsv, out_tsv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

lines = idx_tsv.read_text(encoding="utf-8").splitlines()
head = {c: i for i, c in enumerate(lines[0].split("\t"))}
jcol = "jitter_idx" if "jitter_idx" in head else "jitter_reset_idx"
want = defaultdict(int)
for ln in lines[1:]:
    if not ln.strip():
        continue
    p = ln.split("\t")
    if head.get("armsig") is not None and p[head["armsig"]] != "base":
        continue
    if head.get("has_pkl") is not None and p[head["has_pkl"]] in ("0", "false", "False"):
        continue
    want[(int(p[head["scene_idx"]]), int(p[head[jcol]]))] += 1

with np.load(shard, allow_pickle=False) as z:
    ep, succ, scene, jit = z["ep_id"], z["succ"], z["scene"], z["jitter"]
    n_rec = int(z["X"].shape[0])
seen, cells = {}, defaultdict(lambda: [0, 0, 0])
for e, s, sc_, jj in zip(ep, succ, scene, jit):
    key = (int(sc_), int(jj))
    cells[key][2] += 1
    if int(e) not in seen:
        seen[int(e)] = key
        cells[key][0] += 1
        if int(s) == 1:
            cells[key][1] += 1

bad = [f"s{k[0]}j{k[1]} {cells.get(k, [0])[0]} != {want[k]}"
       for k in sorted(set(want) | set(cells))
       if cells.get(k, [0, 0, 0])[0] != want.get(k, 0)]
n_ep, n_succ = len(seen), sum(v[1] for v in cells.values())
print(f"[audit] {shard.stem}: eps={n_ep}/{sum(want.values())} rec={n_rec} "
      f"succ_eps={n_succ} cells={len(cells)}/{len(want)}", flush=True)
for sc_ in sorted({k[0] for k in want}):
    line = " ".join(f"j{j}:{cells.get((sc_, j), [0])[0]}"
                    for j in sorted(k[1] for k in want if k[0] == sc_))
    print(f"[audit]   s{sc_}  {line}", flush=True)
rows = [] if out_tsv.exists() else ["slug\tscene\tjitter\teps\texpected\tsucc_eps\trecords"]
for k in sorted(set(want) | set(cells)):
    g = cells.get(k, [0, 0, 0])
    rows.append(f"{shard.stem}\t{k[0]}\t{k[1]}\t{g[0]}\t{want.get(k, '')}\t{g[1]}\t{g[2]}")
with out_tsv.open("a", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
if bad:
    sys.exit(f"[audit] 판수 불일치: {bad}")
print("[audit] OK — 셀 단위 일치")
PYEOF
    echo "[wrap] scene shard 완료: $dst ($(du -h "$dst" | cut -f1))"
  done
  # scene shard 는 instruction 이 다 안 모였으므로 AE 로 넘어가지 않는다.
  echo "[wrap] ACTIONPHASE_${TAG}_SCENE_DONE $(date -Is)"
  exit 0
fi

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
# 셀 단위 `<instruction>\t<scene>\t<j>\t<판수>` 표를 넘겨 (키, scene, j) 분포까지 대조하고
# 소비 세션에 그대로 통지할 수 있게 요약 TSV(`audit_cells.tsv`)로도 남긴다.
printf '%s\n' "$CELL_COUNTS" > "$OUT/.expected_cells.tsv"
"$PY" - "$OUT/segA" "$OUT/.expected_cells.tsv" "$OUT/audit_cells.tsv" <<'PYEOF'
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

seg, exp_tsv, out_tsv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def slug_of(s):
    for ch in ("/", " ", "\\", "\t"):
        s = s.replace(ch, "_")
    return s


# 기대: slug → {(scene, j): 판수}
expected = defaultdict(dict)
for line in exp_tsv.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    instr, scene, j, n = line.split("\t")
    expected[slug_of(instr)][(int(scene), int(j))] = int(n)

bad, rows = [], ["slug\tscene\tjitter\teps\texpected\tsucc_eps\trecords"]
for p in sorted(seg.glob("*.npz")):
    with np.load(p, allow_pickle=False) as z:
        ep, succ, scene, jit = z["ep_id"], z["succ"], z["scene"], z["jitter"]
        n_rec_total = int(z["X"].shape[0])
    # record 축 → 판 단위로 접어서 센다 (ep_id 가 판의 고유 키)
    seen, cells = {}, defaultdict(lambda: [0, 0, 0])   # (scene,j) → [eps, succ_eps, recs]
    for e, s, sc, jj in zip(ep, succ, scene, jit):
        key = (int(sc), int(jj))
        cells[key][2] += 1
        if int(e) not in seen:
            seen[int(e)] = key
            cells[key][0] += 1
            if int(s) == 1:
                cells[key][1] += 1
    want_cells = expected.get(p.stem)
    n_ep = len(seen)
    n_succ = sum(v[1] for v in cells.values())
    if want_cells is None:
        print(f"[audit] {p.stem}: eps={n_ep} rec={n_rec_total} — 인덱스에 없는 shard",
              flush=True)
        bad.append((p.stem, "인덱스에 없는 shard"))
        continue
    want_total = sum(want_cells.values())
    print(f"[audit] {p.stem}: eps={n_ep}/{want_total} rec={n_rec_total} "
          f"succ_eps={n_succ} cells={len(cells)}/{len(want_cells)}", flush=True)
    for key in sorted(set(cells) | set(want_cells)):
        got = cells.get(key, [0, 0, 0])
        want = want_cells.get(key)
        rows.append(f"{p.stem}\t{key[0]}\t{key[1]}\t{got[0]}\t"
                    f"{'' if want is None else want}\t{got[1]}\t{got[2]}")
        if want is None:
            bad.append((p.stem, f"s{key[0]}j{key[1]} 인덱스에 없는 셀"))
        elif got[0] != want:
            bad.append((p.stem, f"s{key[0]}j{key[1]} {got[0]} != {want}"))
    # 셀별 판수를 한 줄로 (통지용): s0[j0..j4] s1[...] ...
    for sc in sorted({k[0] for k in want_cells}):
        line = " ".join(f"j{j}:{cells.get((sc, j), [0])[0]}"
                        for j in sorted(k[1] for k in want_cells if k[0] == sc))
        print(f"[audit]   s{sc}  {line}", flush=True)

out_tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
if bad:
    sys.exit(f"[audit] 판수 불일치: {bad[:10]}{' …' if len(bad) > 10 else ''}")
print(f"[audit] OK — shard {len(list(seg.glob('*.npz')))}개 전부 셀 단위 일치 "
      f"({out_tsv.name})")
PYEOF

# 번들 경로 = `<OUT>/ae_k<K>/ae_bundle_k<K>.npz`. OUT 이 이미 라운드를 담고 있으므로
# (기본 `analysis/grid_phase_<TAG>`) 안쪽 이름에 TAG 를 또 넣지 않는다 — 소비 측
# (연산자 fit) 기본 경로와 맞춘 규약이다.
BUNDLE="$OUT/$AE_SUBDIR/ae_bundle_k${K}.npz"
n_all=$(ls "$OUT"/segA/*.npz 2>/dev/null | wc -l)
n_want="${#ALL_INSTRUCTIONS[@]}"
if [[ "$SKIP_AE" == "1" ]]; then
  echo "[wrap] SKIP_AE — ae_cluster 생략 (segA shard ${n_all}/${n_want})"
  echo "[wrap] ACTIONPHASE_${TAG}_PARTIAL_DONE $(date -Is)"
  exit 0
fi
if [[ -z "${PARTIAL_AE:-}" && "$n_all" -ne "$n_want" ]]; then
  echo "[wrap] ERROR: 정식 ae_cluster 는 인덱스의 instruction 전체(${n_want}) 필요 — 현재 ${n_all}" >&2
  echo "[wrap]        완주 전 임시 번들이 필요하면 PARTIAL_AE=1 (산출 ae_k${K}_partial/)" >&2
  exit 13
fi
if [[ -n "${PARTIAL_AE:-}" && "$n_all" -lt 2 ]]; then
  echo "[wrap] ERROR: 임시 AE 도 shard 2개 이상 필요 — 현재 ${n_all}" >&2
  exit 13
fi
# 임시 번들은 shard 가 늘 때마다 다시 만들어야 하므로 존재해도 덮어쓴다(멱등 skip 안 함).
if [[ -z "${PARTIAL_AE:-}" && -s "$BUNDLE" ]]; then
  echo "[wrap] skip ae_cluster — $BUNDLE 존재"
else
  mkdir -p "$(dirname "$BUNDLE")"
  echo "[wrap] ae_cluster 시작 (${PARTIAL_AE:+임시·}shard ${n_all}/${n_want}, $(date +%F' '%T))"
  ls "$OUT"/segA/*.npz | xargs -n1 basename > "$(dirname "$BUNDLE")/PARTIAL_SHARDS.txt"
  "$PY" "$REPO/scripts/analysis/grid_phase/ae_cluster.py" \
    --shard-dir "$OUT/segA" --mode all --dump-labels --k "$K" \
    --out-dir "$(dirname "$BUNDLE")" --export-bundle "$BUNDLE"
fi

echo "[wrap] ACTIONPHASE_${TAG}_DONE $(date -Is)"
