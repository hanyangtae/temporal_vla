#!/usr/bin/env bash
# exp3(구 pq3) C0 — PPCC 신규 물체 2종 seed 스캔 (계획서 v9 §C0, env-only·정책 없음).
# ① 빈도 pass: PPCC seed 대역을 샘플링해 _scan_samples.tsv 의 instruction 분포에서
#    물체 빈도 상위 2종 선정 (bread·potato 제외 — SR 은 절대 보지 않음, Codex R1 #5/#9)
# ② 신규 cells-config 행 생성 (configs/robocasa/exp3_ppcc_new_cells.tsv, index 15/16)
# ③ 타깃 pass: 두 물체를 cell 당 TARGET(기본 60)개까지 정확 일치 스캔
#    (--exclude-selected-seeds 로 기존 선발분 재사용 방지)
#
# 사용: bash exp3_c0_scan.sh  (env: FREQ_N=400 TARGET=60 SEED_START=500000)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../../../.." && pwd)"
FREQ_N="${FREQ_N:-400}"
TARGET="${TARGET:-60}"
SEED_START="${SEED_START:-500000}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/robocasa/groot_n15/steer_eval_exp3/c0_scan}"
EXISTING_TSV="outputs/eval/robocasa/groot_n15/coast4_reused_remote/manifests/selected_instruction_seeds.tsv"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
SCANNER="/temporal_vla/scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py"
mkdir -p "$REPO_ROOT/$OUT_ROOT"

# ── ① 빈도 pass: ppcc_bread cell 로 FREQ_N seed 샘플링 (target 미달 rc 무시 —
#      samples 사이드카가 목적) ────────────────────────────────────────────────
FREQ_TSV="$OUT_ROOT/freq_pass.tsv"
docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
  python "$SCANNER" --cell-id ppcc_bread \
  --output-tsv "/temporal_vla/$FREQ_TSV" --ep-meta-dir "/temporal_vla/$OUT_ROOT/ep_meta_freq" \
  --target-per-cell "$FREQ_N" --max-seeds-per-cell "$FREQ_N" --seed-start "$SEED_START" \
  --resume || true
SAMPLES="$REPO_ROOT/${FREQ_TSV%.tsv}_scan_samples.tsv"
[ -f "$SAMPLES" ] || { echo "[c0] 빈도 pass samples 없음: $SAMPLES"; exit 1; }

# ── ② 물체 빈도 상위 2종 → 신규 cells-config ──────────────────────────────────
NEW_CFG="$REPO_ROOT/configs/robocasa/exp3_ppcc_new_cells.tsv"
python3 - "$SAMPLES" "$NEW_CFG" <<'PYEOF'
import csv, re, sys
from collections import Counter
samples, out_cfg = sys.argv[1], sys.argv[2]
pat = re.compile(r"Pick the (.+?) from the counter and place it in the cabinet\.")
cnt = Counter()
with open(samples, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        m = pat.match((row.get("instruction") or "").strip())
        if m:
            cnt[m.group(1)] += 1
for banned in ("bread", "potato"):
    cnt.pop(banned, None)
if len(cnt) < 2:
    sys.exit(f"[c0] 물체 후보 부족: {dict(cnt)}")
top2 = [obj for obj, _ in cnt.most_common(2)]
print(f"[c0] 물체 빈도: {dict(cnt.most_common())} -> 선정 {top2} (SR 미사용)")
rows = []
for i, obj in enumerate(top2):
    slug = obj.replace(" ", "_")
    rows.append({
        "cell_index": 15 + i,
        "cell_id": f"ppcc_{slug}",
        "task": "PickPlaceCounterToCabinet",
        "env_name": "robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env",
        "canonical_instruction": f"Pick the {obj} from the counter and place it in the cabinet.",
        "target_episodes": 60,
        "seed_search_start": 100000,
    })
with open(out_cfg, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    w.writeheader()
    [w.writerow(r) for r in rows]
print(f"[c0] cells-config -> {out_cfg}")
PYEOF

# ── ③ 타깃 pass: 두 신규 cell 을 TARGET 개까지 정확 일치 스캔 ──────────────────
OUT_TSV="$OUT_ROOT/pq3_ppcc_new_seeds.tsv"  # 기존 on-disk 산출물명 유지 (data key)
for CELL in $(awk -F'\t' 'NR>1{print $2}' "$NEW_CFG"); do
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
    python "$SCANNER" \
    --cells-config "/temporal_vla/configs/robocasa/exp3_ppcc_new_cells.tsv" \
    --cell-id "$CELL" \
    --output-tsv "/temporal_vla/$OUT_TSV" --ep-meta-dir "/temporal_vla/$OUT_ROOT/ep_meta" \
    --target-per-cell "$TARGET" \
    --exclude-selected-seeds "/temporal_vla/$EXISTING_TSV" \
    --resume
done
echo "[c0] 완료 -> $REPO_ROOT/$OUT_TSV (make_exp3_manifests plan 의 --seeds-tsv 로 사용)"
