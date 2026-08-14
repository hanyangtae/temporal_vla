#!/bin/bash
# 클린 replay 영상으로 승준 노드의 kanu 원본 video.mp4 **교체** (러너 완주 후 실행).
#
# 교체 자격 게이트: results.jsonl 에서 error 없고 diverged=false 이며
# **eef_max_dev_m == 0 (비트 일치)** 인 판만. 발산/실패 판은 원본을 그대로 두고 목록만 출력
# (임의 삭제 금지 — 사용자 결정 사항).
#
# 사용:
#   bash scripts/analysis/grid_phase/push_clean_videos.sh <OUT_DIR> <MANIFEST_TSV> [--dry-run]
#     OUT_DIR      = 러너 --out-dir (results.jsonl + <stem>.mp4)
#     MANIFEST_TSV = extract_bundles_kanu.sh 산출 manifest.tsv (stem \t grid 상대경로)
set -euo pipefail

OUT="${1:?OUT_DIR}"
MAN="${2:?MANIFEST_TSV}"
DRY="${3:-}"
T="${REMOTE_USER:-kimseungjun}@${REMOTE_HOST:-166.104.146.37}"
PORT="${REMOTE_PORT:-11112}"
GRID="${REMOTE_GRID:-/home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid}"

ELIG="$OUT/eligible_stems.txt"
python3 - "$OUT/results.jsonl" "$ELIG" "$OUT/rejected_stems.txt" <<'PY'
import json, sys
res, ok_p, bad_p = sys.argv[1], sys.argv[2], sys.argv[3]
seen = {}
for ln in open(res, encoding="utf-8"):
    if ln.strip():
        d = json.loads(ln); seen[d.get("stem", "")] = d       # 마지막 시도 기준
ok, bad = [], []
for s, d in sorted(seen.items()):
    if "error" not in d and not d.get("diverged") and d.get("eef_max_dev_m") == 0:
        ok.append(s)
    else:
        bad.append(f"{s}\t{d.get('error') or ('diverged' if d.get('diverged') else 'eef=' + str(d.get('eef_max_dev_m')))}")
open(ok_p, "w").write("\n".join(ok) + "\n")
open(bad_p, "w").write("\n".join(bad) + ("\n" if bad else ""))
print(f"[push] 교체 자격 {len(ok)} / 미자격 {len(bad)}")
PY

while IFS= read -r stem; do
  [[ -z "$stem" ]] && continue
  src="$OUT/$stem.mp4"
  rel="$(awk -F'\t' -v s="$stem" '$1==s{print $2; exit}' "$MAN")"
  if [[ -z "$rel" || ! -f "$src" ]]; then
    echo "[push] SKIP $stem (src=$src rel=$rel)"; continue
  fi
  if [[ "$DRY" == "--dry-run" ]]; then
    echo "[dry] $src -> $T:$GRID/$rel"
  else
    rsync -a -e "ssh -p $PORT" "$src" "$T:$GRID/$rel"
    echo "[push] $stem -> $rel"
  fi
done < "$ELIG"

echo "[push] 미교체 목록: $OUT/rejected_stems.txt"
