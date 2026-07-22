#!/usr/bin/env bash
# S4 (fit 절단 하드 게이트): --record-start-manifest 절단 후 fit_inputs.json 산술 assert +
# manifest 누락 pkl fail-loud. 입력 = patchceil passB full-token pkl (로컬 기성 자산, GPU 불필요).
# 사용: bash s4_fit_truncation.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/smoke_common.sh"
S4="${SMOKE_ROOT}/s4"; mkdir -p "$S4"
RS_START=6
PASSB="${MAIN_HOST}/outputs/eval/robocasa/groot_n15/patchceil/ppcc_bread_s300033/passB/raw_rollouts"

mapfile -t PKLS < <(find "$PASSB" -name "*.pkl" | sort | head -8)
[ "${#PKLS[@]}" -ge 6 ] || { echo "ABORT: passB pkl ${#PKLS[@]}개 (<6)"; exit 12; }

MAN="$S4/manifest.tsv"; RS="$S4/record_start.tsv"; RS_MISS="$S4/record_start_missing.tsv"
: > "$MAN"; : > "$RS"; : > "$RS_MISS"
for p in "${PKLS[@]}"; do
  succ=0; [[ "$p" == *succ1.pkl ]] && succ=1
  printf '%s\t%s\t300033\n' "$(to_cont "$p")" "$succ" >> "$MAN"
  printf '%s\t%s\n' "$(to_cont "$p")" "$RS_START" >> "$RS"
done
head -n $(( ${#PKLS[@]} - 1 )) "$RS" > "$RS_MISS"   # 마지막 pkl 누락판 (fail-loud 검증)

FIT="${WT_CONT}/scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py"
OUT="$S4/fit_rs${RS_START}"

echo "[s4] fit with --record-start-manifest (${#PKLS[@]} pkl, start=${RS_START})"
docker exec lerobot python "$FIT" \
  --cell "PickPlaceCounterToCabinet/s4smoke" --manifest "$(to_cont "$MAN")" \
  --record-start-manifest "$(to_cont "$RS")" --groups global --layers 8,12 \
  --denoise pool --token-pool mean --min-per-class 3 --out-dir "$(to_cont "$OUT")" \
  2>&1 | tail -5
check "fit 절단 산술 (fit-audit)" judge_c fit-audit "$(to_cont "$OUT")/fit_inputs.json" --expect-start "$RS_START"

echo "[s4] manifest 누락 pkl fail-loud 확인"
if docker exec lerobot python "$FIT" \
  --cell "PickPlaceCounterToCabinet/s4smoke_miss" --manifest "$(to_cont "$MAN")" \
  --record-start-manifest "$(to_cont "$RS_MISS")" --groups global --layers 8 \
  --denoise pool --min-per-class 3 --out-dir "$(to_cont "$S4/fit_miss")" >/dev/null 2>&1; then
  echo "[FAIL] 누락 pkl 인데 fit 이 성공함 (fail-loud 위반)"; N_FAIL=$((N_FAIL+1))
else
  echo "PASS 누락 pkl fail-loud (exit!=0)"; N_PASS=$((N_PASS+1))
fi

summary S4
