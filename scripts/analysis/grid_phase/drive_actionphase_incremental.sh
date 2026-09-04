#!/usr/bin/env bash
# 수집 **진행 중** 증분 파이프 드라이버 (승준 원격, CPU 전용).
#
# 수집 세션이 15분마다 갱신하는 "완료 instruction 만 담긴 부분 인덱스"를 지켜보다가,
# 새로 완료된 instruction 이 보이면 그것만 segA 추출하고, shard 가 늘 때마다 **임시**
# AE·KMeans 번들(ae_k<K>_partial/)을 다시 만든다. 완주 후 정식 번들(ae_k<K>/)은
# run_actionphase_remote.sh 를 PARTIAL_AE 없이 한 번 더 돌려 만든다.
#
# 왜 드라이버가 따로 있나: 추출 러너는 "지금 있는 것"을 한 번 처리하고 끝난다. 수집이
# 몇 시간에 걸쳐 instruction 을 하나씩 끝내므로, 폴링·증분 판정·임시 AE 재생성을 담는
# 층이 필요하다. 여기서 **부분 인덱스를 스냅샷으로 복사해 쓰는** 것이 핵심이다 —
# 원본은 15분마다 원자적으로 덮어써지므로, 추출 도중 바뀌면 감사 모수가 흔들린다.
#
# 실행 (승준):
#   setsid nohup env TAG=v6 \
#     PARTIAL_INDEX=~/workspace/temporal_vla/outputs/collect/partial/index_v6_complete_instr_77e745c37b0f.tsv \
#     bash ~/workspace/temporal_vla/scripts/analysis/grid_phase/drive_actionphase_incremental.sh \
#     > ~/workspace/logs/actionphase_drive.log 2>&1 < /dev/null &
#
# 정지: touch $OUT/STOP  (또는 kill). 완주(모든 instruction 처리) 시 스스로 종료한다.
set -euo pipefail

TAG="${TAG:?TAG 필요 (예: v6)}"
PARTIAL_INDEX="${PARTIAL_INDEX:?PARTIAL_INDEX 필요 — 완료 instruction 만 담긴 index tsv}"

REPO="${REPO:-$HOME/workspace/temporal_vla}"
STORE="${STORE:-$HOME/datasets/temporal_vla_store/groot/n15}"
OUT="${OUT:-$STORE/analysis/grid_phase_${TAG}}"
K="${K:-8}"
POLL_SEC="${POLL_SEC:-600}"
EXPECT_INSTR="${EXPECT_INSTR:-12}"     # 최종 instruction 수 (완주 판정)
RUNNER="$REPO/scripts/analysis/grid_phase/run_actionphase_remote.sh"

mkdir -p "$OUT" "$OUT/segA"
SNAP="$OUT/index_partial_snapshot.tsv"

slug_of() { local s="${1//\//_}"; echo "${s// /_}"; }

echo "[drive] 시작 $(date -Is) — TAG=$TAG poll=${POLL_SEC}s"
echo "[drive] 부분 인덱스: $PARTIAL_INDEX"
echo "[drive] 산출: $OUT (segA/, ae_k${K}_partial/)"

while true; do
  if [[ -e "$OUT/STOP" ]]; then
    echo "[drive] STOP 파일 감지 — 종료 $(date -Is)"; exit 0
  fi
  if [[ ! -s "$PARTIAL_INDEX" ]]; then
    echo "[drive] 부분 인덱스 아직 없음/빈 파일 — 대기"; sleep "$POLL_SEC"; continue
  fi

  # 폴링 중 원본이 덮어써져도 흔들리지 않게 스냅샷을 뜬 뒤 그것만 본다.
  cp -f "$PARTIAL_INDEX" "$SNAP"
  n_rows=$(( $(wc -l < "$SNAP") - 1 ))
  if [[ "$n_rows" -le 0 ]]; then
    echo "[drive] 완료 instruction 0건 — 대기"; sleep "$POLL_SEC"; continue
  fi

  mapfile -t done_instr < <(awk -F'\t' '
    NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
    { print $c["grid_instruction"] }' "$SNAP" | sort -u)

  todo=()
  for instr in "${done_instr[@]}"; do
    [[ -s "$OUT/segA/$(slug_of "$instr").npz" ]] || todo+=("$instr")
  done

  echo "[drive] $(date +%T) 완료 instruction ${#done_instr[@]} / shard 보유 $(ls "$OUT"/segA/*.npz 2>/dev/null | wc -l) / 신규 ${#todo[@]}"

  if [[ "${#todo[@]}" -gt 0 ]]; then
    csv=$(IFS=,; echo "${todo[*]}")
    echo "[drive] 추출 발주: $csv"
    # 추출만 (임시 AE 는 아래에서 한 번에 — instruction 마다 돌리면 낭비다)
    if TAG="$TAG" INDEX="$SNAP" INSTR_CSV="$csv" OUT="$OUT" K="$K" \
         bash "$RUNNER"; then
      echo "[drive] 추출 완료: $csv"
    else
      rc=$?
      echo "[drive] ERROR: 추출 실패 rc=$rc ($csv) — 다음 주기에 재시도" >&2
      sleep "$POLL_SEC"; continue
    fi

    # shard 가 늘었으니 임시 번들 재생성 (2개 이상일 때만)
    n_shard=$(ls "$OUT"/segA/*.npz 2>/dev/null | wc -l)
    if [[ "$n_shard" -ge 2 ]]; then
      echo "[drive] 임시 AE 재생성 (shard $n_shard)"
      if TAG="$TAG" INDEX="$SNAP" OUT="$OUT" K="$K" PARTIAL_AE=1 \
           INSTR_CSV="$csv" bash "$RUNNER"; then
        echo "[drive] 임시 번들 갱신: $OUT/ae_k${K}_partial/ae_bundle_k${K}.npz (shard $n_shard)"
      else
        echo "[drive] ERROR: 임시 AE 실패 — shard 는 남아 있으니 다음 주기 재시도" >&2
      fi
    fi
  fi

  n_shard=$(ls "$OUT"/segA/*.npz 2>/dev/null | wc -l)
  if [[ "$n_shard" -ge "$EXPECT_INSTR" ]]; then
    echo "[drive] instruction ${n_shard}/${EXPECT_INSTR} 전부 추출됨 — 드라이버 종료 $(date -Is)"
    echo "[drive] 다음 단계(사람/세션 판단): 최종 인덱스로 정식 번들"
    echo "[drive]   TAG=$TAG INDEX=<최종 index> bash $RUNNER"
    exit 0
  fi
  sleep "$POLL_SEC"
done
