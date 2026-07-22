#!/usr/bin/env bash
# pq3 read-only 전황 대시보드 — 어디서 무슨 작업이 돌고 있는지 한 화면 요약.
# 사용:  watch -n 30 bash scripts/utils/pq3_dashboard.sh        (30초 갱신)
#        bash scripts/utils/pq3_dashboard.sh                    (1회 출력)
#
# 원칙: 전 조회 read-only (파일 tail / pgrep / nvidia-smi / wc 만). 실험에 무영향.
# 호스트 4개(로컬·승준·a100 50·a100 48)를 병렬 ssh 로 긁어 5초 내 렌더.
# ssh 실패 시 해당 섹션만 "unreachable" — 다른 섹션은 정상 출력.
set -uo pipefail
REPO="${REPO:-/home/dongkyu/pkt_ws/temporal_vla}"
SE="$REPO/outputs/eval/robocasa/groot_n15/steer_eval_pq3"
RR="$REPO/outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts"
RQ='pkt_ws/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval_pq3'
SSH_OPT=(-o ConnectTimeout=5 -o BatchMode=yes)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# ── 원격 수집 (병렬) ──────────────────────────────────────────────────────────
ssh -p 11112 "${SSH_OPT[@]}" kimseungjun@166.104.146.37 '
  p=$(pgrep -af "fit_phase_conceptor|pq3_gated_gate" | grep -v pgrep | sed "s/^/  proc: /" | cut -c1-110)
  [ -n "$p" ] && echo "$p" || echo "  proc: (fit/gate 프로세스 없음)"
  for f in $(ls -t /tmp/remote_compute_logs/*.log 2>/dev/null | head -3); do
    echo "  $(basename $f .log): $(tail -1 $f | cut -c1-95)"
  done
  df -h ~/datasets 2>/dev/null | awk "NR==2{print \"  HDD(datasets) 여유 \"\$4\" (\"\$5\" 사용)\"}"
' > "$TMP/sj" 2>/dev/null || echo "  unreachable" > "$TMP/sj" &

ssh "${SSH_OPT[@]}" junhyeong@166.104.35.50 '
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F", " "\$1==0{print \"  GPU0: \"\$2\" / util \"\$3}"
  q=~/'"$RQ"'/work_queue
  echo "  E큐: 남음 $(grep -c . $q/queue.tsv 2>/dev/null || echo 0) · 실행중 $(ls $q/running 2>/dev/null | wc -l) · 완료 $(wc -l < $q/done/done.tsv 2>/dev/null || echo 0) · 실패 $(wc -l < $q/failed/failed.tsv 2>/dev/null || echo 0)"
  for f in /tmp/pq3_lane_w2_*.log; do [ -f "$f" ] && echo "  $(basename $f .log): $(tail -1 $f | cut -c1-95)"; done
  t=~/'"$RQ"'/c0_scan_remote/pq3_pizza_50_scan_progress.tsv
  [ -f "$t" ] && tail -1 "$t" | awk -F"\t" "{print \"  pizza스캔: found \"\$6\"/\"\$7\" (seed \"\$4\", scanned \"\$5\")\"}"
  pgrep -f select_instruction_seeds >/dev/null && echo "  스캔 프로세스: ALIVE" || echo "  스캔 프로세스: 없음"
' > "$TMP/w50" 2>/dev/null || echo "  unreachable" > "$TMP/w50" &

ssh "${SSH_OPT[@]}" junhyeong@166.104.35.48 '
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F", " "\$1==2{print \"  GPU2: \"\$2\" / util \"\$3}"
  q=~/'"$RQ"'/work_queue
  echo "  E큐: 남음 $(grep -c . $q/queue.tsv 2>/dev/null || echo 0) · 실행중 $(ls $q/running 2>/dev/null | wc -l) · 완료 $(wc -l < $q/done/done.tsv 2>/dev/null || echo 0) · 실패 $(wc -l < $q/failed/failed.tsv 2>/dev/null || echo 0)"
  for f in /tmp/pq3_lane_w48_*.log; do [ -f "$f" ] && echo "  $(basename $f .log): $(tail -1 $f | cut -c1-95)"; done
  for c in beer pizza; do
    t=$(ls ~/'"$RQ"'/c0_scan_remote/pq3_${c}*_scan_progress.tsv 2>/dev/null | head -1)
    [ -f "$t" ] && tail -1 "$t" | awk -F"\t" -v c=$c "{print \"  \"c\"스캔: found \"\$6\"/\"\$7\" (seed \"\$4\")\"}"
  done
  pgrep -f select_instruction_seeds >/dev/null && echo "  스캔 프로세스: ALIVE" || echo "  스캔 프로세스: 없음"
' > "$TMP/w48" 2>/dev/null || echo "  unreachable" > "$TMP/w48" &

# ── 로컬 수집 ────────────────────────────────────────────────────────────────
{
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F', ' '$1<=2 {print "  GPU"$1": "$2" / util "$3}'
  p=$(pgrep -af 'pq3_collect_cell|beta_sweep|pq3_cell_runner|pq3_c0_scan|ppcc_sweep_chain' \
      | grep -vE 'pgrep|dashboard' | awk '{$1="";print "  proc:" $0}' | cut -c1-110 | sort -u | head -6)
  [ -n "$p" ] && echo "$p" || echo "  proc: (수집/sweep 프로세스 없음)"
  for f in $(ls -t "$SE/logs/"*.log 2>/dev/null | head -4); do
    echo "  $(basename "$f" .log): $(tail -1 "$f" | cut -c1-95)"
  done
} > "$TMP/local" 2>/dev/null

# ── sweep 진행 (perinstr β sweep, 로컬 sweep_perinstr) ───────────────────────
# E work_queue 밖 단계 — cell×조건(perm/gated×b01/b03) 별 wrote/목표 집계.
{
  SWDIR="$SE/sweep_perinstr"
  shown=0
  if [ -d "$SWDIR" ]; then
    for cd in "$SWDIR"/*/; do
      [ -d "${cd}logs" ] || continue
      cell=$(basename "$cd")
      man="$SE/manifests/$cell/sweep_manifest.tsv"
      tgt=$(awk -F'\t' '$0!~/^#/ && $1 ~ /^[0-9]+$/{c++} END{print c+0}' "$man" 2>/dev/null)
      case "$tgt" in ''|*[!0-9]*|0) tgt='?';; esac
      line=''
      for cond in perm_b01 perm_b03 gated_b01 gated_b03; do
        if ls "${cd}logs/sweep_${cond}_"w*.log >/dev/null 2>&1; then
          n=$(grep -hE 'wrote task' "${cd}logs/sweep_${cond}_"w*.log 2>/dev/null | wc -l)
          mk=''; [ "$n" = "$tgt" ] && mk='✓'
          line="$line ${cond//_/ } ${n}/${tgt}${mk} ·"
        else
          line="$line ${cond//_/ } —/${tgt} ·"
        fi
      done
      printf '  %-24s%s\n' "$cell" "${line% ·}"
      shown=1
    done
  fi
  [ "$shown" = 1 ] || echo "  (진행 중 sweep 없음)"
} > "$TMP/sweep" 2>/dev/null

# ── 전송 상태 ────────────────────────────────────────────────────────────────
{
  tot=0
  for d in "$RR"/*/pq3_*; do
    [ -f "$d/SHIPPED.tsv" ] || continue
    n=$(wc -l < "$d/SHIPPED.tsv"); tot=$((tot+n))
    printf '  %-28s shipped %s\n' "$(basename "$d")" "$n"
  done
  echo "  합계 shipped: $tot pkl"
  r=$(pgrep -af '[r]sync -a' | head -2 | awk '{print "  rsync 진행: "$NF}')
  [ -n "$r" ] && echo "$r" || echo "  rsync: 진행 중 없음"
} > "$TMP/ship" 2>/dev/null

# ── 대기 중 사용자 게이트 (파일 존재로 자동 파생) ─────────────────────────────
{
  FIT="$SE/fit"
  [ -s "$FIT/gate_PPCC_enforced.json" ] \
    && echo "  PPCC gated 게이트: 완료 ($(python3 -c "import json;d=json.load(open('$FIT/gate_PPCC_enforced.json'));t=list(d)[0];print('pass' if d[t]['pass'] else 'FAIL')" 2>/dev/null))" \
    || echo "  ▷ PPCC gated 게이트(승준) 완료 대기"
  [ -s "$FIT/beta_PPCC.json" ] \
    && echo "  PPCC β/layer: 확정됨" \
    || echo "  ▷ PPCC β sweep + L4/L15 성적 비교 → 사용자 게이트 대기"
  [ -s "$FIT/beta_OpenDrawer.json" ] && echo "  OpenDrawer β: perm 0.1 / gated 0.3 (확정)"
} > "$TMP/gates" 2>/dev/null

wait

# ── 렌더 ─────────────────────────────────────────────────────────────────────
hr() { printf '─%.0s' $(seq 1 100); echo; }
echo "pq3 전황판  $(date '+%F %T')  (read-only · watch -n 30 권장)"
hr
echo "[로컬]   수집/sweep GPU0-2"; cat "$TMP/local"
hr
echo "[sweep]   perinstr β sweep 진행 (wrote/목표 · ✓=완료 · E큐 밖 단계)"; cat "$TMP/sweep"
hr
echo "[승준 .37]   conceptor fit · gated 게이트 · HDD 아카이브"; cat "$TMP/sj"
hr
echo "[a100 50]   E lane(GPU0) · pizza 스캔"; cat "$TMP/w50"
hr
echo "[a100 48]   E lane(GPU2) · beer/pizza 스캔"; cat "$TMP/w48"
hr
echo "[전송]   fit pkl 승준 직송 ledger"; cat "$TMP/ship"
hr
echo "[대기 중 결정]"; cat "$TMP/gates"
