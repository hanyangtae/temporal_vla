#!/usr/bin/env bash
# exp5-2 — 섭동(C1/P1/P2) 하 steering ΔSR eval 러너.
#
# arm = off(개입 없음) | setm(setpoint mean-diff, DiT) | setm_pl(DiT 라벨순열 위약)
#     | setm_vl(VL vlln 토큰평균 setpoint) | setm_vl_pl(VL 위약).
# 같은 (scene seed, inference seed, spec_seed) 로 arm 간 paired — 행 원천은 exp4-2 P0 의
# grid.tsv (config × ep × inference_seed × spec json). spec json 이 spec_seed 를 품고 있어
# arm 을 바꿔도 섭동 실현이 bitwise 동일하다.
#
# 재사용:
#   - collect_perturb_grid.sh : serve 기동/health/kill·docker exec collector·--perturb-spec·
#                               host→container 경로 변환(to_cont)·done_mark resume
#   - exp4_1/run_oracle_rescue.sh : serve_steer_flags(gated NPZ base 스캔 → setpoint_seg 플래그),
#                                   --steer-from-record latch, 사이드카 succ{0,1} 판정
#
# 사용 (worktree 에서, 빈 GPU 확인 후):
#   CFG=c1_s200 ARM=off        GPUS_L="0" PORTS_L="8490" bash run_perturb_steer_eval.sh
#   CFG=p1_d003 ARM=setm    NPZ=<...>/setM    GPUS_L="0 0" PORTS_L="8490 8491" bash ...
#   CFG=p1_d003 ARM=setm_pl NPZ=<...>/setM_pl GPUS_L="0 0" PORTS_L="8490 8491" bash ...
#   CFG=c1_s200 ARM=setm_vl NPZ=<...>/c1_s200/VLsetM/setM   GPUS_L="0" PORTS_L="8490" bash ...
#     (VL arm 은 phase 게이팅·latch 없이 서버 수명 내내 상시 적용 — C1 은 전 구간 섭동)
#   DRY_RUN=1 ... bash run_perturb_steer_eval.sh   # serve/collector 명령만 echo
# detached 권장: setsid nohup bash ... > <OUT>/logs/orch.log 2>&1 < /dev/null &
set -uo pipefail

# ---- 입력 -----------------------------------------------------------------------------
CFG="${CFG:?섭동 config 명 (c1_s200|p1_d003|p2_f040d2 ...)}"
ARM="${ARM:?off|setm|setm_pl|setm_vl|setm_vl_pl}"
case "$ARM" in off|setm|setm_pl|setm_vl|setm_vl_pl) ;; *) echo "ABORT: unknown ARM=$ARM"; exit 2 ;; esac
case "$ARM" in setm_vl|setm_vl_pl) IS_VL=1 ;; *) IS_VL=0 ;; esac

# cell 파라미터 (기본 ppcc_bread — exp4-2 P0 와 동일 cell)
CELL_ID="${CELL_ID:-ppcc_bread}"
TASK="${TASK:-PickPlaceCounterToCabinet}"
ENVN="${ENVN:-robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env}"
CELL_INDEX="${CELL_INDEX:-5}"
SEED="${SEED:-100084}"
INSTR="${INSTR:-Pick the bread from the counter and place it in the cabinet.}"

NAS="${NAS:-5}"; MAXEP="${MAXEP:-720}"
PROFILE="${PROFILE:-configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml}"
SERVE_MODE="${SERVE_MODE:-docker}"          # docker | host
HOST_PY="${HOST_PY:-$HOME/miniconda3/envs/lerobot_050_groot/bin/python}"
BETA="${BETA:-1.0}"                          # setM 기본 β=1.0 (exp4-1 BETA_SETM)
LAYER="${LAYER:-}"                           # 비우면 NPZ 디렉토리에서 자동 감지
NPZ="${NPZ:-}"                               # arm NPZ base (하위 = phase 디렉토리)
NPZ_ROOT="${NPZ_ROOT:-}"                     # 대안: $NPZ_ROOT/$CELL_ID/{setM,setM_pl}
LATCH_OFFSET="${LATCH_OFFSET:-0}"            # P1/P2 latch = trigger_record + offset
DRY_RUN="${DRY_RUN:-0}"
MAX_SERVES="${MAX_SERVES:-2}"                # 원격(pdk_external, GPU 1장 16GB) 기준 상한

# ---- 경로 (스크립트 위치 기준 — REPO 하드코딩 금지) -------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../../../.." && pwd)"
# worktree 면 main-tree 를 유도 (outputs/·grid.tsv 는 main-tree, docker mount 는 main-tree=/temporal_vla)
case "$REPO_ROOT" in
  */.claude/worktrees/*) MAIN_HOST="${REPO_ROOT%%/.claude/worktrees/*}" ;;
  *) MAIN_HOST="$REPO_ROOT" ;;
esac
CONT_REPO="${CONT_REPO:-/temporal_vla}"                 # 컨테이너 안 main-tree mount point
REPO_CONT="${CONT_REPO}${REPO_ROOT#"$MAIN_HOST"}"       # 컨테이너 안 이 worktree 경로
PYPATH="${PYPATH_OVERRIDE:-${CONT_REPO}/src/policies/Isaac-GR00T:${CONT_REPO}/src/benchmarks/robocasa:${CONT_REPO}/src/benchmarks/robosuite:${CONT_REPO}}"

P0="${P0:-${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp42_induced/p0}"
GRID="${GRID:-${P0}/grid.tsv}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/robocasa/groot_n15/exp52}"
case "$OUT_ROOT" in /*) ;; *) OUT_ROOT="${MAIN_HOST}/${OUT_ROOT}" ;; esac

to_cont() { case "$1" in "${MAIN_HOST}"/*) echo "${CONT_REPO}${1#"$MAIN_HOST"}" ;; *) echo "$1" ;; esac; }

GPUS=(${GPUS_L:-0}); PORTS=(${PORTS_L:-8490}); NW=${#PORTS[@]}
[ "$NW" -le "$MAX_SERVES" ] || { echo "ABORT: serve $NW개 > MAX_SERVES=$MAX_SERVES"; exit 2; }
[ "${#GPUS[@]}" -ge "$NW" ] || { echo "ABORT: GPUS_L(${#GPUS[@]}) < PORTS_L($NW)"; exit 2; }

# ---- steering NPZ / serve 플래그 --------------------------------------------------------
# setpoint_seg 는 serve 에서 gated 경로(--steering-phase-npz-base) 전용 (lerobot.py 1058행
# fail loud). permanent arm 도 phase 디렉토리 1개('steer')짜리 gated 등록 + latch POST 로 건다.
# PHASES/LAYER 는 여기서 전역에 확정 (command substitution 안에서 대입하면 subshell 에 갇혀
# 출력 디렉토리명이 'L_' 로 비는 사고가 난다 — DRY_RUN 실측 07-27)
scan_npz_base() {  # $1=NPZ base → PHASES, LAYER 설정
  local base="$1" ph
  [ -d "$base" ] || { echo "ABORT: NPZ base 없음: $base" >&2; return 1; }
  PHASES=""
  for ph in "$base"/*/; do
    [ -d "$ph" ] && ls "$ph"dit_L* >/dev/null 2>&1 && PHASES="${PHASES:+$PHASES,}$(basename "$ph")"
  done
  [ -n "$PHASES" ] || { echo "ABORT: NPZ phase 디렉토리 없음: $base" >&2; return 1; }
  [ -n "$LAYER" ] || LAYER=$(basename "$(ls -d "$base/${PHASES%%,*}"/dit_L* 2>/dev/null | head -1)" | sed 's/dit_L//')
  [ -n "$LAYER" ] || { echo "ABORT: layer 감지 실패: $base" >&2; return 1; }
  [ -f "$base/${PHASES%%,*}/dit_L${LAYER}/conceptors.npz" ] || {
    echo "ABORT: conceptors.npz 없음: $base/${PHASES%%,*}/dit_L${LAYER}" >&2; return 1; }
}
serve_steer_flags() {  # $1=NPZ base (scan_npz_base 선행 필수)
  local base="$1" phases="$PHASES" lyr="$LAYER" serve_base
  serve_base="$base"
  [ "$SERVE_MODE" = host ] || serve_base="$(to_cont "$base")"
  # setpoint_seg 는 --steering-token-select all 필수 (fit=전 토큰 위치별 s_t)
  echo "--steering-phase-npz-base $serve_base --steering-phases $phases --steering-layers $lyr" \
       "--steering-op setpoint_seg --steering-beta $BETA --steering-token-select all"
}

# VL arm(exp5-2): pooled setpoint 단일 NPZ + pathway=vl. phase 디렉토리·layer·latch 없음.
# fit 산출 계약: <fits>/<cfg>/VLsetM/{setM,setM_pl}/conceptors.npz
serve_steer_flags_vl() {  # $1 = conceptors.npz (host 경로)
  local npz="$1" serve_npz="$1"
  [ "$SERVE_MODE" = host ] || serve_npz="$(to_cont "$npz")"
  echo "--steering-npz $serve_npz --steering-op setpoint_vl --steering-pathway vl" \
       "--steering-beta $BETA"
}

STEER_FLAGS=""
if [ "$ARM" != "off" ]; then
  if [ -z "$NPZ" ]; then
    [ -n "$NPZ_ROOT" ] || { echo "ABORT: ARM=$ARM 는 NPZ 또는 NPZ_ROOT 필요"; exit 2; }
    case "$ARM" in
      setm)       NPZ="$NPZ_ROOT/$CELL_ID/setM" ;;
      setm_pl)    NPZ="$NPZ_ROOT/$CELL_ID/setM_pl" ;;
      setm_vl)    NPZ="$NPZ_ROOT/$CFG/VLsetM/setM" ;;
      setm_vl_pl) NPZ="$NPZ_ROOT/$CFG/VLsetM/setM_pl" ;;
    esac
  fi
  case "$NPZ" in /*) ;; *) NPZ="${MAIN_HOST}/${NPZ}" ;; esac
  if [ "$IS_VL" = 1 ]; then
    [ -d "$NPZ" ] && NPZ="${NPZ%/}/conceptors.npz"
    [ -f "$NPZ" ] || { echo "ABORT: VL NPZ 없음: $NPZ"; exit 12; }
    LAYER="VL"
    STEER_FLAGS="$(serve_steer_flags_vl "$NPZ")" || exit 12
  else
    scan_npz_base "$NPZ" || exit 12
    STEER_FLAGS="$(serve_steer_flags "$NPZ")" || exit 12
  fi
else
  LAYER="${LAYER:-NA}"
fi

TAGDIR="${ARM}_L${LAYER}_b${BETA}"
OUT_HOST="${OUT_ROOT}/eval/${CELL_ID}/${CFG}/${TAGDIR}"
OUT_CONT="$(to_cont "$OUT_HOST")"
LOGDIR="${OUT_HOST}/logs"
TSV="${OUT_HOST}/per_episode.tsv"
[ "$DRY_RUN" = 1 ] || mkdir -p "$LOGDIR"

# ---- 행 로드 (grid.tsv 의 실섭동 행만; sham 제외) ----------------------------------------
[ -f "$GRID" ] || { echo "ABORT: grid.tsv 없음: $GRID"; exit 2; }
mapfile -t ROWS < <(awk -F'\t' -v cfg="$CFG" '
  NR==1 { next } $2==cfg && $6 !~ /_sham_/ { print $3 "\t" $4 "\t" $5 "\t" $6 }' "$GRID" | sort -t$'\t' -k1,1n)
if [ -n "${EPS:-}" ]; then
  mapfile -t ROWS < <(printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v f=",${EPS// /,}," 'index(f, ","$1",")')
else
  EP_START="${EP_START:-0}"; N_EP="${N_EP:-0}"
  if [ "$N_EP" -gt 0 ]; then
    mapfile -t ROWS < <(printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v s="$EP_START" '$1>=s' | head -n "$N_EP")
  fi
fi
[ "${#ROWS[@]}" -gt 0 ] || { echo "ABORT: CFG=$CFG 대상 행 0개 ($GRID)"; exit 2; }
echo "[exp52 $CELL_ID/$CFG/$ARM] rows=${#ROWS[@]} layer=$LAYER beta=$BETA mode=$SERVE_MODE gpus=${GPUS[*]:0:$NW} ports=${PORTS[*]}"

# spec json 필드 추출 (trigger_record / spec_seed) — 없으면 NA
spec_field() {  # spec_path key
  python3 - "$1" "$2" <<'PY' 2>/dev/null || echo NA
import json, sys
try:
    print(json.load(open(sys.argv[1])).get(sys.argv[2], "NA"))
except Exception:
    print("NA")
PY
}

# ---- serve ------------------------------------------------------------------------------
start_serves() {
  local i port ok log
  for i in $(seq 0 $((NW - 1))); do
    if [ "$SERVE_MODE" = host ]; then
      local cmd=(env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" PYTHONPATH="$REPO_ROOT/lerobot/src"
                 "$HOST_PY" scripts/serve/lerobot.py --profile "$PROFILE" --host '*'
                 --port "${PORTS[$i]}" --device cuda $STEER_FLAGS)
      if [ "$DRY_RUN" = 1 ]; then echo "[DRY serve host w$i] (cd $REPO_ROOT && setsid nohup ${cmd[*]} > /tmp/exp52_${CFG}_${ARM}_${PORTS[$i]}.log 2>&1 &)"; continue; fi
      ( cd "$REPO_ROOT" && setsid nohup "${cmd[@]}" \
          > "/tmp/exp52_${CFG}_${ARM}_${PORTS[$i]}.log" 2>&1 < /dev/null & )
    else
      local inner="cd ${REPO_CONT} && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} --host '*' --port ${PORTS[$i]} --device cuda ${STEER_FLAGS} > /tmp/exp52_${CFG}_${ARM}_${PORTS[$i]}.log 2>&1 < /dev/null &"
      if [ "$DRY_RUN" = 1 ]; then echo "[DRY serve docker w$i] docker exec -d -e CUDA_VISIBLE_DEVICES=${GPUS[$i]} lerobot bash -lc \"${inner}\""; continue; fi
      docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc "$inner"
    fi
  done
  [ "$DRY_RUN" = 1 ] && return 0
  for port in "${PORTS[@]}"; do
    ok=0
    for _ in $(seq 1 150); do
      curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
      sleep 5
    done
    log="/tmp/exp52_${CFG}_${ARM}_${port}.log"
    echo "[exp52 $CFG/$ARM] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    if [ "$ok" != 1 ]; then
      [ "$SERVE_MODE" = host ] && tail -30 "$log" || docker exec lerobot bash -lc "tail -30 $log" || true
      exit 11
    fi
    # steering 지문 대조 (arm 오배치·무음 identity 방지)
    curl -s -m 3 "http://127.0.0.1:${port}/health" | tr ',' '\n' | grep -E '"op"|"beta"|"mode"' || true
  done
}
kill_serves() {
  local port pid
  for port in "${PORTS[@]}"; do
    if [ "$SERVE_MODE" = host ]; then
      for pid in $(pgrep -f "lerobot.py.*--port ${port}"); do kill "$pid" 2>/dev/null || true; done
    else
      docker exec lerobot bash -lc "pkill -f 'lerobot.py.*--port ${port}'" 2>/dev/null || true
    fi
  done
  sleep 3
}
[ "$DRY_RUN" = 1 ] || trap kill_serves EXIT

# ---- collector --------------------------------------------------------------------------
done_mark() { ls "${OUT_HOST}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${1}--succ"*.json >/dev/null 2>&1; }

run_row() {  # port ep inf spec tag
  local port=$1 ep=$2 inf=$3 spec=$4 tag=$5 latch=() rc trig
  # VL arm 은 gated 등록이 아니라 상시 hook — latch(/steering_phase POST) 없음.
  if [ "$ARM" != "off" ] && [ "$IS_VL" != 1 ]; then
    case "$CFG" in
      c1_*)  latch=(--steer-from-record 0) ;;   # C1 = 전 구간 ON (섭동이 episode 내내 지속)
      p1_*|p2_*)
        trig="$(spec_field "$spec" trigger_record)"
        [ "$trig" = NA ] && { echo "[skip] ep${ep}: trigger_record 없음 ($spec)" >&2; return 1; }
        latch=(--steer-from-record $((trig + LATCH_OFFSET))) ;;
      *) echo "ABORT: CFG=$CFG latch 규약 미정" >&2; return 1 ;;
    esac
  fi
  local cmd=(docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa
    python "${REPO_CONT}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py"
    --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN"
    --output-dir "${OUT_CONT}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX"
    --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1
    --seed "$SEED" --inference-seed "$inf" --n-action-steps "$NAS"
    --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready
    --proximity-phases --no-features --expect-chunk-len 16
    --perturb-spec "$(to_cont "$spec")" --run-tag "exp52_${ARM}_${tag}"
    "${latch[@]}")
  if [ "$DRY_RUN" = 1 ]; then echo "[DRY collect] ${cmd[*]}"; return 0; fi
  "${cmd[@]}" 2>&1 | grep -E "^wrote|Error|Traceback|ABORT"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || echo "[collector FAIL rc=$rc] ep=$ep tag=$tag" >&2
  return "$rc"
}

run_worker() {  # wid
  local wid=$1 i nfail=0 ep inf spec tag
  for i in "${!ROWS[@]}"; do
    [ $((i % NW)) -eq "$wid" ] || continue
    IFS=$'\t' read -r ep inf spec tag <<< "${ROWS[$i]}"
    if [ "$DRY_RUN" != 1 ] && done_mark "$ep"; then echo "[w$wid] skip ep${ep}"; continue; fi
    echo "[w$wid] ${CFG}/${ARM} ep${ep}"
    run_row "${PORTS[$wid]}" "$ep" "$inf" "$spec" "$tag" || nfail=$((nfail + 1))
  done
  [ "$DRY_RUN" = 1 ] || echo "$nfail" > "${LOGDIR}/w${wid}.nfail"
}

start_serves
if [ "$DRY_RUN" = 1 ]; then
  for wid in $(seq 0 $((NW - 1))); do run_worker "$wid"; done
  echo "[DRY] out=$OUT_HOST tsv=$TSV"
  exit 0
fi
for wid in $(seq 0 $((NW - 1))); do run_worker "$wid" > "${LOGDIR}/w${wid}.log" 2>&1 & done
wait
kill_serves; trap - EXIT
# 정리 확인 (cleanup-policy: 종료 보고 전 잔여 프로세스 점검)
for port in "${PORTS[@]}"; do
  if [ "$SERVE_MODE" = host ]; then
    pgrep -f "lerobot.py.*--port ${port}" >/dev/null && echo "[WARN] serve ${port} 잔존"
  else
    docker exec lerobot bash -lc "pgrep -f 'lerobot.py.*--port ${port}' >/dev/null" \
      && echo "[WARN] serve ${port} 잔존 (container)"
  fi
done

# ---- per-episode 결과 -------------------------------------------------------------------
{
  printf 'ep\tsuccess\tarm\tcfg\tlayer\tbeta\tscene_seed\tinference_seed\tspec_seed\ttrigger_record\ttag\n'
  s=0; f=0; miss=0
  for i in "${!ROWS[@]}"; do
    IFS=$'\t' read -r ep inf spec tag <<< "${ROWS[$i]}"
    j=$(ls "${OUT_HOST}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.json 2>/dev/null | head -1)
    case "$j" in
      *succ1.json) succ=1; s=$((s + 1)) ;;
      *succ0.json) succ=0; f=$((f + 1)) ;;
      *) succ=NA; miss=$((miss + 1)) ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ep" "$succ" "$ARM" "$CFG" "$LAYER" \
      "$BETA" "$SEED" "$inf" "$(spec_field "$spec" spec_seed)" "$(spec_field "$spec" trigger_record)" "$tag"
  done
  echo "# n=${#ROWS[@]} succ=$s fail=$f miss=$miss" >&2
  echo "$s $f $miss" > "${LOGDIR}/.summary"
} > "$TSV"
read -r s f miss < "${LOGDIR}/.summary"
echo -e "[exp52 $CELL_ID/$CFG/$ARM] succ=${s}\tfail=${f}\tmiss=${miss}\t/${#ROWS[@]}\t-> $TSV"
if [ "$miss" -ne 0 ]; then
  echo "[exp52 $CELL_ID/$CFG/$ARM] INCOMPLETE — 결측 ${miss}행 (nfail: $(cat "${LOGDIR}"/w*.nfail 2>/dev/null | tr '\n' ' '))"
  exit 13
fi
touch "${LOGDIR}/EXP52_${CELL_ID}_${CFG}_${ARM}_DONE"
echo "[exp52 $CELL_ID/$CFG/$ARM] done"
