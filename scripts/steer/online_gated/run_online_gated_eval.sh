#!/usr/bin/env bash
# online-gated steering closed-loop SR eval 러너
# ("SAFE 감지 → phase 판정 → phase 연산자 개입" 한 판에서 다 도는 arm 비교).
#
# arm (한 task cell = slug 하나에 대해):
#   base           개입 없음 (serve steering 없음, detector 없음)
#   online         setM_seg gated 등록 + serve --failure-detector
#                  → collector --gated-steering-mode online (발화 전 off, 발화 후 phase-follow)
#   online_fut     위와 동일하되 future-only. setpoint_seg 는 변형 NPZ(setM_seg_fut —
#                  make_seg_mask_variant.py), conceptor 계열은 같은 NPZ + token-select future
#   online_pl      setM_seg_pl (라벨순열·dose-match 위약) + 같은 detector
#   oracle_always  setM_seg gated 등록 + collector --gated-steering
#                  (라벨러 phase 를 매 스텝 POST, latch 없이 처음부터 = 개입 상한)
#
# EP_MODE (episode 열거 방식)
#   replay (기본) — grid 수집 셀을 **그대로 재생**. (scene s, noise m) 의 수집 당시
#     env_seed·inference_seed 를 index 에서 읽어 동일 조건으로 재실행한다
#     (`replay_cells.py`). 같은 머신이면 base arm 은 수집 결과와 (거의) 동일하게
#     재현되므로 개입 arm 과의 차이 = 셀 단위 **구제 / 파손**.
#     연산자·detector fit 은 scene 0–4 × noise 0–4 (s5m5) 이고, eval 셀은 10 scene 전체
#     × EVAL_NOISES(기본 0,1,5,6) → 사분면(seen/unseen scene × seen/unseen noise)당
#     10 셀 = 40 판/arm.
#   fresh — 구 방식. grid 와 같은 scene, inference_seed 만 새 대역(EVAL_INF_SEED_BASE).
#
# 발사 (빈 GPU 확인 후, 반드시 setsid — agent 백그라운드 job 은 harness 가 중간에 kill):
#
#   GPUS=4,5 SLUGS=OpenDrawer_left NPZ_ROOT=outputs/steer/online_pipe \
#   INDEX_TSV=outputs/steer/online_pipe/manifests/index_rollouts.tsv \
#   setsid nohup bash scripts/steer/online_gated/run_online_gated_eval.sh \
#     > outputs/eval/robocasa/groot_n15/online_gated/logs/orch.log 2>&1 < /dev/null &
#
#   DRY_RUN=1 GPUS=4 bash scripts/steer/online_gated/run_online_gated_eval.sh   # 명령만 echo
#
# 머신 분산: 이 러너는 **대상 머신 위에서** 돌린다 (ssh 오케스트레이션 없음).
#   srv48/srv50 처럼 lerobot 컨테이너가 없는 머신은 SERVE_MODE=host + SERVE_PY 지정.
#   실제로 어느 머신에서 돌았는지는 결과 옆 MACHINE.txt(hostname) 로 판정한다.
#
# 완료 판정은 로그가 아니라 per_episode.tsv 행 수로 한다.
set -euo pipefail

# ── 입력 ──────────────────────────────────────────────────────────────────────
SLUGS="${SLUGS:-OvenRack_out,DishwasherRack_out,OpenDrawer_left}"
ARMS="${ARMS:-base,online,online_pl,online_fut,oracle_always}"
GPUS="${GPUS:?GPUS (쉼표구분 빈 GPU 번호) 필요 — kanu 규칙: 최대 3 GPU}"
SERVES_PER_GPU="${SERVES_PER_GPU:-2}"          # kanu(A4000 16GB)=2 / A100 80GB(srv48·srv50)=6
PORT_BASE="${PORT_BASE:-8700}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"          # 1 이면 타인 프로세스 있어도 강행(비권장)

NPZ_ROOT="${NPZ_ROOT:-outputs/steer/online_pipe}"   # <NPZ_ROOT>/<slug>/<NPZ_VARIANT>[...]
# 연산자 변형 이름 (fit 산출물 디렉터리). s5m5 = scene 0–4 × noise 0–4 로 fit 한 판.
# 변형 4종 = <V> / <V>_pl(위약) / <V>_fut(future-only) / <V>_fut_pl.
NPZ_VARIANT="${NPZ_VARIANT:-setM_s5m5_seg}"

# ── 연산자 계열 ───────────────────────────────────────────────────────────────
# STEER_OP=setpoint_seg (기본, 기존 동작 불변) — setM 세그먼트 연산자.
#   NPZ 경로 = <NPZ_ROOT>/<slug>/<NPZ_VARIANT>{,_pl,_fut,_fut_pl} (arm 마다 별도 트리),
#   token-select 는 all 고정 (fit 이 토큰 위치별 s_t 를 냈다).
# STEER_OP=conceptor — 수축형 행렬 연산자 (scripts/fit/fit_contraction_ops.py 산출:
#   sconceptor | conceptor | varc). serve 의 op 이름은 셋 다 `conceptor` 이고, 어느
#   연산자인지는 **디렉토리 층**으로 구분한다:
#     <NPZ_ROOT>/<slug>/<NPZ_VARIANT>/<STEER_OP_NAME>{,_pl}/<phase>/dit_L{n}/
#   α 는 fit 이 NPZ 키 alpha0_C_steer 에 구워 넣었으므로 serve 는 --steering-alpha 0.
#   ★ online_fut(_pl) arm 은 **별도 NPZ 트리가 아니라 같은 NPZ + token-select future** 다
#     (행렬 연산자는 세그먼트 마스크 자리가 없어 적용 토큰으로 future-only 를 표현).
STEER_OP="${STEER_OP:-setpoint_seg}"                # setpoint_seg | conceptor
STEER_OP_NAME="${STEER_OP_NAME:-}"                  # conceptor 계열 전용: sconceptor|conceptor|varc
STEER_TOKEN_SELECT="${STEER_TOKEN_SELECT:-all}"     # conceptor 계열 기본 토큰 선택
case "$STEER_OP" in
  setpoint_seg) ;;
  conceptor)
    [ -n "$STEER_OP_NAME" ] || { echo "[online-gated] ABORT: STEER_OP=conceptor 는 STEER_OP_NAME (sconceptor|conceptor|varc) 필요" >&2; exit 2; } ;;
  condg) ;;   # 상태-조건부 대조 guidance (docs/steering/44) — NPZ 는 variant 디렉토리/condg.npz 단일 파일
  *) echo "[online-gated] ABORT: 알 수 없는 STEER_OP=${STEER_OP} (setpoint_seg|conceptor|condg)" >&2; exit 2 ;;
esac
is_conceptor_family() { [ "$STEER_OP" = conceptor ]; }
is_condg() { [ "$STEER_OP" = condg ]; }
CONDG_GATE="${CONDG_GATE:-1}"   # condg 전용: 0 이면 --no-condg-gate (무게이트 ablation arm)
DETECTOR_CKPT="${DETECTOR_CKPT:-}"                  # 지정 시 전 slug 공통 (단일 slug 용)
# slug 별 per-task ckpt 템플릿 (%SLUG% 치환). DETECTOR_CKPT 가 비어 있으면 이걸 사용.
DETECTOR_CKPT_TMPL="${DETECTOR_CKPT_TMPL:-outputs/analysis/grid_phase/detector_sim/detector_pertask_lstm_%SLUG%.pt}"
DETECTOR_LAYERS="${DETECTOR_LAYERS:-12}"            # serve 캡처 layer (ckpt layer 포함 필수)
FAILURE_ALPHA="${FAILURE_ALPHA:-0.1}"               # 5-seed 시뮬 균형점 (docs/steering/42)
FAILURE_TASK="${FAILURE_TASK:-}"                    # 비우면 slug (cp_bands 키 = shard slug)
STEER_BETA="${STEER_BETA:-1.0}"
TOKEN_POOL="${TOKEN_POOL:-all_token_full}"          # detector 계약(고정)

PLAN_JSON="${PLAN_JSON:-configs/collect/n15_grid_v1/collection_plan.json}"
# replay 모드에서는 index 가 **필수** (수집 당시 env_seed·inference_seed 의 유일한 출처).
INDEX_TSV="${INDEX_TSV:-outputs/steer/online_pipe/manifests/index_rollouts.tsv}"
N_SCENES="${N_SCENES:-10}"
N_EP_PER_SCENE="${N_EP_PER_SCENE:-2}"               # fresh 모드 전용
EVAL_INF_SEED_BASE="${EVAL_INF_SEED_BASE:-1400000}" # fresh 모드 전용 (fit noise 1.3M과 분리)

# ── replay 모드 셀 지정 ───────────────────────────────────────────────────────
EP_MODE="${EP_MODE:-replay}"                        # replay | fresh
EVAL_SCENES="${EVAL_SCENES:-0-9}"                   # eval 셀 scene 축 (10 scene 전체)
EVAL_NOISES="${EVAL_NOISES:-0,1,5,6}"               # eval 셀 noise 축 (seen 2 + unseen 2)
FIT_SCENES="${FIT_SCENES:-0,1,2,3,4}"               # 연산자·detector fit 셀 (사분면 태깅용)
FIT_NOISES="${FIT_NOISES:-0,1,2,3,4}"
REPLAY_MACHINE="${REPLAY_MACHINE:-}"                # 비우면 index 단일 machine / hostname
EP_IDX_STRIDE="${EP_IDX_STRIDE:-100}"               # episode_idx = scene*STRIDE + noise
NAS="${NAS:-5}"                                     # GR00T 표준: 16 예측 / 5 실행
MAXEP="${MAXEP:-720}"
# PROX=1 이면 --proximity-phases (PPCC grasp/place 하위 phase). grid 수집·s5m5 fit 은
# 3-phase(off) 였으므로 기본 0 — fit 디렉토리명과 라벨러 어휘 정합. 전용 라벨러
# (drawer·oven·dishwasher)는 이 플래그와 무관.
PROX="${PROX:-0}"
PROX_ARGS=()
[ "$PROX" = "1" ] && PROX_ARGS=(--proximity-phases)
EXPECT_CHUNK="${EXPECT_CHUNK:-16}"

PROFILE="${PROFILE:-configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml}"
# SERVE_MODE=docker(기본): lerobot 컨테이너에서 serve (kanu·pdk_external).
# SERVE_MODE=host: 호스트 conda 로 serve (srv48·srv50 — lerobot 컨테이너 없음).
#   SERVE_PY=파이썬 경로 (collect_grid.sh 와 같은 이름; HOST_PY 도 호환),
#   SERVE_PYTHONPATH=lerobot 패키지 경로(예 ${REPO_ROOT}/lerobot/src).
SERVE_MODE="${SERVE_MODE:-docker}"                  # docker | host
HOST_PY="${HOST_PY:-${SERVE_PY:-$HOME/miniconda3/envs/lerobot_050_groot/bin/python}}"
SERVE_PY="${SERVE_PY:-$HOST_PY}"
SERVE_PYTHONPATH="${SERVE_PYTHONPATH:-}"
PY_HOST="${PY_HOST:-python3}"                       # scene_table/collect_results (numpy 불필요)
OUT_ROOT="${OUT_ROOT:-outputs/eval/robocasa/groot_n15/online_gated}"
DRY_RUN="${DRY_RUN:-0}"

# ── 경로 (스크립트 위치 기준 — REPO 하드코딩 금지) ────────────────────────────
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../.." && pwd)"
# worktree 면 main-tree 를 유도 (docker mount 는 main-tree=/temporal_vla)
case "$REPO_ROOT" in
  */.claude/worktrees/*) MAIN_HOST="${REPO_ROOT%%/.claude/worktrees/*}" ;;
  *) MAIN_HOST="$REPO_ROOT" ;;
esac
CONT_REPO="${CONT_REPO:-/temporal_vla}"
REPO_CONT="${CONT_REPO}${REPO_ROOT#"$MAIN_HOST"}"
PYPATH="${PYPATH_OVERRIDE:-${CONT_REPO}/src/policies/Isaac-GR00T:${CONT_REPO}/src/benchmarks/robocasa:${CONT_REPO}/src/benchmarks/robosuite:${CONT_REPO}}"

abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "${MAIN_HOST}/$1" ;; esac; }
to_cont() { case "$1" in "${MAIN_HOST}"/*) printf '%s\n' "${CONT_REPO}${1#"$MAIN_HOST"}" ;; *) printf '%s\n' "$1" ;; esac; }

OUT_ROOT="$(abspath "$OUT_ROOT")"
NPZ_ROOT="$(abspath "$NPZ_ROOT")"
PLAN_JSON_ABS="$(abspath "$PLAN_JSON")"
[ -n "$INDEX_TSV" ] && INDEX_TSV="$(abspath "$INDEX_TSV")"
[ -n "$DETECTOR_CKPT" ] && DETECTOR_CKPT="$(abspath "$DETECTOR_CKPT")"
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "$LOGDIR"

log() { echo "[online-gated] $(date '+%F %T') $*"; }

# ── 자원 규약 게이트 (kanu: 빈 GPU 만, ≤3 GPU, GPU 당 serve ≤2) ───────────────
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[ "${#GPU_ARR[@]}" -le 3 ] || { log "ABORT: GPU ${#GPU_ARR[@]}개 > 상한 3"; exit 2; }
[ "$SERVES_PER_GPU" -le 6 ] || { log "ABORT: SERVES_PER_GPU=$SERVES_PER_GPU > 상한 6 (A100 규칙)"; exit 2; }
if [ "$DRY_RUN" != "1" ] && command -v nvidia-smi > /dev/null 2>&1; then
  for g in "${GPU_ARR[@]}"; do
    busy="$(nvidia-smi --id="$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
    if [ -n "$busy" ] && [ "$ALLOW_BUSY_GPU" != "1" ]; then
      log "ABORT: GPU ${g} 에 이미 프로세스가 있다 (pid: $(echo "$busy" | tr '\n' ' ')) — 빈 GPU 만 사용"
      exit 3
    fi
  done
fi

PORTS=(); PORT_GPUS=()
p=$PORT_BASE
for g in "${GPU_ARR[@]}"; do
  for _ in $(seq 1 "$SERVES_PER_GPU"); do
    PORTS+=("$p"); PORT_GPUS+=("$g"); p=$((p + 1))
  done
done
NW=${#PORTS[@]}

# ── arm 별 NPZ base / 모드 ────────────────────────────────────────────────────
npz_base_for_arm() {  # slug arm → NPZ base 경로 (base·oracle 도 반환, base 는 빈 문자열)
  if is_condg; then
    # condg: <NPZ_ROOT>/<slug>/<NPZ_VARIANT>/<variant>/condg.npz.
    # hs(성공-모방 ablation)는 같은 W 라 NPZ 는 condg_hs 디렉토리(내용 동일, mode 만 다름).
    local root="${NPZ_ROOT}/${1}/${NPZ_VARIANT}"
    case "$2" in
      base)                              printf '' ;;
      online|online_fut|oracle_always)   printf '%s/condg\n'    "$root" ;;
      online_pl|online_fut_pl|oracle_always_pl) printf '%s/condg_pl\n' "$root" ;;
      online_hs)                         printf '%s/condg_hs\n' "$root" ;;
      *) echo "ABORT: 알 수 없는 arm=$2" >&2; return 2 ;;
    esac
    return 0
  fi
  if is_conceptor_family; then
    # 행렬 연산자: 위약만 별도 트리, future arm 은 같은 NPZ (token-select 로 표현)
    local root="${NPZ_ROOT}/${1}/${NPZ_VARIANT}/${STEER_OP_NAME}"
    case "$2" in
      base)                            printf '' ;;
      online|online_fut|oracle_always) printf '%s\n'    "$root" ;;
      online_pl|online_fut_pl|oracle_always_pl) printf '%s_pl\n' "$root" ;;
      *) echo "ABORT: 알 수 없는 arm=$2" >&2; return 2 ;;
    esac
    return 0
  fi
  case "$2" in
    base)          printf '' ;;
    online)        printf '%s/%s/%s\n'        "$NPZ_ROOT" "$1" "$NPZ_VARIANT" ;;
    online_fut)    printf '%s/%s/%s_fut\n'    "$NPZ_ROOT" "$1" "$NPZ_VARIANT" ;;
    online_fut_pl) printf '%s/%s/%s_fut_pl\n' "$NPZ_ROOT" "$1" "$NPZ_VARIANT" ;;
    online_pl)     printf '%s/%s/%s_pl\n'     "$NPZ_ROOT" "$1" "$NPZ_VARIANT" ;;
    oracle_always) printf '%s/%s/%s\n'        "$NPZ_ROOT" "$1" "$NPZ_VARIANT" ;;
    oracle_always_pl) printf '%s/%s/%s_pl\n'  "$NPZ_ROOT" "$1" "$NPZ_VARIANT" ;;
    *) echo "ABORT: 알 수 없는 arm=$2" >&2; return 2 ;;
  esac
}

# arm 별 적용 토큰. conceptor 계열의 _fut arm 만 future 세그먼트로 좁힌다
# (setpoint_seg 는 fit 이 토큰 위치별 s_t 를 내므로 항상 all — 기존 동작 불변).
token_select_for_arm() {  # arm
  if is_condg; then
    case "$1" in
      online_fut|online_fut_pl) printf 'future\n' ;;
      *) printf '%s\n' "$STEER_TOKEN_SELECT" ;;
    esac
    return 0
  fi
  if is_conceptor_family; then
    case "$1" in
      online_fut|online_fut_pl) printf 'future\n' ;;
      *) printf '%s\n' "$STEER_TOKEN_SELECT" ;;
    esac
  else
    printf 'all\n'
  fi
}
# base 도 detector 를 태운다(steering 미등록 → 전부 identity): eval 대역 failure_scores 를
# sidecar 에 남겨 α sweep(발화 시점·오발화율)을 GPU 재실행 없이 사후 재계산하기 위함.
arm_uses_detector() { case "$1" in base|online|online_fut|online_pl|online_fut_pl|online_hs) return 0 ;; *) return 1 ;; esac; }

# NPZ base 스캔 → PHASES / LAYER 확정.
# ★ command substitution 안에서 대입하면 subshell 에 갇혀 태그가 빈다 (exp5-2 실측) —
# 반드시 이 함수를 직접 호출해 전역에 채운다.
scan_npz_base() {  # base
  local base="$1" ph
  if is_condg; then
    # condg 는 단일 NPZ 파일 — phase 목록·layer 는 NPZ 안(serve preflight 로그)에서 확인.
    [ -f "$base/condg.npz" ] || { echo "ABORT: condg NPZ 없음: $base/condg.npz" >&2; return 1; }
    PHASES="(npz)"; LAYER="${CONDG_LAYER:-12}"
    return 0
  fi
  [ -d "$base" ] || { echo "ABORT: NPZ base 없음: $base" >&2; return 1; }
  PHASES=""
  for ph in "$base"/*/; do
    [ -d "$ph" ] && ls "$ph"dit_L* > /dev/null 2>&1 && PHASES="${PHASES:+$PHASES,}$(basename "$ph")"
  done
  [ -n "$PHASES" ] || { echo "ABORT: NPZ phase 디렉토리 없음: $base" >&2; return 1; }
  LAYER=$(basename "$(ls -d "$base/${PHASES%%,*}"/dit_L* 2>/dev/null | head -1)" | sed 's/dit_L//')
  [ -n "$LAYER" ] || { echo "ABORT: layer 감지 실패: $base" >&2; return 1; }
  [ -f "$base/${PHASES%%,*}/dit_L${LAYER}/conceptors.npz" ] || {
    echo "ABORT: conceptors.npz 없음: $base/${PHASES%%,*}/dit_L${LAYER}" >&2; return 1; }
}

serve_flags_for() {  # slug arm → serve 추가 플래그 (scan_npz_base 선행)
  local slug="$1" arm="$2" base flags="" serve_base
  base="$(npz_base_for_arm "$slug" "$arm")"
  if [ -n "$base" ]; then
    serve_base="$base"
    [ "$SERVE_MODE" = host ] || serve_base="$(to_cont "$base")"
    # setpoint_seg 는 gated 경로(--steering-phase-npz-base) + token-select all 필수
    # (fit 이 전 토큰 위치별 s_t 를 냈다 — last_horizon 이면 좌표가 어긋난다).
    # conceptor 계열은 같은 gated 경로 + --steering-alpha 0 (선택 α 가 NPZ 키에 구워짐).
    if is_condg; then
      local mode=condg
      [ "$arm" = online_hs ] && mode=hs
      flags="--condg-npz ${serve_base}/condg.npz --steering-op condg"
      flags="${flags} --steering-beta ${STEER_BETA} --condg-mode ${mode}"
      flags="${flags} --steering-token-select $(token_select_for_arm "$arm")"
      [ "$CONDG_GATE" = 1 ] || flags="${flags} --no-condg-gate"
    else
      flags="--steering-phase-npz-base ${serve_base} --steering-phases ${PHASES}"
      flags="${flags} --steering-layers ${LAYER} --steering-op ${STEER_OP}"
      flags="${flags} --steering-beta ${STEER_BETA}"
      flags="${flags} --steering-token-select $(token_select_for_arm "$arm")"
      is_conceptor_family && flags="${flags} --steering-alpha 0"
    fi
  fi
  if arm_uses_detector "$arm"; then
    local ckpt
    ckpt="$(detector_ckpt_for "$slug")"
    [ "$SERVE_MODE" = host ] || ckpt="$(to_cont "$ckpt")"
    # detector 전제(serve 가 fail-loud 로 재검증): DiT residual 캡처 + all_token_full.
    flags="${flags} --groot-dit-capture-layers ${DETECTOR_LAYERS} --groot-dit-token-pool ${TOKEN_POOL}"
    flags="${flags} --failure-detector ${ckpt} --failure-alpha ${FAILURE_ALPHA}"
    flags="${flags} --failure-task ${FAILURE_TASK:-$slug}"
  fi
  printf '%s\n' "$flags"
}

# ── preflight: arm 재료 존재 확인 (발사 전에 전부) ────────────────────────────
IFS=',' read -r -a SLUG_ARR <<< "$SLUGS"
IFS=',' read -r -a ARM_ARR <<< "$ARMS"
detector_ckpt_for() {  # slug → ckpt 경로 (DETECTOR_CKPT 지정 시 그것, 아니면 템플릿 치환)
  if [ -n "$DETECTOR_CKPT" ]; then printf '%s\n' "$DETECTOR_CKPT"
  else printf '%s\n' "$(abspath "${DETECTOR_CKPT_TMPL//%SLUG%/$1}")"; fi
}
need_detector=0
for arm in "${ARM_ARR[@]}"; do arm_uses_detector "$arm" && need_detector=1; done
if [ "$need_detector" = 1 ]; then
  for slug in "${SLUG_ARR[@]}"; do
    ck="$(detector_ckpt_for "$slug")"
    [ -f "$ck" ] || { log "ABORT: detector ckpt 없음 (${slug}): $ck"; exit 2; }
  done
fi
for slug in "${SLUG_ARR[@]}"; do
  for arm in "${ARM_ARR[@]}"; do
    base="$(npz_base_for_arm "$slug" "$arm")"
    [ -n "$base" ] || continue
    scan_npz_base "$base" || { log "ABORT: ${slug}/${arm} NPZ 준비 안 됨"; exit 12; }
    log "preflight ${slug}/${arm}: base=$(basename "$base") phases=${PHASES} layer=${LAYER}"
  done
done

# ── episode 표 (slug 당 1회) ──────────────────────────────────────────────────
ep_table_for() { printf '%s/%s_%s.tsv\n' "$LOGDIR" \
  "$([ "$EP_MODE" = replay ] && echo cells || echo scenes)" "$1"; }

case "$EP_MODE" in
  replay)
    [ -n "$INDEX_TSV" ] || { log "ABORT: EP_MODE=replay 는 INDEX_TSV 필수"; exit 2; }
    [ -f "$INDEX_TSV" ] || { log "ABORT: INDEX_TSV 없음: $INDEX_TSV"; exit 2; }
    for slug in "${SLUG_ARR[@]}"; do
      "$PY_HOST" "${REPO_ROOT}/scripts/steer/online_gated/replay_cells.py" \
        --slug "$slug" --index-tsv "$INDEX_TSV" --plan-json "$PLAN_JSON_ABS" \
        --scenes "$EVAL_SCENES" --noises "$EVAL_NOISES" \
        ${REPLAY_MACHINE:+--machine "$REPLAY_MACHINE"} --out "$(ep_table_for "$slug")"
    done
    # 판 수 = 셀 수 (slug 마다 같아야 한다 — 다르면 fail-loud)
    N_EP_TOTAL=""
    for slug in "${SLUG_ARR[@]}"; do
      n=$(wc -l < "$(ep_table_for "$slug")")
      if [ -z "$N_EP_TOTAL" ]; then N_EP_TOTAL="$n"
      elif [ "$n" != "$N_EP_TOTAL" ]; then
        log "ABORT: slug 마다 셀 수가 다르다 (${slug}=${n} vs ${N_EP_TOTAL})"; exit 2
      fi
    done
    log "slugs=${SLUGS} arms=${ARMS} mode=replay cells=${N_EP_TOTAL}판/arm " \
        "(scenes=${EVAL_SCENES} × noises=${EVAL_NOISES}; fit s=${FIT_SCENES} m=${FIT_NOISES})"
    ;;
  fresh)
    for slug in "${SLUG_ARR[@]}"; do
      "$PY_HOST" "${REPO_ROOT}/scripts/steer/online_gated/scene_table.py" \
        --slug "$slug" --plan-json "$PLAN_JSON_ABS" --n-scenes "$N_SCENES" \
        ${INDEX_TSV:+--index-tsv "$INDEX_TSV"} --out "$(ep_table_for "$slug")"
    done
    N_EP_TOTAL=$((N_SCENES * N_EP_PER_SCENE))
    log "slugs=${SLUGS} arms=${ARMS} mode=fresh scenes=${N_SCENES}×${N_EP_PER_SCENE}=${N_EP_TOTAL}판/arm"
    ;;
  *) log "ABORT: 알 수 없는 EP_MODE=${EP_MODE} (replay|fresh)"; exit 2 ;;
esac
log "serve ${NW}개 (gpus=${GPUS} × ${SERVES_PER_GPU}) ports=${PORTS[*]} mode=${SERVE_MODE} beta=${STEER_BETA}"

# 머신 출처 기록 (docs/04 — 결과의 머신 열 원천)
{
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "gpus=${GPUS} serves_per_gpu=${SERVES_PER_GPU} serve_mode=${SERVE_MODE}"
  echo "profile=${PROFILE}"
  echo "detector_ckpt=$(basename "${DETECTOR_CKPT:-none}") alpha=${FAILURE_ALPHA} layers=${DETECTOR_LAYERS}"
  echo "steer_beta=${STEER_BETA} npz_root=${NPZ_ROOT#"$MAIN_HOST"/} npz_variant=${NPZ_VARIANT}"
  echo "steer_op=${STEER_OP} steer_op_name=${STEER_OP_NAME:-NA} token_select=${STEER_TOKEN_SELECT}"
  echo "ep_mode=${EP_MODE} n_ep_total=${N_EP_TOTAL}"
  if [ "$EP_MODE" = replay ]; then
    echo "eval_scenes=${EVAL_SCENES} eval_noises=${EVAL_NOISES}"
    echo "fit_scenes=${FIT_SCENES} fit_noises=${FIT_NOISES} replay_machine=${REPLAY_MACHINE:-auto}"
  else
    echo "eval_inf_seed_base=${EVAL_INF_SEED_BASE} n_ep_per_scene=${N_EP_PER_SCENE}"
  fi
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo NA)"
} > "${OUT_ROOT}/MACHINE.txt"

# ── serve 기동/정리 ───────────────────────────────────────────────────────────
serve_log_host() { printf '%s/serve_%s.log\n' "$LOGDIR" "$1"; }
serve_log_cont() { printf '/tmp/online_gated_serve_%s.log\n' "$1"; }

kill_serve() {  # port
  local port="$1"
  if [ "$SERVE_MODE" = host ]; then
    pkill -f "serve/lerobot.py.*--port ${port}" 2> /dev/null || true
  else
    docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2> /dev/null || true
  fi
}
cleanup() {
  log "cleanup: serve 정리"
  for port in "${PORTS[@]}"; do kill_serve "$port"; done
  sleep 3
  for port in "${PORTS[@]}"; do
    if [ "$SERVE_MODE" = host ]; then
      pgrep -f "serve/lerobot.py.*--port ${port}" > /dev/null 2>&1 && log "WARN: serve ${port} 잔존"
    else
      docker exec lerobot bash -lc "pgrep -f 'serve/lerobot.py.*--port ${port}' > /dev/null" \
        2> /dev/null && log "WARN: serve ${port} 잔존 (container)"
    fi
  done
  command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader 2> /dev/null || true
}
[ "$DRY_RUN" = "1" ] || trap cleanup EXIT INT TERM

health_curl() {  # port
  if [ "$SERVE_MODE" = host ]; then
    curl -s -m 3 "http://127.0.0.1:${1}/health" 2> /dev/null || true
  else
    docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${1}/health 2>/dev/null" || true
  fi
}

start_serve() {  # gpu port extra_flags...
  local gpu="$1" port="$2"; shift 2
  local extra="$*"
  if [ "$SERVE_MODE" = host ]; then
    local hcmd="\"$SERVE_PY\" \"${REPO_ROOT}/scripts/serve/lerobot.py\" --profile \"${REPO_ROOT}/${PROFILE}\" --host '*' --port ${port} --device cuda ${extra}"
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry serve host gpu=${gpu}] ${hcmd} > $(serve_log_host "$port")"
      return 0
    fi
    ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH="${SERVE_PYTHONPATH}${SERVE_PYTHONPATH:+:}${REPO_ROOT}" \
        bash -lc "$hcmd" > "$(serve_log_host "$port")" 2>&1 < /dev/null & )
  else
    local inner="cd ${REPO_CONT} && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} --host '*' --port ${port} --device cuda ${extra} > $(serve_log_cont "$port") 2>&1 < /dev/null &"
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry serve docker gpu=${gpu}] docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} lerobot bash -lc \"${inner}\""
      return 0
    fi
    docker exec -d -e CUDA_VISIBLE_DEVICES="$gpu" lerobot bash -lc "$inner"
  fi
  # health poll (로드 중 VRAM 피크 때문에 순차 기동 — 워커별 1 serve 라 여기서 대기)
  local ok=0
  for _ in $(seq 1 150); do
    if health_curl "$port" | grep -q '"status":"ok"'; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" != 1 ]; then
    log "ABORT: serve ${port} 기동 실패 — 로그 꼬리:"
    if [ "$SERVE_MODE" = host ]; then tail -30 "$(serve_log_host "$port")" || true
    else docker exec lerobot bash -lc "tail -30 $(serve_log_cont "$port")" || true; fi
    return 11
  fi
  return 0
}

serve_preflight() {  # port arm  — 등록 지문 대조 (무음 identity·arm 오배치 방지)
  local port="$1" arm="$2" body reg
  body="$(health_curl "$port")"
  if [ "$arm" != "base" ]; then
    printf '%s' "$body" | grep -q '"steering"' || {
      log "ABORT: serve ${port} /health 에 steering 지문 없음 (arm=${arm})"; return 11; }
    if [ "$SERVE_MODE" = host ]; then
      reg="$(grep -m1 '\[steer-registered\]' "$(serve_log_host "$port")" || true)"
    else
      reg="$(docker exec lerobot bash -lc "grep -m1 '\[steer-registered\]' $(serve_log_cont "$port")" || true)"
    fi
    [ -n "$reg" ] || { log "ABORT: serve ${port} 로그에 [steer-registered] 없음 (arm=${arm})"; return 11; }
    log "  ${reg}"
  fi
  if arm_uses_detector "$arm"; then
    # collector 도 같은 필드로 fail-loud 하지만, 20 판 돌린 뒤가 아니라 여기서 잡는다.
    printf '%s' "$body" | grep -q '"failure_detector"' || {
      log "ABORT: serve ${port} /health 에 failure_detector 없음 (arm=${arm})"; return 11; }
    printf '%s' "$body" | tr ',' '\n' | grep -E '"alpha"|"band_L"|"model"' | head -3 || true
  fi
  return 0
}

# ── 한 판 ─────────────────────────────────────────────────────────────────────
run_episode() {  # port slug arm task env_name instr ep inf_seed env_seed out_host [scene]
  local port="$1" slug="$2" arm="$3" task="$4" envn="$5" instr="$6" ep="$7" inf="$8" seed="$9"
  local out_host="${10}" scene="${11:-}" mode=()
  case "$arm" in
    online|online_fut|online_pl|online_fut_pl) mode=(--gated-steering-mode online) ;;
    oracle_always|oracle_always_pl) mode=(--gated-steering) ;;
    # base: detector ON + steering 미등록 → online 모드라도 전 스텝 identity.
    # failure_scores/trigger_step 사이드카 기록이 목적 (α 사후 sweep 용).
    base) mode=(--gated-steering-mode online) ;;
  esac
  local cmd=(docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa
    python "${REPO_CONT}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py"
    --vla-server "http://127.0.0.1:${port}" --task "$task" --env-name "$envn"
    --output-dir "$(to_cont "$out_host")/raw_rollouts" --cell-id "$slug" --cell-index 0
    --canonical-instruction "$instr"
    --episode-start-idx "$ep" --n-episodes 1
    --seed "$seed" --inference-seed "$inf"
    --n-action-steps "$NAS" --max-episode-steps "$MAXEP"
    --video-fps 20 --steps-per-render 2 --wait-ready
    ${PROX_ARGS[@]+"${PROX_ARGS[@]}"} --no-features --expect-chunk-len "$EXPECT_CHUNK"
    --run-tag "online_gated_${arm}_b${STEER_BETA}"
    ${scene:+--steering-scene "$scene"}
    "${mode[@]}")
  if [ "$DRY_RUN" = "1" ]; then echo "[dry collect] ${cmd[*]}"; return 0; fi
  local rc=0
  "${cmd[@]}" 2>&1 | grep -E "^wrote|Error|Traceback|ABORT|좌표" || true
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || echo "[collector FAIL rc=${rc}] ${slug}/${arm} ep${ep} seed=${seed}" >&2
  return "$rc"
}

done_mark() {  # out_host task slug ep
  ls "${1}/raw_rollouts/${2}/${3}/task0--ep${4}--succ"*.json > /dev/null 2>&1
}

run_job() {  # port gpu slug arm
  local port="$1" gpu="$2" slug="$3" arm="$4"
  local out_host="${OUT_ROOT}/${slug}/${arm}"
  local tsv="${out_host}/per_episode.tsv"
  mkdir -p "$out_host"
  # resume: 표가 이미 목표 행 수를 채웠으면 통째로 건너뛴다 (header 1행 + N).
  if [ "$DRY_RUN" != "1" ] && [ -f "$tsv" ] \
     && [ "$(($(wc -l < "$tsv") - 1))" -ge "$N_EP_TOTAL" ]; then
    log "skip(done) ${slug}/${arm} — ${tsv}"
    return 0
  fi

  local base flags
  base="$(npz_base_for_arm "$slug" "$arm")"
  PHASES=""; LAYER=""
  if [ -n "$base" ]; then scan_npz_base "$base" || return 12; fi
  flags="$(serve_flags_for "$slug" "$arm")"
  log "job ${slug}/${arm} port=${port} gpu=${gpu} flags=${flags:-<none>}"
  start_serve "$gpu" "$port" $flags || return 11
  if [ "$DRY_RUN" != "1" ]; then serve_preflight "$port" "$arm" || { kill_serve "$port"; return 11; }; fi

  local rc_any=0 si ni seed task envn instr ep inf rep csucc
  local tbl; tbl="$(ep_table_for "$slug")"
  if [ "$EP_MODE" = replay ]; then
    # 수집 셀 재생: env_seed·inference_seed 는 index 값 그대로 (새로 만들지 않는다).
    while IFS=$'\t' read -r si ni seed inf csucc task envn instr; do
      [ -n "${si:-}" ] || continue
      ep=$((si * EP_IDX_STRIDE + ni))
      if [ "$DRY_RUN" != "1" ] && done_mark "$out_host" "$task" "$slug" "$ep"; then
        echo "[${slug}/${arm}] skip ep${ep} (s${si} n${ni})"
        continue
      fi
      echo "[${slug}/${arm}] $(date '+%F %T') ep${ep} s${si} n${ni} seed=${seed} inf=${inf} csucc=${csucc}"
      run_episode "$port" "$slug" "$arm" "$task" "$envn" "$instr" "$ep" "$inf" "$seed" "$out_host" "$si" \
        || rc_any=1
    done < "$tbl"
  else
    while IFS=$'\t' read -r si seed task envn instr; do
      [ -n "${si:-}" ] || continue
      for rep in $(seq 0 $((N_EP_PER_SCENE - 1))); do
        ep=$((si * N_EP_PER_SCENE + rep))
        # fit(수집) inference_seed 대역(1.3M noise)과 분리된 결정적 규칙.
        inf=$((EVAL_INF_SEED_BASE + si * 1000 + rep))
        if [ "$DRY_RUN" != "1" ] && done_mark "$out_host" "$task" "$slug" "$ep"; then
          echo "[${slug}/${arm}] skip ep${ep} (s${si} rep${rep})"
          continue
        fi
        echo "[${slug}/${arm}] $(date '+%F %T') ep${ep} s${si} seed=${seed} inf=${inf}"
        run_episode "$port" "$slug" "$arm" "$task" "$envn" "$instr" "$ep" "$inf" "$seed" "$out_host" "$si" \
          || rc_any=1
      done
    done < "$tbl"
  fi

  kill_serve "$port"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  local res=("$PY_HOST" "${REPO_ROOT}/scripts/steer/online_gated/collect_results.py"
    --job-dir "$out_host" --arm "$arm" --slug "$slug" --expect "$N_EP_TOTAL")
  if [ "$EP_MODE" = replay ]; then
    res+=(--cells-tsv "$tbl" --fit-scenes "$FIT_SCENES" --fit-noises "$FIT_NOISES")
  else
    res+=(--scenes-tsv "$tbl")
  fi
  "${res[@]}" || rc_any=13
  cp "${OUT_ROOT}/MACHINE.txt" "${out_host}/MACHINE.txt" 2> /dev/null || true
  return "$rc_any"
}

# ── (slug, arm) 작업 큐 → 워커 fan-out ───────────────────────────────────────
JOBS=()
for slug in "${SLUG_ARR[@]}"; do
  for arm in "${ARM_ARR[@]}"; do JOBS+=("${slug}|${arm}"); done
done
log "jobs=${#JOBS[@]} (${JOBS[*]})"

run_worker() {  # wid
  local wid="$1" i job slug arm nfail=0
  for i in "${!JOBS[@]}"; do
    [ $((i % NW)) -eq "$wid" ] || continue
    job="${JOBS[$i]}"; slug="${job%%|*}"; arm="${job##*|}"
    run_job "${PORTS[$wid]}" "${PORT_GPUS[$wid]}" "$slug" "$arm" || nfail=$((nfail + 1))
  done
  echo "$nfail" > "${LOGDIR}/w${wid}.nfail"
  echo "[w${wid}] DONE nfail=${nfail}"
}

if [ "$DRY_RUN" = "1" ]; then
  for wid in $(seq 0 $((NW - 1))); do run_worker "$wid"; done
  log "[DRY] out=${OUT_ROOT}"
  exit 0
fi
for wid in $(seq 0 $((NW - 1))); do
  run_worker "$wid" > "${LOGDIR}/worker_w${wid}.log" 2>&1 &
done
wait

cleanup; trap - EXIT INT TERM

# ── 최종 요약 (판정은 per_episode.tsv 행 수로) ────────────────────────────────
n_incomplete=0
for slug in "${SLUG_ARR[@]}"; do
  for arm in "${ARM_ARR[@]}"; do
    tsv="${OUT_ROOT}/${slug}/${arm}/per_episode.tsv"
    if [ -f "$tsv" ]; then
      # 열 위치를 헤더에서 찾는다 (열 추가에 안 깨지도록).
      n=$(($(wc -l < "$tsv") - 1))
      s=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next} $c["success"]=="1"' "$tsv" | wc -l)
      fired=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next} $c["trigger_step"]!="NA"' "$tsv" | wc -l)
      printf '%-22s %-14s n=%-3s succ=%-3s fired=%-3s\n' "$slug" "$arm" "$n" "$s" "$fired"
      [ "$n" -ge "$N_EP_TOTAL" ] || n_incomplete=$((n_incomplete + 1))
    else
      printf '%-22s %-14s MISSING\n' "$slug" "$arm"
      n_incomplete=$((n_incomplete + 1))
    fi
  done
done
if [ "$n_incomplete" -gt 0 ]; then
  log "INCOMPLETE — 미충족 ${n_incomplete} 조합 (nfail: $(cat "${LOGDIR}"/w*.nfail 2> /dev/null | tr '\n' ' '))"
  exit 13
fi
touch "${LOGDIR}/ONLINE_GATED_DONE"
log "DONE — ${OUT_ROOT}"
