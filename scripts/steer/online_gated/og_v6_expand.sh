#!/usr/bin/env bash
# v6 β sweep eval 오케스트레이터 — v6_eval_cells.tsv(98판, balance scene 실패 판 전부), α=0.1,
# arm = setm_gt jfair×β{0.6..1.0} + setm_gt plain×β{0.8} + reseed (= 7) [+ setm_ck8×β5 when CK8=1]. rsn_llr 제외(사용자 09-04). base arm 없음.
# ★v6 좌표 = jitter_idx(j). 러너 EVAL_JITTERS 는 replay_cells jitter_idx판(origin/feat/grid-v6-scene-jitter) 반입 후에만 유효 — 반입 전 실행 금지.
# 연산자 = (slug, s, j)당 1개(LOKO): instr_setm_v6_{gt,ck8}/<slug>/s<i>/j<r>/ · rsn_llr_reg_v6[_all]/<slug>/s<i>/j<r>.npz + registry.tsv
# 미등록 (s,j): rsn_llr 셀 skip(분모 제외) · setm phase 부재 = per-step fallback=reseed.
# replay 계약(v6): INDEX_TSV=index_rollouts_v6 · PLAN=n15_grid_v6_scene_jitter · EVAL_JITTERS=j(jitter_idx) · **EP_META_DIR 없음**.
# usage: og_v6_expand.sh <GPU> [GPU2] [GPU3]   — hostname 으로 kanu/worker1/worker2 판별.
#   GPU lease(docs/05 §2)는 스스로 잡는다(with_gpu_lease.sh 재귀). 대상 셀은 index 의 machine 열 = 홈 머신만.
set -u
G1=${1:?GPU}; G2=${2:-$G1}; G3=${3:-$G2}
W=${REPO:-$HOME/pkt_ws/temporal_vla}
case "$(hostname)" in *worker1*|*srv48*) MC=worker1; LM=srv48;; *worker2*|*srv50*) MC=worker2; LM=srv50;; *kanu*) MC=kanu; LM=kanu;; *) MC=${MC:?hostname 판별 실패 — MC 지정}; LM=${LM:-$MC};; esac
if [ "${LEASED:-0}" != 1 ]; then
  exec env LEASED=1 bash "$W/scripts/utils/with_gpu_lease.sh" "$LM" "$(printf '%s\n' "$G1" "$G2" "$G3" | sort -u | tr '\n' ' ')" 전체파이프 v6_expand -- bash "$0" "$@"
fi
B=$W/outputs/eval/robocasa/groot_n15/og_v6_expand
NPZ=$W/outputs/steer/online_pipe_v4_pilot          # v6 연산자도 같은 루트 아래 *_v6 폴더
BUNDLE=${BUNDLE:-$W/outputs/analysis/grid_phase/ae_k8/ae_bundle_k8.npz}
DK=${DK:-$W/outputs/analysis/grid_phase/detector_v6/loko}   # <slug>/s<i>/j<r>/detector_pertask_lstm_<slug>.pt   # fail detector 세션 산출 경로(확정 시 갱신)
EV=${EV:-$W/configs/collect/n15_grid_v6_scene_jitter/v6_loko_cells_eval.tsv}   # 발사판 = 셀표 중 detector+jfair+plain 산출 완비 셀
L=$B/logs; mkdir -p "$L"
echo "machine=$MC alpha=0.1 perstep_n=8 beta_sweep=0.6-1.0 v6 loko capture_hooks=7layer arms=setm_gt_b*(jfair),setm_gtplain_b08,reseed[,setm_ck8_b* if CK8] base=none detector=$DK bundle=$BUNDLE" > "$B/MACHINE.txt"
[ "${CK8:-0}" = 1 ] && { [ -f "$BUNDLE" ] || { echo "[abort] CK8=1 인데 번들 없음: $BUNDLE" | tee -a "$L/run.log"; exit 2; }; }
[ "${CK8:-0}" = 1 ] || BUNDLE=   # gt/plain/reseed 는 cluster 번들 불요(GT phase POST)
if [ "$MC" = kanu ]; then SLOTS=("$G1" "$G2" "$G3" "$G1" "$G2" "$G3"); NSLOT=6
else SLOTS=("$G1" "$G2" "$G1" "$G2" "$G1" "$G2" "$G1" "$G2" "$G1" "$G2" "$G1" "$G2"); NSLOT=$([ "$G1" = "$G2" ] && echo 6 || echo 12); fi
gpu_of() { echo "${SLOTS[$(( ${1:-0} % NSLOT ))]}"; }
HOSTARGS=(); [ "$MC" != kanu ] && HOSTARGS=(SERVE_MODE=host SERVE_PY=${SERVE_PY:-$HOME/miniconda3/envs/lerobot_050_groot/bin/python} SERVE_PYTHONPATH=${SERVE_PYTHONPATH:-$W/lerobot/src})
# 케이스 = (slug, s, k) → noises (awk falsy-0 함정: (key in ns) 판정)
mapfile -t CASES < <(awk -F'\t' -v m="$MC" 'NR>1 && $3==m {key=$2"\t"$4"\t"$5; ns[key]=(key in ns)?ns[key]","$6:$6} END{for(k in ns) print k"\t"ns[k]}' "$EV" | sort)
echo "[v6-expand] $MC cases=${#CASES[@]} eps=$(awk -F'\t' -v m="$MC" 'NR>1&&$3==m' "$EV" | wc -l)" | tee -a "$L/run.log"
p=${PORT_BASE:-8890}; i=0
run() {  # slug s k noises OUT [KEY=VAL...]
  local task=$1 s=$2 k=$3 ns=$4 out=$5; shift 5
  local cs=${task}_s${s}_j${k} sub=s${s}/j${k}
  if [[ "$out" == ps_setm_gtplain* ]] && [ ! -d "$NPZ/instr_setm_v6_gt_plain/$task/$sub" ]; then echo "[defer] $cs $out (plain NPZ 없음)" >> "$L/run.log"; return; fi
  if [[ "$out" == ps_setm_gt_b* ]] && [ ! -d "$NPZ/instr_setm_v6_gt/$task/$sub" ]; then echo "[defer] $cs $out (gt NPZ 없음)" >> "$L/run.log"; return; fi
  if [[ "$out" == ps_setm_ck8* ]] && [ ! -d "$NPZ/instr_setm_v6_ck8/$task/$sub" ]; then echo "[defer] $cs $out (ck8 NPZ 없음)" >> "$L/run.log"; return; fi
  local stem=${task}__s${s}; local det=$DK/$stem/$sub/detector_pertask_lstm_$stem.pt   # fail detector 산출: scene shard stem = <slug>__s<i>, cp_bands 키 동일
  [ -f "$det" ] || { echo "[defer] $cs $out (detector 없음: $det)" >> "$L/run.log"; return; }
  local want; want=$(( $(echo "$ns" | tr -cd , | wc -c) + 1 ))
  local have; have=$(awk 'FNR>1' "$B/$out/$cs/$task"/*/per_episode.tsv 2>/dev/null | wc -l)
  [ "$have" -ge "$want" ] && { echo "[skip] $cs $out ($have/$want)" >> "$L/run.log"; return; }
  local root=$NPZ/instr_roots_v6/$cs; mkdir -p "$root/$task"
  ln -sfn "../../../instr_setm_v6_gt/$task/$sub"  "$root/$task/instr_setm_v6_gt"
  ln -sfn "../../../instr_setm_v6_gt_plain/$task/$sub" "$root/$task/instr_setm_v6_gt_plain"
  ln -sfn "../../../instr_setm_v6_ck8/$task/$sub" "$root/$task/instr_setm_v6_ck8"
  ( env GPUS="$(gpu_of $i)" SERVES_PER_GPU=1 SLUGS="$task" "${HOSTARGS[@]}" \
      EP_MODE=replay EVAL_SCENES="$s" EVAL_JITTERS="$k" EVAL_NOISES="$ns" FIT_SCENES=0-4 FIT_NOISES=0-4 REPLAY_MACHINE="$MC" \
      INDEX_TSV=$W/configs/collect/n15_grid_v6_scene_jitter/index_v6_complete_cells.tsv \
      PLAN_JSON=$W/configs/collect/n15_grid_v6_scene_jitter/collection_plan.json EP_META_DIR= EP_META_LOAD_ENV_NAME= \
      DETECTOR_CKPT="$det" FAILURE_TASK="$stem" \
      FAILURE_ALPHA=0.1 PERSTEP_N=8 DETECTOR_LAYERS=0,2,4,8,10,12,15 TOKEN_POOL=all_token_full \
      NPZ_ROOT="$root" OUT_ROOT="$B/$out/$cs" PORT_BASE=$p SERVE_BOOT_TRIES=360 ALLOW_BUSY_GPU=1 \
      "$@" bash $W/scripts/steer/online_gated/run_online_gated_eval.sh \
      >> "$L/run.log" 2>&1; echo "[v6-exp] $cs $out rc=$?" >> "$L/run.log" ) &
  p=$((p+1)); i=$((i+1))
  [ "$(jobs -rp | wc -l)" -ge "$NSLOT" ] && wait -n
}
for c in "${CASES[@]}"; do
  IFS=$'\t' read -r task s k ns <<< "$c"
  # RUN_BASE=1: 무개입 replay(ps_base) — 결정성 대조용(수집 라벨과 일치해야 함). 본 eval 기본 0.
  [ "${RUN_BASE:-0}" = 1 ] && run "$task" "$s" "$k" "$ns" ps_base ARMS=ps_base CLUSTER_BUNDLE=
  # setm_gt: jfair β 스윕(SETM_JFAIR_BETAS) + plain β 대조(SETM_PLAIN_BETAS, 기본 0.8 하나). ck8 은 정식 AE 번들 후(CK8=1).
  for b in ${SETM_JFAIR_BETAS:-0.6 0.7 0.8 0.9 1.0}; do bt=b${b/./}
    run "$task" "$s" "$k" "$ns" ps_setm_gt_$bt ARMS=ps_setm STEER_OP=setpoint NPZ_VARIANT=instr_setm_v6_gt STEER_BETA=$b PERSTEP_FALLBACK=reseed CLUSTER_BUNDLE=
  done
  for b in ${SETM_PLAIN_BETAS:-0.8}; do bt=b${b/./}
    run "$task" "$s" "$k" "$ns" ps_setm_gtplain_$bt ARMS=ps_setm STEER_OP=setpoint NPZ_VARIANT=instr_setm_v6_gt_plain STEER_BETA=$b PERSTEP_FALLBACK=reseed CLUSTER_BUNDLE=
  done
  if [ "${CK8:-0}" = 1 ]; then for b in ${SETM_JFAIR_BETAS:-0.6 0.7 0.8 0.9 1.0}; do bt=b${b/./}
    run "$task" "$s" "$k" "$ns" ps_setm_ck8_$bt ARMS=ps_setm STEER_OP=setpoint NPZ_VARIANT=instr_setm_v6_ck8 STEER_BETA=$b PERSTEP_FALLBACK=reseed CLUSTER_BUNDLE="$BUNDLE"
  done; fi
  run "$task" "$s" "$k" "$ns" ps_reseed ARMS=ps_reseed CLUSTER_BUNDLE="$BUNDLE"
done
wait
echo "V6_EXPAND_${MC}_DONE $(grep -c 'rc=' "$L/run.log") defer=$(grep -c '^\[defer\]' "$L/run.log") " | tee -a "$L/run.log"
