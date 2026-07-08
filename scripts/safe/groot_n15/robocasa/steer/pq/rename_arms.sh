#!/bin/bash
# arm 디렉토리 명명 개편 (2026-07-08 사용자 지시) — ⚠️ 큐 소진(전 lane 종료) 후에만 실행!
#   ps→per_scene, xb/xa→cross_scene, gx→grand, 숫자→fit{N}(fit 에피소드 수), L 변형 유지.
#   예: ho_permps60→ho_perm_per_scene_fit60, ho_gatedxb30→ho_gated_cross_scene_fit30,
#       ho_gatedL4→ho_gated_per_scene_fit60_L4, ho_gatedxaL812→ho_gated_cross_scene_fit60_L812
# 사용: bash rename_arms.sh          (dry-run: 계획만 출력)
#       bash rename_arms.sh --apply  (실제 mv)
set -u
SE=/home/dongkyu/pkt_ws/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval
APPLY=${1:-}

new_name() {
  local t=$1
  case "$t" in
    ho_base) echo "$t"; return;;
    *6p*|*6pcr*) echo "$t"; return;;   # 구 3-scene 라운드 표기는 보존
  esac
  echo "$t" | sed -E \
    -e 's/^(ho_(perm|gated))ps([0-9]+)$/\1_per_scene_fit\3/' \
    -e 's/^(ho_(perm|gated))x[ab]([0-9]+)$/\1_cross_scene_fit\3/' \
    -e 's/^(ho_(perm|gated))gx([0-9]+)$/\1_grand_fit\3/' \
    -e 's/^(ho_gated)L(4|812)$/\1_per_scene_fit60_L\2/' \
    -e 's/^(ho_gated)x[ab]L(4|812)$/\1_cross_scene_fit60_L\2/'
}

R="lane_""runner"
if pgrep -f "$R" >/dev/null; then
  echo "⚠️ lane runner 실행 중 — rename 금지. 큐 소진 후 다시 실행하세요."; exit 1
fi
for cell in "$SE"/*/; do
  for d in "$cell"ho_*; do
    [ -d "$d" ] || continue
    base=$(basename "$d"); nn=$(new_name "$base")
    [ "$nn" = "$base" ] && continue
    if [ "$APPLY" = "--apply" ]; then mv "$d" "$cell$nn" && echo "mv $base -> $nn"
    else echo "[dry] $base -> $nn"; fi
  done
done
