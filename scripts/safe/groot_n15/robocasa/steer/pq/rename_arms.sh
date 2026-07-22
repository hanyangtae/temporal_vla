#!/bin/bash
# arm 디렉토리 명명 개편 (2026-07-08 사용자 지시, 07-10 확장) — ⚠️ steer_eval 을 쓰는
# arm job/heldout 이 없을 때만 실행 (collect 전용 러너는 phase_event_6p 만 쓰므로 허용).
#   ps→per_scene, xb/xa→cross_scene, gx→grand, 숫자→fit{N}(fit 에피소드 수), L 변형 유지.
#   예: ho_permps60→ho_perm_per_scene_fit60, ho_gatedxb30→ho_gated_cross_scene_fit30,
#       ho_gatedL4→ho_gated_per_scene_fit60_L4, ho_gatedgxL812→ho_gated_grand_fit60_L812
# 구 라운드(6p/6pcr/무접미 ho_perm·ho_gated/annotated_strict/baseline)는 rename 대신
# --deprecate 시 deprecated/ 미러로 이동 (새 라운드와 섞임 방지 — 2026-07-10 사용자 지시).
# 사용: bash rename_arms.sh                      (dry-run: 계획만 출력)
#       bash rename_arms.sh --apply              (rename 만 실제 mv)
#       bash rename_arms.sh --apply --deprecate  (rename + 구 라운드 격리)
# env: SE=<steer_eval 경로> DEP=<deprecated 루트> (기본 로컬 경로 — 승준/w2 에서 override)
set -u
SE=${SE:-/home/dongkyu/pkt_ws/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval}
DEP=${DEP:-$(dirname "$SE")/deprecated}
APPLY=0; DEPRECATE=0
for a in "$@"; do case "$a" in
  --apply) APPLY=1;; --deprecate) DEPRECATE=1;;
  *) echo "unknown arg: $a"; exit 2;;
esac; done
MAP="$SE/RENAME_MAP_$(date -u +%Y%m%d).txt"

new_name() {
  local t=$1
  case "$t" in ho_base) echo "$t"; return;; esac
  echo "$t" | sed -E \
    -e 's/^(ho_(perm|gated))ps([0-9]+)$/\1_per_scene_fit\3/' \
    -e 's/^(ho_(perm|gated))x[ab]([0-9]+)$/\1_cross_scene_fit\3/' \
    -e 's/^(ho_(perm|gated))gx([0-9]+)$/\1_grand_fit\3/' \
    -e 's/^(ho_gated)gxL(4|812)$/\1_grand_fit60_L\2/' \
    -e 's/^(ho_gated)x[ab]L(4|812)$/\1_cross_scene_fit60_L\2/' \
    -e 's/^(ho_gated)L(4|812)$/\1_per_scene_fit60_L\2/'
}

# 구 라운드 산출물 (rename 아닌 격리 대상)
is_deprecated() {
  case "$1" in
    *6p*|*6pcr*) return 0;;
    ho_perm|ho_gated) return 0;;
    annotated_strict|baseline) return 0;;
  esac
  return 1
}

# 가드: steer_eval 에 쓰는 프로세스(heldout arm)만 차단. collect 전용 러너는 허용.
H="heldout_""round_cell"
if pgrep -f "$H" >/dev/null; then
  echo "⚠️ heldout arm 실행 중 — rename 금지."; exit 1
fi
QROOT_DEFAULT=$(dirname "$(dirname "$SE")")/groot_n15/work_queue
QROOT=${QROOT:-$QROOT_DEFAULT}
if ls "$QROOT"/running/*.job >/dev/null 2>&1; then
  if grep -hv '^collect|' "$QROOT"/running/*.job 2>/dev/null | grep -q .; then
    echo "⚠️ arm job 실행 중 ($(grep -hv '^collect|' "$QROOT"/running/*.job | head -1)) — rename 금지."; exit 1
  fi
fi

do_mv() { # src dst kind
  local src=$1 dst=$2 kind=$3 base rel
  base=$(basename "$src"); rel=${src#"$SE"/}
  if [ -e "$dst" ]; then echo "[skip] $rel → 대상 존재: $dst"; return; fi
  if [ "$APPLY" = 1 ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst" && echo "$kind $rel -> ${dst#"$(dirname "$SE")"/}" | tee -a "$MAP"
  else
    echo "[dry-$kind] $rel -> ${dst#"$(dirname "$SE")"/}"
  fi
}

for cell in "$SE"/*/; do
  cellname=$(basename "$cell")
  for d in "$cell"*/; do
    [ -d "$d" ] || continue
    base=$(basename "$d")
    # 재사용 심링크(ps30/60↔6p 등, 함정 §8-7)는 대상(6p)과 함께 deprecated 로 — 같은
    # 미러 cell dir 안에서 대상 basename 상대 링크로 재작성해 유효성 유지.
    if [ -L "${d%/}" ]; then
      if [ "$DEPRECATE" = 1 ]; then
        tgt=$(basename "$(readlink "${d%/}")")
        if [ "$APPLY" = 1 ]; then
          mkdir -p "$DEP/steer_eval/$cellname"
          rm "${d%/}" && ln -s "$tgt" "$DEP/steer_eval/$cellname/$base" \
            && echo "dep-link $cellname/$base (target=$tgt)" | tee -a "$MAP"
        else
          echo "[dry-dep-link] $cellname/$base (target=$tgt)"
        fi
      fi
      continue
    fi
    if is_deprecated "$base"; then
      [ "$DEPRECATE" = 1 ] && do_mv "${d%/}" "$DEP/steer_eval/$cellname/$base" dep
      continue
    fi
    case "$base" in ho_*) ;; *) continue;; esac
    nn=$(new_name "$base")
    [ "$nn" = "$base" ] && continue
    do_mv "${d%/}" "$cell$nn" mv
  done
done
[ "$APPLY" = 1 ] && [ -f "$MAP" ] && echo "매핑 기록: $MAP"
exit 0
