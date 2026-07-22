#!/usr/bin/env bash
# S1 (Track P 하드 게이트): sham ≡ baseline bitwise / double-run 결정론 / perturb 키 산술 /
# 실섭동 실효(≠ baseline) / 캡처↔no-features 판정 필드 일치.
# 사용: GPU=<idx> PORT=8470 bash s1_perturb.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/smoke_common.sh"
GPU="${GPU:?빈 GPU 번호 (nvidia-smi 확인 후)}"
PORT="${PORT:-8470}"
S1="${SMOKE_ROOT}/s1"; mkdir -p "$S1/specs"

spec() { echo "$1" > "$S1/specs/$2.json"; }
spec '{"mode":"C1_camera","sham":true,"spec_seed":11,"scale":1.0,"tag":"sham_c1"}' sham_c1
spec '{"mode":"G1_gripper_init","sham":true,"spec_seed":12,"sigma_xyz_m":0.10,"tag":"sham_g1"}' sham_g1
spec '{"mode":"P1_displace","sham":true,"spec_seed":13,"trigger_record":8,"magnitude":0.08,"tag":"sham_p1"}' sham_p1
spec '{"mode":"P1_displace","sham":true,"sham_forward":true,"spec_seed":13,"trigger_record":8,"magnitude":0.08,"tag":"sham_p1f"}' sham_p1f
spec '{"mode":"P2_force","sham":true,"spec_seed":14,"trigger_record":10,"duration_records":2,"magnitude":15.0,"tag":"sham_p2"}' sham_p2
spec '{"mode":"C1_camera","sham":false,"spec_seed":21,"scale":1.0,"tag":"real_c1"}' real_c1
spec '{"mode":"G1_gripper_init","sham":false,"spec_seed":22,"sigma_xyz_m":0.10,"tag":"real_g1"}' real_g1
spec '{"mode":"P1_displace","sham":false,"spec_seed":23,"trigger_record":8,"magnitude":0.08,"tag":"real_p1"}' real_p1
spec '{"mode":"P2_force","sham":false,"spec_seed":24,"trigger_record":10,"duration_records":2,"magnitude":15.0,"tag":"real_p2"}' real_p2

start_serve "$GPU" "$PORT" --collect --groot-dit-capture-layers 15
trap 'kill_serve "$PORT"' EXIT
wait_health "$PORT"

run_one() {  # name [spec_name] [extra...]
  local name=$1; shift
  local out="$S1/$name"
  if [ -n "$(ep_file "$out" pkl)$(ep_file "$out" json)" ]; then echo "[s1] skip $name"; return; fi
  local extra=()
  if [ $# -ge 1 ] && [ -n "$1" ]; then
    extra=(--perturb-spec "$(to_cont "$S1/specs/$1.json")"); shift
  fi
  echo "[s1] run $name"
  run_ep "$PORT" "$out" 0 0 "${extra[@]}" "$@"
}

run_one base_cap ""
run_one base_cap2 ""
run_one base_nf "" --no-features --expect-chunk-len 16
for s in sham_c1 sham_g1 sham_p1 sham_p1f sham_p2 real_c1 real_g1 real_p1 real_p2; do
  run_one "$s" "$s"
done
run_one real_c1_rerun real_c1
kill_serve "$PORT"; trap - EXIT

BASE_CSV="$(ep_file "$S1/base_cap" csv)"
[ -n "$BASE_CSV" ] || { echo "ABORT: baseline csv 없음"; exit 12; }

check "harness double-run"  judge csv-bitwise "$BASE_CSV" "$(ep_file "$S1/base_cap2" csv)"
for s in sham_c1 sham_g1 sham_p2; do
  check "sham≡base [$s]"    judge csv-bitwise "$BASE_CSV" "$(ep_file "$S1/$s" csv)"
done
check "sham≡base [sham_p1]" judge csv-bitwise "$BASE_CSV" "$(ep_file "$S1/sham_p1" csv)"
# sham_p1f(δ=0+forward)는 warmstart 실측 — bitwise 면 승격, 아니면 skip판을 사양으로 동결
check_soft "sham_p1f warmstart 실측" judge csv-bitwise "$BASE_CSV" "$(ep_file "$S1/sham_p1f" csv)"
for s in real_c1 real_g1 real_p1 real_p2; do
  check "실효 [$s]"          judge csv-bitwise "$BASE_CSV" "$(ep_file "$S1/$s" csv)" --expect-diff
done
check "perturbed 결정론 [real_c1]" judge csv-bitwise "$(ep_file "$S1/real_c1" csv)" "$(ep_file "$S1/real_c1_rerun" csv)"
for s in sham_c1 sham_g1 sham_p1 sham_p2 real_c1 real_g1 real_p1 real_p2; do
  check "perturb-audit [$s]" judge_c perturb-audit "$(to_cont "$(ep_file "$S1/$s" pkl)")" --nas "$NAS"
done
check "capture↔no-features 판정 일치" judge_c fields \
  "$(to_cont "$(ep_file "$S1/base_cap" pkl)")" "$(to_cont "$(ep_file "$S1/base_nf" json)")" \
  --fields episode_success,event_steps,grasp_steps,drop_steps

summary S1
