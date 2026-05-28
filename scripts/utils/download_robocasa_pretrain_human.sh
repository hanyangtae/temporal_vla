#!/usr/bin/env bash
# RoboCasa pretrain × human × atomic 데이터 10개 task 를 cache datasets/robocasa/ 로 다운로드.
#
# 출력 구조: <cache>/datasets/robocasa/v1.0/pretrain/atomic/<Task>/<date>/lerobot/
# robocasa.macros.DATASET_BASE_PATH 를 <cache>/datasets/robocasa 로 두면 그대로 인식됨.
#
# Box 공유 링크 변환 규칙: https://utexas.box.com/s/<id> -> https://utexas.box.com/shared/static/<id>.tar
# 출처: src/benchmarks/robocasa/robocasa/scripts/download_datasets.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/utils/cache_env.sh"
BASE_DIR="${VLA_DATASETS_ROOT}/robocasa/v1.0/pretrain/atomic"
mkdir -p "${BASE_DIR}"

# task | date | box-id  (human pretrain atomic)
ENTRIES=(
  # 기존 10 task
  "OpenDrawer|20250819|1wh3r8fumly55x8orq4hynzdvui61adj"
  "CloseDrawer|20250819|nvrtrm29t0i438c1g1erlvqhxttnl46f"
  "OpenCabinet|20250819|xfg5pamn63h8r0mfjgx04sxi3jpyvy58"
  "CloseCabinet|20250819|56pi6wwsjhqzza7ezke6jxz66kf6ph9d"
  "OpenFridge|20250819|8b1qfa4ql1zubpufz0xhobh9f8rslz1c"
  "CloseFridge|20250819|kd9kx4l400jyq06uyc0cqdmfclc1s7gx"
  "OpenMicrowave|20250819|hsndmkr9gm4q00ay1sidjpbhixe6b82u"
  "CloseMicrowave|20250819|wiunvswubst88t8x7cy00udi4k7q78jr"
  "PickPlaceCounterToStove|20250819|ddqy3klmtxeai90ujtheqgqqydk0ngji"
  "PickPlaceCounterToSink|20250819|9cv0clw8zw4sfahtu81g4uizpb0stpwa"
  # 신규 20 task — 모두 fixed-base atomic. single-door (cabinet 류 제외).
  # Doors/Lids (7) — Open 위주, motion 다양성
  "OpenOven|20250820|wm696hf03usgrjsqjkdmnbchfpf6ujvu"
  "OpenDishwasher|20250820|cn13od342z0046si983deodgtxkimor9"
  "OpenFridgeDrawer|20250821|zfes13imtdxt0csy46cga4b0qi3ipr2w"
  "OpenToasterOvenDoor|20250820|elr8b557cy5vc0pfgwt4wwt9lwj8sltj"
  "OpenBlenderLid|20250822|hwmpnoj1vyfcs7gs8j8wpdivk7laimh8"
  "OpenElectricKettleLid|20250820|xywmovoafrygd4ry2mxdy56o9y3pt805"
  "OpenStandMixerHead|20250820|bbcwsuneusxu2sx9skuai8oglkio8nca"
  # PickPlace (7) — 다양한 source/target 조합
  "PickPlaceCounterToCabinet|20250819|myhvbjtfii7at6itjepbivcuo1y8krbp"
  "PickPlaceCabinetToCounter|20250819|u4jnblpqftlg2e3nafmh426sjm3xdo3r"
  "PickPlaceCounterToDrawer|20250821|ptft0eb0gc0q4o4k3zsl886a2zq18t2n"
  "PickPlaceCounterToMicrowave|20250819|h1c72tmzbsmi28c5k0k7n5xa3houq9cm"
  "PickPlaceCounterToOven|20250819|ts6s7u1xvzicffn0m7y4hkiqxnf6z17i"
  "PickPlaceSinkToCounter|20250819|yeglgmnrbgv1omfsqr986llfxby06x2b"
  "PickPlaceStoveToCounter|20250819|h1te73m68o9qkbqewf8vl3z1uqsuagwy"
  # Turn/Switch (6) — 다양한 control type (faucet/knob/button/lever)
  "TurnOnSinkFaucet|20250819|3anr1aa9obr6surhz83npcsz77fevfsb"
  "TurnOnStove|20250819|l8my16o129y0yjyo5t16t6ijud2kmz7g"
  "TurnOnMicrowave|20250819|7alz6aevqinrny6ggbl83ld3ty3w27lf"
  "TurnOnBlender|20250822|76lkwmzhcekpffs3cq7hbybsvdd55btl"
  "TurnOnElectricKettle|20250820|183i98pu2opkn92cpihhhwiv0xiek9bc"
  "TurnOnToaster|20250820|scjjcg9tw9o9sf3rlk2gcc4vapnv7pfn"
)

download_one () {
  local task="$1" date="$2" box_id="$3"
  local target_dir="${BASE_DIR}/${task}/${date}"
  local lerobot_dir="${target_dir}/lerobot"
  local tar_path="${target_dir}/lerobot.tar"
  local url="https://utexas.box.com/shared/static/${box_id}.tar"

  if [[ -d "${lerobot_dir}" ]]; then
    echo "[skip] ${task} -> ${lerobot_dir} already exists"
    return 0
  fi

  mkdir -p "${target_dir}"
  echo "[get ] ${task} <- ${url}"
  # -L: follow redirect (Box 는 redirect 됨), -f: HTTP 에러시 fail, --retry: 재시도
  curl -fL --retry 3 --retry-delay 5 -C - -o "${tar_path}" "${url}"

  echo "[tar ] extract -> ${target_dir}"
  tar -xf "${tar_path}" -C "${target_dir}"
  rm -f "${tar_path}"
  echo "[done] ${task}"
}

# 선택적 task 필터: 인자 주면 해당 task 만, 안 주면 10개 전체.
SELECTED=("$@")

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r task date box_id <<< "${entry}"
  if [[ ${#SELECTED[@]} -gt 0 ]]; then
    skip=1
    for s in "${SELECTED[@]}"; do
      if [[ "${s}" == "${task}" ]]; then skip=0; break; fi
    done
    [[ ${skip} -eq 1 ]] && continue
  fi
  download_one "${task}" "${date}" "${box_id}"
done

echo "[ALL ] complete -> ${BASE_DIR}"
