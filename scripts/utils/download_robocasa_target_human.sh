#!/usr/bin/env bash
# RoboCasa target × human × atomic 데이터 15 task 를 data/robocasa/ 로 다운로드.
#
# 출력 구조: data/robocasa/v1.0/target/atomic/<Task>/<date>/lerobot/
# pretrain 다운로드와 같은 구조의 target split.
#
# Box id 출처: src/benchmarks/robocasa/robocasa/models/assets/box_links/box_links_ds.json
# (target/atomic/<Task>/<date>/lerobot.tar entry 의 box shared id)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/data/robocasa/v1.0/target/atomic"

mkdir -p "${BASE_DIR}"

# task | date | box-id  (human target atomic, 15 task)
ENTRIES=(
  "TurnOnElectricKettle|20250817|ctirlg0mryjz511rb2pntdgmwir2m3vw"
  "CloseToasterOvenDoor|20250818|86caoedy7compqenfh5a8ry62xebruyh"
  "OpenCabinet|20250813|b15gbqq99i6kx8bogkdkq9lvurhbg9xw"
  "SlideDishwasherRack|20250820|4mrm925oekfrlfyjlg9c8wflqktwla6y"
  "PickPlaceToasterToCounter|20250817|4gn9j4llpxb2v8pk2r00im1a1zz7qqqm"
  "TurnOnMicrowave|20250813|ta8tufw9408tzx3fxffwr5lh1hknp8fz"
  "OpenDrawer|20250816|mpztcjagk0fqs7e5360znjvhs08hnyjp"
  "PickPlaceSinkToCounter|20250813|gtxm57dfse72c6zwbel05zhjritt1scq"
  "PickPlaceCounterToStove|20250818|9zy06e4dcaxezsq35l6qpe39zi0uaqvh"
  "CloseFridge|20250816|6yw8qwkwufg7di063v7e5zbcyhgn8nlf"
  "TurnOnSinkFaucet|20250812|g0gamv85pvi4ab220szfhr72wg7tioap"
  "PickPlaceCounterToCabinet|20250811|ersnvqs6rcy6yge2fy5hmna7d954v9ty"
  "CoffeeSetupMug|20250813|rhrr67a994a16o25iprsfec07a1wmwm2"
  "PickPlaceDrawerToCounter|20250820|xec45dl0meig2yl1v4cmi1hqxw3eu74l"
  "TurnOffStove|20250812|xsvs8uazyfwoeqij298gsax3b400b4h8"
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
  curl -fL --retry 3 --retry-delay 5 -C - -o "${tar_path}" "${url}"

  echo "[tar ] extract -> ${target_dir}"
  tar -xf "${tar_path}" -C "${target_dir}"
  rm -f "${tar_path}"
  echo "[done] ${task}"
}

# 선택적 task 필터: 인자 주면 해당 task 만, 안 주면 15개 전체.
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
