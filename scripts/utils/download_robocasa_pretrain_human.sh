#!/usr/bin/env bash
# RoboCasa pretrain × human × atomic 데이터 10개 task 를 data/robocasa/ 로 다운로드.
#
# 출력 구조: data/robocasa/v1.0/pretrain/atomic/<Task>/<date>/lerobot/
# robocasa.macros.DATASET_BASE_PATH 를 data/robocasa 로 두면 그대로 인식됨.
#
# Box 공유 링크 변환 규칙: https://utexas.box.com/s/<id> -> https://utexas.box.com/shared/static/<id>.tar
# 출처: src/benchmarks/robocasa/robocasa/scripts/download_datasets.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/data/robocasa/v1.0/pretrain/atomic"
mkdir -p "${BASE_DIR}"

# task | date | box-id  (human pretrain atomic)
ENTRIES=(
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

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r task date box_id <<< "${entry}"
  download_one "${task}" "${date}" "${box_id}"
done

echo "[ALL ] complete -> ${BASE_DIR}"
