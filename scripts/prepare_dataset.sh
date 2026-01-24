#!/bin/bash

# =============================================================================
# Dataset Preparation Script for X-VLA Fine-tuning
# =============================================================================
# 이 스크립트는 LIBERO HDF5 데이터를 LeRobot 형식으로 변환하고
# 시간 정보를 추가합니다.
#
# 사용법: ./scripts/prepare_dataset.sh
#
# 주의: Docker 컨테이너 내부에서 실행하거나,
#       docker exec vla_container ./scripts/prepare_dataset.sh 로 실행
# =============================================================================

set -e  # Exit on error

# =============================================================================
# 경로 설정
# =============================================================================
# Docker 컨테이너 내부 경로 (서버/PC1 공통)
WORKSPACE="/workspace/vla_tset"

# 입력 데이터 경로 (LIBERO HDF5 파일들)
INPUT_DIR="${WORKSPACE}/data/datasets/libero_goal"

# 중간 출력 (ee6d 형식 변환된 데이터셋)
INTERMEDIATE_DIR="${WORKSPACE}/data/lerobot_libero_goal_ee6d"
INTERMEDIATE_REPO_ID="lerobot_libero_goal_ee6d"

# 최종 출력 (시간 정보 추가된 데이터셋)
OUTPUT_DIR="${WORKSPACE}/data/lerobot_libero_goal_ee6d_time"
OUTPUT_REPO_ID="lerobot_libero_goal_ee6d_time"

# FPS 설정
FPS=10

# =============================================================================
# Step 1: LIBERO HDF5 → LeRobot ee6d 형식 변환
# =============================================================================
echo "=============================================="
echo "🔄 Step 1: Converting LIBERO HDF5 to LeRobot ee6d format"
echo "   Input:  ${INPUT_DIR}"
echo "   Output: ${INTERMEDIATE_DIR}"
echo "=============================================="

# 입력 디렉토리 확인
if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ Error: Input directory not found: ${INPUT_DIR}"
    echo "   Please download LIBERO dataset first."
    exit 1
fi

# 기존 출력 디렉토리가 있으면 삭제
if [ -d "${INTERMEDIATE_DIR}" ]; then
    echo "⚠️  Removing existing intermediate directory: ${INTERMEDIATE_DIR}"
    rm -rf "${INTERMEDIATE_DIR}"
fi

python3 ${WORKSPACE}/scripts/convert_libero_to_lerobot.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${INTERMEDIATE_DIR}" \
    --repo_id "${INTERMEDIATE_REPO_ID}" \
    --fps ${FPS} \
    --force_override

echo "✅ Step 1 complete!"
echo ""

# =============================================================================
# Step 2: 시간 정보 추가 (time_to_go_task)
# =============================================================================
echo "=============================================="
echo "⏱️  Step 2: Adding time features"
echo "   Input:  ${INTERMEDIATE_DIR}"
echo "   Output: ${OUTPUT_DIR}"
echo "=============================================="

# 기존 출력 디렉토리가 있으면 삭제
if [ -d "${OUTPUT_DIR}" ]; then
    echo "⚠️  Removing existing output directory: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

python3 ${WORKSPACE}/scripts/add_time_features.py \
    --repo_id "${INTERMEDIATE_REPO_ID}" \
    --root "${INTERMEDIATE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --new_repo_id "${OUTPUT_REPO_ID}"

echo "✅ Step 2 complete!"
echo ""

# =============================================================================
# Step 3: 데이터셋 검증
# =============================================================================
echo "=============================================="
echo "🔍 Step 3: Validating dataset"
echo "=============================================="

python3 -c "
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
info_path = output_dir / 'meta' / 'info.json'

if info_path.exists():
    with open(info_path) as f:
        info = json.load(f)
    
    print('Dataset Info:')
    print(f'  Total episodes: {info.get(\"total_episodes\", \"N/A\")}')
    print(f'  Total frames: {info.get(\"total_frames\", \"N/A\")}')
    print(f'  FPS: {info.get(\"fps\", \"N/A\")}')
    print()
    print('Features:')
    for key, val in info.get('features', {}).items():
        shape = val.get('shape', 'N/A')
        dtype = val.get('dtype', 'N/A')
        print(f'  {key}: shape={shape}, dtype={dtype}')
else:
    print('❌ Error: info.json not found')
    exit(1)
"

echo ""
echo "=============================================="
echo "🎉 Dataset preparation complete!"
echo "   Final dataset: ${OUTPUT_DIR}"
echo ""
echo "   Expected features:"
echo "   - action: shape=[20] (ee6d format)"
echo "   - observation.state: shape=[20]"
echo "   - time_to_go_task: shape=[1]"
echo "=============================================="
