#!/bin/bash
# 학습용 8개 태스크 Subtask 분해 실행 (백그라운드)
# 
# 사용법:
#   컨테이너 내부에서:
#   chmod +x scripts/run_decompose_all.sh
#   nohup scripts/run_decompose_all.sh > logs/decompose.log 2>&1 &
#
# 학습 태스크 (8개):
#   1. open_the_middle_drawer_of_the_cabinet
#   2. open_the_top_drawer_and_put_the_bowl_inside
#   3. put_the_bowl_on_the_plate
#   4. put_the_bowl_on_the_stove
#   5. put_the_bowl_on_top_of_the_cabinet
#   6. put_the_cream_cheese_in_the_bowl
#   7. put_the_wine_bottle_on_top_of_the_cabinet
#   8. turn_on_the_stove
#
# 테스트 태스크 (2개) - 나중에 따로 실행:
#   - push_the_plate_to_the_front_of_the_stove
#   - put_the_wine_bottle_on_the_rack

set -e

# 환경 변수 로드
export OPENAI_API_KEY=$(grep OPENAI_API_KEY /workspace/vla_tset/.env | cut -d'=' -f2)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not found in .env"
    exit 1
fi

echo "✅ API Key loaded: ${OPENAI_API_KEY:0:15}..."

# 디렉토리 설정
cd /workspace/vla_tset
mkdir -p logs
mkdir -p data/datasets/libero_goal_subtasks

# 기존 데이터 삭제
rm -rf data/datasets/libero_goal_subtasks/*

# pandas 설치 (Excel 출력용)
pip install pandas openpyxl -q

# 학습용 8개 태스크 목록
TRAIN_TASKS=(
    "open_the_middle_drawer"
    "open_the_top_drawer"
    "put_the_bowl_on_the_plate"
    "put_the_bowl_on_the_stove"
    "put_the_bowl_on_top_of_the_cabinet"
    "put_the_cream_cheese"
    "put_the_wine_bottle_on_top"
    "turn_on_the_stove"
)

echo ""
echo "=========================================="
echo "🚀 Starting Subtask Decomposition"
echo "   Model: gpt-4o"
echo "   Tasks: ${#TRAIN_TASKS[@]}"
echo "   Time: $(date)"
echo "=========================================="
echo ""

# 실행
python3 scripts/decompose_subtasks.py \
    --input data/datasets/libero_goal \
    --output data/datasets/libero_goal_subtasks \
    --provider openai \
    --model gpt-4o \
    --tasks ${TRAIN_TASKS[@]}

echo ""
echo "=========================================="
echo "✅ Decomposition Complete!"
echo "   Time: $(date)"
echo "   Results: data/datasets/libero_goal_subtasks/"
echo "   Analysis: data/datasets/libero_goal_subtasks/analysis_results.xlsx"
echo "=========================================="
