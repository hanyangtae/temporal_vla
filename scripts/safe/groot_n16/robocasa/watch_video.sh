#!/bin/bash
# 원격 서버에서 mp4를 로컬로 당겨와서 재생.
# 사용법:
#   bash watch_video.sh task7--ep53--succ0.mp4 SlideDishwasherRack
#   bash watch_video.sh task2--ep11--succ0.mp4 OpenCabinet

FILENAME=${1:?"파일명 필요 (예: task7--ep53--succ0.mp4)"}
TASK=${2:?"task 폴더명 필요 (예: SlideDishwasherRack)"}

REMOTE_BASE="kimseungjun@166.104.146.37:/home/kimseungjun/workspace/temporal_vla/outputs/eval/robocasa/groot_n16/target_atomic_moderate10_pathway_pertoken_100ep/raw_rollouts"
LOCAL_TMP="/tmp/watch_videos"

mkdir -p "$LOCAL_TMP"
LOCAL_PATH="$LOCAL_TMP/$FILENAME"

echo "다운로드 중: $TASK/$FILENAME ..."
scp -P 11112 "$REMOTE_BASE/$TASK/$FILENAME" "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    # IDE에서 바로 열 수 있도록 workspace에 복사
    WORKSPACE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/${FILENAME}"
    cp "$LOCAL_PATH" "$WORKSPACE_PATH"
    echo ""
    echo "✓ 재생 방법:"
    echo "  1) IDE 파일 탐색기에서 클릭: ${FILENAME}"
    echo "  2) 파일 위치: $WORKSPACE_PATH"
    # 사용 가능한 플레이어 자동 선택
    if command -v mpv &>/dev/null; then
        mpv "$LOCAL_PATH"
    elif command -v vlc &>/dev/null; then
        vlc "$LOCAL_PATH"
    elif command -v ffplay &>/dev/null; then
        ffplay -autoexit "$LOCAL_PATH" 2>/dev/null
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$LOCAL_PATH"
    fi
else
    echo "다운로드 실패"
fi
