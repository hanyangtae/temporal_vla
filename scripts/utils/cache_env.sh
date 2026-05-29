# VLA cache 루트 해석 헬퍼. 체크포인트/데이터셋은 repo 트리 밖 cache 에 둔다.
#
# 사용법: 스크립트 상단에서
#   source "${REPO_ROOT}/scripts/utils/cache_env.sh"
# 그러면 아래 3개 변수가 export 된다.
#   - VLA_CACHE_ROOT       (컨테이너=/cache, 호스트 기본=~/.cache/temporal_vla)
#   - VLA_CHECKPOINTS_ROOT (${VLA_CACHE_ROOT}/checkpoints)
#   - VLA_DATASETS_ROOT    (${VLA_CACHE_ROOT}/datasets)
#
# 컨테이너에서는 docker-compose 가 VLA_CACHE_ROOT=/cache 를 주입한다.
: "${VLA_CACHE_ROOT:=${HOME}/.cache/temporal_vla}"
export VLA_CACHE_ROOT
export VLA_CHECKPOINTS_ROOT="${VLA_CACHE_ROOT}/checkpoints"
export VLA_DATASETS_ROOT="${VLA_CACHE_ROOT}/datasets"
