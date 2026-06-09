# Cache Paths (체크포인트 · 데이터셋 경로 규칙)

체크포인트와 데이터셋은 **repo 트리 밖 cache** 에 둔다. 이 문서는 그 위치와 코드에서
참조하는 방법(단일 소스)을 정리한 기준 문서다. 새 스크립트/설정/문서를 만들 때 여기 규칙을 따른다.

## 요약 (TL;DR)

- 체크포인트·데이터셋은 git 에 안 들어가고 `~/.cache/temporal_vla/{checkpoints,datasets}` 에 산다.
- 컨테이너 안에서는 그 cache 가 `/cache` 로 마운트되고 `VLA_CACHE_ROOT=/cache` 가 주입된다.
- 경로를 **문자열로 하드코딩하지 말 것**. Python 은 `scripts/path_setup.py`, Shell 은
  `scripts/utils/cache_env.sh` 한 곳에서만 cache 루트를 해석한다.
- 학습 **산출물**(파인튜닝 ckpt, rollout, eval 결과)은 cache 가 아니라 `outputs/` 에 그대로 둔다.

## 왜 이렇게 했나

- 체크포인트/데이터셋(수~수십 GB)을 repo 트리 안(`checkpoints/`, `data/`)에 두던 구조에서
  cache 로 분리했다. repo 가 가벼워지고, 같은 머신의 여러 워크스페이스가 한 cache 를 공유할 수 있다.
- 이전엔 `/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B` 같은 경로가 코드 29곳에 하드코딩돼
  실제 파일 위치(`checkpoints/nvidia/GR00T-N1.6-3B`)와 어긋난 stale 가 생겼다. 그래서 경로 해석을
  단일 헬퍼로 모았다.

## 레이아웃

```
호스트:  ~/.cache/temporal_vla/
컨테이너: /cache/                     (docker-compose 가 위 디렉토리를 bind-mount)
│
├── checkpoints/
│   ├── nvidia/GR00T-N1.6-3B/         # 베이스 모델 (canonical, 로컬 1벌)
│   └── ttt_module/...                # 사전학습 모듈 ckpt
├── datasets/
│   ├── robocasa/v1.0/{pretrain,target}/atomic/<Task>/<date>/lerobot/
│   ├── datasets/                     # merge 산출 LeRobot v2.1 등 (구 data/datasets)
│   ├── robocasa_eagle_pre_llm/<Task>/embeddings.pt
│   ├── huggingface/                  # 일부 컨테이너의 HF_HOME (아래 'HF 캐시' 참고)
│   └── bridge_v2_*, calvin* ...      # (legacy / phase1)
└── safe_groot_n16_data.tar.gz
```

> 학습 산출물은 여기 두지 않는다 → `outputs/checkpoints/<run>`, `outputs/train/`,
> `outputs/eval/`, `outputs/rollouts/` 는 그대로 repo `outputs/` 에 남는다.

## 경로 참조 방법 (단일 소스)

### Python

```python
from scripts.path_setup import CHECKPOINTS_ROOT, DATA_ROOT
base  = CHECKPOINTS_ROOT / "nvidia/GR00T-N1.6-3B"
atomic = DATA_ROOT / "robocasa/v1.0/pretrain/atomic"
```

- `scripts/path_setup.py` 가 `VLA_CACHE_ROOT` env 를 읽는 **유일한 곳**이다
  (컨테이너=`/cache`, 호스트 기본=`~/.cache/temporal_vla`). 노출 심볼:
  `CACHE_ROOT`, `CHECKPOINTS_ROOT (=CACHE_ROOT/"checkpoints")`, `DATA_ROOT (=CACHE_ROOT/"datasets")`.
- import 가 되려면 repo root 가 `sys.path` 에 있어야 한다(대부분 스크립트가 이미
  `sys.path.insert(0, <repo_root>)` 후 `from src.* import` 하는 패턴이라 동일하게 동작).
- 개별 스크립트에서 `os.environ.get("VLA_CACHE_ROOT", ...)` 를 새로 쓰지 말 것 — 헬퍼를 import 한다.

### Shell

```bash
source "${REPO_ROOT}/scripts/utils/cache_env.sh"
echo "${VLA_CHECKPOINTS_ROOT}"   # = ${VLA_CACHE_ROOT}/checkpoints
echo "${VLA_DATASETS_ROOT}"      # = ${VLA_CACHE_ROOT}/datasets
```

- `cache_env.sh` 가 `VLA_CACHE_ROOT`(기본 `~/.cache/temporal_vla`),
  `VLA_CHECKPOINTS_ROOT`, `VLA_DATASETS_ROOT` 를 export.
- `cd /temporal_vla` 하고 도는 **컨테이너 전용 스크립트**는 `/cache/...` 리터럴을 써도 된다
  (컨테이너에 `VLA_CACHE_ROOT=/cache` 가 항상 있으므로).

### configs/checkpoints/*.yaml

로컬 체크포인트는 `checkpoint_source.id` 에 컨테이너 경로로 적는다:

```yaml
checkpoint_source:
  type: local
  id: /cache/checkpoints/nvidia/GR00T-N1.6-3B
```

HF hub ID(`nvidia/GR00T-N1.6-3B`, `moojink/...` 등)는 경로가 아니므로 그대로 둔다.

### docker-compose.yml

모든 서비스에 다음이 들어가 있다:

```yaml
volumes:
  - .:/temporal_vla:rw
  - ${VLA_CACHE_ROOT:-${HOME}/.cache/temporal_vla}:/cache:rw
environment:
  - VLA_CACHE_ROOT=/cache
```

## HF 캐시 (참고, cache 이동 범위 밖)

HuggingFace hub 캐시는 이번 이동의 1차 대상이 아니다.

- 모델 서버 컨테이너(groot/dreamvla/openvla_oft/xvla): `HF_HOME=~/.cache/huggingface`
  (호스트 HF 캐시를 그대로 bind-mount). 베이스 GR00T-N1.6-3B 는 HF 가 아니라 위 로컬
  `/cache/checkpoints/nvidia/GR00T-N1.6-3B` 를 canonical 로 쓴다.
- lerobot 통합 컨테이너: `./data`를 `/cache`에 mount하고 `HF_HOME=/cache/huggingface`를
  사용한다.
- robocasa/calvin/groot_n15/upvla: 과거 `/temporal_vla/data/huggingface` 를 쓰던 것을
  `HF_HOME=/cache/datasets/huggingface` 로 옮겼다(faithful swap). 필요하면 추후 통일 가능.

## 구 경로 → 신 경로 매핑

| 구 (repo 내부) | 신 (cache) |
|---|---|
| `temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B` | `~/.cache/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B` (컨테이너 `/cache/checkpoints/...`) |
| `temporal_vla/checkpoints/<X>` | `<cache>/checkpoints/<X>` |
| `temporal_vla/data/<X>` | `<cache>/datasets/<X>` (faithful prefix swap; `data/datasets/foo` → `datasets/datasets/foo`) |
| `/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B` (stale, 29곳) | `/cache/checkpoints/nvidia/GR00T-N1.6-3B` 로 통일 |
| 루트 `safe_groot_n16_data.tar.gz` | `<cache>/safe_groot_n16_data.tar.gz` |

## 자주 하는 실수 (바꾸면 안 되는 것)

- `<dataset>/data/chunk-XXX/episode_*.parquet` 의 `data/chunk-...` 는 **LeRobot 내부 구조**다.
  repo `data/` 디렉토리가 아니다 → 절대 cache 로 바꾸지 말 것.
- `configs/checkpoints/*.yaml` 경로(프로파일 설정 파일)는 repo 안에 남는 설정이다 → 그대로.
- `outputs/...` (학습 산출물)은 이동 대상이 아니다. 베이스 GR00T-N1.6-3B 만 예외적으로 cache 로 갔다.
- HF hub ID(`org/name` 형태, 경로 prefix 없음)는 경로가 아니다 → 그대로.
- 서브모듈 내부 경로(`src/benchmarks/robocasa/datasets/`, `src/policies/dreamvla/.../checkpoints/` 등)는 그대로.
