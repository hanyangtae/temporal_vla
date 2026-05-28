# 🔍 Temporal VLA — Read-Only 상태 점검 보고서

> 생성: 2026-05-26 / CLI: Claude Code 2.1.150 / 모델: Opus 4.7 (1M) / effort: xhigh
> 아무것도 수정하지 않음. 조사 전용.
> 작업 머신: 네이티브 Linux (OEM 커널), 호스트 패키지 매니저 = conda(miniconda3)

## 1. 작업 디렉토리 & 저장소 구조

- pwd: `/home/dongkyu/pkt_ws/temporal_vla`
- git branch: `refactor/groot-n16-vis-core` (main/PR 대상 = `dev`)
- uncommitted (파일 목록만):
  ```
  M  scripts/eval/libero.py
  M  scripts/eval/openvla_oft_libero_rollouts.sh
  M  scripts/eval/robocasa_eval.py
  M  scripts/serve/openvla_oft.py
  M  src/policies/openvla-oft            (submodule pointer)
  ?? SAFE_GROOT_N16_DATA_BUNDLE_README.md
  ?? docs/SAFE.pdf
  ?? docs/groot_n16_robocasa_safe_report.md
  ?? docs/robocasa_env_reproducibility.md
  ?? docs/safe_groot_n16_robocasa_wiring.md
  ?? scripts/eval/test_robocasa_env_reproducibility.py
  ?? src/benchmarks/LIBERO/             (untracked submodule dir)
  ?? src/benchmarks/calvin/            (untracked submodule dir)
  ```
- git log --oneline -n 20:
  ```
  c12c87c docs: GR00T-N1.6 datapoint semantics + 환경 재현 가이드 + 용어 통일
  023106a chore: dev merge 잔존 superseded 파일 정리
  8055574 refactor: vis/ cluster_static 통합 — 11개 스크립트 → 1 CLI + 2 모듈
  321428a Merge branch 'dev' of ...temporal_vla into refactor/groot-n16-vis-core
  ab87fd1 Merge pull request #44 from hanyangtae/pdk/safe-groot-n16-followups
  640a312 Document SAFE RoboCasa runbook flow
  2c481f1 Generalize RoboCasa collection wrappers
  9e6d1f6 Refactor SAFE RoboCasa run configuration
  c445b7c Add SAFE N1.6 detector followup tooling
  a85ca37 refactor: vis/ 공통 helper를 core/ 패키지로 추출, progress 시각화 통합
  cf284a7 chore: GR00T-N1.6 SAFE vis/ 분석 스크립트 일괄 기록
  02986ed docs: update GR00T N1.6 SAFE wiring documentation ...
  1d3d471 chore: organize GR00T N1.6 SAFE artifact paths
  eb2bc6f docs: document GR00T N1.6 SAFE detector results
  e4b2682 feat: add GR00T N1.5 SAFE split helper
  ca6d69f feat: add GR00T N1.6 SAFE detector analysis tools
  f36b43f feat: add GR00T N1.6 SAFE rollout collection
  02682ff 문서 정리
  0104816 chore: Update .gitignore to include CONTEXT.md
  341f712 Merge pull request #43 ...n1.5_1.6_tuning_eval_docs_refactor
  ```
- top-level (`ls -la`) 주요 항목: `scripts/ src/ docs/ configs/ docker/ data/ outputs/ lerobot/ tests/ checkpoints/ logs/ temp/`, `docker-compose.yml`, `CLAUDE.md`, `README.md`, 그리고 루트에 `safe_groot_n16_data.tar.gz` (2.36 GB) 가 untracked로 존재.
- 메인 코드 root: 프로젝트 자체 코드 = `scripts/` + `src/` (단, `src/policies/*`, `src/benchmarks/*`, `lerobot/` 는 git submodule = 외부 코드).
- `src/` 2단계 depth:
  ```
  src/
  ├── benchmarks/   (submodules) calvin/ LIBERO/ robocasa/ robosuite/
  ├── datasets/     adapters/  __pycache__/      (학습용 dataset+adapter)
  ├── deprecated/
  ├── policies/     (submodules) dreamvla/ groot/ Isaac-GR00T/
  │                 Isaac-GR00T-N1.5/ openvla-oft/ UP-VLA/
  ├── processor/    action/  obs/  __pycache__/  (추론용 pipeline)
  ├── ttt/          integrations/  __pycache__/  (TTA/TTT 모듈)
  └── utils/        common/
  ```

## 2. 키워드별 기존 코드 위치 + 첫 30줄

### [A] `z_mean` / `pooled_hidden_states`
주로 `scripts/safe/groot_n16/robocasa/vis/**` 에 분포. 대표: `scripts/safe/groot_n16/robocasa/safe_feature_vectors.py` (172 lines)
```python
"""SAFE feature vector loading and aggregation for GR00T N1.6 RoboCasa rollouts."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_aggregation_command(value: str) -> float | str:
    if value == "mean" or value.startswith("concat"):
        return value
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Unknown aggregation command: {value}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"Relative aggregation index must be in [0, 1], got: {value}")
    return parsed
```

### [B] `safe` / `SAFE-LSTM` / `failure_detector`
대표: `scripts/safe/groot_n16/robocasa/vis/core/lstm.py` (48 lines) — SAFE-LSTM detector 정의
```python
"""SAFE-LSTM detector definition + load + causal inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class LSTMDetector(nn.Module):
    """Single-layer LSTM + linear head with sigmoid output (SAFE-LSTM)."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


def load_detector(ckpt_path: Path, hidden_dim: int = 256, device: str = "cpu") -> LSTMDetector:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    model = LSTMDetector(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
```
관련 문서: `docs/safe/groot_n16_robocasa_safe_report.md`, `docs/safe/groot_n16_robocasa_wiring.md`, `docs/adr/0001-dedicated-safe-groot-n16-zmq-server.md`.

### [C] `groot` / `GR00T` / `n1_6` / `n16`
대표: `src/ttt/integrations/groot_wrapper.py` (471 lines) — TTT × GR00T N1.6 통합
```python
"""GR00T N1.6 × TTT integration wrapper.

Stage 2 학습/추론: TTT module 은 Phase 1 ckpt 로 **outer-frozen** 으로 로드되고,
각 sample 마다 episode prefix Eagle pre-LLM 시퀀스 ``z_0..z_t`` 를 sequential 하게
inner-loop SSL 로 누적 → 마지막 timestep 의 TTT 출력 = **Latent History Token (LHT)**.
LHT 는 ``.detach()`` 후 DiT cross-attention KV 에 token 으로 **direct prepend**
(projector 없음) — action loss 의 gradient 가 TTT 로 흐르지 않게 차단. TTT 는
오로지 inner-loop SSL 로만 적응하고 outer 학습은 안 받음.

Architecture (Stage 2 forward, episode-prefix mode)
---------------------------------------------------
::

    inputs:
      ├─ ttt_z_seq      [B, T_max, 2048]   (Eagle pre-LLM 캐시 0..t 까지)
      ├─ ttt_valid_mask [B, T_max] bool
      └─ <GR00T 표준 obs at frame t (image, state, lang)>

    ttt_z_seq, valid_mask
        │
        ▼ predictor.meta_forward (inner-loop SSL, create_graph=False)
    ttt_outputs_seq [T_max, B, 2048]
        │
        ▼ gather last-valid timestep per item
    LHT [B, 2048]
        │
        ▼ .detach().unsqueeze(1)
    ttt_token [B, 1, 2048]

    <obs at frame t> ─► Eagle.model(hidden_states[-1]) ─► post_llm [B, T_vl, 2048]
```
기타: `scripts/serve/groot.py` (317 lines), `scripts/eval/groot_robocasa_zmq_eval.py`, `scripts/train/phase1_groot_robocasa.py`, `configs/checkpoints/groot__robocasa_panda_omron.yaml`.

### [D] `robocasa` / `rollout`
대표: `scripts/eval/robocasa_eval.py` (597 lines, 현재 modified 상태)
```python
"""
VLA closed-loop 평가 스크립트 (RoboCasa, 모델 무관).

통일 API를 따르는 어떤 VLA 서버든 --vla-server URL만 바꾸면 평가 가능.

사용법:
  # DreamVLA로 평가
  python scripts/eval/robocasa_eval.py \
    --task TurnOnMicrowave \
    --vla-server http://localhost:8200
  ...
출력 구조 (--output-dir 지정 시):
  {output_dir}/
    {TaskName}.json   ← 태스크 완료마다 즉시 저장
    summary.json      ← 전체 완료 후 종합 요약
"""
```
rollout 수집 본체: `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`. dataset reader: `src/datasets/robocasa_v21_reader.py` (174 lines).

### [E] `silhouette` / `mahalanobis` / `cluster_analysis`
대표: `scripts/safe/groot_n16/robocasa/vis/core/metrics.py` (119 lines) — silhouette/centroid/ROC-AUC
```python
"""Cluster metrics: silhouette, centroid distance, per-point a-vs-b, ROC-AUC."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .distance import l2_normalize


def silhouette_safe(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    sample_size: int | None = None,
    random_state: int = 0,
) -> float | None:
    """sklearn silhouette_score with safe fallbacks (None when undefined)."""
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return None
    ss = sample_size if (sample_size is not None and X.shape[0] > sample_size) else None
    try:
        return float(
            silhouette_score(X, labels, metric=metric, sample_size=ss, random_state=random_state)
        )
    except ValueError:
        return None
```
- silhouette: 위 파일에 구현. 클러스터링 CLI = `vis/cluster_static.py`, 거리 helper = `vis/core/distance.py`.
- mahalanobis: 코드에서 미확인 (위 grep 매칭은 silhouette/cluster_analysis 때문이며, mahalanobis 키워드를 코드에서 직접 확인하지 못함).

### [F] `conceptor` / `activation_steering` / `steering_hook` / `COAST`
- 프로젝트 코드·문서 어디에도 존재하지 않음 (0 matches). 파일명, 본문, TODO/FIXME 모두 없음.
- → Task B는 백지에서 시작해야 함.

## 3. Python / 의존성 환경

- 호스트 패키지 매니저: conda (miniconda3). venv/uv 흔적 없음 (`uv` 미설치).
- conda envs:

  | env | python | numpy | scipy | sklearn | torch | pytest |
  |---|---|---|---|---|---|---|
  | `base` (활성) | 3.13.2 | ❌ | ❌ | ❌ | ❌ | ❌ |
  | `hyundai_aigs` | 3.10.16 | 2.0.1 | 1.15.3 | 1.7.2 | 2.7.1+cu126 | ❌ |
  | `myenv` | — | python 바이너리 없음 (깨진 env) | | | | |

  - `requests` 2.32.3, `urllib` 은 base 에 존재. PDF 라이브러리(`fitz`/`pypdf`/`pdfplumber`)·`pdftotext` 바이너리·`tree` 는 미설치.
- pyproject.toml / setup.py / requirements.txt (루트): 없음. 의존성은 Docker 이미지별로 분리:
  `docker/{upvla,groot,openvla_oft,dreamvla,robocasa,xvla}/requirements.txt`.
- lint config (ruff/black/isort/flake8/pre-commit): 없음 (확인된 범위 내).
- pytest 설정 (`pytest.ini`/`tox.ini`/`[tool.pytest]`): 없음 (pyproject 부재). 단 `tests/` 에 `test_processor.py`(25 KB), `test_serve_lerobot.py`(20 KB) 존재 — test_ 네이밍.
- ※ CLAUDE.md 가 말하는 robocasa=3.11 / calvin=3.8 환경은 호스트 conda가 아니라 Docker 컨테이너 내부 env로 보임.

## 4. 데이터 위치

- rollout/feature 데이터 root: `outputs/eval/robocasa/groot_n16/`
  - `rollouts_n16_seen5_20ep_upstream_video_20260519/{Task}/task#--ep#--succ{0,1}.pkl`
  - `safe_split_seen4_unseen2_openDrawer_pnpCab_100ep/` → `manifest.tsv`, `summary.tsv` (split 정의). 총 600 rollouts / 6 tasks / task당 100.
  - `safe_feature_vis/seen4_unseen2_.../` → 다수 `*.tsv` 분석 산출물(cluster, silhouette, online_detection 등).
- 파일 개수: `*.pkl` = 1204개 (rollout feature). `*.npz` = 1061개 — 단 이건 `data/robocasa/v1.0/pretrain/.../extras/episode_*/states.npz` (RoboCasa env state, array 이름 `['states']`)이며 hidden state 아님.
- pkl 한 개 구조 (`PnPCounterToStove/task4--ep59--succ1.pkl`):
  ```python
  {
    'task_suite_name': 'groot_n16_robocasa', 'task_id': 4,
    'task_description': 'pick the potato ... place it in the pan',
    'episode_idx': 59, 'episode_success': 1,
    'hidden_states':  list[21],  # T_env=21, 각 원소 Tensor (4,16,1024) float16
    'actions':        list[21],
    'action_vectors': ndarray (21, 12) float32,
    'action_keys': ['action.end_effector_position','action.end_effector_rotation',
                    'action.gripper_close','action.base_motion','action.control_mode'],
    'feature_kind': 'groot_n16_dit_valid_action_tokens_pre_velocity',
    'feature_axes': ['denoising_step','valid_action_step','feature_dim'],
    'feature_slice': 'valid', 'exported_action_token_count': 16,
    'valid_action_horizon': 16, 'model_action_horizon': 50,
    'num_inference_timesteps': 4,
    'env_name': 'robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env',
    'video_source': 'groot_upstream_video_recording_wrapper',
  }
  ```
- hidden state schema: per-step `[K=4 denoising, H=16 valid_action, D=1024]` float16. GR00T N1.6 action head DiT output (action decoder 직전, SAFE의 `pre_velocity` 위치 대응). detector 입력 `z_t` = K·H축 평균 → [1024] (`diff_idx_rel=mean, horizon_idx_rel=mean`). (상세: `SAFE_GROOT_N16_DATA_BUNDLE_README.md`)
- 참고 결과치 (README): val_unseen end max-so-far ROC-AUC ≈ 0.849±0.121; split CP α=0.2 bal-acc ≈ 0.634. silhouette success/fail ≈ 0.008 (분리 거의 없음).

## 5. Paper 디렉토리

- `C:\Users\rudxo\OneDrive\바탕 화면\data\papers\`: 존재하지 않음. 이 머신은 네이티브 Linux(OEM 커널)이고 WSL이 아니므로 `/mnt/c/...`, `/c/...` 모두 MISSING. → 그 Windows 경로는 이 세션에서 접근 불가.
- 로컬에 있는 paper PDF (프로젝트 내부):
  ```
  docs/references/VITA.pdf
  docs/references/CoT-VLA.pdf
  docs/references/robocasa365.pdf
  docs/references/Scaling World Model.pdf
  docs/SAFE.pdf
  ```
- .txt 변환본: 없음 (`pdftotext` 미설치).
- paper 요약/리뷰 노트: RoboMD/PPGuide/COAST 관련 노트는 없음. docs/ 의 .md 는 전부 프로젝트 자체 문서(SAFE/GR00T/robocasa wiring·report 등). 스타일 reference로 쓸 만한 기존 분석 문서 예: `docs/groot_n16_robocasa_safe_report.md` (한글 기술 보고서 스타일, 16 KB).

## 6. conceptor / steering 흔적

- 코드·문서·TODO/FIXME 전부 0건 (`conceptor|COAST|activation_steering|steering_hook`).
- worktree: 추가 worktree 없음 (메인 1개만).
- 관련 브랜치: conceptor/steering 명칭 브랜치 없음. 동료 작업 추정 브랜치는 전부 GR00T/SAFE/finetune 계열:
  `feat/groot-ttt-phase1-integration`, `origin/pdk/0526/groot_rollout`, `origin/pdk/safe-groot-n16-followups`, `origin/exp/groot-robocasa-unified-path` 등.
- 결론: COAST/conceptor 작업의 선행 코드 없음 → Task B는 신규 모듈.

## 7. CLI 자체 상태

- 버전: Claude Code 2.1.150. 모델 Opus 4.7 (1M context) (`claude-opus-4-7[1m]`). `CLAUDE_EFFORT=xhigh`.
- 시작 working directory: `/home/dongkyu/pkt_ws/temporal_vla`.
- 로드된 context 크기: 정확 측정 불가(확인 안 됨). 대략 수만 토큰(roughly 30–50K) 수준 추정, 1M 한도 대비 여유 큼.

## 8. 막힌 곳 / 우려사항 (Task A·B 착수 전)

### Task A — RoboMD(2412.02818) & PPGuide(2603.10980) 정독 + md 요약
- ✅ 네트워크 접근 수단 존재: `requests` 2.32.3, `urllib` 사용 가능. (WebFetch/WebSearch 툴도 사용 가능.)
- ⚠️ PDF→텍스트 변환 수단 없음: `pdftotext` 미설치 + base에 PDF 파이썬 라이브러리 없음. → arxiv abs/ar5iv HTML을 fetch하거나 WebFetch 툴로 본문 추출하는 방식 권장(로컬 PDF 파싱 의존 회피).
- ⚠️ arxiv ID 확인 필요: `2603.10980` = 2026년 3월 등록 의미. 오늘(2026-05-26) 기준 가능하나 매우 최신이라 ID 오타 여부 재확인 권장. `2412.02818`(2024-12)은 정상.
- ℹ️ 두 논문 로컬 사본 없음 → 전적으로 네트워크 의존.

### Task B — COAST Appendix A.9 pseudocode → Python 구현 + pytest
- 🚫 COAST 논문/pseudocode 원문이 repo·로컬에 전혀 없음. → 구현 시작하려면 사용자가 COAST PDF(또는 Appendix A.9 텍스트/arxiv ID)를 제공해야 함. 현재 최대 블로커.
- 🚫 pytest가 어느 env에도 미설치 (base·hyundai_aigs 모두). → `pip install pytest` 필요. (numpy/scipy/torch 는 hyundai_aigs에 이미 있음.)
- ⚠️ 활성 env(base, py3.13)에는 numpy/scipy/torch 없음. 구현·테스트는 `conda activate hyundai_aigs` (py3.10) 에서 수행해야 함. 단 거기에도 pytest 설치 선행 필요.
- ❓ 신규 모듈 디렉토리 미결정 — 사용자 입력 필요: conceptor/steering 코드를 `src/conceptor/`(신규) vs `src/ttt/`(기존 TTA 모듈 옆) vs `scripts/safe/...` 어디에 둘지. 테스트 위치도 `tests/`(기존) 컨벤션 따를지 확인.

### 착수 전 사용자에게 받아야 할 결정
1. COAST 논문 소스 (PDF/arxiv ID/A.9 텍스트) — Task B 필수.
2. `2603.10980` ID 정확성 재확인.
3. 신규 conceptor 모듈/테스트 디렉토리 위치.
4. pytest 설치 + 작업 env: hyundai_aigs에 설치해도 되는지.
