# GR00T RoboCasa Loop 분석 실행 계획

> Legacy note: 이 문서는 초기 N1.6 local rollout / loop analysis 계획이다. 현재 권장 N1.6 RoboCasa 평가는 `docs/groot_robocasa_eval_setup.md`의 Docker ZMQ workflow를 기준으로 한다. 체크포인트 위치는 현재 repo 규칙에 맞춰 `/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B`를 사용한다.

## 목표
GR00T N1.6-3B 모델을 RoboCasa 환경에서 rollout하여 실패 케이스를 영상으로 확인하고,
VLA 모델의 **실패 루프(loop) 발생 조건**을 탐색한다.

## 아키텍처 (B 방식: groot 컨테이너 단독 실행)

```
[groot 컨테이너]
├── GR00T N1.6-3B 모델 (GPU, cuda:0)
├── 포크 robocasa + robosuite (환경)
├── GrootRoboCasaEnv (gymnasium 래퍼)
└── rollout_policy.py → 모델 + 환경 한 프로세스에서 실행
```

robocasa 컨테이너는 사용하지 않음 (기존 다른 모델 평가용으로만 유지).

---

## 진행 상황

### [완료] Phase 0: 환경 셋업

| 항목 | 상태 | 비고 |
|------|------|------|
| groot 컨테이너 기동 | ✅ | docker-compose up |
| Isaac-GR00T 패키지 설치 (groot) | ✅ | `pip install -e .../Isaac-GR00T --no-deps` |
| mujoco 설치 (groot) | ✅ | `mujoco==3.2.6` (포크 robocasa 요구) |
| robosuite 설치 (groot) | ✅ | `ARISE-Initiative/robosuite@master` (v1.5.2) |
| 포크 robocasa 설치 (groot) | ✅ | `external_dependencies/robocasa` (v0.2.0) |
| numpy 버전 | ✅ | `numpy==1.26.4` (gr00t 요구) |
| lxml 설치 | ✅ | robocasa 의존 |
| GR00T 모델 다운로드 | ✅ | `nvidia/GR00T-N1.6-3B` → `/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B` |
| GR00T 모델 로딩 테스트 | ✅ | `Gr00tPolicy` 정상 로딩, modality configs OK |
| Kitchen 에셋 다운로드 | ✅ | HF datasets 캐시로 다운로드 완료 |
| HF_HOME 경로 해결 | ✅ | docker-compose.yml에 `/D/temporal_vla_data` 볼륨 마운트 추가 |
| HF_MODULES_CACHE | ✅ | 외장 드라이브 chmod 문제 → `/tmp/hf_modules`로 우회 |

### 핵심 해결된 문제들

1. **HF_HOME 경로**: `/temporal_vla/data` → `/D/temporal_vla_data` 심링크인데 외장 드라이브 미마운트.
   - **해결**: `HF_HOME=/tmp/hf_cache` 사용 (컨테이너 재시작 시 캐시 사라짐, 재다운로드 필요)
   - **영구 해결**: docker-compose.yml에서 `HF_HOME` 변경하거나 `/temporal_vla/data` 심링크 수정

2. **transformers 모델 인식**: `AutoModel.from_pretrained`이 `Gr00tN1d6` 타입 못 찾음
   - **원인**: HF_HOME 경로 에러로 config.json 다운로드 실패 → 모델 타입 매칭 안 됨
   - **해결**: HF_HOME 수정 후 정상 동작. `import gr00t.model`이 auto-registration 트리거.

3. **포크 robocasa 호환성**: 우리 robocasa (`src/benchmarks/robocasa`)에는 `gym_utils`, `models.robots` 모듈 없음
   - **해결**: groot 컨테이너에 포크 robocasa (`external_dependencies/robocasa`) 직접 설치

---

### [완료] Phase 1: 통신 테스트 → 단일 rollout

**결과: OpenDrawer 5 에피소드 → 4/5 성공 (success rate 0.8)**

```bash
# groot 컨테이너에서 직접 실행 (로컬 모드, model_path는 로컬 체크포인트 경로 사용)
docker exec -e HF_MODULES_CACHE=/tmp/hf_modules -e MUJOCO_GL=egl groot bash -c "
python /temporal_vla/src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py \
    --model_path /temporal_vla/outputs/checkpoints/GR00T-N1.6-3B \
    --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n_episodes 5 \
    --n_envs 1 \
    --n_action_steps 8 \
    --max_episode_steps 720
"
```

> **주의**: `--model_path`에 HF hub ID (`nvidia/GR00T-N1.6-3B`) 사용 시 `split('/')[-3]` 에러 발생.
> 반드시 로컬 경로 (`/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B`) 사용.
> 영상은 `/tmp/sim_eval_videos_*` 에 자동 저장됨.

### [진행중] Phase 2: 배치 평가

태스크 이름 확인 결과: **수정 불필요**. `PnPCounterToCab` 등이 실제 robocasa 등록 이름과 일치.

eval_groot_robocasa.sh에 `local` / `local-batch` 모드 추가 (ZMQ 서버 없이 로컬 실행).

```bash
# 로컬 배치 평가 (groot 컨테이너 단독)
docker exec -e HF_MODULES_CACHE=/tmp/hf_modules -e MUJOCO_GL=egl groot \
    bash /temporal_vla/scripts/eval/groot_robocasa.sh local-batch 20
```

> 로그: `outputs/groot_eval_batch_*.log`

### [준비완료] Phase 3: Loop 패턴 분석

**스크립트 작성 완료**, Phase 2 배치 평가 완료 후 실행.

1. **Trajectory 수집**: `scripts/analysis/collect_groot_trajectories.py`
   - 매 스텝 action/state를 `.npz`로 저장
   - 동시에 비디오도 저장

2. **Loop 분석**: `scripts/analysis/analyze_loop_patterns.py`
   - action cosine similarity 연속 구간 탐지 (threshold=0.98, min_run=10)
   - state 변화량 정체 구간 탐지
   - gripper open/close 반복 패턴
   - near-zero action (frozen) 탐지
   - 자동 분류: stuck_loop, action_repetition, state_stagnation, gripper_oscillation, frozen, other_failure

3. **배치 스크립트**: `scripts/analysis/collect_and_analyze_groot.sh`
   - 수집 + 분석 일괄 실행

```bash
# Phase 2 완료 후 실행
docker exec -e HF_MODULES_CACHE=/tmp/hf_modules -e MUJOCO_GL=egl groot \
    bash /temporal_vla/scripts/analysis/collect_and_analyze_groot.sh 20
```

> 결과: `outputs/trajectories/` (npz), `outputs/analysis/` (json)

---

## 환경 변수 (groot 컨테이너)

```bash
export MUJOCO_GL=egl          # headless rendering
export HF_HOME=/tmp/hf_cache  # HuggingFace 캐시 (외장 드라이브 미마운트 대응)
export PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T  # docker-compose에 이미 설정됨
```

## 패키지 버전 (groot 컨테이너)

| 패키지 | 버전 | 비고 |
|--------|------|------|
| Python | 3.10.12 | |
| torch | 2.7.1 (CUDA 12.6) | |
| transformers | 4.51.3 | 모델 학습 시 사용된 버전과 동일 |
| mujoco | 3.2.6 | 포크 robocasa 요구 (robosuite는 >=3.3 원하지만 동작함) |
| robosuite | 1.5.2 | ARISE-Initiative/robosuite@master |
| robocasa | 0.2.0 | squarefk/robocasa 포크 |
| gymnasium | 1.2.2 | setup_RoboCasa.sh는 0.29.1 사용하지만, 먼저 1.2.2로 시도 |
| numpy | 1.26.4 | gr00t 요구 |
| gr00t | 0.1.0 | Isaac-GR00T editable install |

## 핵심 파일

| 파일 | 용도 |
|------|------|
| `src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py` | rollout 실행 (모델+환경) |
| `src/policies/Isaac-GR00T/gr00t/eval/run_gr00t_server.py` | ZMQ 서버 (필요시) |
| `src/policies/Isaac-GR00T/gr00t/policy/gr00t_policy.py` | Gr00tPolicy 클래스 |
| `src/policies/Isaac-GR00T/external_dependencies/robocasa/` | 포크 robocasa (환경) |
| `scripts/eval/groot_robocasa.sh` | 평가 스크립트 (local/local-batch 모드 추가) |
| `scripts/analysis/collect_groot_trajectories.py` | trajectory 수집 (action/state .npz 저장) |
| `scripts/analysis/analyze_loop_patterns.py` | Loop 패턴 분석 (cosine sim, stagnation, gripper) |
| `scripts/analysis/collect_and_analyze_groot.sh` | 수집 + 분석 배치 스크립트 |
| `outputs/checkpoints/GR00T-N1.6-3B/` | 모델 체크포인트 (로컬) |
| `docs/plan_groot_loop_analysis.md` | 이 문서 |

## 주의사항

- **HF_HOME**: docker-compose.yml에 `/D/temporal_vla_data` 볼륨 마운트 추가하여 해결. 외장 드라이브 마운트 필수.
- **HF_MODULES_CACHE**: 외장 드라이브에서 chmod 불가 → `/tmp/hf_modules`로 우회. 모든 실행 시 `HF_MODULES_CACHE=/tmp/hf_modules` 설정 필요.
- **model_path**: HF hub ID (`nvidia/GR00T-N1.6-3B`) 사용 시 rollout_policy.py에서 `split('/')[-3]` IndexError. 로컬 경로 `/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B` 사용.
- **컨테이너 재시작 시**: robocasa, robosuite, mujoco, lxml 재설치 필요 (Dockerfile에 미포함). 체크포인트는 영구 볼륨에 저장되어 유지됨.
- gymnasium 1.2.2와 rollout_policy.py 호환 확인됨 (동작 OK).
- mujoco 3.2.6 / robosuite 1.5.2 버전 충돌 경고 있으나 동작 중.
- `eval_groot_robocasa.sh` 태스크 이름은 수정 불필요 (실제 등록 이름과 일치 확인).
