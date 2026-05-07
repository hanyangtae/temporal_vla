---
name: VLA 체크포인트 온보딩 진행 상태 (2026-04-24 시점)
description: 프로파일 기반 serve 리팩터 작업의 현재 위치, 커밋 이력, 각 serve 별 완료/보류 상태, 다음 할 일. dreamvla 작업 중단 지점 재개용.
type: project
---

# VLA 체크포인트 온보딩 — 진행 상태 (2026-04-24)

## 한 줄 요약
`feat/vla-checkpoint-manager` 브랜치 위에 프로파일 시스템 + 에이전트 + **5개 serve 리팩터** (openvla_oft / xvla / lerobot / dreamvla / groot) + openvla_oft EGL Docker fix + RoboCasaActionProcessor 확장 까지 완료. upvla 는 skip. RLinf CALVIN (openvla_oft × CALVIN) 는 별도로 보류. groot 정확도 (axisangle↔env 입력) 추가 검증은 별건 작업.

## 브랜치 & 커밋 (feat/vla-checkpoint-manager)

dev 에서 분기. push 안 함. 시간순:

1. `be1c776` feat: openvla-oft 서브모듈 및 컨테이너 Docker 셋업
2. `3a51c0c` script: RoboCasa 태스크 목록 확인 분석 스크립트 추가
3. `dae8ed6` feat: VLA 체크포인트 프로파일 시스템 추가
4. `4643d84` refactor: openvla_oft serve 프로파일 기반 전환 및 docker-compose 환경 정비
5. `127d181` feat: .claude 에이전트 정의/메모리 git 추적 시작 (`.gitignore` 에서 `.claude/agents/`, `.claude/agent-memory/` negate)
6. `6aed3dd` feat: 체크포인트 프로파일 LoRA adapter 지원 추가
7. `bf6d2b2` refactor: xvla serve 를 프로파일 기반으로 전환
8. `efdcb1c` fix: Calvin Dockerfile pip / cmake 버전 호환성 수정
9. `5f54660` refactor: lerobot serve 를 프로파일 기반으로 전환

## 완료 & 검증

| serve | 체크포인트 | 벤치 | smoke | rollout | video path |
|---|---|---|---|---|---|
| openvla_oft | moojink/...-spatial-object-goal-10 | LIBERO | ✅ | **10/10 = 100%** native eval (libero_spatial 전 task) | `src/policies/openvla-oft/rollouts/2026_04_26/*.mp4` |
| xvla | 2toINF/X-VLA-Calvin-ABC_D | Calvin | ✅ | **5/5 = 100%** (18.7s) | `outputs/eval/calvin/xvla/260424022221/seq0000_result5.mp4` |
| lerobot (pi05) | checkpoints/pi05-calvin-sft | Calvin | ✅ | **0/5** (체크포인트 자체 이슈, 과거 log 도 동일) | (video 저장 안 됨) |
| dreamvla | checkpoints/dreamvla/...dynamic_depth_semantic-001.pth | Calvin | ✅ | **1/5** (1 seq, Avg seq length 1.000) | `outputs/eval/calvin/dreamvla/260506101245/seq0000_result1.mp4` |
| groot | nvidia/GR00T-N1.6-3B (ROBOCASA_PANDA_OMRON) | RoboCasa | ✅ | 통신/변환 경로 ✅. 정확도 미해결 — 별건 이슈 분리 (`issue_groot_robocasa_unified_path.md`). native eval 은 **CloseDrawer 1/1 = 100%** 검증 완료. | `outputs/eval/robocasa/groot/260506144254/CloseFridge.mp4` 외 |

## 보류

### openvla_oft × CALVIN (RLinf/RLinf-OpenVLAOFT-CALVIN-SFT)
- 프로파일 작성 완료 (`configs/checkpoints/openvla_oft__rlinf_calvin_sft.yaml`, 상태 코멘트 포함)
- RLinf 체크포인트에는 OpenVLA-OFT 의 `action_head--*.pt` / `proprio_projector--*.pt` 파일이 **없음**
- config.json 에 `n_action_bins: 256` 단서 — 원본 OpenVLA 의 **token-based action** 방식일 가능성
- 현재 `scripts/serve/openvla_oft.py` 는 OFT-L1 regression 경로 전용
- 재개 시: RLinf GitHub examples 에서 CALVIN inference 코드 파악 후 serve 에 `inference_mode: token_based` vs `oft_l1_regression` 분기 추가 필요

### pi05-calvin-sft rollout success = 0%
별건 이슈로 분리됨. 상세 디버깅 노트, 추정 원인 5종, 액션 후보:
**`.claude/agent-memory/vla-checkpoint-manager/issue_pi05_calvin_zero_success.md`** 참조.
요약: serve 레이어는 깨끗(smoke pass) → 체크포인트 + 프로파일 fallback config 매칭 문제.

## 남은 작업 / 후속 fix

### upvla × Calvin — **사용자 결정으로 SKIP**
- `scripts/serve/upvla.py` — Calvin spec(7D relative) 에 맞춤. 체크포인트 미확인.
- 사용자가 이번 온보딩 사이클에서는 skip. 추후 필요 시 재개.

### groot × RoboCasa 통일 API path 정확도 (별건 분리)
별건 이슈로 분리됨 → **`.claude/agent-memory/vla-checkpoint-manager/issue_groot_robocasa_unified_path.md`**
요약:
- native eval: CloseDrawer 1/1 = 100% (모델/체크포인트 정상)
- 통일 API path: 0% (obs/action 매핑이 native 와 다름)
- R3 commit (`c74c1c8` in robocasa submodule) 으로 GrootRoboCasaEnv 사용자 fork 에 이식 완료. 의존성 호환 (mujoco) + robosuite robot 초기화 실패 진단 미완.
- 다음 우선순위: `composite_controller None` 진단 (robocasa_models 미설치 / mink 호환 의심).

### openvla_oft × CALVIN (RLinf/RLinf-OpenVLAOFT-CALVIN-SFT) — 보류
- 프로파일 작성 완료 (`configs/checkpoints/openvla_oft__rlinf_calvin_sft.yaml`)
- RLinf 체크포인트는 token-based action 방식일 가능성 (OFT-L1 regression 경로 안 맞음)
- 재개 시 RLinf inference 코드 파악 후 serve 에 inference_mode 분기 추가 필요

### pi05 × Calvin 0% rollout
- 별건 이슈 → `.claude/agent-memory/vla-checkpoint-manager/issue_pi05_calvin_zero_success.md`

## 기술 포인트 (재개 시 참고)

### 프로파일 스키마 (이미 확립)
- 정의: `scripts/utils/checkpoint_profile.py` — `CheckpointProfile` dataclass + `load_profile`
- 필드: name, base_model, checkpoint_source{type, id}, compatible_benchmarks, action_type, action_layout[{name, dims}], rotation_encoding, gripper_encoding{range, binarize, sign_flip, threshold}, normalization{scheme, stats_file, key_selection}, observation_requirements{images, state, allow_conversions}, n_action_steps, image_preprocess{resolution, rotate_180, center_crop}, emits_subkeys, lora?{subfolder}, model_specific: Dict[str, Any]
- 문서: `configs/checkpoints/README.md`
- dry-run: `python scripts/utils/checkpoint_profile.py <yaml>`

### serve 리팩터 공통 패턴
1. `from checkpoint_profile import CheckpointProfile, load_profile`
2. `_profile: Optional[CheckpointProfile] = None` 전역
3. `--profile` 필수 CLI 인자
4. `load_model` 을 try/except 로 감싸고 stderr 로 traceback 강제 출력 (uvicorn startup hook 이 exception 삼킴)
5. `/health` 응답이 프로파일의 `action_type`, `emits_subkeys`, `n_action_steps` 그대로 반영
6. `/act` sub-key emit 은 `action_layout` 의 dim_slice 로 일반화

### docker-compose 설정 공통 주의
- HF_HOME 은 NTFS(/D) 에서 chmod 미지원 → ext4 로 이동:
  ```yaml
  volumes:
    - .:/temporal_vla:rw
    - /D/temporal_vla_data:/D/temporal_vla_data:rw
    - ${HOME}/.cache/huggingface:/home/${USER_NAME}/.cache/huggingface:rw
  environment:
    - HF_HOME=/home/${USER_NAME}/.cache/huggingface
    - HF_TRUST_REMOTE_CODE=1
    - HF_TOKEN=${HF_TOKEN}
    - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    - PYTHONPATH=<기존 경로들>:/temporal_vla/scripts/utils
  ```
- `.env` 에는 `HF_TOKEN` 만 있음. 토큰 매핑 필수.

### Docker 특유 함정 (이번 세션에서 겪은 것)
- **uvicorn startup hook 의 exception silencing**: `@app.on_event("startup")` 안에서 발생한 exception 은 stdout 에 traceback 안 찍힘. try/except + stderr 수동 출력 필수.
- **docker compose run -d --rm** 조합: 컨테이너 exit 시 즉시 삭제되어 logs 소실. 디버깅 중에는 `--rm` 빼기.
- **PYTHONUNBUFFERED=1** 필수 (부모가 auto 감지 안 함).
- **Calvin pip/cmake 호환성** (이미 efdcb1c 로 fix): `pip<24.1`, `cmake==3.18.4.post1`.
- **lerobot torch ABI 불일치** (이미 5f54660 로 fix): pi05 는 transformers fork(fix/lerobot_openpi) 필요, tokenizers<0.22, requirements 설치 중 torch 2.5.1→2.10 업그레이드로 flash-attn ABI 깨짐 → flash-attn 제거 (eager attention fallback).
- **mujoco/robosuite EGL 초기화 실패** (openvla_oft LIBERO native eval 진행 중 만남, 이번 세션 fix): PyOpenGL 이 `libEGL.so.1` 을 dlopen 못 해 `_p.PLATFORM.EGL = None`. 해결:
  1. Dockerfile 에 `libegl1 libglvnd0 libglx0 libgles2` 추가
  2. docker-compose env 에 `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=all`, `MUJOCO_GL=egl`
  3. 이 조합으로 NVIDIA EGL ICD 가 컨테이너에 mount 됨. Default `compute,utility` 만으론 graphics lib 못 들어옴.
  cf. 호스트 노트: `docs/docker로 ai2thor cloudrendering(headless)돌릴때 생긴문제(v 27a63918d42a803ea893cf610b8a6c7c.md`
- **LIBERO native eval 추가 함정**:
  - `pip install -e LIBERO` 가 user-site 에 등록 → `PYTHONPATH=/temporal_vla/src/benchmarks/LIBERO` 로 직접 보강 필요
  - LIBERO `__init__.py` 가 첫 import 때 `input(...)` 으로 dataset 경로 묻는 이슈 → `~/.libero/config.yaml` 사전 작성 필요
  - 두 fix 모두 `scripts/eval/openvla_oft_libero.sh` 에 반영됨 (이번 세션 modify)

### 에이전트 인프라
- 정의: `.claude/agents/vla-checkpoint-manager.md`
- 메모리 디렉토리: `.claude/agent-memory/vla-checkpoint-manager/` (이 파일 위치)
- 가이드: `docs/adding_checkpoint.md` (한글 체크리스트)
- CLAUDE.md 에 "새 체크포인트 추가" 절 포함

## 재개 체크리스트

재개 세션에서 처음 할 일:
1. 현재 브랜치 확인: `git branch --show-current` — `feat/vla-checkpoint-manager` 여야 함
2. 이 메모리 파일 읽기 (MEMORY.md 에 인덱스 있음)
3. `git log --oneline -10` 으로 커밋 상태 재확인
4. groot 체크포인트 확인: `ls /temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B/`
5. **groot × RoboCasa 부터 재개**

## 미커밋 변경사항 (이번 세션, 별도 commitor 처리 예정)

**Submodule commit (`src/benchmarks/robocasa`):**
- `c74c1c8` feat: GR00T 호환 RoboCasa env wrapper 추가 (R3 — KeyConverter, GrootRoboCasaEnv, RoboCasaEnv)

**Main repo:**
- `M docker-compose.yml` — openvla_oft (EGL caps + MUJOCO_GL=egl), dreamvla (HF ext4 + scripts/utils + GPU deploy + HF_TOKEN), groot (HF ext4 + scripts/utils + HF_TOKEN)
- `M docker/openvla_oft/Dockerfile` — libegl1 libglvnd0 libglx0 libgles2
- `M docker/groot/Dockerfile` — fastapi / uvicorn / opencv-python-headless / pyyaml
- `M scripts/eval/openvla_oft_libero.sh` — PYTHONPATH 보강, ~/.libero/config.yaml 사전 작성, MUJOCO_GL/PYOPENGL_PLATFORM
- `M scripts/eval/robocasa_eval.py` — `--use-groot-env` flag + `run_vla_rollouts_groot` 함수 + GR00T schema rename 매핑 + action dump (env VLA_ACTION_DUMP)
- `M scripts/serve/dreamvla.py` — 프로파일 기반 풀 리팩터
- `M scripts/serve/groot.py` — 프로파일 기반 리팩터, native dict → 통일 sub-key 매핑, 누락 video/state 키 zero fallback (state dim 은 statistics.json 에서 로드)
- `M src/processor/action/robocasa.py` — eef_axisangle / base_motion / torso 옵션 키 처리, missing 키 1회 warning
- `M src/processor/obs/robocasa.py` — 3-camera 모드 키 정정 (static + left + right + wrist)
- `M src/benchmarks/robocasa` — submodule pointer → c74c1c8 (R3)
- `?? configs/checkpoints/dreamvla__calvin_dynamic_depth_semantic.yaml`
- `?? configs/checkpoints/groot__robocasa_panda_omron.yaml`
- `?? .claude/agent-memory/vla-checkpoint-manager/issue_pi05_calvin_zero_success.md`
- `?? .claude/agent-memory/vla-checkpoint-manager/issue_groot_robocasa_unified_path.md`

## 컨테이너 함정 (이번 세션 추가)
- **robocasa Dockerfile entrypoint = `start_vnc.sh`**: `docker compose run --rm robocasa <cmd>` 시 entrypoint 가 cmd 무시하고 KasmVNC 만 띄운 후 `tail -f /dev/null` 로 hang. 우회: `docker compose run --rm --entrypoint "" robocasa <cmd>`.
- **GR00T modality_keys 의 prefix 누락**: `_modality_configs["video"].modality_keys` 가 `'res256_image_side_0'` 형태 (prefix 없음) 인데 GR00T 검증은 `'video.res256_image_side_0'` (prefix 있음) 으로 함. fallback 시 prefix 자동 추가 필요.
- **GR00T state dim 정보 출처**: `_modality_configs` 에는 dim 없음. `<checkpoint>/statistics.json` → `embodiment_value/state/<key>/mean` 에서 추출.
- **`docker system prune -af` 주의**: stopped 컨테이너의 모든 unused 이미지 삭제. dreamvla / xvla / openvla_oft 등 active container 없는 이미지 다 날아감.
