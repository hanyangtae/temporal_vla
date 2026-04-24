---
name: VLA 체크포인트 온보딩 진행 상태 (2026-04-24 시점)
description: 프로파일 기반 serve 리팩터 작업의 현재 위치, 커밋 이력, 각 serve 별 완료/보류 상태, 다음 할 일. dreamvla 작업 중단 지점 재개용.
type: project
---

# VLA 체크포인트 온보딩 — 진행 상태 (2026-04-24)

## 한 줄 요약
`feat/vla-checkpoint-manager` 브랜치 위에 프로파일 시스템 + 에이전트 + 2개 serve 리팩터(openvla_oft / xvla / lerobot) 까지 완료. **dreamvla 리팩터 시작하기 직전에서 중단**. groot / upvla 대기. RLinf CALVIN (openvla_oft × CALVIN) 는 별도로 보류.

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
| openvla_oft | moojink/...-spatial-object-goal-10 | LIBERO | ✅ | (native eval, 이번 범위 외) | — |
| xvla | 2toINF/X-VLA-Calvin-ABC_D | Calvin | ✅ | **5/5 = 100%** (18.7s) | `outputs/eval/calvin/xvla/260424022221/seq0000_result5.mp4` |
| lerobot (pi05) | checkpoints/pi05-calvin-sft | Calvin | ✅ | **0/5** (체크포인트 자체 이슈, 과거 log 도 동일) | (video 저장 안 됨) |

## 보류

### openvla_oft × CALVIN (RLinf/RLinf-OpenVLAOFT-CALVIN-SFT)
- 프로파일 작성 완료 (`configs/checkpoints/openvla_oft__rlinf_calvin_sft.yaml`, 상태 코멘트 포함)
- RLinf 체크포인트에는 OpenVLA-OFT 의 `action_head--*.pt` / `proprio_projector--*.pt` 파일이 **없음**
- config.json 에 `n_action_bins: 256` 단서 — 원본 OpenVLA 의 **token-based action** 방식일 가능성
- 현재 `scripts/serve/openvla_oft.py` 는 OFT-L1 regression 경로 전용
- 재개 시: RLinf GitHub examples 에서 CALVIN inference 코드 파악 후 serve 에 `inference_mode: token_based` vs `oft_l1_regression` 분기 추가 필요

### pi05-calvin-sft rollout success = 0%
- serve 레이어는 정상 (smoke test 통과, action shape/값 반환 OK)
- rollout 0% 는 체크포인트/설정 문제. 추정 원인: state dim 불일치, norm_stats 로딩 실패, policy_type 오매칭, fallback external_config 의 image/state/action shape 오설정
- 과거 `outputs/calvin_pi05_eval.log` 도 10 seq × 0%. 별개 디버깅 이슈로 분리.

## 남은 작업 (C1 나머지)

우선순위 순:

### 1. dreamvla × Calvin  ← **다음 재개 지점**
- 파일: `scripts/serve/dreamvla.py` (322 lines, 이미 구조 파악 완료)
- 체크포인트: `/temporal_vla/checkpoints/dreamvla/dreamvla_dynamic_depth_semantic-001.pth` (로컬)
- vit_checkpoint: `/temporal_vla/checkpoints/mae_pretrain_vit_base.pth`
- docker-compose 확인됨: PYTHONPATH 에 `/temporal_vla/scripts/utils` **없음** → 추가 필요
- HF_HOME 은 NTFS 경로(`/temporal_vla/data/huggingface`) → ext4 로 이동 여부는 HF 다운로드 여부에 따라 결정 (로컬 ckpt 만 쓰면 당장 문제 없지만 일관성 위해 바꾸는 게 나음)
- dreamvla 이미지 아직 빌드 안 됨

**할 일**:
1. `scripts/serve/dreamvla.py` 리팩터 (--profile 필수화, 모든 하이퍼파라미터를 `model_specific` 으로, `_profile` 전역 추가, load_model try/except 패턴)
2. `configs/checkpoints/dreamvla__calvin_dynamic_depth_semantic.yaml` 작성 — action_type relative, eef_pos(3)+eef_euler(3)+gripper(1), `observation_requirements.state: [eef_pos, eef_euler, gripper_action]`, `observation.state.gripper_qpos` fallback 로 gripper 유추, `model_specific`: vit_checkpoint, precision=fp32, sequence_length=10, action_pred_steps=3, num_resampler_query=16, num_obs_token_per_image=9, image_size=224, patch_size=16, transformer_layers=24, hidden_dim=1024, transformer_heads=16, obs_pred=true, depth_pred=true, sam_feat_pred=true, use_dit_head=true, pred_num=1, attn_implementation=sdpa, dit_type=DiT-B, phase=evaluate, history_len=10, atten_goal=0
3. `docker-compose.yml` dreamvla 서비스 수정 — PYTHONPATH 에 `scripts/utils` 추가, HF_HOME → ext4, HF 토큰 env 추가
4. dreamvla 컨테이너 빌드 (`docker compose build dreamvla` — 첫 빌드 10~20분 예상)
5. serve 기동 + smoke test + Calvin rollout 1 seq

### 2. groot × RoboCasa
- 체크포인트: `/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B` (로컬)
- eval: `scripts/eval/groot_robocasa.sh`
- 특이: GR00T 는 embodiment_tag 기반, action 이 dict 로 반환 (native `action.base_motion`, `action.end_effector_position` 등)
- 프로파일 action_layout 표현이 다른 serve 들과 다름 — dict 출력을 그대로 emit 하는 구조라 `action_layout` 을 어떻게 표현할지 검토 필요
- embodiment_tag 는 `model_specific.embodiment_tag` 로
- 벤치마크: RoboCasa (Calvin 아님) — `src/processor/action/robocasa.py` 와 계약 확인

### 3. upvla × Calvin
- `scripts/serve/upvla.py` — Calvin spec(7D relative) 에 맞춤. 체크포인트 미확인.
- HF 에서 체크포인트 받아야 할 수도. 가장 후순위.

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
4. dreamvla 컨테이너 이미지 여부: `docker images | grep dreamvla`
5. **dreamvla 리팩터부터 재개**
