# 새 VLA 체크포인트 추가 체크리스트

`vla-checkpoint-manager` 에이전트 워크플로우의 한글 요약. 사람이 직접 할 때도 이 순서 그대로.

## 0. 사전 확인

- **대상 벤치마크**: Calvin / RoboCasa / LIBERO 중 어디서 평가할지
- **모델 아키텍처**: 이미 지원하는 `scripts/serve/<model>.py` 가 있는지, 아니면 신규인지
- **체크포인트 위치**: HF repo 인지 로컬 경로인지 (가중치는 git 커밋 금지)

## 1. 체크포인트 아티팩트 조사

체크포인트 루트에서 다음 파일을 찾는다:

- `config.json` — 아키텍처 정보 (action dim, vision encoder, LLM base 등)
- `dataset_statistics.json` (OpenVLA 계열) 또는 `norm_stats.json` (LeRobot/pi0 계열)
- `processor_config.json` — GR00T 계열 modality config
- `*.safetensors` / `*.bin` — 모델 최종 레이어에서 action tensor shape 확인

HF repo 이면 모델 카드 텍스트 + local cache (`data/huggingface/hub/models--<org>--<name>/`) 동시 확인.

## 2. 프로파일 7항목 확정

`configs/checkpoints/README.md` 의 스키마 참고. 각 필드 값을 근거와 함께 한 줄씩 정리:

| 항목 | 확인 포인트 |
|---|---|
| `action_type` | 출력이 delta(상태 대비 변화량)인지 absolute 좌표인지. 학습 loss/label 식에서 확인. |
| `action_layout` | 출력 벡터의 차원별 의미 (pos/rot/gripper 순서). 논문 또는 코드의 action encoder 주석. |
| `rotation_encoding` | euler / quat (xyzw·wxyz) / rot6d / axisangle 중 하나. 학습 데이터 전처리 코드 확인. |
| `gripper_encoding` | 값 범위([-1,1] / [0,1] / binary), 이진화 여부, sign flip 유무. 보통 가장 함정이 많음. |
| `normalization` | stats 파일 경로 + 선택할 key fallback chain. OpenVLA-OFT 는 `unnorm_key` 개념. |
| `observation_requirements` | 모델이 forward 시 요구하는 state sub-key (eef_quat vs eef_euler 등). |
| `n_action_steps` / `image_preprocess` | 반환 action 개수, 이미지 resolution / rotate_180 / center_crop. |

### 주의 포인트

- **gripper sign_flip**: LIBERO 는 필요, CALVIN 은 보통 불필요. 학습 데이터 gripper 값 분포 확인.
- **image rotate_180**: LIBERO bddl 환경은 업스트림에서 회전됨. 벤치마크마다 다름.
- **`unnorm_key` fallback**: 체크포인트가 multi-task 학습이면 정확한 key 를 첫 번째에 넣어야 한다. 잘못된 key 로 denormalize 하면 action 이 조용히 0 근처가 되거나 발산.

## 3. 프로파일 YAML 작성

`configs/checkpoints/<base_model>__<variant>.yaml` 생성. 파일명 stem 은 프로파일의 `name` 필드와 동일해야 함.

검증:

```bash
python scripts/utils/checkpoint_profile.py configs/checkpoints/<name>.yaml
```

## 4. serve 스크립트 수정

- **같은 아키텍처 지원**: 기존 `scripts/serve/<model>.py` 에 프로파일 기반 분기 추가. 단일 파일에서 여러 체크포인트 지원.
- **새 아키텍처**: `scripts/serve/<model>.py` 신규 작성 (참고: `scripts/serve/lerobot.py` 가 외부 체크포인트 norm_stats 로딩 패턴을 잘 보여줌) + 필요 시 `docker-compose.yml` 서비스 + `docker/<model>/Dockerfile` 신규.

`/health` 응답은 프로파일의 `action_type`, `emits_subkeys`, `n_action_steps` 를 그대로 반영할 것.

## 5. 벤치마크 쪽 계약 확인

`src/processor/action/<bench>.py` 가 소비하는 sub-key 조합에 프로파일의 `emits_subkeys` 가 부합하는지 확인.

eval 스크립트의 `make_*_processors(action_type=..., gripper_threshold=...)` 호출을 프로파일과 일치시킨다.

## 6. Smoke test

1. 올바른 컨테이너에서 serve 기동:
   ```bash
   docker compose exec <container> python /temporal_vla/scripts/serve/<model>.py \
       --profile /temporal_vla/configs/checkpoints/<name>.yaml
   ```
2. `curl :<port>/health` → JSON 이 프로파일과 일치하는지
3. 해당 벤치 eval 1 episode 실행
4. 초반 5~10 step action 이 상식적인 범위인지 (position delta 가 갑자기 수십 수백이면 normalization 오류)

### 실패 시 진단 순서

1. **Sub-key 매핑** — 서버가 반환한 키와 processor 가 기대한 키 일치 여부
2. **Normalization key** — `key_selection` 의 첫 번째가 실제 stats 파일에 존재하는지
3. **Gripper sign / threshold** — sign_flip 과 gripper_threshold 조합
4. **Rotation 인코딩** — euler/quat/rot6d 변환 경로
5. **Image preprocess** — rotate_180, resolution, center_crop

## 7. 문서화

- `configs/checkpoints/<name>.yaml` 은 git 커밋
- 가중치는 `.gitignore` 유지 (커밋 금지)
- 비직관적 발견(예: "이 체크포인트는 모델 카드와 달리 sign_flip 불필요") 은 agent memory 의 `project` 타입으로 기록
