# Detector–Steering 입출력 계약 초안

시간 기준은 **inference step**이며 현재 step의 action을 환경에서 실행하기 전에 판정해야 한다.
모델 종류나 activation 선택은 유연하게 두고 이 실행 순서를 고정한다.

## 입력 범위

입력은 inference step마다 **`[layer, denoise_step, token, feature]`** 형태로 제공한다.
Detector는 아래 범위 안에서 layer·denoise step·token을 선택하거나 pooling할 수 있다.
시간 이력을 쓸 경우 여러 inference step을 별도 history 축으로 묶는다. denoise 축이 rollout 시간 축은 아니다.

대상은 GR00T N1.5 RoboCasa365 `checkpoint-120000`의 **DiT block 출력 residual stream**이다.
아래 최대 범위는 이 checkpoint와 현재 4-step 추론 설정에서의 DiT 캡처 범위이며,
VLM의 모든 내부 activation까지 포함하는 모델 전체의 최대 범위라는 뜻은 아니다.

| 축 | 기존 저장 설정 및 확인 표본 | 현재 코드로 캡처 가능한 최대 범위 | 의미 |
|---|---|---|---|
| layer | 물리 ID `[0,2,4,8,10,12,15]`, 7개 | 물리 ID `0–15`, 총 16개 block | 저장 배열의 index `0–6`과 물리 layer ID를 구분한다. 예: 배열 index 5가 L12. |
| denoise_step | `0,1,2,3`, 4개 모두 저장 | 현재 추론 설정의 4개 step 모두 | 호출 순서 기준. flow time은 `0,0.25,0.5,0.75`; 시간 bucket은 `0,250,500,750`. 마지막 slot은 3이며 적분 종료 t=1을 별도로 저장한 슬롯은 없다. |
| token | `0–48`, 전체 49개 (`all_token_full`) | 동일하게 49개 모두 | `0`: state 1개, `1–32`: learned future token 32개, `33–48`: action token 16개. |
| feature | `0–1535`, 1536차원 | 동일하게 1536차원 모두 | DiT block residual의 차원. action decoder 출력 차원이나 실제 action 차원과 다르다. |

- **기존 저장본:** 표본의 1 record는 `[7,4,49,1536]`, `float16`이다.
  이 범위 안의 선택·pooling은 기존 activation으로 가능하다.
- **전체 DiT layer 캡처:** `[16,4,49,1536]`을 코드상 지원한다. capture layer 목록을
  `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`로 명시해야 한다.
  이번 확인에서는 전체 layer 캡처를 새로 실행하지 않았으며, 기존 저장본에 누락된 layer가 생기는 것은 아니다.
  누락 layer가 필요하면 별도의 캡처가 필요하고, 재실행 데이터를 과거 rollout과 자동으로 동일시하지 않는다.
- `all_token_full`은 state/future/action token 전체를 보존한다. learned future token은
  모델의 학습된 embedding이며 실제 미래 관측이나 과거 영상 프레임이 아니다.
- action token 16개는 예측 horizon이다. 확인 표본은 그중 5 action을 환경에 실행했다.
  이를 token 수 5개 또는 denoise step 5개로 해석하지 않는다.
- 기존 모든 데이터의 공통 범위를 전수 확인한 것은 아니다. 전달 manifest에 포함할 각 artifact의
  `capture_layers`, `record_shape`, dtype 및 checkpoint를 확인하고 이 표와 다른 파일을 구분해야 한다.
- denoise 횟수를 바꾸는 것은 저장 슬롯 선택이 아니라 추론 설정 변경이다. 동일 조건 실험으로 취급하지 않는다.

### 추가 입력: VL 및 과거 activation

확인 표본에는 `vl_hidden_states`도 record당 `[2048]`, `float16`으로 저장돼 있다.
이는 `vlln_mean` 위치의 sequence mean이며 DiT의 `[L,K,T,D]` 배열과 별도 입력이다.
이 pooled 벡터로 원래 VL token들을 복원할 수 없다. 코드에는 `post_vl_sa_full` 캡처 옵션도 있지만
전체 VL token 수·저장 가용성은 이번 표본으로 확정하지 않았으므로 제공 확정 범위에 포함하지 않는다.

과거 activation history는 같은 episode의 현재 및 이전 record만 사용한다.
history 길이는 detector가 선언하며 episode 경계에서 reset한다.
미래 activation, 아직 실행하지 않은 action의 결과, 최종 성공 라벨은 온라인 입력으로 사용하지 않는다.

### 확인 근거 (2026-09-18)

- Checkpoint config: `robocasa/robocasa365_checkpoints`,
  `gr00t_n1-5/multitask_learning/checkpoint-120000/config.json`.
  로컬 HF snapshot `14895998fe7c8f8f2441cc8957ec2c510302758b` 및
  `c484448aba1a9b60a04c9b0ca117241518ea69f3`의 action-head 설정이 위 축 값에 대해 일치한다.
  `diffusion_model_cfg.num_layers=16`, `num_inference_timesteps=4`,
  `num_target_vision_tokens=32`, `action_horizon=16`, `input_embedding_dim=1536`.
- Token 구성·denoise 순서: `lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py`,
  `get_action()`의 `cat(state_features, future_tokens, action_features)`와 denoise loop.
- 저장 경로: `scripts/serve/safe_hooks.py`의 `SafeFeatureCapture.__enter__()` 및
  `assemble_blocks()`. block post-hook에서 `[K,B,T,D]`를 쌓아 batch 0을 선택하고
  `[L,K,T,D]` float16으로 반환한다. denoise/token 평균을 미리 하지 않는다.
- 기존 plan: `configs/collect/n15_grid_v6_scene_jitter/collection_plan.json`의
  `capture_layers`, `denoise_k=4`, `token_mode=all_token_full`.
- 실제 파일: `outputs/collect/kanu_base_recovery_20260916/grid/4fa6496cd684/kanu/PPCC/marshmallow/s3/j4/n0/base/`
  의 `meta.json`과 `rollout.pkl`. 직접 로드해 144개 hidden-state record 중 첫 tensor의
  `[7,4,49,1536]`/float16 및 첫 VL tensor `[2048]`/float16을 확인했다.
  이는 재현 수집 표본의 스키마 확인이며 과거 원본과 궤적이 같다는 증거가 아니다.

Detector artifact는 다음 input specification을 함께 제공한다.

- policy/checkpoint 및 feature schema 버전
- 물리 layer ID와 denoise 슬롯 의미, token 범위와 pooling
- history 길이·순서, 시작 구간 padding/mask 및 episode reset 규칙
- normalization/PCA/AE 등 전처리 artifact와 hash, 학습 split
- 선택 입력 필드, dtype/shape, feature가 준비되는 시점
- threshold/calibration 설정 및 사용 데이터

입력 누락이나 schema 불일치를 조용히 0 또는 no-fire로 처리하지 않는다.
구체적인 오류 시 실행 정책은 통합 단계에서 정한다.

## 최소 adapter 제안

아래는 언어·클래스 구현을 고정하지 않는 논리 API이다.

```text
reset(episode_id)
detect(episode_id, inference_step, activation_bundle) -> decision
```

`decision`의 필수 값은 동일한 episode_id, inference_step과 `fire: bool`이다.
권장 로그에는 detector/model version, 입력 artifact/spec version, score(s), threshold(s),
reason, latency가 포함된다. 서로 다른 detector의 score를 같은 확률로 간주하지 않는다.
내부적으로 2개 MLP가 success/failure score를 내더라도 환경 실행 측에는 최종 fire를 제공한다.

## 실행 순서

```text
현재 관측 → 정책 후보 activation/action 생성
          → detector의 현재 step 판정 완료
          → fire이면 steering/fallback 처리
          → 최종 action 확정 → 환경 실행
```

정책 후보 action이 이미 계산돼 있어도 환경에 실행되기 전이면 계약에 맞는다.
같은 forward 중간 layer에 개입해야 하는 steering은 이후 layer에서 얻은 detector 입력을
거슬러 사용할 수 없으므로, 중간에서 대기하거나 후보 pass 이후 재-forward하는 실행 구성이 필요하다.
입력 availability와 개입 위치를 adapter 통합 시 함께 확인한다.
비동기 구현이라도 판정 완료 전에 해당 action chunk를 환경으로 보내지 않는다.
허용 지연 상한의 숫자는 아직 합의하지 않았다. target 환경에서 p50/p95와 최대 지연을 먼저 측정한다.

## 미결: 개입 후 상태 정렬

현재 LSTM 경로는 사용자의 설명에 따르면 steered activation으로 계산한 hidden state로
다음 상태를 교체한다. 이 방식을 새로운 모든 detector의 공통 규칙으로 강제하지 않는다.
새 detector/steering 조합마다 다음을 결정해야 한다.

- detector가 개입 전 후보 activation과 실제 사용 activation 중 무엇을 기억할지
- reseed/재-forward 시 상태를 commit, rollback 또는 재계산할지
- 같은 inference step을 두 번 진행시키지 않는 방법
- 반복 발화/cooldown 및 개입 직후 판정의 처리

논리 inference_step은 환경으로 다음 action chunk를 실행할 때 진행한다.
후보 재계산은 별도 attempt ID로 기록할 수 있으며 새로운 시간 step으로 취급하지 않는다.
이 항목들은 이번 초안에서는 미결로 유지한다.

## 통합 검증과 평가 보고

- reset 후 episode 간 state 누출이 없어야 한다.
- offline 입력과 online adapter가 동일한 전처리·판정값을 내야 한다.
- detector 판정 전에 환경 step이 진행되지 않는지 실행 trace로 확인한다.
- invocation과 최종 decision을 구분하고, 최초 발화 시점과 사건 전/후 시차를 기록한다.
- 성공 rollout 전체 발화율과 그중 사건 없는 성공판의 발화율을 구분한다.
- 실패 rollout 발화율, 미발화 수, 사건 전/후 발화 비율과 검토 불가 분모를 함께 보고한다.
- 구제/파괴는 동일한 원본 좌표·재현 계약을 확인한 steering 비교에서 별도로 집계한다.

현재 코드 참고: `scripts/serve/lerobot.py`의 feature extraction, failure detector와
`features.perstep_fired` 처리. 새 adapter 연결과 모델 branch 선택은 아직 수행하지 않았다.
