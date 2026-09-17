# CosmosPolicy로 v6 형태의 activation grid를 수집할 수 있는가

작성: 2026-09-07. 성격: 환경·코드 호환성 분석 보고서. 설치/실행 runbook이나 성능 평가 결과가 아니다.

## 1. 판단

**같은 실험 구조의 CosmosPolicy rollout+activation 데이터는 수집 가능하다. 다만 현재 GR00T 수집기에 체크포인트만 바꾸는 방식은 불가능하다.** 환경 재현·격자·무결성 관리 구조는 재사용하고, 정책 IO·feature capture·일부 저장/분석 코드를 새로 연결해야 한다.

두 목표를 구분해야 한다.

| 목표 | 판단 |
|---|---|
| 같은 형태의 실험: 주방 × 세계 변형 × 정책 noise, 시점별 activation, phase, 성공 여부 | 공개 CosmosPolicy RoboCasa 체크포인트로 구현할 수 있다. 먼저 지원 과제 소규모 수집으로 검증한다. |
| 기존 v6의 12 instruction·36 scene·180개 초기 세계를 그대로 쓰는 모델 교체 비교 | 현재 RoboCasa365 환경을 유지하는 별도 이식이 필요하다. 공식 Cosmos 환경으로 바꾸면 기존 seed의 세계가 보존되지 않는다. |
| 기존 GR00T pkl을 현재 분석 코드에 넣던 그대로 Cosmos pkl도 투입 | 불가. token 의미·차원·denoise 축이 다르고 현재 분석기에 고정 상수가 있다. |
| 전체 1,800판을 지금 바로 실행 | 준비되지 않았다. 서버·profile·capture 구현이 없고 Cosmos runtime smoke도 미실시다. |

이번 조사 대상은 **`nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`와 NVlabs/cosmos-policy**다. 일반 Cosmos video checkpoint나 다른 세대 모델에 이 결론을 자동 적용하지 않는다. 공개 모델은 RoboCasa 24과제, 과제당 50개 demonstration으로 학습된 모델이다. [공식 모델 카드](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B)

## 2. 현재 데이터를 왜, 어떻게 모으는가

### 실험 목적

[v5 handoff §0.2](../collab_within_claude/history/handoff_20260902_grid_recollect_v5.md)는 “같은 작업장의 작은 변화 때문에 실패할 때, 적은 rollout과 activation으로 감지·steering을 시도”하는 시나리오를 정한다. 이 grid는 정책 실행 데이터다. Expert demonstration은 별도로 확보해야 하며, grid가 expert 데이터를 대신하지 않는다.

같은 `(instruction, scene, jitter)`는 같은 초기 세계이고 `noise`만 바꿔 정책의 확률적 실행을 얻는다. 물체 위치가 달라진 효과와 정책 noise가 달라진 효과를 분리해서 추적하려는 설계다. 성공·실패 label은 환경 판정이며 phase는 simulator state로 만든 oracle annotation이다. 이 보고서는 분리도·detector 성능·steering 효과를 주장하지 않는다.

### v5에서 v6로 바뀐 이유

- v5의 scene은 사실상 seed 목록이었다. 같은 layout에 몰릴 수 있었으므로 v6는 scene을 `(layout, style)` 주방으로 명시했다.
- ep_meta를 고정한 채 reset만 반복하면 pull task의 base까지 고정된다. 따라서 v6는 rack task의 지터를 base offset으로 정의했다.
- 저장된 ep_meta JSON을 첫 `reset(seed)` 전에 넣는 replay는 수집과 다른 세계를 만들었다. 현재 계약은 **seed reset으로 ep_meta 재획득 → plan의 지터 재적용**이다.
- 일부 초기 세계의 관측 freeze 때문에 다음 reset 후보로 교체한 이력이 있다. 실패한 policy rollout과 깨진 관측 데이터는 구별한다.

근거: [v6 설계](../collab_within_claude/handoff_20260903_grid_v6_scene_jitter.md), [09-04 운영 handoff](../collab_within_claude/handoff_20260904_grid_collection.md). 과거 handoff의 실행 상태·용량·완료 숫자는 현재 실측으로 간주하지 않았다.

### 실제 격자

현재 [collection_plan.json](../../configs/collect/n15_grid_v6_scene_jitter/collection_plan.json)의 기계 판독 결과는 plan `77e745c37b0f`, **12키 × scene 3 × jitter 5 × noise 10 = 1,800판**이다. `note`/`extra.scenario`에는 아직 `n5=900` 설명이 남아 있다. 실행 계약은 실제 배열과 셀 열거 결과를 따른다.

| 계열 | instruction 키 수 | 같은 scene의 jitter |
|---|---:|---|
| oven rack out / dishwasher rack out | 각 2: 로봇 스폰 left/right | reset index 0 고정. `(lat, back)` = `(0,0)`, `(0,.05)`, `(0,.10)`, `(.05,.10)`, `(.05,.15)` m. lat은 fixture 중심 방향. |
| OpenDrawer | 2: 대상 서랍 left/right | 해당 문장을 유지하는 reset index 목록 + back `[0,.05,.10,.05,.10]` m. 두 요인이 결합된 지터다. |
| PPCC | 5: apple/jug/candle/bread/marshmallow | ep_meta 고정 후 reset index 0–4로 물체 배치·팔 상태 변형. |
| CoffeeSetupMug | 1 | PPCC와 같은 원리. 무효 세계 교체는 scene별 plan의 실제 reset index를 따른다. |

환경 주방 후보는 `[[1,1], …, [10,10]]`, noise seed는 `1300000..1300009`. scene/jitter는 index이고 실제 seed/reset 횟수는 별도 값이다. v5의 5주방 후보 목록과 섞으면 같은 seed라도 같은 세계가 아니다.

### 실행 코드 연결

| 역할 | 실제 파일과 핵심 동작 |
|---|---|
| scene/plan 구성 | [build_v6_plan.py](../../scripts/collect/build_v6_plan.py), [scan_fixture_groups.py](../../scripts/collect/scan_fixture_groups.py): scene 선택표·reset/offset·머신 배정을 plan에 고정 |
| 격자·경로 | [src/collect/plan.py](../../src/collect/plan.py): `CollectionPlan`, `resolve_grid`, `grid_dir_for`; 계획 밖 좌표 거부 |
| 작업 분배 | [collect_grid.sh](../../scripts/safe/groot_n15/robocasa/collect/collect_grid.sh): DONE_LIST/로컬 meta에서 결손 계산, GPU별 서버 기동, 셀 수집, staging backpressure |
| 정책 서버 | [lerobot.py](../../scripts/serve/lerobot.py), [lerobot_adapters/groot.py](../../scripts/serve/lerobot_adapters/groot.py): N1.5 checkpoint 로드, HTTP 추론 |
| activation | [safe_hooks.py](../../scripts/serve/safe_hooks.py): action-head DiT block output hook, layer별·denoise별 token 보존; 별도 VL capture |
| 수집 본체 | [http_feature_collect.py](../../scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py): `make_env`, `_v6_apply_jitter`, `N15LerobotHttpFeatureClient.get_action`, `run` |
| 환경 입출력 | [gymnasium_basic.py](../../src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_basic.py), [gymnasium_groot.py](../../src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_groot.py), [io.py](../../src/policies/groot/robocasa/io.py) |
| phase/GT | [event_labeler.py](../../src/collect/robocasa/event_labeler.py), [step_phase.py](../../src/collect/robocasa/step_phase.py): 추론 직전 phase와 env-step GT 정렬 |
| 저장 | [artifacts.py](../../src/collect/artifacts.py), [schema.py](../../src/collect/schema.py): pkl·action CSV·video·meta 및 지문 |
| 검증·이관 | [first_cell_gate.sh](../../scripts/collect/first_cell_gate.sh), [compare_cell_runs.py](../../scripts/collect/compare_cell_runs.py), [scan_video_integrity.py](../../scripts/collect/scan_video_integrity.py), [build_grid_index.py](../../scripts/collect/build_grid_index.py), [verify_grid.py](../../scripts/collect/verify_grid.py), [ship_to_archive.sh](../../scripts/safe/groot_n15/robocasa/collect/ship_to_archive.sh) |

한 셀의 실행은 다음 순서다.

1. plan에서 환경 이름·env seed·문장·jitter·inference seed를 결정한다.
2. 현재 RoboCasa365 `GrootRoboCasaEnv` 생성 및 seed reset으로 ep_meta를 획득한다. 생성자의 내부 reset도 재현 경로의 일부다.
3. `_v6_apply_jitter`에서 문장을 대조하고 base offset을 계산한다. ep_meta 주입 후 plain reset을 `reset_idx+1`회 수행한다. 최종 base 일치, 새 접촉/관입 증가를 검사한다.
4. 관측을 HTTP 입력으로 바꾸고 `/act_with_features`를 호출한다. 추론 r의 seed는 **noise seed + r**다.
5. N1.5는 **16개 action을 예측하고 앞 5개를 실행**한다. 최대 720 env-step, 성공 시 종료. 한 feature record는 실행 chunk 시작 관측에 대응한다.
6. capture layer `[0,2,4,8,10,12,15]`, denoise 4, full token으로 record `[7,4,49,1536]`를 저장한다. 49는 state 1 + future 32 + action 16이다. 이 future token을 Cosmos의 미래 이미지 frame과 같은 의미로 가정하면 안 된다.
7. phase·action·state·성공 여부·ep_meta·모델/서버 지문과 함께 저장하고 QA 후 이관한다.

`--no-features` 평가도 `/act_with_features + skip_features`의 같은 chunk 추론을 사용한다. 기존 `/act`의 action queue-pop으로 바꾸면 추론 주기와 seed 소비가 달라진다.

셀 저장 구조는 `<model>/<version>/grid/<plan_id>/<machine>/<instruction>/s<sid>/j<jid>/n<nid>/base/{rollout.pkl,traj.csv,video.mp4,meta.json}`이다. pkl은 시점별 activation·예측 action chunk·state·phase/GT를 담는다. **traj.csv는 관측된 EEF 궤적 전체가 아니라, record별 예측 action의 첫 스텝을 뽑은 7차원 CSV**다. 재현 검증을 강화할 때 이 차이를 반영해야 한다.

## 3. CosmosPolicy 이식에서 바뀌는 계약

### 환경·task

공식 설치 경로는 `moojink/robocasa-cosmos-policy` fork다. 조사한 fork는 version 0.2.0이며, 공식 의존성은 robosuite 1.5.1 / MuJoCo 3.2.6이다. 현재 repo의 RoboCasa365 fork·assets·reset 동작과 동일하지 않다. [공식 설치/평가 문서](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/ROBOCASA.md), [의존성](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/pyproject.toml), [환경 fork](https://github.com/moojink/robocasa-cosmos-policy/tree/edd9a328b3ec98050f42d194c1419307a79c4d87)

| v6 task | 공식 Cosmos 환경에서 확인한 대응 | 이식 판단 |
|---|---|---|
| OpenDrawer | OpenDrawer | 첫 통합 후보. 버전 간 fixture·문장·reset 상태 동일성은 별도 확인. |
| CoffeeSetupMug | CoffeeSetupMug | 후보. 같은 이름만으로 성공 predicate·배치가 같다고 가정하지 않는다. |
| PickPlaceCounterToCabinet | PnPCounterToCab | 개념상 대응. 객체 지정, 문장, split, 목표 cabinet/성공 조건을 대조해야 한다. |
| SlideOvenRack | 해당 class 및 공식 24과제 목록에 없음 | 공개 체크포인트의 학습 과제로 볼 근거가 없다. 현재 환경에서 실행하더라도 task transfer 조건이다. |
| SlideDishwasherRack | 해당 class 및 공식 24과제 목록에 없음 | 위와 같음. |

공식 runner는 10 trial마다 지정 scene을 고르는 경로와 최초 10회 zero-action settling을 포함한다. v6의 seed/reset/jitter 경로와 다르므로 runner 전체를 그대로 돌려 v6 paired data라고 부를 수 없다. [공식 runner](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py)

**추천:** 공식 환경에서 정상 동작 기준선을 먼저 확보한 후, 현재 RoboCasa365를 유지한 adapter에서 OpenDrawer·Coffee·PPCC의 동작을 검증한다. 성공적으로 이식되면 v6 초기 세계의 동일성을 확인한 subset부터 수집한다. 공식 환경만 사용하는 방식을 선택하면 scene scan·jitter 검증·plan·phase predicate를 새로 만들어야 하며 기존 v6와 별도 데이터셋이다.

### action: profile만 보고 변환하면 위험하다

현재 [N1.5 profile](../../configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml)은 `absolute`로 선언돼 있다. 그러나 현재 native 경로의 [PandaOmron 기본 controller](../../src/benchmarks/robosuite/robosuite/controllers/config/robots/default_pandaomron.json)는 `input_type=delta`, `input_ref_frame=base`이며 [key converter](../../src/benchmarks/robocasa/robocasa/models/robots/__init__.py)의 arm metadata도 `absolute=False`다. **따라서 “기존이 절대 action이니 Cosmos delta를 절대 pose로 바꾸면 된다”는 판단은 성립하지 않는다.** 실제 로드된 controller·checkpoint action 전처리를 함께 대조해야 한다. 이번 조사에서 기존 profile을 수정하지 않았다.

Cosmos artifact의 stats는 action 7차원, proprio 9차원이다. 공식 controller 파일을 안전한 pickle disassembly로 확인했으며 `OSC_POSE`, `control_delta=True`, 입력 ±1, 위치 출력 ±0.05, 회전 출력 ±0.5다. 이는 controller 입력 단위의 상대 명령이며 unnormalize 직후 값을 곧바로 미터/radian의 절대 pose로 취급하면 안 된다. [공식 controller artifact](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl), [실제 stats](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/blob/4b2a04c80d97202f86127ebec80461e8016ec1dc/robocasa_dataset_statistics.json)

공식 runner는 7D `[arm6, gripper1]`에 `[0,0,0,0,-1]`을 붙여 `[arm6, gripper1, base3, torso1, base_mode1]`의 12D를 만든다. gripper 부호를 임의로 반전하거나 2관절이라고 action을 복제하지 않는다. 새 버전에서 OSC 기준 좌표계/scale까지 같아야 같은 명령이다.

특히 [generic RoboCasaActionProcessor](../../src/processor/action/robocasa.py)는 flat 7D 입력에서 gripper를 복제하는 경로가 남아 있다. 이 경로는 Cosmos의 공식 12D 조립과 다르다. 이식 시 `action.base_motion=[0,0,0,0]`, `action.control_mode=-1`까지 명시한 sub-key 경로 또는 검증된 전용 adapter를 사용한다. GR00T native gripper/mode의 0.5 threshold도 그대로 물려받지 않는다.

### 관측·전처리·언어

| 항목 | Cosmos용 요구사항 |
|---|---|
| 카메라 | left/right agentview + wrist 세 시점. view 순서와 카메라 extrinsic도 확인. |
| proprio | **`[gripper_qpos(2), world eef_pos(3), eef_quat(4)]` 순서**. GR00T HTTP의 base-relative eef state를 그대로 쓰면 안 됨. native obs에 있는 absolute state를 별도로 전달 가능. quaternion convention도 검증. |
| 정규화 | Cosmos 자체 stats의 min/max로 proprio normalize, action unnormalize. GR00T stats 재사용 불가. |
| 이미지 | 224×224, JPEG quality 95, 학습에 대응하는 90% 면적 center crop. 현재 512→256 GR00T 영상 경로와 공식 raw 224 렌더의 차이를 검증. |
| 상하 반전 | 공식 raw env 경로는 flipud 적용. 현재 repo의 `get_basic_observation`은 이미 상하 반전하므로 이 입력에 공식 flip을 더 적용하면 두 번 뒤집힘. 180° 회전과도 구별. |
| 언어 | exact instruction의 T5 embedding이 필요. scene별 실제 `lang` 전체를 준비. |

근거: [공식 관측 준비 코드](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py#L268), [Cosmos 전처리/정규화](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L493).

cache miss 시 upstream는 `google-t5/t5-11b` encoder를 GPU에 추가 로드할 수 있다. 2B 정책의 추론 VRAM만 보고 용량을 잡으면 안 된다. 수집 전에 plan 문장 embedding을 별도 과정에서 생성·검증하고 수집 서버에서는 cache miss를 명확히 검출하는 구성이 적합하다. [T5 loader](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/_src/predict2/inference/get_t5_emb.py)

### horizon·noise

공식 Cosmos RoboCasa는 **32 predicted / 16 executed / action denoise 5**다. 모델 config와 eval chunk size 일치를 assert하므로 GR00T의 16 predicted를 그대로 강제할 수 없다. [공식 config artifact](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/blob/4b2a04c80d97202f86127ebec80461e8016ec1dc/config.json)

기존 N1.5의 16/5 계약은 유지한다. Cosmos에서 기존 phase 시간 해상도를 맞추려면 **32 predicted / 5 executed인 별도 수집 계약**을 검토할 수 있다. 이는 공식 32/16 baseline과 정책 실행이 다르므로 별도 검증·plan에 기록해야 한다. 공식 성능 숫자를 이 설정에 전용할 수 없다. `model_action_horizon`, `executed_action_steps`, `env_step_start/end`, `control_freq`를 구분해 저장한다.

공식 serial runner는 매 query에 `cfg.seed + query_idx`를 전달한다. 기존 수집기의 `noise seed + inference index`와 다르다. Cosmos adapter가 매 요청 seed를 직접 받도록 만들고, episode reset마다 query counter를 초기화해야 한다. `randomize_seed=True`는 `secrets` 기반 재추첨 경로이므로 grid 재현에 쓰지 않는다. noise 숫자가 두 모델에서 같아도 동일한 latent noise 또는 동일한 행동을 의미하지는 않는다. [공식 get_action](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py#L851)

첫 수집은 best-of-N=1, search depth=1로 한다. future/value를 생성하는 것과 그것으로 여러 action 후보를 선택하는 planning은 구분해야 한다. collector가 의도치 않게 planning 데이터를 기본 policy 데이터에 섞지 않도록 설정을 기록한다.

## 4. activation 계약과 저장량

Cosmos는 GR00T action-head hook을 그대로 달 수 없다. `CosmosPolicyVideo2WorldModel.net`의 `blocks` output이 hook 후보다. base 2B config는 28 blocks, hidden width 2048이고 block tensor는 `(B,T,H,W,D)` 구조다. RoboCasa latent frame은 blank/current proprio/current wrist/current left/current right/action/future proprio/future wrist/future left/future right/value의 11개다. action frame index는 5다. [network](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/_src/predict2/networks/minimal_v4_dit.py), [net config](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/_src/predict2/configs/text2world/defaults/net.py), [RoboCasa config](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py#L228)

수집할 정보는 다음과 같다. 아래는 구현 제안이며 아직 확정한 스키마가 아니다.

- `hidden_states`: record별 layer × 실제 denoiser call × 선택한 frame/patch × hidden dim. episode 평균으로 축약하지 않는다.
- `feature_axes`, 실제 shape/dtype, module path, layer index, latent frame 역할·patch 좌표, capture 위치(pre/post).
- `denoise_call_index`, 실제 sigma/timestep, conditional/unconditional 또는 다른 forward branch 식별. sampler step 수와 hook 호출 수가 같다고 가정하지 않는다.
- `model_actions`(32×7), `env_actions`(실제 소비한 12D), world proprio, record 시작/끝 env-step, GT phase, episode success, seed.
- 기존 identity 외 Cosmos commit/weight revision/stats hash/T5 cache hash, 환경·controller·assets 정보.

Cosmos에는 N1.5의 Eagle 기반 `vl_hidden_states`와 동일한 표현이 없다. **T5 text embedding을 VL state의 대용으로 넣으면 안 된다.** 같은 instruction에서 이미지 변화를 반영하지 않는 입력이기 때문이다. 필요한 경우 현재 이미지에 대응하는 VAE/DiT feature를 별도 이름으로 수집하고 역할 차이를 명시한다. QA도 raw observation hash/픽셀 변화와 모델 입력을 함께 보도록 바꾼다.

전체 token 저장 비용을 먼저 계산해야 한다. 224 입력→28×28 latent→2×2 patch라면 frame당 196 patch다. **선택 layer 7개, 실제 capture call 5회, fp16, flatten된 11×196 tokens**를 가정하면:

| 저장 범위 | record당 feature만 | 최대 144 records/episode | 1,800 episodes가 모두 최대 길이일 때 |
|---|---:|---:|---:|
| 11 frame 전체 | 약 294.8 MiB | 약 41.45 GiB | 약 72.86 TiB |
| action frame의 196 patches만 | 약 26.8 MiB | 약 3.77 GiB | 약 6.62 TiB |

위 값은 **shape 기반 산술 추정**으로 runtime 측정이 아니다. 압축·조기 종료·layer/call 선택에 따라 달라지고 metadata/video/직렬화 복사 메모리는 제외했다. fp32면 두 배다. 현재 N1.5 full-token 수집량 추정을 Cosmos에 사용할 수 없다.

따라서 소규모 진단에서 patch/slot을 보존한 뒤, 실험 질문에 필요한 layer·denoise·frame 범위를 정하고 새 plan을 확정한다. action frame만으로는 world/future steering 분석을 할 수 없다. 전체 episode를 RAM list에 쌓는 기존 writer보다 record 단위 chunk 저장·sidecar·HTTP payload 상한/timeout 점검이 필요할 수 있다. 공간 token pooling을 택한다면 기존 full-token 계약과 별도 representation으로 명시해야 한다.

## 5. 재사용과 구현 변경 범위

| 구분 | 조치 |
|---|---|
| 재사용 | `src/collect/plan.py`의 좌표/해시/경로, 결손 관리, index/QA/이관의 기본 틀, 동일 RoboCasa365를 쓸 때 scene 선택·jitter 재현 원리 |
| 서버 신규 | 제안 `scripts/serve/cosmos_policy.py`: `/health`, `/reset`, `/act`, `/act_with_features`, 요청 seed, skip_features 동일 chunk 경로 |
| runtime 신규 | 제안 `docker/cosmos_policy/` + compose service. 공식 환경을 위한 별도 컨테이너가 필요하면 기존 robocasa 컨테이너와 분리 |
| profile 신규 | 제안 `configs/checkpoints/cosmos_policy__robocasa_predict2_2b.yaml`. 상대 OSC·7D 모델 출력/12D 환경 출력·world proprio·32 horizon·stats·이미지 변환을 명시 |
| capture 신규 | 제안 `scripts/serve/cosmos_policy_hooks.py`. 기존 GR00T `safe_hooks.py`/`steering_hooks.py`의 layer/sequence 가정 복사 금지 |
| collector 수정/추출 | `http_feature_collect.py`에서 grid/replay loop와 GR00T IO를 분리. `_v6_apply_jitter`는 동일 환경에서 공유하는 helper로 추출하는 방향 |
| launcher 수정 | `collect_grid.sh`는 profile을 읽지만 실제 서버/컨테이너/플래그는 `lerobot`, `--groot-dit-*`, `--capture-vl` 고정. profile 교체만으로 해결되지 않음 |
| writer 수정 | `write_safe_triplet`의 필수 policy 속성, `groot_action_vector`, `GROOT_ACTION_KEYS` 의존 제거/adapter 제공. GR00T를 사칭하는 key로 Cosmos payload를 저장하지 않음 |
| downstream 수정 | `extract_grid_matrix.py`는 layer 목록·K=4·T=49·D=1536·state/future/action slice 고정. metadata 기반 loader 또는 Cosmos 전용 loader 필요 |
| operator 재학습 | 기존 conceptor/SAE/detector/readout을 Cosmos hidden에 적용할 수 없음. Cosmos 데이터로 다시 fit하고 같은 hook 위치로 intervention 연결 |
| archive 분리 | 제안 `cosmos/v1/grid/<new_plan_id>/...`. 원래 v6 초기 세계를 공유하면 `world_plan_id` 등 출처도 기록. 기존 GR00T archive나 완료 목록에 합치지 않음 |

인덱서가 재사용 가능하다는 것과 pkl 내부까지 모델 무관하다는 것은 다르다. [기존 matrix extractor](../../scripts/analysis/grid_phase/extract_grid_matrix.py), [writer](../../src/collect/artifacts.py)가 현재 가장 직접적인 schema 종속 지점이다.

## 6. 환경 세팅 판단

조사한 현재 host는 RTX A4000 16GB ×8, NVIDIA driver 550.144.03이다. 실행 중인 robocasa 컨테이너에서 확인한 버전은 Python 3.11.15, torch 2.7.1+cu126, NumPy 2.2.5, MuJoCo 3.3.1이다. Dockerfile의 torch 2.5.1/CUDA12.1 선언과 실제 설치 상태가 다르므로 Dockerfile만 보고 실행환경을 판단하면 안 된다.

Cosmos 공식 cu128 경로는 Python 3.10, torch 2.7.0, transformers 4.57.1, transformer-engine 2.2.0 등 별도 dependency set이다. 공식 Docker base는 CUDA12.8.1/Ubuntu24.04다. **기존 LeRobot/GR00T/환경 컨테이너에 pip로 덮어 설치하지 않고 별도 모델 서버 환경을 만든다.** 모델 서버와 simulator를 HTTP로 분리하면 Python·torch 버전이 같을 필요는 없다. 공식 parity 확인용 환경과 현재365 수집용 환경도 구분한다. [공식 Dockerfile](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/docker/Dockerfile), [dependency lock 입력](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/pyproject.toml)

공식 모델 카드의 RoboCasa 추론 메모리는 8.9GB이며 검증 hardware는 H100/A100다. A4000 16GB에서 수용 가능성이 있지만 **activation capture·T5 추가 로딩·allocator·kernel/driver 호환을 포함한 실행 가능성을 보증하지 않는다**. 기존 “kanu 2 serves/GPU, srv 6/GPU”를 이 모델에 그대로 적용할 근거가 없다. 1 serve/GPU부터 peak VRAM/host RAM/latency를 측정한다. A4000/550에서 공식 CUDA12.8 이미지와 attention backend가 작동하는지도 실제 검증 항목이다. [메모리와 검증 hardware](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B#system-requirements-and-performance)

필요 artifact는 정책 `.pt`, config/stats, T5 embedding cache, VAE/tokenizer 등 loader의 전이 의존성이다. 학습 demonstration 전체는 rollout 실행만 할 때 필수는 아니지만, handoff의 expert 대조 실험에는 Cosmos가 학습한 demonstration을 별도로 마련해야 한다.

repo에서는 Cosmos 서버/profile/compose service를 찾지 못했다. 검사한 local cache에서도 Cosmos 이름의 checkpoint를 확인하지 못했다. 다른 머신·별도 경로에 보유한 checkpoint까지 없다고 단정하지 않는다.

## 7. 본수집 전 확인 순서

1. **공식 모델의 정상 실행 확보:** 격리된 Cosmos 환경에서 공식 지원 task 1개, `/health` 및 실관측 1회 추론, 1 episode. action 32×7·정규화·gripper·정책/환경 input contract 확인.
2. **기존 세계에 이식:** 현재 RoboCasa365의 OpenDrawer/Coffee/PPCC 초기 scene을 사용해 controller, world proprio, 이미지 방향/전처리, task 조건을 대조. 초기 qpos/qvel·물체 pose·base·문장·관측 지문을 수집.
3. **캡처 경로 확인:** 선택 DiT block의 실제 shape/call 수/sigma를 측정. capture on/off에서 동일 action이 나오는지 확인. full-token 소량 수집으로 저장량/지연 측정.
4. **재현 게이트:** A=수집, B=fresh 재수집, D=feature 저장 없는 평가의 동일 세계·action·성공 여부 대조. seed reset 전에 JSON 주입하는 C는 거부돼야 함. 기존 gate 셸에는 `A=B=C`라는 과거 주석과 C 산출물까지 comparator에 넘기는 배선이 남아 있으므로, Cosmos용 게이트는 최신 A=B=D 계약으로 명확히 작성.
5. **작은 grid:** 예컨대 지원 과제 1개 × scene 3 × jitter 2 × noise 2의 12판. 동일 `(scene,j)`의 noise 간 초기 세계 일치, n 변화에 따른 정책 noise 변화, 12판 모두 파일/feature/phase 정렬 및 freeze 검사. 충분한 성공·실패 사례 확보 여부는 이때 관찰하며 미리 가정하지 않음.
6. **본수집 계약 확정:** 32/5 또는 다른 실행 horizon, capture 범위·dtype·frame 의미, 신규 plan_id·staging·archive·machine home 확정 후 확대. 수집과 후속 eval이 동일한 실행 계약을 사용해야 함.

비교는 **같은 Cosmos 모델의 수집 vs replay**에서 재현을 확인하는 것이며, GR00T action과 Cosmos action이 같아야 한다는 뜻이 아니다. 환경 fork를 바꾸는 경우는 초기 세계의 동일성부터 성립하지 않을 수 있다.

## 8. 확인 범위와 미실시 사항

- 사용자가 지정한 세 handoff, 현 plan, collector/launcher/server hook/writer/processor/분석기와 공식 Cosmos source/artifact를 대조했다.
- 공식 source는 별도 `/tmp` checkout에서 읽었다. NVlabs commit `18a2accadf4e7a3531e56754102af5a24d2316da`, Cosmos 환경 fork `edd9a328b3ec98050f42d194c1419307a79c4d87`, HF artifact revision `4b2a04c80d97202f86127ebec80461e8016ec1dc`.
- 현재 local RoboCasa submodule HEAD `3431d2e3cff7cb9759bce8279f7101824f3eefce`, robosuite `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`. HEAD만으로 working tree/asset/env 지문 전체를 대신할 수 없다.
- 실제 컨테이너 package/import 경로와 host GPU 정보를 읽기 전용으로 확인했다. robosuite distribution metadata는 설치 형태상 없었으나 source import 경로는 확인했다.
- 정책 가중치 다운로드·설치·GPU inference·새 환경 reset·rollout 수집은 실행하지 않았다. 따라서 결론은 코드/환경 계약에 근거한 구현 가능성 분석이며, 이 host에서의 runtime 성공이나 task 성능 검증이 아니다.
- 변경물은 이 보고서뿐이다. 기존 사용자 작업 파일·profile·수집 plan·환경·archive는 변경하지 않았다.
