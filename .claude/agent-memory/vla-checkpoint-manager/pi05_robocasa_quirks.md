---
name: pi05-robocasa-quirks
description: pi0.5 RoboCasa365 체크포인트 온보딩 발견사항 + SR 게이트 진단 — openpi→LeRobot 변환 ckpt 서빙 시 주의점, serve/profile 검증 결과, SR=0 원인
metadata:
  type: project
---

# pi0.5 RoboCasa365 체크포인트 온보딩 발견사항

체크포인트: `/cache/checkpoints/pi05-robocasa-75000-lerobot` (config.json + model.safetensors 16G + norm_stats.json)
프로파일: `configs/checkpoints/lerobot_pi05__robocasa365_75000.yaml` (base_model=lerobot, policy_type=pi05)
serve: `scripts/serve/lerobot.py` external_config 경로 (신규 코드 불필요 — 기존 pi.py 어댑터 재사용)

## 1. config.json visual feature 누락 → camera_keys 로 _force_external
openpi→LeRobot 변환이 `input_features` 에 camera key 를 안 넣음. profile `external_config.camera_keys`
dict 선언 시 `pi.py` 가 `_force_external=True` 로 config.json 있어도 external 경로 강제.

## 2. QUANTILES → MEAN_STD undershoot (pi.py 자동 치환) — 실재하나 SR 원인은 아님
config.json `normalization_mapping={STATE,ACTION:QUANTILES}` 이고 **PI05Config() 기본값도 QUANTILES**.
그러나 norm_stats 는 mean/std 만(q01/q99=null) → pi.py(111-117) 가 QUANTILES→MEAN_STD 무조건 치환 + None 필터.
lerobot QUANTILES(normalize.py:362-377) 는 [q01,q99]→[-1,1], 역정규화 스케일=(q99-q01)/2=2.326·std(Gaussian).
MEAN_STD 로 치환하면 스케일이 std 라 **action·state 가 ~2.326배 undershoot**.
- **교정법**: (1) norm_stats.json q01/q99 를 Gaussian 근사(mean∓2.326·std, zero-std dim 은 eps floor) patch,
  (2) profile `external_config.normalization_mapping` 을 non-None 으로 명시 → pi.py 치환 skip(cfg 기본 QUANTILES 유지).
- **★ 실측(2026-07-25)**: 교정 후 action 2.3배↑·gripper -2.1·eef_pos peak 1.56([-1,1] clip) = QUANTILES 활성 확인.
  **그러나 OpenDrawer SR 0/10 그대로** → undershoot 은 실재하나 SR=0 의 충분원인 아님. 모델이 scene 에서 저-commit
  (action |mean| 0.082, arm 0.05m/60step).
- **★ 결론(가설 기각·되돌림)**: HF 원본(robocasa/robocasa365_checkpoints pi05_pretrain_human300/.../75000/
  assets/norm_stats.json)**에도 q01/q99=None** → quantile 결측은 변환 손실 아니라 원본에도 없음. openpi 가 이 ckpt 에
  mean/std 를 쓴 것 → **pi.py 기본 MEAN_STD 치환이 원본과 정합 = 충실**. Gaussian-QUANTILES 는 과대스케일이었음.
  → norm_stats.json.orig 복원, profile scheme=mean_std·normalization_mapping override 제거 (현재 상태). **정규화는 red herring.**
- profile `normalization.stats_file` 는 문서용 — load_dataset_stats 는 `dataset_stats_path` 없으면 None→rglob 경로.

## 3. dtype float32 → bfloat16 강제 (OOM 회피)
config dtype=float32 는 16GB GPU OOM(~13.4GB). `external_config.dtype: bfloat16` → 실측 로딩 9.5GB.

## 4. state 16D → 32D zero-pad (norm_stats 로 확정)
학습 state = [base_to_eef_pos(3), base_to_eef_quat(4, xyzw), base_pos(3), base_quat(4, xyzw), gripper_qpos(2)] = 16D.
norm_stats mean/std 로 검증됨(state[16:31]=pad, mean0/std1). `_apply_input_remap` 가 max_state_dim(32)까지 zero-pad.
profile `observation_requirements.state` 에 위 5키를 이 순서로. RoboCasaObsProcessor 가 robot0_base_to_eef_pos/_quat 직접 emit.

## 5. 12D action layout + control_mode (SR=0 gripper fix 는 코드에 이미 존재)
`RobocasaOutputs[:12]` = [eef_pos(3), eef_axisangle(3), gripper(1,[-1,1]), base_motion(4=vx,vy,w,torso), control_mode(1)].
모델은 32D(max_action_dim) 출력 → 앞 12D 만 emit. action_type=absolute(config use_relative_actions=false).
**`RoboCasaActionProcessor._process_subkeyed` (robocasa.py:149) 이 `action.control_mode` 있으면 1D gripper+base_mode
조립** → 프로파일에 control_mode(dims:1) 포함하면 generic 경로 정상. (구 §8 "SR=0/--use-groot-env 권장" 은 fix 이전 기록, 낡음.)

## 6. PI05Config() 기본값이 이 ckpt 와 완전 일치 → external 경로 안전
external 경로는 config.json 무시하고 `config_class()` 기본값 사용. 확인: gemma_2b/gemma_300m/chunk50/
max_state_dim32/max_action_dim32/num_inference_steps10/224x224/empty_cameras0 전부 일치. n_action_steps 는 profile 값으로 override(≤chunk_size 50).

## 7. eval 은 generic 경로 (--use-groot-env 금지)
`make_robocasa_processors(static_cam2=agentview_right)` + `--three-cameras`, robocasa_eval `run_vla_rollouts`(create_eval_env).
--use-groot-env 는 pi05 state 키와 불일치. instruction 은 `env.get_ep_meta()["lang"]` 로 정상 전달됨(확인).

## ★ serve/profile 는 전부 검증됨 (2026-07-24)
직접 진단으로 full HTTP 경로가 올바름을 확인:
- 3 카메라 실이미지가 모델 도달(mean~150 정상, black 아님), state 16D 가 norm_stats 분포와 일치.
- 역정규화 정상(postprocess action 이 실스케일; norm std~0.37 규모).
- **task conditioning 작동**: "Open the left drawer" vs "right drawer" 에 방향이 다른 대형 action 생성.
  arm 이 env 에서 실제로 이동(base_to_eef 0.265m/40step).
- **진단 기법 — 빈 task 시그니처**: `task=""` 이면 모델이 near-zero(~1e-3) action 으로 freeze.
  이게 SR eval 초기 near-zero dump 와 헷갈릴 수 있으나, eval 은 instruction 을 정상 전달함(별개).

## ★ SR 게이트 실패 (2026-07-24~25) — serve 버그 아님. 4 task 전부 0
generic 경로(MEAN_STD 충실, --three-cameras, seed 100000) 실측:
| task | rollouts | SR |
|---|---|---|
| OpenDrawer | 20+10+6 (2 mapping·const/vary seed) | 0 |
| CloseDrawer | 10 | 0 |
| OpenStandMixerHead (mixer) | 10 | 0 |
| PickPlaceCounterToCabinet (bread) | 10 | 0 |
(mixer·bread 는 N1.5/N1.6 atlas 셀과 동일 — cross-model 비교용 최우선이었으나 0.)
모델이 접근은 하나 완결 못함(프레임상 arm 이 counter 부근). serve/profile 은 검증됨.
→ **다음 단계: openpi 네이티브 서빙**으로 lerobot 변환 weight 문제 배제 필요(원본 params 12.44GB + openpi 설치, 미실행).
- 원인 후보: 변환 weight 손상 / step75000 undertrained / **camera 위치 OOD**(학습 ep_meta `cam_configs` vs generic
  create_eval_env fresh reset; --use-groot-env 는 cam_configs 적용하나 pi05 state 키 금지) / camera slot 미확정.
- **camera slot 미해결**: MINE(wrist→right_wrist_0_rgb) vs alt1(wrist→left_wrist_0_rgb, openpi single-arm 관례) 둘 다
  task-differentiated action 생성하나 SR=0. 정답 slot 은 학습 dataset info.json(부재) 또는 SR>0 task 로만 확정 가능.
  현재 프로파일 = alt1(더 원칙적). convert_robocasa_to_lerobot.py 는 이 머신에 없음.

## 비자명 함정 (재현 시 참고)
- **PandaMobile == PandaOmron 별칭**: create_eval_env(robots="PandaMobile") 이 default_pandaomron.json 로드,
  robots 를 PandaOmron 으로 바꿔도 fixed seed 에서 obs 완전 동일. 초기 "robot 차이" 관찰은 noise 아티팩트였음.
- **flow-matching noise 민감성**: inference_seed 미고정 시 같은 obs 라도 call 마다 다른 action → single-scene 프로브 신뢰불가.
  프로브는 fixed inference_seed 로. (단 constant seed 로 full rollout 해도 SR=0 — noise 는 근본 원인 아님.)
- pi.py external INFO 로그(norm_stats loaded 등)는 stdout 에 안 찍힘(logger 설정) — 로딩 성공은 /health + GPU 메모리로 확인.

## 다음 조치 후보
1. robocasa365 가 잘하는 task 탐색(전 task-set 소량 rollout 스윕) — exp4-3 는 SR>0 task 아무거나면 됨.
2. camera 위치 OOD 검증: 학습 collection cam_configs vs generic reset 비교. 필요시 generic 경로에 cam_config 주입.
3. 정확한 camera slot 을 학습 dataset info.json / convert 스크립트로 확정.
4. 더 학습된 pi05 robocasa 체크포인트 확보.
