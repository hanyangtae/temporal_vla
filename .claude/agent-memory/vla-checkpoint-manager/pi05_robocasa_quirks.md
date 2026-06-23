---
name: pi05-robocasa-quirks
description: pi0.5 RoboCasa 체크포인트 온보딩 비자명 발견사항 — openpi→LeRobot 변환 체크포인트 서빙 시 주의점
metadata:
  type: project
---

# pi0.5 RoboCasa 체크포인트 온보딩 발견사항

체크포인트: `/cache/checkpoints/pi05-robocasa-75000-lerobot`
프로파일: `configs/checkpoints/lerobot_pi05__robocasa365_75000.yaml`

## 1. config.json visual feature 누락

openpi → LeRobot 변환 스크립트가 생성하는 `config.json`의 `input_features`에 camera key가 없다
(변환 아티팩트). `pi.py` 어댑터가 `from_pretrained`로 로딩하면 no-camera 모델이 만들어진다.

**Fix**: `external_config.camera_keys` dict를 profile에 선언하면 `pi.py`가 `_force_external=True`로
`config.json`이 있어도 external_config 경로를 강제 사용한다.

## 2. QUANTILES → MEAN_STD 강제 변환

`config.json`의 `normalization_mapping: {STATE: QUANTILES, ACTION: QUANTILES}` 이지만
openpi `norm_stats.json`은 `mean/std`만 있고 `q01/q99=null`. lerobot QUANTILES normalizer가
`q01/q99` 필수 요구 → `ValueError`.

**Fix**: `pi.py` external_config 경로에서 `normalization_mapping` QUANTILES → MEAN_STD 자동 override.
null 값은 `if sv is not None` 필터로 dataset_stats 구성 시 제외.

## 3. bf16 OOM 이슈

`config.json`의 `dtype: float32` → PaliGemma-2B 풀 로딩 시 ~13.4 GB (A4000 16GB 초과).

**Fix**: external_config 경로에서 `cfg.dtype = ext.get("dtype", "bfloat16")`로 bfloat16 강제.
profile `model_specific.external_config.dtype` 필드로 override 가능.

## 4. 16-dim state layout (32D 패딩)

`convert_robocasa_to_lerobot.py` 기준:
- [0-2]  `robot0_base_to_eef_pos` (base-frame EEF pos, 3D)
- [3-6]  `robot0_base_to_eef_quat` (xyzw, 4D)
- [7-9]  `robot0_base_pos` (3D)
- [10-13] `robot0_base_quat` (xyzw, 4D)
- [14-15] `robot0_gripper_qpos` (2D)
- [16-31] 패딩 (mean=0, std=1)

총 16D → 32D zero-pad. `RoboCasaObsProcessor`가 `robot0_base_to_eef_pos/_quat`를
직접 observable로 expose하므로 세계좌표계 fallback 불필요.

## 5. state 32D zero-padding (`_apply_input_remap`)

`input_features.observation.state.shape=[32]`인데 profile state는 16D.
`_apply_input_remap`에서 `cur_dim < _state_dim`인 경우 zero-pad 추가.

## 6. 3-camera 필수

pi05는 `base_0_rgb` + `left_wrist_0_rgb` + `right_wrist_0_rgb` 세 카메라 모두 필수.
RoboCasaObsProcessor `--three-cameras` 옵션으로 `observation.images.right` emit 활성화.

## 7. base_to_eef_pos/quat robosuite 가용성 확인

`robot0_base_to_eef_pos`, `robot0_base_to_eef_quat`는 PandaOmron env에서 valid observable임
(2026-06-22 확인). `RoboCasaObsProcessor._extract_named_states`가 직접 추출.
세계좌표계 fallback은 불필요하지만 `_build_state_from_profile`에 fallback 코드는 안전망으로 남김.

## 8. 12-dim action layout (SR=0 원인)

`RobocasaOutputs[:12]`:
- [0-2]  eef_pos (3D)
- [3-5]  eef_axisangle (3D)
- [6]    gripper ([-1,1], 1D)
- [7-10] base_motion (vx, vy, w, torso, 4D)
- [11]   control_mode (1D)

**SR=0 원인**: `RoboCasaActionProcessor._process_subkeyed` (line 145-150)가 gripper를 2D로 복제:
`[arm(6), [grip, grip](2), base_torso(4)]` → 12D. 그러나 PandaMobile 컨트롤러 레이아웃은
`[right(6), right_gripper(1), base(3), torso(1), base_mode(1)]` = 12D. 복제된 grip이
position[7]에 가서 base_x 속도 슬롯을 오염시키고, control_mode는 완전히 누락된다.

**Fix 필요 (비자명)**:
- Option A (최소): `RoboCasaActionProcessor`에 `gripper_dim=1` 파라미터 추가 — `src/processor/` 수정 필요
- Option B: `--use-groot-env` 플래그 + `GrootRoboCasaActionProcessor` 사용 — 올바른 단일 gripper + control_mode 변환, 단 obs pipeline도 groot 버전 필요
- Option C (미검증): 서브키 대신 flat 12D를 직접 패스스루

**GR00T와 차이**: GR00T N1.5는 `gripper_qpos` 2D를 정말 2-finger로 내보냄 → `[grip, grip]` 복제가 일치.
pi05는 1D gripper → 복제 시 레이아웃 불일치.

**Why**: pi05 COAST 재현 온보딩 시 발생한 비자명 문제들 기록. SR=0 smoke eval(2026-06-22).

**How to apply**: pi05 eval 수정 전까지 SR=0 예상. `--use-groot-env` 전환이 가장 안전한 경로.
다음 openpi 변환 체크포인트 온보딩 시: gripper_dim 확인 필수.
