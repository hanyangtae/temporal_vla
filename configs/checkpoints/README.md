# Checkpoint Profiles

VLA 체크포인트별 **정규화 · action sub-key 매핑 · state 요구사항**을 선언적으로 기록하는 YAML 프로파일 디렉토리.

`scripts/serve/*.py` 는 `--profile configs/checkpoints/<name>.yaml` 으로 프로파일을 로드해 동작을 결정한다. 즉 serve 스크립트에는 **모델 아키텍처 고유 코드**(weight 로딩, forward) 만 남고, 체크포인트별 가정(LIBERO 180° rotation, gripper sign flip, unnorm_key 등)은 모두 프로파일로 분리된다.

## 파일명 규칙

```
<base_model>__<dataset_or_variant>.yaml
```

- `base_model` 은 `scripts/serve/<base_model>.py` 와 매칭.
- 예: `openvla_oft__rlinf_calvin_sft.yaml`, `pi05__calvin_abc_sft.yaml`, `xvla__calvin_default.yaml`.

## 스키마

```yaml
name: openvla_oft__rlinf_calvin_sft      # 파일명 stem 과 일치해야 함
base_model: openvla_oft                  # scripts/serve/<base_model>.py
checkpoint_source:
  type: hf_repo                          # hf_repo | local
  id: RLinf/RLinf-OpenVLAOFT-CALVIN-SFT  # HF repo id 또는 로컬 절대경로
compatible_benchmarks: [calvin]          # calvin | robocasa | libero ...

action_type: relative                    # relative | absolute
action_layout:                           # 모델 출력 벡터의 차원별 의미 (순서대로)
  - {name: eef_pos,       dims: 3}
  - {name: eef_axisangle, dims: 3}
  - {name: gripper,       dims: 1}
rotation_encoding: axisangle             # euler | quat_xyzw | quat_wxyz | rot6d | axisangle | none
gripper_encoding:
  range: "[-1,1]"                        # "[-1,1]" | "[0,1]" | "{-1,1}"
  binarize: true                         # 이진화 여부
  sign_flip: false                       # 반환 전 부호 반전 여부
  threshold: 0.0                         # binarize 임계값

normalization:
  scheme: q01_q99                        # none | min_max | q01_q99 | mean_std
  stats_file: dataset_statistics.json    # 체크포인트 루트 상대경로
  key_selection:                         # stats JSON 내 key fallback chain
    - calvin_abc_d
    - calvin
    - <first_available>                  # sentinel: 첫 번째 사용 가능한 key

observation_requirements:
  images: [static, wrist]                # 모델이 요구하는 image view
  state: [eef_pos, eef_euler, gripper_qpos]  # 요구 state sub-key
  allow_conversions: [quat_to_euler, quat_to_axisangle]

n_action_steps: 1                        # /act 가 한 번에 반환할 action 개수
image_preprocess:
  resolution: 256
  rotate_180: false                      # LIBERO 등 특수 규칙
  center_crop: true

emits_subkeys:                           # /act 응답에 포함될 action.* sub-key
  - action.eef_pos
  - action.eef_euler                     # axisangle → euler 변환해서 emit
  - action.gripper

# 선택 필드 — 체크포인트가 LoRA 어댑터를 포함할 때
lora:
  subfolder: lora_adapter                # checkpoint_source 기준 서브폴더

# 선택 필드 — 모델 아키텍처 고유 설정. 각 serve 가 필요한 key 만 참조.
# 예시 (X-VLA): domain_id, denoising_steps, max_views, tokenizer, image_normalization, proprio_layout
# 예시 (GR00T): embodiment_tag
model_specific:
  domain_id: 2
  denoising_steps: 10
```

## 로더

`scripts/utils/checkpoint_profile.py` 의 `load_profile(path)` 이 위 YAML 을 `CheckpointProfile` dataclass 로 반환한다. 유효성은 로드 시 assert 로 검증.

```python
from scripts.utils.checkpoint_profile import load_profile

profile = load_profile("configs/checkpoints/openvla_oft__rlinf_calvin_sft.yaml")
print(profile.action_dim, profile.emits_subkeys)
```

dry-run:

```bash
python scripts/utils/checkpoint_profile.py configs/checkpoints/<name>.yaml
```

## 벤치마크 측 계약

프로파일의 `emits_subkeys` 가 해당 벤치 ActionProcessor 의 소비 가능한 조합을 **반드시 포함**해야 한다.

| 벤치 | 파일 | 필요 sub-key 조합 |
|---|---|---|
| Calvin | `src/processor/action/calvin.py` | `action.eef_pos` + (`action.eef_euler` \| `action.eef_rot6d` \| `action.eef_quat`) + `action.gripper` |
| RoboCasa | `src/processor/action/robocasa.py` | `action.eef_pos` + `action.eef_euler` + `action.gripper` |
| LIBERO | (OpenVLA-OFT 기준) | `action.eef_pos` + `action.eef_euler` + `action.gripper` (relative) |

eval 스크립트의 `make_*_processors(action_type=..., gripper_threshold=...)` 호출은 프로파일과 일치시켜야 한다. 일반적으로 `action_type=relative` 면 `gripper_threshold=0.0`, `absolute` 면 `0.8` (X-VLA 패턴).

## 온보딩 절차

새 체크포인트 추가 시에는 `vla-checkpoint-manager` 에이전트를 호출하거나 `docs/adding_checkpoint.md` 체크리스트를 따른다.
