# GR00T N1.5 RoboCasa — LeRobot Internal Parity

이 문서는 SR(success rate) 비교가 아니라, LeRobot HTTP 경로가 RoboCasa365
GR00T N1.5 checkpoint를 올바르게 로드하는지 확인하는 진단 문서다.

파일 수를 줄이기 위해 action/preprocess/ZMQ-vs-HTTP one-off probe scripts는 제거했다.
현재 유지하는 parity 표면은 checkpoint-load verifier 하나다.

## 유지하는 진단 스크립트

| Script | 역할 |
|---|---|
| `scripts/eval/lerobot_groot_n15_internal_parity.py` | raw Isaac-GR00T checkpoint prefix/value parity verifier |

Runtime compatibility glue는 `scripts/utils/lerobot_groot_n15_runtime.py` 한 곳에만 둔다.
Import shadowing을 피하기 위해 예전 eval-side helper duplicate는 제거했다.

## 현재 결론

현재 재현 가능한 retained verifier로 말할 수 있는 것은 아래까지다.

| Layer | 결론 | 근거 |
|---|---|---|
| checkpoint key/shape load | OK | 764 tensors checked, missing/shape/unexpected 0 |
| vision tower weights | exact | 448 tensors, `max_abs=0.0` |
| Eagle projector `backbone.eagle_model.mlp1` | exact | 2 tensors, `max_abs=0.0` |
| action head weights | dtype-materialized equivalent | 314 tensors, raw fp32-vs-bf16 `max_abs=0.007767`, checkpoint cast to model dtype gives mismatch 0 |

현재 주장할 수 있는 것은 "LeRobot과 native가 완전히 동일하다"가 아니다. 방어 가능한
결론은, 이전 loader gap에서 중요했던 raw checkpoint prefix들이 LeRobot-wrapped model 안에
들어갔고 shape 및 dtype-cast value가 맞는다는 것이다. Closed-loop SR과 activation-level
identity는 별도 질문으로 남긴다.

## 체크포인트 로드 검증

검증 명령:

```bash
python scripts/eval/lerobot_groot_n15_internal_parity.py \
  --device cpu \
  --output outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json
```

2026-06-09에 사용한 container 명령:

```bash
docker exec temporal_vla-lerobot-run-0b327aff8915 bash -lc '
cd /temporal_vla &&
PYTHONPATH=/temporal_vla/scripts/eval:/temporal_vla/scripts:/temporal_vla/scripts/utils:/temporal_vla/scripts/serve:/temporal_vla/lerobot/src:/temporal_vla \
python scripts/eval/lerobot_groot_n15_internal_parity.py \
  --device cpu \
  --output outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json'
```

산출물:

```text
outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json
```

요약:

```json
{
  "checked_count": 764,
  "missing_count": 0,
  "shape_mismatch_count": 0,
  "value_mismatch_count": 314,
  "dtype_cast_value_mismatch_count": 0,
  "unexpected_count": 0,
  "max_abs": 0.007767438888549805,
  "dtype_cast_max_abs": 0.0,
  "mismatch_count": 314
}
```

Prefix별 breakdown:

| Prefix | Checkpoint tensors | Checked | Missing | Shape mismatch | Raw value mismatch | Dtype-cast mismatch |
|---|---:|---:|---:|---:|---:|---:|
| `backbone.eagle_model.vision_model.` | 448 | 448 | 0 | 0 | 0 | 0 |
| `backbone.eagle_model.mlp1.` | 2 | 2 | 0 | 0 | 0 | 0 |
| `action_head.` | 314 | 314 | 0 | 0 | 314 | 0 |

Action head raw mismatch는 fp32 checkpoint tensor와 bf16 loaded model tensor를 직접 비교해서
생긴 차이다. Checkpoint tensor를 loaded model tensor dtype으로 cast하면 모든 action-head
tensor가 정확히 맞는다.

## 보존하지 않는 과거 근거

진단 과정에서 제거한 one-off probe들은 OpenFridge 첫 observation을 native/LeRobot boundary에서
비교했다. 최종 기록은 아래와 같다.

- 저장된 fixture key 기준 `GR00T prepare_input` boundary가 맞았다
  (`backbone_input__*`, `action_input__*` 모두 `max_abs=0.0`; bool mask mismatch 0).
- 선택한 parameter fingerprint가 정확히 맞았다.
- 첫 observation action chunk 차이는 bf16-scale increment 수준이었다
  (`flat_max_abs=0.00390625`; gripper/control mode exact).
- activation tensor는 bit-exact가 아니었으므로 full activation parity는 주장하지 않았다.

해당 scripts는 의도적으로 보존하지 않는다. 유지하는 regression surface는 위의
checkpoint-load verifier와 adapter state/action contract unit test다.

## `mlp1`만으로 단정하지 않는 이유

`backbone.eagle_model.mlp1`는 구체적으로 검증된 mismatch였다. Raw Isaac-GR00T checkpoint는
이 tensor를 `backbone.eagle_model.mlp1.*` 아래에 저장하지만, LeRobot wrapper는 해당 module을
`policy._groot_model.backbone.eagle_model.mlp1`로 노출한다. Repo-local adapter는 이제 이 두
tensor를 명시적으로 로드하고 검증한다.

하지만 이것이 `mlp1`이 유일한 문제였다는 뜻은 아니다. 현재 evidence는 아래까지다.

- `mlp1`은 이제 정확히 로드된다.
- 선택한 vision tower weight가 정확히 로드된다.
- action head key/shape는 로드됐고 dtype-cast 기준으로 동등하다.
- 과거 first-observation input/action evidence는 일관되게 보였다.
- 저장된 dump에서 activation tensor는 bit-exact가 아니었다.

따라서 올바른 결론은 이렇다. `mlp1`은 확인된 high-impact loader gap 중 하나였고, 이를
수리한 뒤 관측 가능한 action mismatch는 큰 semantic mismatch에서 작은 bf16-scale 차이로
줄었다. Runtime activation parity에는 아직 추가 차이가 남아 있을 수 있다.

## 검증 명령

재사용 checker unit test:

```bash
python -m pytest tests/test_lerobot_groot_n15_internal_parity.py -q
```

결과:

```text
5 passed
```

Target env 체크포인트 로드 verifier:

```bash
docker exec temporal_vla-lerobot-run-0b327aff8915 bash -lc '
cd /temporal_vla &&
PYTHONPATH=/temporal_vla/scripts/eval:/temporal_vla/scripts:/temporal_vla/scripts/utils:/temporal_vla/scripts/serve:/temporal_vla/lerobot/src:/temporal_vla \
python scripts/eval/lerobot_groot_n15_internal_parity.py --device cpu \
  --output outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json'
```

결과:

```text
checked_count=764
mismatch_count=314
dtype_cast_value_mismatch_count=0
```
