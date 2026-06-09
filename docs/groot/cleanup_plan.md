# GR00T / SAFE 코드 정리 계획

> 🛠️ **작업용 문서 (working doc)** — groot reading order(번호 prefix)에 포함하지 않는다.
> 정리 실행 후 폐기하거나 `_legacy/`로 옮긴다. README 인덱스에 등록하지 않는다.

작성: 2026-06-10. 목적: GR00T / SAFE 관련 코드의 구조·중복·dead-code 정리.

## 범위 (2026-06-10 확정)

- **대상(in-scope)**: GR00T(N1.5/N1.6) + SAFE + serving + processor + adapter 코드 **전부**
  (본인 do-dong-park / 동료 hanyangtae 소유 무관).
- **제외(steering 연구 라인만)**: `src/conceptor/*`,
  `scripts/safe/groot_n16/robocasa/steer/*`, `scripts/safe/groot_n16/robocasa/analyze/*`
  (vl_dit / pathway / LSTM detector **연구 분석** 스크립트), `scripts/serve/steering_hooks.py`,
  `docs/steering/*`. → COAST / conceptor / hook 만 제외. **SAFE 는 steering 과 무관하므로 포함**
  (`safe_hooks.py`, `feature_server.py`, `vis/`, collect, split 모두 정리 대상).

> 동료 파일도 정리 허용됨(사용자 승인). 단 큰 구조 변경(Tier 2~3)은 **import 파급이 동료 코드에
> 닿으므로 실행 전 hanyangtae 와 공유** 권장. Tier 1 은 기계적·국소라 단독 진행 가능.

## 이번 적용 목표 — N1.5 RoboCasa 집 만들기

사용자 결정: `lerobot_groot_n15` / GR00T N1.5 RoboCasa 파일이 흩어지지 않게
`scripts/safe/groot_n15/robocasa/`를 canonical home으로 쓴다. `scripts/safe/groot_n16`와
같은 축(`model/benchmark/function`)으로 맞춘다.

- Canonical tree: `scripts/safe/groot_n15/robocasa/{eval,split,utils}`.
- `scripts/eval`, `scripts/utils`, `scripts/data` 아래의 N1.5 RoboCasa 파일은 제거하고
  compatibility wrapper는 두지 않는다.
- `scripts/serve/lerobot_adapters/groot.py`는 shared LeRobot serving adapter surface라서 유지한다.
  대신 runtime helper만 `scripts/safe/groot_n15/robocasa/utils/runtime.py`로 이동한다.
- `docs/groot/cleanup_plan.md`를 포함한 `docs/groot` 문서는 새 경로만 active runbook으로 쓴다.
- 회귀 방지: `tests/test_groot_n15_layout.py`로 canonical 파일 존재와 old path 부재를 검증한다.

## 왜 (배경)

코드 리뷰 진단: 핵심 추상화(`service`/`robocasa_io`/`safe_features`)는 재사용 양호하나, 주변에
① split 로직 복제, ② processor 회전변환 중복, ③ dead-code, ④ `src/policies/groot` flat·혼재,
⑤ `vis/` 반쪽 리팩터, ⑥ N1.5 GR00T 산재, ⑦ 폴더 구성 축 불일치가 있다. `좀 정리` 수준이므로
**Tier 1(저위험 기계적)** 부터 하고, 구조 변경은 deliberate 하게.

---

## Tier 1 — 저위험·기계적, 지금 실행

### T1-A. split 공통 로직 추출 (최우선)
- **문제**: `groot_n15/.../prepare_seen5_trainval_cp_test_split.py`(210줄)와
  `groot_n16/.../prepare_seen4_unseen2_split.py`(292줄)가 `ROLLOUT_RE`, `class RolloutFile`,
  `parse_rollout`, `collect_rollouts`, `link_rollout_files`, `symlink_relative`,
  `safe_remove_split_root`, `count_success` 등 **8+ 심볼 복제**.
- **조치**: 공통 심볼 → 신규 `scripts/safe/_common/split_lib.py`. 두 스크립트는 split별
  TASKS/비율만 보유.
- **blast radius**: 두 importer 모두 본인 소유 + 신규 파일 → 기존 import 0 영향.
- **검증**: 추출 후 두 스크립트 산출(파일 목록/symlink) 기존과 diff 0.

### T1-B. dead-code 제거 (검증 완료)
- **`src/policies/groot/core/schema.py:23` `UNIFIED_TO_VIDEO_KEY`** — repo 전체 참조 0 (정의뿐). 삭제.
- **`scripts/serve/groot.py`의 미사용 헬퍼 3개** — `_language_keys_from_modality_config`,
  `_convert_native_action_to_subkeys`, `_feature_config`. 모두 호출부 0 (service 메서드에 가려짐). 삭제.
- **(audit) `src/policies/groot/core/preprocess.py`** — 직접 importer grep 0건. relative import(`from .`)
  가능성 있으니 **삭제 전 패키지 내부 사용 재확인**. 진짜 미사용이면 제거.
- **blast radius**: 삭제 대상 모두 미참조 확인됨 → 0 (preprocess 만 확인 후).
- **검증**: 삭제 후 `serve/groot.py` import/기동 smoke, `python -c "import src.policies.groot.core.schema"`.

### T1-C. processor 회전변환 중복 제거
- **문제**: `_rot6d_to_euler`, `_quat_to_euler` 가 `src/processor/action/calvin.py`(L184,L220)와
  `src/processor/action/robocasa.py`(L184,L202)에 **중복 정의**.
- **조치**: 두 구현이 동일한지 먼저 확인 → 동일하면 `src/processor/action/_rotation.py`(신규)로
  추출, 양쪽 import. 미세 차이 있으면 통합하지 말고 주석으로 사유 명시.
- **blast radius**: 두 파일 모두 in-scope(non-steering). calvin/robocasa action pipeline.
- **검증**: 기존 단위테스트(`tests/test_processor.py`) 통과 + 변환 수치 동일성 spot-check.

### T1-D. N1.5 vs N1.6 범위 명시 (docstring, 이동 없음)
- `src/policies/groot/__init__.py`: "이 패키지 = GR00T **N1.6** core. N1.5 LeRobot 경로는
  `scripts/serve/lerobot_adapters/groot.py`." 명시.
- `scripts/serve/lerobot_adapters/groot.py`: "GR00T **N1.5** via lerobot 서브모듈." 명시.
- **blast radius**: 0.

---

## Tier 2 — 구조 변경(이동/리네이밍), import 파급 게이트 — deliberate, 동료 공유 권장

### T2-A. `src/policies/groot` 정리 (N1.6 명확화 + 관심사 그룹핑)
- **문제**: flat 10파일에 model-core(`loader/schema/preprocess/service/rng`) + robocasa 전용
  (`robocasa_io/robocasa_env_wrappers/scenario_replay`) + SAFE(`safe_features`) 혼재. 사실상 N1.6.
- **조치(안)**: `robocasa/`, `safe/` 서브패키지로 그룹 또는 헤더 정리. 물리 이동 선택 시 importer 갱신:

  | 모듈 | importer 수 | 이동 시 갱신 대상 |
  |---|---|---|
  | `robocasa_io` | **8** | processor(4: __init__,obs,action,factory), collect(3), eval(1) |
  | `scenario_replay` | 4 | collect(3), robocasa_eval |
  | `safe_features` | 3 | feature_server, collect_policy_clients, serve/groot |
  | `schema` | 2 | collect_policy_clients, serve/groot |
  | `service`/`robocasa_env_wrappers`/`loader`/`rng` | 1~2 | 각 해당 |

- **권장**: `robocasa_io`(8곳)·`scenario_replay`(4곳)는 churn 크고 동료 파일 포함 → 물리 이동은
  **마지막에, 한 commit**으로. 우선은 **헤더/docstring 정리 + import alias 유지**로 비용 최소화.
- **검증**: 이동 시 전체 import smoke (`serve/groot`, `feature_server`, collect, eval) + 단위테스트.

### T2-B. `scripts/safe/.../vis/` 반쪽 리팩터 통합
- **문제**: 모던 `vis/{core,analyses}` 플러그인 구조와 레거시 standalone ~12개
  (`compute_*`,`plot_*`,`cluster_static`,`progress_evolution`,`run_feature_visualization` 등,
  각자 plotting 복붙)가 공존. SAFE detector/feature 시각화 → in-scope.
- **조치**: 레거시 스크립트를 `analyses/`(NAME/HELP/add_args/run) + `core/` 기초층으로 점진 이주.
  재사용 빈도 높은 것부터(silhouette, online_detection, conformal).
- **blast radius**: vis/ 내부 한정(외부 import 거의 없음). 큰 노력 → 점진.
- **검증**: 이주 전후 동일 입력으로 plot/metric 산출 동일성.
- ⚠️ vis 스크립트 중 steering/pathway 결과 전용 시각화가 섞여 있으면 그건 제외.

### T2-C. N1.5 GR00T 산재 정리
- **문제**: N1.6 core 는 `src/policies/groot/`로 모였으나 N1.5 RoboCasa helper는 여러
  script tree에 산재했다. "집이 없다."
- **확정 조치**: N1.5 RoboCasa 파일은 `scripts/safe/groot_n15/robocasa/` 아래로 이동한다.
  - `eval/`: `native_official_zmq_eval.py`, `native_zmq_eval.py`, `lerobot_http_eval.py`,
    `internal_parity.py`, `run_target15_seedpairs.sh`
  - `utils/`: `runtime.py`, `prepare_base_new_embodiment.py`
  - `split/`: `prepare_seen5_trainval_cp_test_split.py`, `build_safe_splits.py`,
    `merge_seen60_source.py`
  - `README.md`: N1.5 RoboCasa local index
- **게이트**: `lerobot_adapters/groot.py` 는 **동료 `serve/lerobot.py`가 패키지로 import**
  (`lerobot.py:47,53`) → shared serving adapter 위치는 유지한다. eval/utils/split 파일만 safe tree로
  모은다.
- **검증**: shell wrapper, tests, docs 8개(`n15_02/03/04/05/07/08`, flow_map,
  steering_explorer.html)와 `cleanup_plan.md` 경로를 동시 갱신한다.

---

## Tier 3 — 폴더 구성 축(axis) 통일

- **문제**: 트리마다 축이 다름 — `src/policies/groot`(model flat), `src/processor`(function→bench),
  `scripts/serve`(model file), `scripts/safe`(model→bench→function).
- **전체 적용 시에도 "그냥 적용" 불가 — 두 제약 (아래 "전체 적용 모드" Hard limit 2 참조)**:
  1. **목표 taxonomy 가 미정** = 설계 결정. 기계적 적용 대상이 아니다.
  2. `scripts/safe` 축을 바꾸려면 `groot_n16/robocasa/{steer,analyze}`(steering **제외 영역**)를
     같이 재배치해야 함 → **steering 제외 원칙과 충돌**.
- **이번 적용에서 실제로 하는 것**: steering 을 건드리지 않는 부분 정렬만 — `scripts/safe/_common`
  신설(T1-A), `src/policies/groot` 그룹핑(P1), N1.5 집 만들기(T2-C). **repo 전역 단일 taxonomy 는
  보류**(별도 설계 + hanyangtae 합의).

---

## 폴더 구조 / 파일명 제안

진단의 구조 이슈(flat·혼재, N1.5 산재, SAFE 공용 부재)에 대한 **구체적 target**. 각 항목에
risk/tier 표기. import 파급이 in-scope 안에서 닫히는 것만 "clean", steering importer 를 건드리면
"blocked".

### P1. `src/policies/groot/` flat → 관심사 그룹핑  (Tier 2, churn 큼)

```
src/policies/groot/                  src/policies/groot/
  __init__.py                          __init__.py        ← "N1.6 core" 명시
  loader.py                            core/
  schema.py                              loader.py
  preprocess.py            ─────▶        schema.py
  rng.py                                 preprocess.py
  service.py                             rng.py
  robocasa_io.py                         service.py
  robocasa_env_wrappers.py             robocasa/
  scenario_replay.py                     io.py            (← robocasa_io.py)
  safe_features.py                       env_wrappers.py  (← robocasa_env_wrappers.py)
                                         scenario_replay.py
                                       safe/
                                         features.py      (← safe_features.py)
```
- **rename 파급**: `robocasa_io`(importer 8), `scenario_replay`(4), `safe_features`(3) 등 →
  processor·collect·serve·eval 갱신 필요(전부 in-scope, 동료 파일 포함). **한 commit, 마지막에.**
- **대안(저비용)**: 물리 이동 보류하고 `__init__.py` 에 grouping 의도만 문서화 → 비용 0.
  실익(탐색성) 대비 churn 이 커서 **권장: 대안부터, 이동은 동료 합의 후**.

### P2. serve adapter 위치 — 유지하고 helper만 이동  (Tier 1, clean)

- `scripts/serve/lerobot_adapters/groot.py`는 shared LeRobot adapter registry의 public surface라
  위치/파일명을 유지한다.
- N1.5 runtime helper는 `scripts/safe/groot_n15/robocasa/utils/runtime.py`로 이동하고, adapter는
  해당 파일을 path-based loader로 가져온다.
- N1.6 HTTP 서버 `scripts/serve/groot.py` 는 그대로 두되 모듈 docstring 에 "N1.6" 명시(T1-D).

### P3. `scripts/safe/` 공용 레이어 신설  (T1-A 는 clean / 일부 blocked)

```
scripts/safe/
  _common/                ← 신규 (cross-model 공용)
    split_lib.py          ← T1-A (clean: importer 둘 다 in-scope)
  groot_n15/robocasa/split/...
  groot_n16/robocasa/{collect,serve,split,vis,train}
  lerobot/
```
- **`safe_feature_vectors.py` 를 `_common/` 로 올리는 것은 BLOCKED**: importer 에 `steer/`,
  `analyze/diagnose_*`, `src/conceptor`(steering 제외 영역)가 포함 → 이동하면 제외 파일을 수정해야
  함. **제자리 유지**. (lerobot 재사용은 collect_common.py 가 이미 우회 중.)

### P4. eval N1.5 파일명 — safe tree 안에서 역할 기반 이름 사용

- `eval/native_zmq_eval.py`: N1.5 ZMQ protocol + repo rollout helper.
- `eval/native_official_zmq_eval.py`: benchmark-style official RoboCasa env + N1.5 ZMQ.
- `eval/lerobot_http_eval.py`: benchmark-style official RoboCasa env + LeRobot HTTP `/act`.
- `eval/internal_parity.py`: checkpoint/model parity diagnostic.

### P5. processor 회전 util 위치  (T1-C 와 동일)

- `_rot6d_to_euler`/`_quat_to_euler` 중복 → `src/processor/action/_rotation.py` 신설로 단일화.

> **요약**: 저비용 clean = P2(adapter 유지 + helper 이동), P3 의 split_lib, P4(eval 역할명), P5. 큰 것 = P1
> (groot 그룹핑, churn 큼 → 합의 후). P3 의 feature_vectors 이동·P1 의 물리 이동은 steering/동료
> 파급 때문에 **신중**.

---

## 코드 중복 / 재사용성 제안 (lerobot + groot 스택 한정)

지금 주의 깊게 보는 **lerobot · groot 서빙/어댑터 스택** 안의 중복·reuse 만 다룬다.
xvla/dreamvla/upvla/openvla_oft 광범위 리팩터는 이번 범위 밖(다만 RU1·RU2 는 그들도 나중에
채택 가능하게 설계).

| # | 중복/기회 | 위치 | 제안 | tier/risk |
|---|---|---|---|---|
| **RU1** | base64 이미지 디코드 복붙 | `serve/lerobot.py:100`(인라인) vs `src/policies/groot/core/preprocess.py:42 decode_b64_image`(헬퍼) | groot 의 `decode_b64_image` 를 공용 위치(예: `scripts/serve/_imgutil.py`)로 올려 **groot·lerobot 둘 다 사용**. groot 은 service 경유라 무변경에 가깝고 lerobot 인라인만 교체 | Tier 1, 낮음 |
| **RU2** | 회전/쿼터니언 헬퍼 분산 | `serve/lerobot.py:110 _quat_xyzw_to_axisangle` + `lerobot_adapters/groot.py:106 _quat_wxyz_to_rotation_6d` | 한 모듈(`lerobot_adapters/rotation.py`)로 **모아 두기**. 입력 format(xyzw/wxyz)·출력(axisangle/6d)이 달라 함수는 별개로 두되 한 곳에서 발견 가능하게 | Tier 1, 낮음 |
| **RU3** | **SAFE feature 캡처 평행 중복** | groot `safe_features.py:SafeFeatureExtractor` vs lerobot `serve/safe_hooks.py:SafeFeatureCapture` | 둘 다 "추론 1회 pre-velocity feature 1점" 개념 + `_pre_hook`/`_post_hook`/`assemble` 동형 + `feature_kind`/`feature_axes` 평행. **캡처 lifecycle/contract 를 공유 base 로** 추출하고 groot=DiT block, lerobot=action_expert 만 specialize. (모델 내부 hook 지점이 달라 완전 통합은 불가) | Tier 2, 중간 — **lifecycle 동일 형태인지 먼저 확인** |
| **RU4** | `/health`·`/reset` 핸들러 골격 복붙 | `serve/groot.py` ↔ `serve/lerobot.py` | 두 서버만 쓰는 작은 헬퍼(`/health` 응답 빌더 + `/reset` no-op 패턴)로 공유. profile→health 필드는 이미 계약 있음 | Tier 2, 낮음 |

**RU3 보조 — 네이밍 계약 통일(저비용)**: 캡처 base 추출이 무겁다면, 최소한 `feature_kind`/
`feature_axes` 생성 규칙(`*_pre_velocity` + axes 이름)만 공통 함수로 빼서 두 경로가 같은 계약을
만들게 한다. (`01_serving_interface.md` 의 `/health` feature 메타와도 정합.)

### 이미 잘 되는 reuse (건드리지 말 것)
- `lerobot_adapters/common.py` (`STATE_DIM`, `preprocess_image_numpy`, `load_dataset_stats`)
  — `lerobot.py` + 어댑터들이 공유. 양호.
- `serve/groot.py` 가 `GrootPolicyService`(service.py) 를 **얇게 위임** — 좋은 구조, 유지.
- `SafeFeatureExtractor` 가 groot **HTTP serve + ZMQ feature_server** 공유 — 양호(이건 groot 내부).

> **lerobot+groot reuse 우선순위**: RU1·RU2 (저위험, 지금) → RU3 의 네이밍 계약 통일 → (여유 시)
> RU3 캡처 base / RU4. RU3 의 캡처 base 는 두 SAFE 경로를 동시에 건드리니 신중.

---

## 제외 (steering — 손대지 않음)

`src/conceptor/*`, `scripts/safe/groot_n16/robocasa/{steer,analyze}/*`,
`scripts/serve/steering_hooks.py`, `docs/steering/*`. (COAST/conceptor/pathway/steer-eval.)

---

## ⚠️ 전체 적용(Tier 1~3) 모드 — 결정 확정 & 한계

Tier 3까지 전부 적용하기로 함. 이전의 "either/or·deliberate·협의 후" 망설임을 아래로 확정한다.
단 **두 hard limit 은 전체 적용에서도 깨지지 않는다.**

### 확정 결정 (무거운 쪽 채택)
- **P1 = T2-A**: `src/policies/groot` **물리 이동**(core/robocasa/safe). importer 전부 갱신, 한 commit.
- **T2-C**: N1.5 RoboCasa 파일을 `scripts/safe/groot_n15/robocasa/`로 물리 정리.
  shell 1 + docs 8 + `cleanup_plan.md` 동시 갱신.
- **RU3**: 네이밍 계약 통일 **+** 캡처 base 추출 둘 다.
- **중복 항목은 한 번만**: P1≡T2-A, P3 split_lib≡T1-A, P5≡T1-C. 아래 실행 순서 기준으로만 수행.

### Hard limit 1 — steering 제외 절대
- `safe_feature_vectors.py` `_common/` 이동 **금지**(importer 에 steer/analyze/conceptor). 제자리 유지.

### Hard limit 2 — Tier 3 축 통일 부분만
- repo 전역 단일 taxonomy 는 (a) 목표 미정 + (b) `scripts/safe` 재배치가 steer/analyze(제외)를
  건드림 → **보류**. 실제 적용은 steering 안 건드리는 부분 정렬(_common, groot 그룹핑, N1.5 집)만.

## 실행 순서

1. **T1-A** split 공통 모듈 `_common/split_lib.py` (= P3). 위험 0.
2. **T1-B** dead-code 제거(schema `UNIFIED_TO_VIDEO_KEY`, serve/groot 헬퍼 3 / preprocess 확인 후).
3. **T1-C** processor 회전 dedup(= P5) + **RU2** serve 스택 회전 헬퍼 한 모듈로.
4. **RU1** base64 디코드 공용화(groot↔lerobot) + **RU3 네이밍 계약 통일**.
5. **T1-D** N1.5/N1.6 헤더 + **P2** adapter 위치 유지/helper 이동 + **P4** eval 역할명 정리.
6. **T2-A(=P1)** groot 물리 그룹핑 → **T2-C** N1.5 정리(shell+docs 8+cleanup_plan) → **T2-B** vis 이주 →
   **RU3 캡처 base / RU4**. **각 단계 후 import smoke + `tests/` 실행** gate.
7. **Tier 3**: 위에서 흡수한 부분 외 repo 전역 축 통일은 **보류**(Hard limit 2).

> 각 항목 별도 commit. 한글 메시지 `refactor:`. 브랜치 `refactor/groot-safe-cleanup` 분기.
> 동료 소유 파일 수정 commit 은 PR 본문에 사유·blast-radius 명시. 전체가 큰 PR이므로 tier 경계마다
> import smoke + 단위테스트로 회귀 확인.
