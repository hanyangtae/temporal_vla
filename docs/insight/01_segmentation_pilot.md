# INSIGHT VLM primitive-segmentation 파일럿 — 방법 & 현황

INSIGHT(Steerable VLAs, Stanford)의 **Stage 1 자동 primitive segmentation**을 우리 robocasa
GR00T rollout 에 적용해 보는 소규모 파일럿. "우리 데이터에서 VLM 분해가 얼마나 되나"를 실측한다.

- 단일 method 출처(상위): [`../references/`](../references) 의 `Steerable VLAs.pdf`, 코드 출처 = github.com/insight-vla/insight (Apache-2.0).
- 자매 문서: progress 예측 설명/재현성은 [`02_progress_prediction.md`](02_progress_prediction.md).

## 핵심 아이디어 (INSIGHT Stage 1)

primitive 단위 데이터가 **처음엔 없다**. INSIGHT는 사람 full-task demo 를 VLM 으로 **찢어서**
primitive 라벨 trajectory 를 만든다(수집 아님, relabel/split). 3단계:
1. VLM 이 task 설명 → 정렬된 primitive sequence 생성(예시 primitive 로 granularity 고정).
2. subsample 한 video 를 프레임별로 보며 각 프레임을 primitive 에 배정 + **per-frame EE motion 캡션**
   (dominant 축 dx/dy/dz/회전)과 교차검증 → primitive 사이 boundary frame 반환.
3. 각 boundary 를 EE delta 변화점 + 시각적으로 명확한 가장 이른 프레임으로 미세조정.
- gripper task 면 gripper open/close 명령 속도로 경계를 추가 시드.

우리는 **expert teleop 이 아니라 VLA rollout(실패 포함)** 을 찢는다는 차이가 있다(아래 caveat).

## 우리 데이터에서의 현실 (실측)

`outputs/eval/robocasa/groot_n15/.../*.{pkl,mp4,csv}` triplet 기준:

| 항목 | 상태 |
|---|---|
| video(프레임) | ✅ mp4 = **3뷰 가로결합 768×256 = [side_0\|side_1\|wrist_0]**, cv2 로 디코드(ffmpeg 불필요) |
| task 언어 | ✅ `ep_meta['lang']` / `task_description` |
| 성공 플래그 | ✅ `episode_success` (episode 단위) |
| EE pose | △ **상대(base 기준) `eef_pos_rel`/`eef_quat_rel`** 만 (`states` 있을 때). 절대 EE pose 없음 → EE-caption 은 relative 로 근사 |
| gripper | △ `action_vectors[:,6]`. **robocasa kitchen(서랍/냉장고 등)은 grasp 가 없어 거의 상수** → gripper-velocity 경계 무력 |

→ **결론**: robocasa kitchen task 는 INSIGHT 의 gripper-segmentation 가정이 안 맞는다.
**video-VLM 라벨링(+ EE/state changepoint)** 경로가 메인이다. grasp 가 있는 **PickPlace*/CoffeeSetupMug**
류만 gripper 경로도 의미가 있다(`has_gripper=True` 로 자동 판별됨).

## 파이프라인 (구현)

코드: `scripts/analysis/insight_seg/` (env = `~/miniconda3/envs/lerobot_safe/bin/python`).

```
rollout_adapter.load_episode   # 우리 pkl+mp4 -> EpisodeData(frames/뷰, ee_pos, ee_delta, gripper, has_gripper, task, success)
  -> segmentation.segment_episode(ep, vlm, SegConfig)   # INSIGHT 이식: video-VLM 라벨링 + EE 캡션 + state changepoint 보정
      -> segments.json (+ annotated.mp4)  per rollout
          -> run_pilot.py 가 집계 -> summary.json / summary.tsv
```

- `vlm_client.get_vlm_client()`: `GEMINI_API_KEY`(env) 또는 `~/.config/temporal_vla/gemini_api_key`
  파일이 있으면 **Gemini(google-genai)**, 없으면 **MockVLMClient**(플레이스홀더, plumbing 검증용).
  기본 모델 `gemini-2.5-flash`(논문은 gemini-3-flash; `--model` 로 override).
- segmentation/prompts 는 INSIGHT `densely_label_dataset.py` + `src/insight/prompts.py` 를 출처주석과 함께 이식.

## 현황 (실 VLM 실행 완료)

- ✅ 전체 파이프라인 구현 + **mock 7개**(`pilot00_mock`)로 plumbing 검증.
- ✅ **실제 Gemini(`gemini-2.5-flash`) 분해 10 rollout 완료** — 3 task(coffee_setup_mug grasp /
  close_fridge no-grasp / turn_on_electric_kettle) × 성공·실패 섞음. **10/10 full-coverage**,
  segment 2~7개. 키는 repo `.env`(`GEMINI_API_KEY`)에서 source. 결과: `outputs/analysis/insight_seg/pilot01/`.
  annotated.mp4 에 primitive 라벨 + frame counter + task 가 번인됨(확인).
  ```
  set -a; source .env; set +a
  ~/miniconda3/envs/lerobot_safe/bin/python scripts/analysis/insight_seg/run_pilot.py \
    --root outputs/eval/robocasa/groot_n15/coast4_instruction_pathway_50ep/raw_rollouts \
    --task coffee_setup_mug:2:2 --task close_fridge:2:1 --task turn_on_electric_kettle:2:1 \
    --no-mock --out outputs/analysis/insight_seg/pilot01
  ```

## 평가 결과 (pilot01)

**1. pick-place 류는 INSIGHT 분해가 거의 교과서적으로 됨.** coffee_setup_mug:
`move gripper to object → close gripper → lift upward → move gripper to target → lower gripper → open gripper`
— INSIGHT 의 6-primitive pick-place 시퀀스와 일치. 짧은 primitive(close gripper=2 step)도 잡음.

**2. robocasa articulated/kitchen 은 known-primitive 어휘 bias 가 보임.** close_fridge 를
`move gripper to object → close gripper → pull object` 로 라벨 — 냉장고 닫기는 grasp 없고 "push" 인데
기본 어휘가 grasp 쪽으로 치우쳐 의미가 느슨함. → **도메인 맞춤 primitive 어휘**(`SegConfig.known_primitives`)로
교정 필요(예: "approach handle", "push door closed"). kettle 은 `... → push object` 로 lever 누르기를 합리적으로 잡음.

**3. ★ 실패 rollout 의 phase-localized stall 이 segmentation 으로 드러남(메인 method 와 직결).**
성공은 짧고 깔끔, 실패는 항상 44 step(=timeout)인데 **어디서 멈췄는지가 segment 에 보인다**:
- kettle FAIL: 44 step 중 **38 step 을 `move gripper to object`(approach)** 에 소모 → lever 도달 실패.
- coffee FAIL: 끝에서 `open gripper → move gripper to object` 로 **재접근 loop** (재시도하다 timeout).

즉 segmentation 이 "실패가 **어느 phase 에서** 났나"(approach 고착 vs 말단 retry-loop)를 라벨로 노출 →
이는 우리 메인 method 의 **online phase / failure-type 식별**(가장 중요한 미해결 문제) 타당성에 직접 연결되는
긍정적 신호다. 단 현재는 offline(전체 trajectory 본 뒤) 분해라, online 화는 별도 과제.

**4. 길이 confound 가 분해에도 그대로.** 성공=조기종료(14~33 step)/실패=44(timeout)
([[seen18-rollout-length-confound]]). segment 길이·개수 해석 시 길이 통제 필수.

### 요약 (pilot01 summary.tsv)

| task | grasp | 성공 분해 | 실패 분해 특징 |
|---|---|---|---|
| coffee_setup_mug | ✅ | 6-primitive pick-place(교과서) | 말단 재접근 loop, 44 timeout |
| turn_on_electric_kettle | ✅ | approach→push(→lift) | approach 에 38/44 고착 |
| close_fridge | ✗ | approach→(close)→pull(어휘 bias) | open gripper/재이동 추가, 44 timeout |

## 정정 — "expert vs rollout" 은 분해 품질의 driver 가 아니다

처음엔 pilot02(expert)가 pilot01(rollout)보다 "잘 나뉜다"고 봤으나, 이는 **교란된 비교**였다.
두 pilot 은 source(rollout/expert)뿐 아니라 **task(혼합 vs PickPlace-only)** 와
**성공/실패(혼합 vs 전부성공)** 가 동시에 다르다. 입력을 직접 까보면:

| | exterior view | VLM keyframe | task |
|---|---|---|---|
| ROLLOUT coffee 성공 | 256×256 | 24장(video 128f) | ✓ |
| ROLLOUT coffee 실패 | 256×256 | 24장(video 350f) | ✓ |
| EXPERT PickPlace | 256×256 | 24장(video 252f) | ✓ |

**VLM 입력 종류·해상도·뷰·keyframe 수가 동일.** rollout 도 control-step 당 ~8 video frame 으로 촘촘하다
(n_steps=16 은 control-step 수일 뿐, video 는 128 frame). apples-to-apples 로 맞추면 — **성공 + pick-place
rollout 도 expert 와 똑같이 교과서 분해**가 된다:
- ROLLOUT coffee 성공(ep24,16step): `move→close→lift→move→lower→open`
- EXPERT PickPlace(ep0): `move→close→lift→move→lower→open` (동일)

→ 분해 품질의 실제 driver 는 **source 가 아니라**: (1) **task↔`known_primitives` 어휘 매칭**
(close_fridge 의 "close gripper/pull object" 헛라벨 = grasp 어휘 강제 → config 문제), (2) **성공/실패**
(실패의 "지저분함"은 분해 오류가 아니라 로봇이 실제로 loop/stall 한 걸 충실히 반영). 유일한 진짜 입력
차이는 **경계 정밀도 단위**(성공 rollout 은 step 16개라 경계 양자화 거칠고 expert 252개라 촘촘 — 라벨 품질엔 무관).

## expert LeRobot demos (pilot02) — 깨끗한 분해 품질용 데이터원

위 정정대로 source 자체가 우월한 건 아니지만, expert demo 는 **"task 매칭 + 전부성공 + 긴 궤적"
세 조건을 항상 만족**해 분해 품질 평가에 편하다. 우리 레포 cache 에 있다:

`~/.cache/temporal_vla/datasets/robocasa/v1.0/pretrain/atomic/<Task>/<date>/lerobot/`
(GR00T 학습에 쓰인 robocasa v1.0 pretrain. 10 task: PickPlace×2, Open/Close articulated×8.)

편의 이점(분해 품질의 driver 는 아님 — 위 정정 참조, 단지 깨끗한 조건을 항상 만족):
- **전부 성공·깨끗** — 궤적에 primitive 가 실제로 다 들어있음(실패 loop/stall 없음).
- **task 가 pick-place** — 기본 `known_primitives` 어휘와 자동 매칭.
- **뷰 분리 mp4**(agentview_left/right, eye_in_hand 각 256×256) — 768 가로결합 해체 불필요(편의).
- **경계 정밀도**: frame=step 1:1, ~150-274 step → 경계가 촘촘(짧은 성공 rollout 의 거친 양자화 대비; 라벨 품질 자체는 동일).
- 보너스: 데이터에 **task-level `progress` 컬럼**이 이미 있음(0→1 단조) → [`02_progress_prediction.md`](02_progress_prediction.md) 참고.

어댑터: `lerobot_adapter.py` (`run_pilot.py --lerobot-root <atomic_base> --lerobot-task PickPlace --episodes auto:3`).

**결과(pilot02, PickPlace expert 6 demo)**: 6/6 모두 교과서적 pick-place 분해로 수렴
(`move→close→lift→move→lower→open`, 경계도 phase 정렬). 단 이 "일관성"은 expert 라서가 아니라
**task(PickPlace)+outcome(성공)을 통제**했기 때문 — 같은 조건의 성공 pick-place rollout 도 동일하게 수렴한다.
(`outputs/analysis/insight_seg/pilot02_lerobot_expert/`)

- 잔여 artifact: 매 demo 끝에 release 후 home 복귀 동작이 7번째 segment("move..."/"lift upward")로 한 번 더 붙음(경미; 프롬프트 Rule 또는 trim 으로 정리 가능).

→ **권고**: INSIGHT segmentation 의 "분해 품질" 평가는 **이 expert LeRobot demo 로**, "실패가 어느
phase 에서 나나"(메인 method online phase 식별)는 **rollout(실패 포함)으로** — 두 데이터원을 용도 분리.
articulated kitchen task 는 양쪽 다 `SegConfig.known_primitives` 를 도메인 어휘로 교정 후 사용.

## 주의/한계

- robocasa kitchen 은 INSIGHT 의 tabletop pick-place/pour 와 도메인이 달라 primitive 어휘·경계 의미가 흐릴 수 있음(video-VLM 위주).
- 절대 EE pose 미저장 → EE-caption boundary refinement 는 relative 근사(또는 생략). 필요 시 수집 때 `states` 항상 기록.
- frame↔step 정렬은 근사(`frame_to_step = round(f·n_steps/T)`).
- VLM 비용/레이트리밋 → 파일럿 규모(~7-10 rollout)로 제한.
