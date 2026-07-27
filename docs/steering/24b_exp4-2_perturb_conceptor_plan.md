# exp4-2 실행 계획 — perturbation 유도 실패 생성 → conceptor (확정판)

작성 2026-07-21, **확정 2026-07-22** (Codex Gate1 반영 + 사용자 최종 결정. 원장: `docs/collab/2026-07-22-exp4-plan-gate1.md`). 공유 배경·arm 정의·게이트는 [`24_exp4_shared_plan.md`](24_exp4_shared_plan.md) (이하 "공유문서") — 반드시 먼저 읽을 것.

확정 스코프 주석: exp4 task 4종(OpenStandMixerHead/OpenDrawer/ppcc_bread/ppcc_beer — 07-22 재결정, CloseFridge 탈락) 중 exp4-2 파일럿·본수집은 ppcc_bread 중심, P2 확장 후보가 나머지 3종. **섭동 대상 seed도 scene feasibility 필터 선행**(공유문서 §5) — 기하 불가 scene에 섭동을 주면 그 실패는 latent가 아니라 기하라 fail 클래스를 오염시킴. steering 축의 cross-instruction 유예는 **fit/eval 축** 기준이며, B1(타 instruction donor 주입)은 실패 *생성* 메커니즘이라 유지 — 단 사용자가 B1도 유예를 원하면 P0에서 제외 가능(스모크 비용만 소액).

## 목표·가설

자연(내재적) 실패 대신 **만들어낸 실패**로 succ/fail activation 분포를 벌린다.
- **가설 H1**: 유도 실패는 자연 실패보다 분포가 확연히 갈라져, AND-NOT conceptor가 **비퇴화**로 산출된다 (exp3의 구조적 퇴화 탈출).
- **가설 H2 (headline)**: 유도실패-fit conceptor가 **자연 실패**에도 전이된다 (분리·rescue).
- **부가 산출**: Track I는 pathway별 주입이라 유도 실패에 **TYPE ground-truth**(goal vs motor)가 설계상 부여됨 — online type 식별 검증의 정답지.
- 직접 선행: WA-LQR(arXiv:2607.14943) — clean-vs-perturbed 대조로 DiT residual에 diff-of-means steering, 교란 하 SR +11~40pp(위약·유의성 검정 없음). 우리와 차이: 우리는 위약·비퇴화 게이트·rollout-phase 축 보유. 노트: docs/references/reading_notes/steering_robustness_wam_lqr.md.

## 1. Track P — 물리 섭동 (신규 모듈)

### 1.1 섭동 메뉴

> **개정 (2026-07-22, 사용자 결정 — 아래 5모드 표를 대체)**: WAM_Steer(WA-LQR, arXiv:2607.14943)의
> perturbation 메뉴를 따라 **C1 카메라 pose 변경**(MuJoCo cam_pos/cam_quat/cam_fovy, agentview
> 좌+우, δpos σ=0.10m·δrot σ=8°·δfov σ=5°, reset 시 1회·에피소드 내내 지속) + **G1 그리퍼 초기
> 위치 이동**(reset 직후 scripted OSC delta 액션으로 EE를 δxyz(σ=10cm) 실이동 후 policy 시작)을
> 채택. 기존 표에서 **P1 pre-grasp displace·P2 in-hand force는 유지**, **P3 forced gripper-open·
> P4 swap·이미지 노이즈는 제외**. 최종 메뉴 = **C1·G1·P1·P2**. 이 개정으로 §1.1의 "시각·센서
> perturbation 비채택" 결정은 C1에 한해 뒤집힘 — 유도축↔자연축 정렬은 bridge 게이트가 실측
> 판정하고, C1은 지속형이라 primary 대조(perturbed-fail vs perturbed-succ, 동일 config)가 섭동
> 서명을 양 클래스에서 통제한다. P5(drawer 저항)는 drawer 확장 시 재고. 나머지 절의 P3/P4/P5
> 언급은 이 개정 기준으로 읽을 것 (P0 우선순위 = C1 → G1 → P1 → P2).

(아래는 개정 전 원문 표 — P1/P2 행과 주의 문구는 유효)

mid-rollout 실패 주입 선행(RePO-VLA 4종, FailGen 7종) 기준 2모드는 적음. 자연 실패 3대장(grasp miss / drop / wrong object) + drawer timeout을 커버하는 5모드:

| 모드 | 방법 | 유발 실패 | 적용 phase | 선행 |
|---|---|---|---|---|
| **P1 pre-grasp displace** | trigger step에 target free-joint qpos에 seeded δ(3~15cm, xy) + `sim.forward()` | grasp miss·재탐색 | reach (자유물체 한정) | RePO E3 |
| **P2 in-hand force** | 파지 중 `xfrc_applied[body_id,:3]`에 수평 wrench, N env-step 후 명시적 0 해제 | slip·drop | lift/transport/place | DynamicVLA |
| **P3 forced gripper-open** | stepping loop에서 action의 gripper 채널을 K step 동안 open으로 **오버라이드** (mujoco API 불필요) | in-hand drop (**가장 자연스러운 낙하**) | lift/transport | RePO E2 |
| **P4 target↔distractor swap** | reach 전 두 물체 qpos 교환 | wrong-object | reach 전 | FailGen 유사 |
| **P5 drawer 저항** | drawer prismatic joint `mod_damping/mod_frictionloss` ↑ (robosuite `DynamicsModder`, `src/benchmarks/robosuite/robosuite/utils/mjmod.py:1405` — runtime-mutable) | 안 열림 → timeout | drawer pull (drawer task 전용) | COLOSSEUM 계열 |

- **force vs displace의 본질** (사용자 질문 답): "집기 전후"가 아니라 **물리를 통과하는가**. displace=상태 점프(접촉·관성 무시, 파지 중엔 접촉 폭주로 불가 — 그래서 pre-grasp 전용처럼 보임), force=연속 wrench가 physics로 적분(전 phase 가능, 파지 중 개입의 유일한 물리적 경로).
- **비채택**: 시각·센서 perturbation(조명/카메라/이미지 노이즈) — 유발되는 activation signature가 "지각 OOD" 축이라 자연 실패(조작 실패)와 다름. WA-LQR이 정확히 그 축이므로 우리와 상보 관계로 문서에만 기록.
- **P0 파일럿 우선순위**: P3(구현 최소·signature 최자연) → P1 → P2 → P4, P5는 drawer 확장 시.
- ⚠️ **P1 주의**: teleport의 시각 불연속이 activation에 순간 OOD 스파이크를 만들 수 있음(자연 실패에 없는 signature) → trigger 직후 2 record는 fit window에서 추가 제외.

### 1.2 구현

- 신규 `scripts/safe/groot_n15/robocasa/collect/perturbation.py` (robocasa 컨테이너 py3.11, torch 불요):
  - `PerturbSpec(mode, target, trigger={env_step k | event "grasp+m"}, duration, magnitude|delta|damping, direction, spec_seed)` — 전 모드 이 스키마로 로깅.
  - `Perturber(env, spec)`: `find_kitchen_env` 패턴(rejudge_success.py:26)으로 kenv 획득, `maybe_apply(step_i, action) -> action` (P3는 action 수정 반환, P1/P2/P4/P5는 sim 조작).
- `http_feature_collect.py` 연결: `--perturb-spec '<json>'`, stepping loop의 `env.step` 직전 호출. pkl/사이드카 신규 키: `perturb_spec`(resolved), `perturb_applied_env_steps`, `perturb_record_window [r0,r1]`.
- trigger: 1차는 고정 env-step(baseline feature_phases의 phase 앵커 ×5로 산출). event trigger(contact 검사)는 결정적이므로 2차 옵션.
- **결정론 검증**: double-run bitwise + **sham spec(크기 0) ≡ 무섭동 baseline bitwise** (배선 무오염 증명).

### 1.3 캘리브레이션 게이트

- 대상: 6p ppcc_bread 성공 seed 12개 (결정 재현 가능, feature_phases로 앵커 산출 가능).
- grid: P3 open-window {3,6,10} step / P1 δ {3,8,15}cm / P2 {5,15,40}N × dur {10,25}.
- **채택 기준: 유도 실패율 40–70% (Gate1 반영, 구 50–90%에서 하향)** — 상한을 낮춘 이유: primary fit 대조가 perturbed-fail vs **perturbed-succ**(§4.1)라 성공 대조군이 충분해야 하고, 12-seed grid에서 1판=8.3pp라 90% 설정은 극단 섭동 선호 편향이 있음. 채택 config는 **독립 calibration seed**에서 실패율 CI 재확인 후 동결. 모드당 1 config.

## 2. Track I — activation 주입 (patchceil 포팅·확장)

### 2.1 공통 설계

- 포팅 원본: `.claude/worktrees/patching-ceiling/scripts/serve/{patching_hooks.py, patchceil_serve.py}` (PatchSteering h'=donor 대입, `/patch_arm`의 start_record/patch_len 창, `--patch-allow-collect` 캡처 병행 — anchors stage 실사용 이력). worktree 무접촉, `exp/exp4-2-induced-failures` 브랜치로 복제.
- serve: `--collect --capture-vl --patch-layers 15 --patch-allow-collect`, 캡처 kind는 기존 fit 호환(`action_token_mean`+`vlln_mean`).
- **주입 길이(확정)**: **짧은 창 W ∈ {3,6} inference steps**(=env-step 15~30). 창 안에서는 각 inference의 denoise 4 step 전부 교체, **창 종료 후 주입 중단 → closed-loop 복귀, max step까지 자유 실행**. 전창(고갈까지)은 patchceil에서 open-loop replay화로 실패(2/77) — 배제. W는 P0에서 확정. 예외: B3만 행동 관찰용 W=12 탐색 소수.
- 창 위치: `start_record` = baseline feature_phases의 phase 앵커 (B1=reach 초입, B2/B3=transport 초입).
- **anti-circularity 3중 방어**: ① 시간분리 — fit은 `record ≥ 창끝+2`만 ② layer분리 — 주입 L15, fit L≤12·VL ③ `/patch_status` fired_records 대조, 불일치 rollout 폐기.

### 2.2 변형별

| 변형 | 주입 | donor | TYPE | 상태 |
|---|---|---|---|---|
| **B1** | VL: `vl_self_attention` 출력 통째 교체(inference당 1회) | 같은 scene **타 instruction** 성공 4ep (신규 `post_vl_sa_full` 수집; distractor 지시 + "Open the drawer" 2종) | **goal** | 유일한 실배선 신작: `PatchSteeringVL` ~90 LOC + serve `--patch-pathway` ~60 + `extract_vl_donor_npz.py` ~80 (0.5–1일). T_vl 길이 상이 허용(cross-attn 임의 길이) |
| **B2** | DiT L15 | 같은 task **타 phase** (passB full-token succ 8ep 로컬, `donors/ep*_L15.npz` 기추출 재사용; donor_start=insert-settle 시작) | **motor** | 재사용 |
| **B3** | DiT L15 | **타 task**: OpenDrawer 성공 full-token 신규 4ep(~4GB) | **motor** | 수집만 신규. 관전: patchceil A2상 창 내 action은 donor를 그대로 냄 → **창 이후 closed-loop에서 서랍식 행동 잔존 여부** (eef 궤적 DTW·서랍 핸들 접근 판정·mp4 육안) |
| **B4** | DiT L15 | donor 통계 매칭 gaussian noise({0.5,1,2}×σ, 고정 seed, `make_noise_npz.py` ~60 LOC) | unstructured | 구조화 신호 vs 무구조 파괴의 dose-matched 대조 |

## 3. 수집 매트릭스 (pilot-first, ppcc_bread 중심)

| 단계 | 내용 | ep | 게이트 |
|---|---|---|---|
| P0 | Track P 캘리브레이션 grid(P3/P1/P2) ~54 + B2/B4 창·scale ~32 + B1/B3 스모크 ~12 | ~98 | smoke S1–S5 통과 후; 실패율 40–70% config 선정 |
| **bridge 게이트** | P0 데이터만으로 유도실패 축 ↔ 자연실패 축 정렬 검사: induced/natural mean-diff cosine, cross-AUROC(held-out), WA-LQR식 SVM 분리도 사전 측정(24c §3) | 0 (분석) | **정렬 신호 없으면 P1 중단** — 자연실패와 직교한 연산자에 200ep를 쓰지 않음 (Gate1) |
| P1 | Track P 2~3모드 × 40+ + B군 통과 변형 × 40 | ~200+ | P0·bridge 통과분만; **진입 변형은 fail ≥20ep 목표** (Gate1 — 10ep는 episode-level AUROC·McNemar에 과소) |
| P2 (조건부) | 최우수 1–2 변형을 잔여 task(OpenDrawer[P5 포함]·ppcc_beer·OpenStandMixerHead) 확장 | ~40/task | P1 유의 시; 대상 seed는 feasibility 통과분만 |

- fail class = 유도실패(변형·intervention-source 라벨 부착). **primary fit 대조 = perturbed-fail vs perturbed-succ (동일 변형·dose·trigger, 동등 가중) — Gate1 반영**: clean succ 460개를 섞으면 "perturbation 받음 ≈ fail"이 되어 물리 서명을 학습함. 자연 성공(clean succ 460, perturbed-succ 별도 라벨 보존)은 **secondary fit·transfer 평가 전용**.
- **데이터 3분할 계약 (Gate1)**: calibration(P0) / fit(P1 일부) / locked test(P1 잔여 + 자연실패) — episode 교집합 0을 manifest hash로 강제. 변형×layer×α 선택은 fit split에서만, headline(H2·TYPE)은 locked test에서 1회.
- 캡처 ON(수집이므로 정책 위반 아님), 디스크: full-token donor ~5GB + action_token_mean 수집 수 GB.

## 4. Fit + 평가

### 4.1 Fit

- `fit_phase_conceptor_n15.py` 확장(~50 LOC): `--record-start-manifest <tsv>` (pkl별 fit 시작 record 절단 — 시간분리). 값은 `/patch_status`·`perturb_record_window`에서 `build_induced_fit_manifest.py`가 산출(창끝+2; P1은 +2 추가).
- 변형별 fit (primary): `--manifest`(**succ=perturbed-succ, fail=동일 변형 유도실패** — §3 계약; secondary로 자연성공 포함판 병행), `--layers "8,12,VL"`(주입 L15 제외), α 그리드 0.3 아래 확장, `--quota-floor` 유지.
- **mean-diff(setpoint) 병행**: 같은 데이터로 `fit_mean_diff.py`(24a §4.1)도 돌려 유도실패 기반 r̂·s 산출 — conceptor·setpoint 두 연산자 모두 유도 데이터 버전 확보.

### 4.2 비퇴화 진단 = 1차 가설 검정 (공유문서 §3)

- 유도실패 fit C_steer의 R-가중 이득 vs 자연실패 fit 기준값(~0.006). **유의 판정은 held-out episode + label-permutation null 기준** (공유문서 §3 — 자기참조 지표라 sanity gate로만). 크지 않으면 그 변형은 SR eval 없이 종료 (H1 기각 데이터로 보고 — 이것도 결과).

### 4.3 통과분 평가 (confound-audit 필수; per-record·phase-bin 조건부, 시간 pooling 금지)

공통 통제 (Gate1 반영): **공통 post-trigger horizon으로 절단 + episode×phase당 동일 record 수 샘플링 + episode 동등 가중** (실패=timeout·정체로 phase 내 record 수가 달라 dwell/길이 confound가 phase-bin만으로 안 잡힘). CI는 episode/seed/donor **cluster bootstrap**.

1. **분리**: induced vs natural vs perturbed-succ 점수 AUROC + **metadata-only baseline**(kinematics·phase·시간 특징만의 분류기) 병기 — latent가 행동 메타데이터 이상을 설명하는지 확인.
2. **주입 흔적 잔존 검사 (Gate1)**: perturbed-succ에서 injected vs sham 분류 AUROC를 창 이후 lag별로 측정 — chance로 떨어지는 cutoff 이전 record는 fit 제외, 끝까지 안 떨어지면 해당 변형은 "perturbation aftermath" 등급으로 해석 강등.
3. **TRANSFER (headline, H2)**: 유도-fit 연산자를 (a) held-out **자연** succ/fail 분리 (b) **exp4-1 배선으로 자연실패 rescue** (공유문서 §4 계약으로 NPZ 전달, exp4-1 arm B). locked test에서 1회.
4. **intervention-source 검증 (구 "TYPE", Gate1 반영 — 라벨이 주입 경로·donor와 혼입돼 failure-type의 GT로 승격 불가)**: B1 vs B2/B3 × VL-layer vs DiT-layer 2×2 AUROC 교차 우위 + **leave-one-donor-out** 재현 + 행동 phenotype(영상) 대조. 교차 우위가 LODO에서도 유지될 때만 "goal/motor 축 후보"로 보고. B4는 양쪽 낮아야 정상.
5. **B3 행동 판독**: 창 내 action↔donor 일치 확인(예측: A2상 ≈일치) + 창 후 eef 궤적 DTW·서랍 접근 판정·mp4 육안(±10 record).

## 5. Smoke (본수집 전 하드 게이트)

- **S1** (Track P): sham bitwise ≡ baseline / double-run 결정론 / pkl perturb 키·record_window 산술 검증.
- **S2** (DiT 주입): self-donor W=3 ≈ baseline / fired_records == 기대 창 / 캡처 ON에서 record 수 정합.
- **S3** (VL 주입): self-VL-donor bitwise ≡ baseline / cross-instruction donor(T_vl 상이) 무에러 실행 + action 변화 / fire 수 == record 수.
- **S4**: `--record-start-manifest` 절단 후 fit_inputs.json record 수 unit check.
- **S5**: 산출 NPZ를 exp4-1 serve 계약(`--steering-phase-npz-base`)으로 로드 + gated 1ep 실행.

## 6. 리스크

- **섭동 과강 → '물리 서명' 학습**: 실패율 40–70% 게이트 + perturbed-fail vs perturbed-succ primary 대조 + "perturbed-succ vs 자연succ" 분리 감시(높으면 오염 신호 → 세기 완화/P3 위주 전환) + 시간분리.
- **주입 창 누출**: 3중 방어(§2.1) — 그래도 남으면 layer별 진단으로 검출 가능(주입 layer 인접에서만 이득 급증 시 의심).
- **donor phase 오정렬**: donor NPZ meta_json feature_phases에서 프로그램적 산출 + assert.
- **VL 배선 미검증**: S3 하드 게이트.
- **P1 시각 불연속 스파이크**: fit window 추가 제외(§1.1) + 분리 분석에서 P1만 별도 층 확인.
- **Transfer 실패는 버그가 아니라 결과** — "induced ≠ natural 분포"가 H2의 부정적 답. confound-audit 표와 함께 등급 명시 보고.

## 7. 예산

- 코드: Track P 모듈+연결 ~1일 / VL 배선 0.5–1일 / fit 절단·유틸 0.5일 / 러너 개조 0.5일 (2트랙 병렬 ~2일).
- 수집: P0 ~98ep ≈ 4–6h, P1 ~200ep ≈ 9–12h, donor 수집 ~1h (ep당 3–6분, 빈 GPU × serve 2).
- fit·분석: CPU 0.5일(로컬; 자연실패 결합 분석은 remote_compute 경유 가능). transfer eval은 exp4-1 세션 예산.
- 총 wall-clock **~4–5일**.

## 8. 새 세션 시작 프롬프트 (복붙용)

```
exp4-2(perturbation 유도 실패 → conceptor)를 실행한다.
계획: docs/steering/24b_exp4-2_perturb_conceptor_plan.md (+ 공유 24_exp4_shared_plan.md) — 먼저 정독.
worktree를 만들어 그 안에서 작업한다 (브랜치 exp/exp4-2-induced-failures, dev에서 분기) —
exp4-1 세션이 본 트리를 쓰므로 (공유문서 §5 동시 실행 규칙). patchceil worktree는 포팅
원본으로만(무접촉).
순서: ① Track P perturbation.py + collect 연결, Track I 포팅(+PatchSteeringVL 신작) 병렬 구현
② smoke S1–S5 (하드 게이트, 통과 못 하면 중단·보고) ③ P0 파일럿 → 실패율 40–70% 게이트
→ **bridge 게이트(유도축↔자연축 정렬, 분석만)** → 사용자 보고 후 P1 본수집(3분할 계약 준수)
④ fit + 비퇴화 진단(결과 먼저 보고) → 통과분만 분리/잔존검사/transfer/intervention-source
평가, exp4-1로 NPZ 전달. GPU는 비어있는 것 확인 후 사용(GPU당 serve 2), exp4-1 세션과
자원 겹침 확인. 문제·불확실성 발생 시 중단하고 보고.
```
