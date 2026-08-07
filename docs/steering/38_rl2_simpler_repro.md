# 38. RL2-VLA SIMPLER 재현 결과 (Stage 1)

2026-08-05 축소본(1b) → **2026-08-07 논문 프로토콜 풀사이즈 완료(1c)**. 계획 =
`~/.claude/plans/iridescent-tickling-sunrise.md`, 배경 =
[`../references/reading_notes/rl2_vla_adaptive_steering.md`](../references/reading_notes/rl2_vla_adaptive_steering.md).

## ★ 최종 판정 (Stage 1c, 3,600 에피소드)

논문과 **동일 프로토콜**(3 seed × α top-3 사후선택 × 4 arm)로 재실행한 결과:

| arm | 우리 (3 seed) | 논문 Fig 8 | 차이 |
|---|---|---|---|
| Vanilla | 38.0 | 36.0 | +2.0 ✓ |
| Rephrase | **45.3** | **45.3** | 0.0 ✓ |
| Compose-Always | 47.7 | 46.5 | +1.2 ✓ |
| **Compose-Adaptive** | **46.7** | **53.7** | **−7.0 ✗** |

**4 arm 중 3개가 논문값에 일치(Rephrase는 소수점까지)하는데 adaptive만 7pp 미달.**
α를 task별로 고르는 반칙성 oracle을 써도 48.2로 always와 동급이고, α 평균은 44.4로 rephrase 이하다.
→ **논문 헤드라인("게이팅이 이득의 원천", always +1.2 / adaptive +8.4)은 재현되지 않는다.**
축소 프로토콜 탓이라는 초기 가설은 **기각**됐다.

## 셋업

- 코드: `RL2-VLA/` 서브모듈 (hanyangtae fork of marmotlab/RL2-VLA, vanilla — 수정 0줄).
  러너: `scripts/rl2_vla/stage1_simpler/run_arm.sh` (원본 eval 플래그 유지, 절대경로·1 seed만 변경).
- 부품: pi0 = `juexzz/INTACT-pi0-finetune-bridge`(6.1G), QAM bridge 500k(367M),
  CoVer verifier(302M), SAFE ckpt(동봉, combined CP band). 전부 pretrained — 학습 0.
- 범위: **OOD 환경 suite** 4 task(orange_juice/spoon_google/tape_measure/toy_dinosaur)
  × 3 arm × seed 42 × 50판 = 600 ep. 대조 대상 = 논문 Fig 8 (π0 OOD env).
  ⚠ 공개 코드의 base 지시문은 **원문**(red-team 아님, 코드 실측) — Fig 7(OOD prompt)이 아니라
  Fig 8 프로토콜에 해당.
- 실행: A4000 3장(GPU 1/2/3), 판당 rephrase 1.3분 / adaptive 1.9분 / always 2.9분,
  전체 wall ~7h45m. VRAM 피크 13.3/16.4GB (단일 A4000 수용). Traceback 0, 600/600 완주.

## Stage 1c 상세 (풀사이즈, 3 seed × α top-3)

task별 3-seed 평균 SR (adaptive = seed별 α 사후선택 = 논문 프로토콜):

| arm | orange_juice | spoon_google | tape_measure | toy_dinosaur | 평균 |
|---|---|---|---|---|---|
| Vanilla | 30.0 | 52.0 | 19.3 | 50.7 | 38.0 |
| Rephrase | 36.0 | 48.0 | 53.3 | 44.0 | 45.3 |
| Compose-Always | 35.3 | 50.7 | 53.3 | 51.3 | **47.7** |
| Compose-Adaptive | 37.3 | 50.7 | 54.0 | 44.7 | 46.7 |
| *논문 Adaptive* | *43.3* | *59.3* | *59.3* | *53.3* | *53.7* |

- **adaptive 미달이 4 task 전부에서 균일**(−6.0/−8.6/−5.3/−8.6) → 특정 task 아티팩트 아님.
- α 9조합 중 최고 lane도 48.5(seed0 α0.2)로 always 평균(47.7)과 동급 — 어느 α에서도 게이팅 우위 없음.
- α 선택 프로토콜별: seed별 사후선택 46.7 / task별 oracle 48.2 / 단순 평균 44.4.
- 실행: 15 lane × 200 ep = 3,000 + Stage 1b 600 = **3,600 ep**, Traceback 0, 15/15 DONE.
  A4000 4장 큐 러너(`run_full_repro.sh`)로 wall 18.2h. 집계 = `aggregate.py`.

## Stage 1b 상세 (축소본, seed 42 × α top-1) — 참고용

우리 (SR %, 괄호 = Rephrase 대비 ΔSR):

| task | Rephrase | Compose-Always | Compose-Adaptive |
|---|---|---|---|
| orange_juice_on_plate | 32.0 | 32.0 (+0.0) | 28.0 (−4.0) |
| spoon_on_towel_google | 46.0 | **60.0 (+14.0)** | 46.0 (+0.0) |
| tape_measure_in_basket | 48.0 | 54.0 (+6.0) | 50.0 (+2.0) |
| toy_dinosaur_on_towel | 44.0 | 48.0 (+4.0) | 48.0 (+4.0) |
| **평균** | **42.5** | **48.5 (+6.0)** | **43.0 (+0.5)** |

논문 Fig 8 (같은 4 task·같은 세팅, 3 seed; 막대에서 읽은 근사값 —
본문의 "평균 +8.5pp, spoon_google 최대 +14.6pp"와 정합):

| task | Vanilla | Rephrase | Compose-Always | Compose-Adaptive |
|---|---|---|---|---|
| orange_juice_on_plate | 31.3 | 35.3 | 41.3 (+6.0) | 43.3 (+8.0) |
| spoon_on_towel_google | 46.0 | 44.6 | 54.0 (+9.4) | **59.3 (+14.7)** |
| tape_measure_in_basket | 18.7 | 52.0 | 44.6 (−7.4) | 59.3 (+7.3) |
| toy_dinosaur_on_towel | 48.0 | 49.3 | 46.0 (−3.3) | 53.3 (+4.0) |
| **평균** | **36.0** | **45.3** | **46.5 (+1.2)** | **53.7 (+8.4)** |

(n=200/arm, 평균 SR의 SE ≈ 3.5pp; per-task n=50, SE ≈ 7pp.)

게이트(adaptive): 판정 스텝 5,843 중 발동 654 = **11.2%** (비퇴화). 단 발동이
**에피소드 후반(t≥116, timeout 근접)에 집중** — 검출 시점이 늦어 개입 여지가 작았음.

## 논문 대조·판정

> ⚠ 초판(08-05) 기록은 우리 **always**(+6.0)를 논문 **adaptive**(+8.5) 값과 비교하는 오류가 있었다.
> Fig 8 실물 확인 후 아래로 정정한다 — 논문의 always는 +1.2pp에 불과하다.

arm별 대조 (평균 SR):

| arm | 우리 | 논문 | 차이 |
|---|---|---|---|
| Rephrase | 42.5 | 45.3 | −2.8 (노이즈 범위) |
| Compose-Always | 48.5 | 46.5 | +2.0 (노이즈 범위) |
| **Compose-Adaptive** | **43.0** | **53.7** | **−10.7 ← 불일치 집중** |

- **arm 순서 역전**: 논문 adaptive ≫ always ≈ rephrase / 우리 always > adaptive ≈ rephrase.
  → **논문 헤드라인(게이팅이 이득의 원천)은 우리 축소 프로토콜에서 재현되지 않았다.**
- 3 arm 중 2 arm이 논문값에 붙고 불일치가 adaptive 하나에 집중 → 부품·배선 오류가 아니라
  **게이팅 설정(α)·검출 타이밍** 쪽 문제로 국소화된다.
- 공통점: 두 실험 모두 spoon_google이 최대 개선 여지 task (우리 +14.0 always / 논문 +14.7 adaptive).
- 게이트 발동률 11.2%는 비퇴화(0%/100% 아님) — 게이트 자체는 작동.

**종합(1b 시점): 파이프라인 계약 검증 통과, 수치 재현은 부분 성공.** 원인 후보 (a) α top-1만
(b) 게이트 후반 발동 (c) 1 seed → **(a)(c)는 Stage 1c 에서 제거했고 결과는 그대로였다.**
남은 후보는 (b) 게이트 타이밍, 그리고 공개 코드/ckpt 와 논문 실행 환경의 미기재 차이다.

## 시사점 (우리 연구 관점)

1. **"언제 개입할까" 축은 생각만큼 확립돼 있지 않다** — 저자 코드·저자 ckpt·저자 프로토콜로도
   게이팅 우위가 재현되지 않았다. RL2 가 이 축을 선점했다는 우리 판단은 **약화**되며, 조기 검출·
   phase 라우팅으로 더 나은 답을 낼 여지가 크다.
2. **검출 타이밍이 유력 원인** — 게이트 발동(11.2%)이 에피소드 후반(t≥116, timeout 근접)에 집중.
   늦게 감지하면 개입 여지가 없다 → 우리 "조기 online 검출" 축을 직접 지지하는 실측.
3. **합성(다양성 주입)은 재현된다** — always 47.7 vs 논문 46.5. 즉 velocity 합성 자체는 작동하며,
   Stage 2 이식의 기술적 토대는 유효하다.
4. α 사후선택은 약한 oracle이고 그마저도 이득을 만들지 못했다 — Stage 2 에서는 α(또는 우리
   게이트 threshold)를 명시적 통제 변수로 두고, 사후선택 없는 값을 primary 로 보고할 것.

## 재현 인프라

- 러너: `scripts/rl2_vla/stage1_simpler/run_arm.sh` (arm×GPU 단위, IID/OOD suite 지원)
- 로그·MP4: worktree `RL2-VLA/experiments/stage1b_OOD_seed42/<arm>/` + `stage1b_launch_<arm>.log`
- conda env `rl2` (py3.10, torch 2.5.1+cu121; 설치 = `env_simpler_pi.sh`에서 sudo 줄만 제외)
- 함정 기록: 스크립트의 `INFERENCE_ROOT=$REPO_ROOT/CoVer_VLA`는 낡은 경로(실제는
  `RL2_CoVer_VLA`) — PYTHONPATH 직접 지정으로 우회. wandb는 `WANDB_MODE=offline` 필수.
  SAPIEN/Vulkan이 GPU 0에 3MiB 컨텍스트를 여는 것은 무해(프로세스 종료 시 자동 회수).

## 다음 (Stage 2 게이트 — 사용자 결정 대기)

① verifier 전략(verifier-free / CoVer 재학습 / oracle) ② QAM 학습 데이터(로컬 atomic 데모 /
우리 수집 rollout / RoboCasa365 다운로드) ③ eval task set. 추가 후보: 1b에서 얻은 교훈으로
**α 스윕 재실행**(adaptive 재평가, ~5.5h) 또는 IID suite sanity — 필요시.
