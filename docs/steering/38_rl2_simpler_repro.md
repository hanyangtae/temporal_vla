# 38. RL2-VLA SIMPLER 축소 재현 결과 (Stage 1b)

2026-08-05. 계획 = `~/.claude/plans/iridescent-tickling-sunrise.md`, 배경 =
[`../references/reading_notes/rl2_vla_adaptive_steering.md`](../references/reading_notes/rl2_vla_adaptive_steering.md).
**탐색적 축소 재현** — 1 seed × top-1 α라 점추정 대조가 아니라 방향·순서 검증이 목적.

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

## 결과

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

**종합: 파이프라인 계약 검증은 통과, 수치 재현은 부분 성공.** 전 경로(latent 추출 → SAFE+CP 게이트
→ velocity 합성 → verifier 선별)가 실측으로 작동하고 두 arm이 논문값에 정합하므로 이식 기반으로는
충분하다. 단 adaptive 미달은 미해결이며 원인 후보는 (a) α **top-1만** 실행(논문은 top-3 사후선택)
(b) 게이트 후반 발동 (c) 1 seed. **α top-3 스윕으로 adaptive만 재평가**하면 공정 비교가 된다(~5.5h).

## 시사점 (우리 연구 관점)

1. **검출이 늦으면 게이팅 이득이 사라진다** — 발동의 후반 집중이 adaptive≈rephrase의 직접 원인
   후보. 우리 "언제" 축(조기 online 검출·phase-매칭)의 가치를 역설적으로 지지하는 데이터.
2. **기여 분해가 세팅에 따라 뒤집힌다** — 논문에서는 게이팅이 이득의 원천(always +1.2 / adaptive +8.4),
   우리 재현에서는 합성이 원천(always +6.0 / adaptive +0.5). 같은 코드·같은 ckpt·같은 task에서
   이렇게 갈린다는 것 자체가 **이 방법의 이득 귀속이 불안정**함을 시사한다.
3. α top-3 사후선택은 사실상 약한 oracle — Stage 2 이식 시 α 프로토콜을 명시적 통제 변수로.

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
