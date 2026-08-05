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

| task | Rephrase | Compose-Always | Compose-Adaptive |
|---|---|---|---|
| orange_juice_on_plate | 32.0 | 32.0 | 28.0 |
| spoon_on_towel_google | 46.0 | **60.0** | 46.0 |
| tape_measure_in_basket | 48.0 | **54.0** | 50.0 |
| toy_dinosaur_on_towel | 44.0 | **48.0** | 48.0 |
| **평균** | **42.5** | **48.5** | **43.0** |

Δ(always−rephrase) = **+6.0pp**, Δ(adaptive−rephrase) = **+0.5pp**, Δ(adaptive−always) = −5.5pp.
(n=200/arm, 평균 SR의 SE ≈ 3.5pp; per-task n=50, SE ≈ 7pp — +6pp는 ~1.7σ, 유의성 주장 없음.)

게이트(adaptive): 판정 스텝 5,843 중 발동 654 = **11.2%** (비퇴화). 단 발동이
**에피소드 후반(t≥116, timeout 근접)에 집중** — 검출 시점이 늦어 개입 여지가 작았음.

## 논문 대조·판정 (계획의 기준 1~3)

| 기준 | 결과 |
|---|---|
| ① arm 순서 adaptive ≥ always | **✗ 역전** (43.0 < 48.5). 단 논문 스스로 π0에서는 "non-adaptive가 adaptive와 거의 동률"이라 인정(Sec VI-C1) + 우리는 α **top-1만** 사용 (논문 수치는 α top-3 스윕 후 최선 선택 = 사후 선택 프로토콜) |
| ② compose 이득의 부호·자릿수 | **✓** always +6.0pp — 논문 Fig 8 평균 +8.5pp와 같은 부호·자릿수. task별 최대 이득도 spoon_google(+14.0pp)로 논문의 최대 이득 task(spoon_google +14.6pp)와 **일치** |
| ③ 게이트 발동률 합리성 | **✓** 11.2%, 0%/100% 붕괴 아님 |

**종합: 파이프라인 계약 검증 통과.** velocity 합성+verifier의 이득(+6pp)과 최대 이득 task까지
재현됐고, 전 부품(pi0·QAM·CoVer·SAFE·CP)이 계약대로 작동함을 실측 확인. adaptive 열세는
(a) α 사후선택 미수행 (b) 1 seed (c) **게이트 후반 발동**의 합성으로 설명 가능 — 재현 실패라기보다
"adaptive 이득이 α 튜닝에 민감하다"는 논문 자신의 한계(α 휴리스틱 필요)의 실측 확인에 가깝다.

## 시사점 (우리 연구 관점)

1. **검출이 늦으면 게이팅 이득이 사라진다** — 발동의 후반 집중이 adaptive≈rephrase의 직접 원인
   후보. 우리 "언제" 축(조기 online 검출·phase-매칭)의 가치를 역설적으로 지지하는 데이터.
2. **이득의 주성분은 합성(다양성 주입) 쪽** — always가 +6pp를 다 가져감. RL2의 기여 분해
   (steering vs gating)에서 이 세팅은 steering 성분이 지배적.
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
