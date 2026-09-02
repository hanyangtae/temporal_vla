# resample-only + verifier 라운드 (SIMPLER, 2026-08-21)

요청: '연산자 설계' 세션. 질문 = RL² 사다리에서 **합성·rephrase를 다 빼고 "샘플링+선별"만**
남기면 어디에 서는가. 우리 차기 연산자(재샘플 + 내부 read 선별)의 기준선이자, RL² 이득
귀속(합성 vs 선별)의 마지막 분해 조각.

재현 라운드 본문 = `38_rl2_simpler_repro.md` (인프라·함정·baseline 단일 출처).

## 설계

- arm 2종 (둘 다 rephrase 0, QAM 미적재 — 후보 다양성은 **denoise noise 재샘플에서만**):
  - **resample (상시)**: 원문 지시문 1종 × noise 후보 **N=40** → CoVer best-of-40, 매 chunk.
    N=40은 RL² 기본 예산(8 rephrase × 5 sample)과 동일 후보수.
  - **resample_gated (발화 후만)**: prefail = vanilla(1×1), 재학습 SAFE + CP 밴드가 실패를
    예측한 스텝에서만 1×40 재샘플 + CoVer. α=0.2 고정(사후선택 없음).
- 공통: OOD env suite 4 task × 50판 × seed {42, 0, 7} = arm당 600판, 총 1,200판.
  SAFE = 우리 rollout 재학습본(`open_pizero-bridge-lstm-ours_cpTrue/20260807/123421`).
- 판정: pooled SR로는 부족 → **vanilla와 (task, episode_idx) 셀 짝대응**. env reset seed가
  episode_idx에 결정적(`itertools.count(1000)`, cfg.seed 무관)이라 같은 셀 = 같은 초기조건.
  집계 = `scripts/rl2_vla/stage1_simpler/analyze_paired.py`.

## 결과

vanilla 대비 짝대응 (각 seed 200판):

| seed | vanilla | resample(상시) | Δ | 구제/파손 | resample_gated | Δ | 구제/파손 |
|---|---|---|---|---|---|---|---|
| 42 | 41.0 | 34.0 | −7.0 | 22 / 36 | 37.5 | −3.5 | 26 / 33 |
| 0 | 36.5 | 33.5 | −3.0 | 24 / 30 | 39.0 | +2.5 | 29 / 24 |
| 7 | 36.5 | 38.0 | +1.5 | 25 / 22 | 35.5 | −1.0 | 24 / 26 |
| **평균** | **38.0** | **35.2** | **−2.8** | **71 / 88** | **37.3** | **−0.7** | **79 / 83** |

게이트 발화율(gated arm, 결정 시점 기준): seed 42 21.9% / seed 0 24.3% / seed 7 24.6% —
0%·100% 붕괴 없이 정상 작동.

사다리 위치 (같은 suite·조건당 50판, seed 3):
**vanilla 38.0 → resample 35.2 → resample_gated 37.3 → rephrase 45.3 → always 47.7 →
adaptive(재학습 SAFE) 52.7**.

## 판정

1. **"샘플링+선별만"은 이득이 없다** — 상시 −2.8pp, 게이팅 −0.7pp, 둘 다 seed별 부호가
   갈리는 null 패턴. 뒤집힘 총량은 크다(상시 159셀, 게이팅 162셀 = 전체의 약 27%)므로
   개입이 약해서가 아니라 **선별 방향이 무작위에 가깝다**.
2. **RL² 이득은 선별이 아니라 언어 축·합성에 실린다** — 같은 후보 예산 40에서 rephrase는
   +7.3pp(45.3), 우리 순수 재샘플은 −2.8pp. 특히 rephrase가 크게 이겼던 tape_measure
   (seed42 20→48)는 재샘플로 전혀 회수되지 않음(18).
3. **게이팅이 상시보다 낫다(+2.1pp)** — 부호를 뒤집지는 못하나, 개입 횟수를 1/4로 줄이면서
   손해가 줄었다. "상시 개입은 성공 상태를 망친다"는 우리 관측(구제 103/망침 89, docs/38)과
   같은 방향.
4. **해석 한계 — verifier 미스매치 가능성**: CoVer는 rephrase 계층 채점과 함께 학습·검증된
   verifier라, 고정 지시문 단독에서 판별력이 떨어질 수 있다. CoVer 논문 자체가 π0 위에
   RoboMonkey(순수 재샘플+7B verifier)를 얹었을 때 41.5→7.5 붕괴를 보고했고 원인을
   정책-verifier 미스매치로 지목했다. 우리 결과(−2.8pp)는 그 정도 붕괴는 아니지만,
   "재샘플 축이 원래 약하다"와 "이 verifier가 이 조건에서 약하다"를 이 라운드만으로는
   분리할 수 없다.

## 문헌 위치 (2026-08-21 조사)

"합성 정책 없이 policy 재샘플 N + 학습된 verifier best-of-N"은 **기존재** — 우리 실험은
신규가 아니라 **재현/대조군**:

- **V-GPS** (2410.13816, CoRL24): frozen 정책 K=50 재샘플 + Cal-QL value re-rank, 상시.
  SIMPLER 평균 27→34. 사실상 원조.
- **RoboMonkey** (2506.17811, CoRL25): N̂5~9 → Gaussian 적합 → K̂16~32 + LLaVA-7B verifier,
  상시. SIMPLER 38.5→47.5. rephrase 없음.
- **RL²의 "Repeated" arm** (2607.26991): 같은 지시문 반복 샘플 + verifier — 우리 resample과
  동형. 단 **절대 SR 미공개**(그림의 delta만; adaptive가 Repeated 대비 평균 +7.5%).
- **CoVer** (2602.12281): rephrase가 본체. 순수 재샘플 축은 action-error 곡선에서만 열세로
  제시(power-law 기울기 −0.15 vs rephrase −0.18)고, **고정 지시문 + CoVer의 rollout SR arm은
  없음**. 즉 우리 resample arm이 그 논문이 안 돌린 clean control.
- **MG-Select** (2510.05681): verifier-free(KL confidence) 변형.

→ 미측정 빈칸이었던 두 칸을 이 라운드가 채운다: ① CoVer × 고정 지시문 × rollout SR,
② **게이팅된 best-of-N**(위 논문 전부 상시 적용. RL²의 게이트도 샘플링이 아니라 steering
종류를 전환할 뿐). related works 서술 시 "우리가 열었다" 금지 — 재현 위에 얹은 대조군.

## 파일

- 러너: `scripts/rl2_vla/stage1_simpler/run_arm.sh` (arm `resample`·`resample_gated`,
  env `RESAMPLE_N` 기본 40), `run_resample_round.sh` (GPU당 seed 하나 연쇄)
- 집계: `scripts/rl2_vla/stage1_simpler/analyze_paired.py` (셀 짝대응 구제/파손)
- 로그: worktree `rl2-vla-port`의 `RL2-VLA/experiments/stage1b_OOD_seed{42,0,7}/`
  (`resample/`, `resample_gated_a0.2_ours/`), lane 로그 `resample_round_s*.log`
