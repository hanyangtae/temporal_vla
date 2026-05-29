# PPGuide: Steering Diffusion Policies with Performance Predictive Guidance

> arXiv ID **2603.10980** 검증 완료 — 유효하며, 제공된 제목과 일치하는 paper로 확인됨.

## 메타 정보
- **arXiv ID**: 2603.10980 (v1, 2026-03-11 제출)
- **저자 / 소속**:
  - Zixing Wang — Purdue University, Dept. of Computer Science
  - Devesh K. Jha — Mitsubishi Electric Research Laboratories (MERL)
  - Ahmed H. Qureshi — Purdue University, Dept. of Computer Science
  - Diego Romeres — Mitsubishi Electric Research Laboratories (Cambridge, MA)
- **게재 venue**: ICRA 2026 채택(accepted) — abstract 페이지 기준
- **게재 날짜**: 2026-03-11
- **프로젝트 페이지**: 명시되지 않음
- **코드 공개 여부**: 명시되지 않음 (License: CC BY 4.0)

## 한 줄 요약 (1 sentence)
사전학습된 diffusion policy를 **추론시(inference-time)** 실패 모드에서 멀어지도록 steering하는 경량 classifier 기반 guidance — attention MIL로 obs-action chunk를 SR/FR/IR로 self-labeling하고, performance predictor의 gradient로 denoising 과정을 조정한다.

## 문제 정의
diffusion policy는 error accumulation으로 인해 복구 불가능한 실패 상태에 빠지는 경향이 있다. 이를 해결하려는 기존 접근(데이터셋 증강, 비싼 world model 구축)은 비용이 크다. PPGuide는 추가 데이터나 world model 없이, 사전학습 정책을 그대로 두고 추론시에 실패 모드로부터 밀어내는 경량 방법을 제안한다.

## 방법 (Method)
- **핵심 아이디어**: 3단계 파이프라인. (1) 여러 학습 단계 checkpoint에서 다양한 롤아웃 수집. (2) attention-based Multiple Instance Learning(MIL)으로 trajectory(=bag)와 obs-action chunk(=instance)를 묶고, gated attention weight로 어떤 chunk가 outcome에 기여하는지 자동 식별. (3) 그 pseudo-label로 lightweight classifier f_guide 학습. 추론시 f_guide의 gradient로 denoising noise estimate를 조정.
- **입력 / 출력**: 입력 = observation chunk(길이 j) os_t^j + action chunk(길이 k) as_t^k 쌍, MLP 인코더 φ로 공유 임베딩 사상. 출력 = SR/FR/IR 3-class 예측, action 차원에 대한 guidance gradient.
- **모델 아키텍처**: gated attention MIL 모듈 + MLP classifier(f_guide) + 사전학습 Diffusion Policy(DP, Chi et al. 2023).
- **학습 / 추론 단계 구분**: 둘 다.
  - *Training-time(offline)*: epoch 250/300/350/400/450 롤아웃 수집 → MIL 학습(trajectory 이진 성공/실패 라벨) → attention weight로 instance pseudo-label 생성 → f_guide를 cross-entropy로 supervised 학습.
  - *Inference-time(online)*: 사전학습 f_guide로 action에 대한 gradient 계산, denoising에 주입. 연산 절감을 위해 짝수 step에만 적용하는 alternating schedule 사용.
- **실패 정의 / 발견 방식**: trajectory-level **이진 성공/실패 라벨**("the only readily available supervisory signal")만 사용하는 self-supervised 방식. attention weight의 z-score > τ인 chunk를, 성공 trajectory면 SR(Success-Relevant), 실패 trajectory면 FR(Failure-Relevant)로 pseudo-label, 나머지는 IR(Irrelevant). 즉 **categorical 3-class**이며, cluster나 OOD 기반이 아님.
  - Success bag = SR instance를 1개 이상 포함; Failure bag = irrecoverable error로 이어진 FR instance를 1개 이상 포함.
- **작동 공간**: **observation-action sequence(action level)**. raw policy latent도, VL embedding도 아님 — obs+action chunk를 MLP φ로 별도 임베딩한 공간에서 작동하고, guidance gradient는 **action 차원에 대해서만** 계산.
- **Mode-conditional 여부**: **단일 global guidance**. SR로 끌어당기고(w_sr, 보통 작음) FR에서 밀어냄(w_fr, 보통 큼)으로 비대칭. 실패 mode별 조건부 분기는 없음(FR을 하나로 통합).
- **Pseudo code / 알고리즘**: 명시적 의사코드 형태는 없음. 절차:
  ```
  # Offline
  1. 정책 checkpoint들에서 trajectory T 수집
  2. MIL 학습: 인코더 φ → attention weight α_t → bag classifier g
  3. 각 trajectory에서 α_t > τ 인 instance 추출
  4. outcome+weight로 SR/FR/IR 라벨링
  5. f_guide를 cross-entropy로 학습
  # Online (denoising step k 마다)
  6. g_sr = ∇_{as_t^k} log P(y=SR|·),  g_fr = ∇_{as_t^k} log P(y=FR|·)
  7. ε̂_θ = ε_θ + w_sr·g_sr − w_fr·g_fr      (Eq.5)
  8. (옵션) 짝수 step에만 적용 (alternating)
  ```
  - 핵심 식: gated attention(Eq.1) `α_t ∝ exp(w⊤(tanh(V h_t⊤) ⊙ sigm(U h_t⊤)))`, guidance(Eq.5) `ε̂_θ = ε_θ + w_sr·g_sr − w_fr·g_fr`.

## 실험
- **사용한 모델**: Diffusion Policy(DP, Chi et al. 2023) **단일**. (π0/DP3 등 다른 family는 다루지 않음.)
- **벤치마크**: Robomimic + MimicGen — Stack D1, Stack Three D1, Coffee D2, Coffee Preparation D1, Kitchen D1, Mug Cleanup D1, Square, Transport.
- **설정**: 원본 demo의 10% subset, epoch 250–450 롤아웃으로 guidance 학습, eval은 epoch 500/550.
- **Baseline 비교 대상**: DP(base), DP-SS(stochastic sampling), PPGuide-SS, PPGuide-CG(매 step constant guidance), PPGuide(alternating).
- **핵심 수치 (Table II)**:
  - Stack D1: 92% → 94% (+2%)
  - Coffee D2: 28–30% → 32–36% (+4–8%)
  - Kitchen D1: 40–52% → 44–58% (+4–12%)
  - Transport: 60–68% → 68–76% (+8%)
  - **Heterogeneous(Table III)**: epoch 250–450로 학습한 *단일* PPGuide를 epoch 1300–1600 정책에 적용해도 +4–18% 개선(다른 학습 단계 정책에 전이 가능).
- **Ablation**: guidance strength 민감도(Fig.4 — mid-range에서 최고), z-score threshold(Fig.5 — ~2.0 최적, 민감한 파라미터). alternating(PPGuide)이 constant(PPGuide-CG)와 동등/우월하면서 연산 절감.

## 우리 wedge와의 정확한 차이
우리 wedge:
- **Frozen VLA action expert residual stream(z_mean 1024-d)에서의 unsupervised failure mode discovery**
- **Mode-conditional conceptor steering at inference time**
- **Latent 기반 auto-labeling (VLM 기반 labeling baseline과 대비)**

1. **Failure mode 분해 여부 + 방법**: PPGuide는 chunk를 SR/FR/IR 3-class로 분류한다 — 성공/실패 *관련성* 분해이지, "여러 종류의 실패 mode"를 분해하지는 않는다(FR을 단일 class로 통합). 우리는 실패를 여러 mode로 unsupervised 분해한다. **이것이 가장 본질적인 차이.**
2. **작동 공간**: PPGuide = obs-action sequence를 MLP φ로 임베딩한 별도 공간, guidance는 action 차원. 우리 = frozen VLA action expert residual stream(1024-d 정책 내부 latent).
3. **Training-time vs Inference-time**: 둘 다 inference-time steering이라는 점에서 **가장 가까운 이웃**. PPGuide도 우리처럼 정책 본체를 건드리지 않고 추론시 개입한다(단 별도 classifier f_guide 학습 필요, 우리도 conceptor 추출에 데이터 필요).
4. **Mode-conditional intervention 유무**: PPGuide = 단일 global guidance(SR attract / FR repel), mode별 조건부 분기 없음. 우리 = mode별 conceptor를 선택해 조건부 steering. **핵심 차별점.**
5. **VLA에 적용 가능한가**: PPGuide는 diffusion policy의 denoising gradient guidance에 의존 — diffusion action expert를 쓰는 VLA에는 적용 여지가 있으나 논문은 DP만 실험. discretized/autoregressive 출력 VLA에는 직접 적용 불가. 우리는 frozen latent steering으로 정책 family에 더 일반적.

## Reviewer 대응 시나리오
> "이건 PPGuide와 같지 않냐?"

"PPGuide는 diffusion policy의 denoising gradient를 단일 SR/FR guidance로 미는 방법으로, (a) 실패를 하나의 FR class로만 다루고, (b) 정책 내부 latent가 아닌 obs-action MLP 임베딩에서 작동하며, (c) diffusion 정책에 국한된다. 우리는 frozen VLA action expert latent에서 실패를 *여러 mode*로 비지도 분해하고 mode별 conceptor로 조건부 steering하므로, 실패 mode 분해의 다중성·정책 내부 latent 작동·diffusion 비종속이라는 세 축에서 다르다. 다만 'inference-time classifier/predictor guidance' 패러다임의 가장 가까운 선행 연구로 인정하며, 우리의 multi-mode·mode-conditional 확장이 차별점이다."

## 핵심 인용 포인트
- **Related Work**: "inference-time guidance / steering for policies" 단락 — PPGuide를 "single-mode(global) classifier guidance"의 대표로 위치시키고, 우리를 "multi-mode, mode-conditional" 확장으로 대비.
- **Experiments (baseline/비교)**: 우리 정책이 diffusion action expert를 쓰는 경우 직접 baseline으로 비교 가능. 혹은 "global vs mode-conditional guidance" ablation의 근거 인용.
- **Discussion**: 실패를 단일 FR로 보는 한계 → 실패 mode 분해 필요성 논증의 근거.

## 한계 / 우리에게 유리한 점
- **PPGuide 명시 한계**:
  - *Cold-start*: 거의 성공하지 못하는 정책에는 적용 곤란("a policy that rarely succeeds presents a 'cold start' problem").
  - *Spurious correlation*: 무관하지만 반복되는 feature를 relevant로 오인할 위험.
  - *하이퍼파라미터 민감*: z-score threshold, guidance strength가 task별 튜닝 필요.
  - offline self-labeling은 online 적응 불가, 느린 error accumulation의 credit assignment 한계, diffusion 전용.
- **우리에게 유리**: multi-mode 분해로 단일 FR의 over-repelling/spurious correlation 위험을 분산 주장 가능; frozen latent steering으로 정책 family 일반성 확보; mode-conditional 개입으로 단일 global guidance의 한계 완화. 단 **cold-start는 우리도 공유하는 한계**이므로 정직히 인정한다(실패 sample이 거의 없으면 mode discovery·conceptor 추출 모두 어려움).
