# SEGA: Instructing Text-to-Image Models using Semantic Guidance (Brack et al. 2023)

- 출처: arXiv:2301.12247 (v2, 2023-11-02) · DFKI/TU Darmstadt/Hessian.AI/LAION (Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, Kristian Kersting) · NeurIPS 2023
- PDF: `docs/Activation_steering_basic/SEGA_2301.12247.pdf`
- §5파트: 이미지 생성 도메인의 도구 생태계 탑재 사례
- 3축: 쓰기(write, additive score/noise-estimate 항) · inference-time(학습·아키텍처 확장 없이 매 denoising step 계산) · 도구탑재(Diffusers 공식 파이프라인 `SemanticStableDiffusionPipeline`)
- 한줄 역할: LLM의 additive activation steering(ActAdd/CAA류 "방향 벡터를 더한다")을 diffusion의 classifier-free guidance(CFG)로 확장해, 텍스트만으로 임의 개념 방향을 뽑아 노이즈 추정치에 더하는 형태로 실제 도구 배포까지 간 사례.

## 문제·동기

텍스트-이미지 diffusion model은 prompt를 조금만 바꿔도 완전히 다른 이미지가 나오는 fragile한 특성 때문에 "한 번에 의도한 이미지"를 얻기 어렵다. 기존 fine-grained 제어 방법(segmentation mask, 아키텍처 확장, 모델 파인튜닝, embedding 최적화 — Blended Latent Diffusion, Prompt-to-Prompt cross-attn control, Imagic, Wu et al. disentanglement 최적화 등)은 diffusion의 강점인 "빠른 탐색적 워크플로"를 깨뜨린다(개념·이미지마다 추가 학습/최적화 필요). 배경 논쟁: Kwon et al.(2022)은 diffusion의 noise-estimate 공간 자체가 semantic 조작에 부적합하며 U-Net bottleneck에 별도 학습된 매핑이 필요하다고 주장했는데, SEGA는 이 주장을 정면 반박한다.

## 핵심 아이디어

CFG가 이미 "무조건 추정치에서 조건부 추정치 방향으로 미는" 연산이므로, 이 arithmetic을 일반화해 prompt 조건뿐 아니라 임의 concept 텍스트(e)로 조건화한 추정치와 무조건 추정치의 차를 방향 벡터로 뽑고, 이를 CFG 항에 추가로 더한다. word2vec의 "king − male + female = queen" 직관을 diffusion score space로 그대로 옮긴 것(Fig1). 방향은 sparse(전체 차원의 1~5%만 유효, percentile tail thresholding)해서 여러 개념을 동시에 걸어도 서로 간섭하지 않는다(isolation).

## 방법(semantic guidance: CFG 분해 확장, 개념 방향 항, 매 denoising step)

- 배경 CFG: ϵ~θ(zt,cp) = ϵθ(zt) + sg·(ϵθ(zt,cp) − ϵθ(zt)) — 무조건 추정치를 prompt 조건부 방향으로 민다.
- SEGA는 여기에 semantic guidance 항 γ(zt,ce)를 추가: ϵ~θ(zt,cp,ce) = ϵθ(zt) + sg·(ϵθ(zt,cp) − ϵθ(zt)) + γ(zt,ce).
- γ(zt,ce) = μ(ψ; se, λ)·ψ(zt,ce), ψ = ϵθ(zt,ce) − ϵθ(zt)(positive) 또는 그 음수(negative guidance) — 부호로 "개념을 향함/멀어짐"을 결정.
- μ는 element-wise 마스킹: |ψ|가 percentile λ 이상인 차원만 scale se로 남기고 나머진 0 — concept-relevant 차원만 골라 쓰는 게 isolation의 근거(경험적으로 1~5% 차원이면 충분).
- warm-up δ: 초기 diffusion step엔 γ=0(구도가 정해지기 전엔 개입 안 함, 값이 클수록 detail만 변경). momentum ν로 여러 step에 걸쳐 같은 방향으로 밀리는 차원을 가속.
- 다중 개념: 여러 ei 각각 독립적으로 γ^i_t 계산 후 가중합(Σ gi·γ^i_t) — additive steering의 다중개념 확장판.
- 매 denoising step마다 개념 텍스트 수만큼 U-Net forward pass가 추가(개념 1개당 pass 1회) — 학습/최적화 없이 순수 추론 시 계산, 아키텍처 무관(CFG 쓰는 모든 diffusion에 적용).
- 이론적 근거: μ 없는 SEGA 식이 implicit classifier gradient ∇zt log p(ce|zt) 기반 classifier guidance와 유사함을 유도하나, 저자 스스로 "실제 diffusion 출력은 classifier gradient가 아니므로 성능 보장은 없다"고 인정(경험적 근거로 대체).

## 실험·결과

- CelebA 속성 10종(안경/미소/대머리/수염/모자/곱슬머리/화장/새치/앞머리/성별) 250 seed 기준 positive guidance 성공률 평균 95%, negative(속성 제거) 92%.
- 4개 속성 동시 편집에서도 per-attribute 성공률 유지(최대 91%) — 방향 간섭 없음(isolation) 실증.
- 원본 SD 얼굴 이미지 FID 117.73(조악) → SEGA 편집 후 59.86으로 개선(생성 결함 제거 부수효과).
- I2P(inappropriate-image-prompt) 벤치마크: SD 부적절 콘텐츠 확률 0.38 → SEGA 적용 0.11, DeepFloyd-IF 0.38 → 0.15 — Safe Latent Diffusion(SLD, 동일 저자군 §5 자매논문) 수준 억제를 아키텍처 무관하게 재현.
- Composable Diffusion/Prompt-to-Prompt/Disentanglement(Wu et al.) 대비 user study: multi-conditioning 80% vs 35%/35%, minor changes 91% vs 72%/68%/65%, 종합 72.7% vs 60.5%/43.3%/41.4% — 대부분 범주에서 우세(style transfer·object removal만 Composable Diffusion과 comparable).
- Robustness/Uniqueness/Monotonicity 정성 실증: 같은 seed면 한 번 계산한 방향 벡터를 다른 프롬프트에도 재계산 없이 전용 가능, guidance scale에 효과가 단조 비례.

## §5(산업)에서의 위치(Diffusers/ComfyUI 탑재 vs 상용서비스 미확인)

- Diffusers 라이브러리에 `SemanticStableDiffusionPipeline`으로 정식 탑재(HuggingFace 공식 문서 API 노출) — 연구 프로토타입을 넘어 `pip install diffusers`만으로 누구나 즉시 쓸 수 있는 도구 생태계 배포 사례. Stable Diffusion, Paella, DeepFloyd-IF 등 아키텍처 무관 동작.
- 다만 이는 "OSS 표준 라이브러리 탑재"이지, 특정 상용 서비스(예: 이미지 생성 SaaS API)에 실제 배포됐다는 근거는 논문에 없다 — Circuit Breakers(안전 gate, 실 프로덕션 배치)나 Gemma Scope(연구 인프라 오픈)와 결이 다른 **중간 지대**: 코드는 표준 라이브러리에 박제돼 도구화됐지만, 내부 프로덕션 파이프라인에 상시 가동 중이라는 증거는 없음.
- 동일 저자군(Schramowski/Brack/Kersting)의 자매 논문 Safe Latent Diffusion은 SEGA 수학의 특수화(부적절 콘텐츠 하나만 억제하는 좁은 응용)로 볼 수 있어, "범용 steering 도구(SEGA)"와 "안전 제품(SLD)"이 같은 수학에서 갈라져 나온 형제 사례라는 점이 §5 구조상 흥미롭다.

## 우리 프로젝트 연결(additive vs projective, image vs action)

- SEGA는 순수 **additive** steering이다: 개념 방향(ϵθ(zt,ce) − ϵθ(zt))을 조건부 추정치에 그대로 더한다(부호만 반전해 negative guidance) — ActAdd/CAA와 수학적으로 동형인 diffusion판 additive steering. 우리 conceptor(C_steer = C_success ∧ ¬C_failure)는 **projective/subspace 연산**(h' = h·Mᵀ, 다차원 부분공간으로 투영·억제)이라 이 additive-vs-projective 축에서 정확히 대비되는 진영에 선다.
- SEGA의 "sparse 차원만 골라 쓴다(1~5%, percentile threshold)"는 개념을 isolate한다는 목적은 우리 conceptor의 저랭크 subspace 선택과 비슷하지만 구현이 다르다: SEGA는 **차원별 hard mask**(기저 회전 없는 element-wise 선택)이고 우리는 **공분산 기반 soft subspace**(임의 방향 조합, aperture로 연속 제어) — 이 대비는 Conceptors 노트([notes/Conceptors.md](Conceptors.md))가 이미 다룬다.
- 도메인·시간축이 근본적으로 다르다: SEGA는 image(매 denoising step마다 텍스트 재조건화로 방향을 다시 계산)이고 우리는 action(연속 제어, DiT motor pathway). SEGA의 "매 step 개입"은 COAST/우리 taxonomy의 **denoising-step 축**에 대응할 뿐, 우리가 문제삼는 **rollout task-phase 축**과는 다른 시간축이다.
- warm-up δ(초기 step엔 개입 안 함, 구도가 정해진 후 디테일만 편집)는 표면적으로 "정확한 타이밍에만 개입"이라는 phase-matched 문제의식과 닮았지만, SEGA의 δ는 **생성마다 고정된 스케줄**이지 online 신호로 라우팅되는 게 아니다 — 단일 forward 생성이라 "실패 상태를 온라인에 인식"한다는 개념 자체가 없다.

## 면접 포인트(Q→A)

Q1. SEGA가 activation steering 계보에서 왜 중요한가?
A. LLM의 additive steering(ActAdd/CAA: hidden state에 방향 벡터를 더함)이 diffusion에도 그대로 통함을, diffusion 고유 메커니즘인 CFG를 확장해 보여준 사례다. "표현공간 방향=개념"이라는 activation steering의 핵심 가설이 modality(텍스트→이미지)를 넘어 성립함을 실증했고, Diffusers 표준 라이브러리에 탑재돼 도구 생태계 배포까지 간 흔치 않은 케이스다.
Q2. SEGA는 왜 학습이 필요 없는가?
A. 개념 방향을 모델 자신의 CFG 메커니즘(무조건 vs 조건부 noise estimate 차)으로 매 step 즉석에서 뽑기 때문이다. 새 개념을 걸려면 텍스트만 바꾸면 되고, 이 forward pass는 사전학습 때 이미 있던 CFG 인프라를 재사용한다 — Kwon et al.의 "noise estimate 공간은 semantic 조작에 못 쓴다(학습된 매핑 필요)"는 주장에 대한 직접 반박이다.
Q3. 우리 conceptor 방법과 뭐가 다른가?
A. 축이 둘 다르다. (1) 연산: SEGA는 additive(방향을 더함) vs 우리는 projective(부분공간으로 투영·억제). (2) 도메인·시간축: SEGA는 image·매 denoising-step 재조건화 vs 우리는 action·rollout task-phase 조건부 고정 연산자. 공통점은 "학습 없이 추론 시 개입"이라는 철학뿐이다.

## 한계·비판

- 이론적 정당화가 약함: 저자 스스로 "실제 classifier gradient(ϵ*)와 SEGA 실제 출력(ϵθ)은 근본적으로 다르며 성능 보장은 없다"고 인정 — 순수 경험적 방법.
- hyperparameter(se, λ, δ, momentum)가 개념·이미지마다 손튜닝 필요(App. C grid ablation) — "학습 없음"이 "튜닝 없음"을 뜻하지 않는다.
- 평가가 거의 전부 인간 정성평가(user study)에 의존, 정량 자동 metric은 FID/I2P 확률 정도 — 재현성·객관성 논쟁 여지.
- 단일 forward-pass 생성 태스크라 "실패 후 복구"·"online 상태 추적" 개념이 없음 — rollout 중 실패를 인지해 즉시 steering해야 하는 우리 문제로 옮기려면 시간축 자체를 새로 설계해야 한다.
- Broader Impact에서 저자도 인정: 같은 메커니즘으로 부적절 콘텐츠를 억제도 생성도 할 수 있는 양날의 검이며, gender 등 학습표현의 편향을 그대로 상속한다.
