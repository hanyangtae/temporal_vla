# Reducing Hallucinations in Vision-Language Models via Latent Space Steering (Liu et al. 2024, VTI)

- 출처: arXiv 2410.15778v2 [cs.CV] (Stanford, Liu/Ye/Xing/Zou) · PDF: docs/Activation_steering_basic/VTI_VLMHallucination_2410.15778.pdf · 섹션=§3 Mechanism of Hallucination(동기)·§4 Method(핵심)·§5 Experiments · tier=must · 한 줄 역할: RepE(Zou 2023) 계열 activation steering을 LVLM(vision encoder+text decoder 이종 구조)에 최초로 이식한 대표 사례 — LLM steering과 우리 VLA pathway steering 사이의 멀티모달 다리.

## 문제·동기 (LVLM 환각)
LVLM의 환각(hallucination)은 LLM의 언어적 환각과 메커니즘이 다르다 — vision encoder(CLIP류)와 text decoder(LLM)가 각각 별도로 사전학습되고 소량만 함께 파인튜닝되므로, text decoder가 vision feature의 미세한 불안정성에 과민 반응한다. 저자들은 원본 이미지에 노이즈(Gaussian, random mask 등)를 주입해 vision feature의 분산을 측정했고, 대부분의 feature는 안정적이지만 약 15%가 롱테일 형태로 큰 분산을 보이며, 이 불안정 feature가 환각과 상관됨을 관측(§3, Fig.2). 여러 perturbation의 vision feature를 평균하면 perturbation 종류와 무관하게 환각률이 감소하지만(noise 자체가 아니라 averaging이 원인), naive averaging은 (1) 매 쿼리마다 다중 forward pass가 필요해 비효율적이고 (2) 정보 손실을 유발한다. 이 트레이드오프를 해소하는 것이 VTI의 출발점이다.

## 핵심 아이디어
naive feature averaging을 실제로 수행하는 대신, "averaging이 유도하는 방향"을 소수 예제(N=50)로 사전계산해 그 방향으로 latent를 shift한다 — RepE(Zou et al. 2023)·저자 본인의 이전 연구(In-context vectors)에서 착안한 latent space steering을 vision encoder와 text decoder 양쪽에 각각 적용(Visual + Textual Intervention). 두 개입은 서로 다른 실패 메커니즘을 겨냥한다: vision 쪽은 "불안정한 feature를 안정화"하고, text 쪽은 "언어 사전확률에 치우친(image를 무시하는) decoder를 다시 이미지에 근거하게" 만든다. 방향은 대조 클래스(성공/실패) 평균차가 아니라, perturbation-averaged feature와 original feature의 차이를 PCA 1주성분으로 축약해 image-specific 노이즈를 제거한 것 — 50개 예제로 뽑은 방향을 이후 모든 task/dataset/query에 고정 적용(task-agnostic, 추가 학습·추가 추론비용 없음)한다는 실용성이 헤드라인 주장이다.

## 방법 (Visual·Textual Intervention: latent 방향 추출·주입으로 안정화)
- Visual direction: 이미지 v에 랜덤마스크 C_i(mask ratio 0.99, m=50개)를 적용한 corrupted copy들의 vision encoder 활성화 h^{C_i(v)}_{l,t}를 평균해 h̄^v_{l,t}를 얻고, Δv_{l,t} = h̄^v_{l,t} − h^v_{l,t}(식1)를 계산. N개 예제의 Δv들을 모아 PCA 1주성분을 d^vision_{l,t}(레이어 l·토큰 t별)로 사용.
- Textual direction: hallucination 없는 caption x와 GPT로 합성한 hallucination 버전 x̃의 paired 캡션(Zhou et al. 2023 방식)을 만들어, 같은 이미지 v를 조건으로 두 캡션 생성 시의 마지막 토큰 활성화 차이 Δ^{x,v}_{l,t} = h^{x,v}_{l,t} − h^{x̃,v}_{l,t}(식2)를 계산 → PCA로 d^text_{l,t}. 텍스트 디코더는 causal이므로 last-token 활성화만 사용.
- 개입: vision encoder는 non-causal이라 전체 layer·전체 토큰에 h^v_{l,t} := h^v_{l,t} + α·d^vision_{l,t}(식3); text decoder는 causal이라 매 생성 스텝의 마지막 토큰에만 h^{x,v}_{l,t} := h^{x,v}_{l,t} + β·d^text_{l,t}(식4) — vision은 전체 토큰 additive, text는 last-token만이라는 비대칭이 멀티모달 구조 특유의 설계.
- 하이퍼파라미터: mask ratio 0.99, 50 random masks 평균, N=50 예제로 방향 사전계산. α,β는 {0.1,...,1.0} 그리드서치; CHAIR에서는 vision-only α=0.4, text-only β=0.4, 결합 VTI는 α=0.2·β=0.4, 그 외 실험은 α=β=0.9.

## 실험·결과
- 백본 3종(LLaVA-1.5, InstructBLIP, Qwen-VL) × 벤치마크 3종(POPE, CHAIR, MMHAL-Bench), baseline은 디코딩 기반 개입 OPERA·VCD.
- POPE(Table1): 전 백본에서 accuracy·F1 최고 (LLaVA-1.5 acc 79.8→86.5, F1 79.4→85.9), OPERA/VCD보다 우수.
- CHAIR(Table2, LLaVA-1.5): CHAIRs 51.0→35.8, CHAIRi 15.2→11.1, recall 유지(75.2→76.8). vision-only는 CHAIRi(이미지-레벨)에, text-only는 CHAIRs(문장-레벨)에 더 강함 — 상보적 역할 분담을 실증(§5.2).
- MMHAL-Bench(Table3): 평균 스코어 1.99→2.90(VCD 2.69·OPERA 2.60보다 우수), hallucination rate 0.62→0.51. vision-shift는 attribute/comparison 같은 vision-centric task에, text-shift는 adversarial/counting 같은 text-centric(언어추론 필요) task에 강함(Fig.4).
- 분석: (a) vision intervention이 5종 perturbation 전반에서 실제 feature variance를 낮춤(Fig.5, "안정화"가 실재함을 확인). (b) textual intervention은 text→vision attention을 늘리고 text→text attention을 줄임(Fig.6left, 이미지 근거 강화). (c) vision+text 결합은 생성 길이를 줄이지 않고도 환각을 낮춤(단순히 "짧게 말해서" 좋아진 게 아님, Fig.6right). (d) simple averaging은 linear probe 정확도가 62%로 떨어져 정보손실이 크지만, visual intervention은 probing accuracy를 거의 유지하며 variance만 낮춤(Fig.8left, "정보보존+안정화" 동시 달성 주장). (e) α=β=0(무개입)이 최악이고 강도를 올릴수록 개선(Fig.8right).

## activation-steering 흐름에서의 위치 (텍스트 steering의 멀티모달 이식)
RepE(Zou et al. 2023)·저자 본인의 In-context Vectors 계열 "latent space steering"을 vision encoder + text decoder라는 이종 하위모듈에 나눠 적용한 사례 — CAA/ActAdd류의 "차이벡터를 층별로 additive"라는 개입 형식 자체는 그대로 계승하지만, 방향의 출처가 클래스 대조(성공/실패, truthful/untruthful)가 아니라 "perturbation-averaging이 유도하는 안정화 방향"이라는 점이 다르다. 즉 vision 쪽은 레이블 없는 self-supervised 방향(마스킹 평균), text 쪽만 supervised paired-caption 대조 방향 — 한 논문 안에 서로 다른 방향 추출 방식이 공존한다. causal(text, last-token만) vs non-causal(vision, 전체 토큰) 구조 차이에 따라 개입 스코프를 다르게 설계한 것이 LLM 단일 스트림 steering(ITI/CAA/RepE)에는 없는 멀티모달 특유의 기여.

## 우리 프로젝트 연결 (VLM→VLA, VL pathway 개입)
- VTI의 "vision encoder에 안정화 방향 주입 + text decoder에 grounding 방향 주입"이라는 두 갈래 개입은, 우리 pathway 분리(VL=goal "what" / DiT=motor "how") steering의 직접 선례다 — VTI도 입력측 표현(vision)과 출력측 표현(text decoding)을 **서로 다른 방향·서로 다른 스코프**로 개입하고, Table2/Fig.4에서 vision-shift와 text-shift가 서로 다른 hallucination 유형(이미지-레벨 vs 문장/추론-레벨)에 특화됨을 실증했다 — 이는 우리의 "VL 실패(goal 오인식) vs DiT 실패(motor 실행)가 서로 다른 개입 지점을 요구한다"는 가설과 구조적으로 유사한 상보성 패턴이다.
- 다만 VTI는 **완전히 정적**이다: 50개 예제로 뽑은 방향·고정 α,β를 모든 task/query/timestep에 동일하게 적용한다(task-agnostic이 강점이자 한계). 우리가 미해결로 남긴 "phase-matched"(rollout 진행에 따라 개입 대상·강도가 바뀜)나 "online pathway/실패유형 식별"에 해당하는 라우팅이 VTI에는 전혀 없다 — VTI는 "어디"(vision 대 text)만 구조적으로 고정 분리했을 뿐, "언제" 축은 다루지 않는다.
- 방향 추출 방식(perturbation-averaging PCA 주성분)은 우리 conceptor(C_success ∧ ¬C_failure, 대조 통계 기반)와 문제 정의가 다르다 — VTI는 "무엇이 안정된 표현인가"를 최적화하고, 우리는 "무엇이 성공으로 이어지는 표현인가"를 최적화한다. 안정성과 성공이 반드시 같은 방향이라는 보장은 없어, VTI 방식을 그대로 차용하기보다 "성공/실패 대조"라는 우리 축을 유지하는 게 맞다는 근거로 재확인된다.
- vision(non-causal, 전체 토큰) vs text(causal, last-token) 개입 스코프의 비대칭은, 우리 VL(non-causal에 가까운 attention 구조)과 DiT(denoising step 구조, 토큰 causal도 아니고 diffusion-step 축)에 개입 스코프를 각각 어떻게 설계할지 참고할 수 있는 선례 — 단 diffusion 기반 DiT는 이 논문의 causal-LM 가정이 그대로 대응되지 않는다는 점에 유의해야 한다(아래 한계 참고).

## 면접 포인트 (Q→A)
1. Q: "VTI가 LLM steering(CAA, RepE 등)과 구조적으로 다른 점은?" A: "VLM은 vision encoder와 text decoder가 별도 사전학습된 이종 모듈이라, VTI는 개입을 두 갈래로 나눈다. vision encoder는 non-causal이라 전체 layer·전체 토큰에 stability 방향을 더하고, text decoder는 causal이라 매 생성 스텝의 마지막 토큰에만 grounding 방향을 더한다. 이 causal/non-causal 비대칭에 따른 개입 스코프 차이가 LLM 단일 스트림 steering에는 없는 멀티모달 특유의 설계다."
2. Q: "왜 vision-shift와 text-shift가 서로 다른 hallucination 지표(CHAIRi vs CHAIRs)에 강한가?" A: "vision feature의 불안정성은 이미지-레벨 정보 오류(어떤 객체가 있는지)를 만들고, text decoder의 언어 사전확률 편향은 문장 생성 패턴(이미지를 무시하고 그럴듯한 다음 단어를 잇는 것)을 만든다. 서로 다른 원인이라 서로 다른 개입 지점이 필요하다는 걸 실증한 것 — 우리 프로젝트의 goal(VL) 대 motor(DiT) 실패 분리 가설과 같은 논리 구조다."
3. Q(우리 프로젝트 관점): "VTI를 VLA steering에 어떻게 참고하나?" A: "pathway별 분리 개입이 상보적 효과를 낸다는 실증 선례로 쓴다. 다만 VTI는 방향·강도가 완전히 정적(50개 예제로 한 번 고정, 모든 query에 동일 적용)이라 우리가 풀어야 할 online phase/실패유형 라우팅은 다루지 않는다 — VTI는 '어디'만 구조적으로 고정 분리했고, '언제 얼마나' 축은 열려 있는 문제로 남는다."

## 한계·비판
- 방향-환각 인과의 mechanistic 설명이 약하다: feature stability와 환각의 상관은 §3에서 보였지만, PCA 1주성분이 "왜" 안정화·grounding 방향인지에 대한 이론적 근거는 없고 실험적 상관에 의존한다(correlational, ITI의 probe-vs-mass-mean 논쟁과 유사한 사후적 정당화 문제).
- textual direction은 GPT-3.5로 합성한 paired hallucinated/non-hallucinated caption이라는 supervised 대조쌍이 필요한 반면, visual direction은 label-free self-supervised(랜덤마스킹 평균)라 두 방향의 획득 난이도·신뢰도가 비대칭이다.
- 완전히 정적인 개입: 모든 task·dataset·query·생성 스텝에 동일한 방향과 고정 α,β를 적용 — query-adaptive나 context-adaptive 라우팅이 전혀 없다(저자도 이를 "task-agnostic"이라는 장점으로만 프레이밍하고, 온라인 적응은 스코프 밖).
- mask ratio 0.99(거의 전체 마스킹)라는 극단적 corruption에서 유도한 방향이 "본질적 semantic 안정화 방향"인지 단순 "정보손실 방향"인지 완전히 분리되지 않는다 — linear probe accuracy로 정보보존을 간접 검증했을 뿐, 직접적 증명은 아니다.
- 평가가 대부분 object-presence/object-hallucination 벤치마크(POPE, CHAIR)에 치우쳐 있고, MMHAL-Bench는 GPT-4 judge에 의존 — judge 노이즈와 평가 범위의 한계.
- 백본이 모두 autoregressive discrete-token 생성 LVLM(LLaVA/InstructBLIP/Qwen-VL)에 한정 — VLA의 DiT 같은 diffusion 기반 continuous action head는 causal last-token 개입이라는 전제 자체가 성립하지 않아, text-intervention 설계를 그대로 이식할 수 없고 vision-intervention(전체 토큰 additive) 쪽 설계만 참고 가능하다.
- "안정성(stability)"을 대리 목표로 최적화하는데, 안정성이 항상 정답(비환각)으로 이어진다는 보장은 없다 — 과도한 smoothing이 세부정보 손실로 이어질 위험을 저자도 트레이드오프로 일부 인정(Fig.8left)했을 뿐 완전히 해소하지 못한다.
