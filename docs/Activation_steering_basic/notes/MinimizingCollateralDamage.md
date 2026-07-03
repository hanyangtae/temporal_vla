# Minimizing Collateral Damage in Activation Steering (Nguyen, Nguyen, Alemohammad & Baraniuk 2026)

- 출처: arXiv:2605.01167v1 [cs.LG] (2026-05-01, preprint under review) · Tam Nguyen(Rice ECE), Tu Anh Nguyen(Rice CAAM), Sina Alemohammad(UT Austin ECE), Richard G. Baraniuk(Rice ECE)
- PDF: `docs/Activation_steering_basic/MinimizingCollateralDamage_2605.01167.pdf`
- §5파트: 이 논문 전체가 "장벽(capability 손상)과 그 완화"에 해당 — collateral damage를 수학적으로 formalize하고 이를 최소화하는 geometry-aware 최적화(저자 자칭 COAST = COllateral-damage Minimizing Activation STeering)를 제시하는 것이 논문의 전부.
- 3축: **이론+벤치마크 방법론 논문**(실배포 사례 보고 아님, ICLR류 preprint) / **범용 LLM 안전 steering**(jailbreak ASR을 testbed로 쓰지만 프레임 자체는 도메인 무관) / **inference-only, 가중치 재학습 없음**(SLERP/ActAdd/Angular Steering의 drop-in 대체).
- 한줄역할: steering의 "성공률 vs 성능 유지" trade-off를 "비target feature 공간의 등방성(isotropy) 암묵 가정 오류"로 최초 수학적 진단하고, 경험적 2차모멘트(공분산) 가중치로 anisotropy를 반영한 constrained optimization(구면 위 geodesic 최적화)으로 SLERP를 strict하게 일반화하는 완화 기법.

**주의(동명이인 경고)**: 이 논문의 자칭 방법명도 "COAST"이지만, 우리 저장소의 기존 `COAST.md` 노트(Miao, Kim, Yang & Ungar 2026, arXiv:2605.17144, "Contrastive Conceptor Activation Steering" — VLA action expert에 conceptor를 이식한 논문)와는 **완전히 다른 논문·저자·도메인**이다. 같은 달(2026-05) 공개된 두 무관한 논문이 우연히 같은 약어를 채택했다. 인용 시 반드시 arXiv ID로 구분할 것 (이 문서 = 2605.01167 = LLM 안전 steering / collateral damage 논문. 2605.17144 = VLA conceptor 논문).

## 문제·동기

기존 activation steering(ActAdd, SLERP/Norm-Preserving Steering, Angular Steering)은 "steering 성공률(ASR 등) vs 무관한 downstream 성능" 사이 뚜렷한 trade-off를 보인다 — 예: ActAdd로 Gemma-2-9B-IT에서 ASR 30-40% 달성 시 정확도가 10-20%p 하락, 반대로 Angular Steering은 정확도는 지키지만 ASR이 10% 미만에 그친다. 선행연구들은 이 trade-off를 "downstream benchmark를 모니터링해 증상을 완화"하는 식으로만 다뤘지 근본 원인(기하학적 원인)을 짚지 않았다. 저자들의 핵심 진단: 표준 steering 방법은 암묵적으로 "비target feature 방향으로의 변화는 어느 방향이든 비용이 같다(등방적)"고 가정하는데, 실제 LLM 표상 공간은 anisotropic(feature가 군집·상관)하므로 target 방향과 얽힌 feature를 건드리면 큰 collateral damage가 발생한다.

## 핵심 아이디어

Collateral damage를 "steering 후 비target feature 방향으로의 정렬(alignment) 변화 제곱의 기댓값"으로 공식 정의하고, steering을 "고정된 alignment budget(목표 코사인 유사도 α) 하에서 collateral damage를 최소화하는 constrained optimization"으로 재구성한다. 기하학적으로는: 목표 방향 d와의 코사인 유사도가 α로 고정된 초구면 위의 한 "위도(latitude)"에서, collateral-damage-가중 거리 상 가장 안전한 착지점을 찾는 문제. 등방(isotropic) 특수 케이스에서는 이 문제가 정확히 SLERP로 환원되고, anisotropic한 일반 경우에는 Riemannian 최적화로 풀어야 한다 — 즉 COAST는 SLERP의 strict generalization이다.

## 방법(부작용 최소화 기법)

- **정식화**: 활성화 h, 목표 방향 d, 목표 활성화 x 모두 단위구 위(노름 보존, 사후 원래 노름으로 재스케일). Alignment budget 제약: d^T x = α. Collateral damage = E[(f^T(x-h))^2] over 비target feature 집합 F, 이를 (x-h)^T Σ (x-h) 형태로 쓰고 Σ := E[ff^T]를 "collateral weighting matrix"라 명명.
- **Σ의 두 가지 균일성 구분(Remark 1)**: "uniform importance"(모든 방향을 동등하게 중요시하겠다는 처방적 의도, 목적함수 설계 선택)와 "uniform geometric distribution"(feature가 실제로 등방적으로 분포한다는 서술적 사실)을 구분 — 전자를 가정해도 후자가 성립하지 않으면(feature가 실제로 클러스터링되어 있으면) 여전히 anisotropic한 비용 지형이 나온다는 것이 핵심 통찰.
- **Σ 추정**: SAE 사전(dictionary)으로 개별 feature 중요도를 매기는 방법은 feature 독립성을 가정해 co-activation을 무시하는 한계가 있음 → 대신 참조 코퍼스(C4) 활성화의 **경험적 2차모멘트** Σ_href = E[h_ref h_ref^T]를 그대로 사용, 이는 개별 feature moment와 co-activation 항을 모두 암묵적으로 포함하며 SAE 학습 없이 anisotropy를 반영.
- **Feasible set과 Riemannian 최적화**: M = {x: ||x||=1, d^T x=α}는 αd를 중심으로 반지름 r=sqrt(1-α^2)인 (p-2)-구면. Tangent space 투영 Π_x, Riemannian gradient = Π_x(2Σ(x-h)), 지수사상(exponential map)의 closed form(Lemma 1)으로 매 iteration이 budget 제약을 정확히 만족하며 geodesic step 진행(Algorithm 1). Σ가 등방(3가지 특수 케이스: d에 정확/근사 직교, 구면 균등 분포)이면 closed-form 해가 정확히 SLERP.
- **이론 보장**: gradient flow가 단일 정류점으로 수렴(Lemma 2, LaSalle invariance + 임계점 유한성), Euclidean 사영 초기화 시 그 정류점이 전역 최적임을 증명(Theorem 1, topological separation 논증), 스텝사이즈 상한 하에서 단조 감소(Prop.1, L-smooth 상수 명시), sublinear O(1/sqrt(T)) 정류 수렴률 + 정규화 시 국소 선형 수렴(Remark 3). KKT 기반 대안 closed-form(Appendix E)도 제시(고유분해 후 1차원 root-finding).
- **실전 세팅**: 2L개 개입 지점(각 layer attn/MLP LayerNorm 직후), d_ℓ은 harmful(AdvBench)/harmless(Alpaca) 프롬프트 대조 difference-in-means, Σ는 top eigenvalue로 정규화해 스케일 불변 확보 → η∈(0,0.5]이면 항상 하강 보장, 실제론 η=0.3, T=1(단 한 번의 geodesic step)으로 충분. Adaptive alignment budget(C.7): 토큰별 α_{ℓ,t}=α·|<h,d>|로 이미 target feature를 강하게 갖는 토큰엔 budget을 더 허용하고 무관한 토큰은 강하게 제약.

## 실험·결과

- 4개 instruction-tuned 모델(Llama-3.2-3B/3.1-8B, Qwen2.5-14B, Gemma-2-9B-it), 목표=harmful 프롬프트에 대한 jailbreak(ASR, HarmBench 채점), 성능 보존=tinyBenchmarks(ARC/GSM8K/HellaSwag/MMLU/TruthfulQA/Winogrande).
- Fig.1: Qwen2.5-14B에서 collateral damage 메트릭과 6개 태스크 정확도 간 강한 음의 상관(Pearson r<-0.9 전 태스크) — 제안 메트릭이 실제 성능 저하의 유효한 proxy임을 검증.
- Fig.3: ActAdd는 ASR 상승에 정확도가 급락하는 급경사 trade-off, Angular Steering은 ASR<10%에 갇힘, COAST는 4개 모델 전부에서 baseline 수준 정확도를 유지하며 높은 ASR 달성 — Angular 대비 ASR 최대 +30%p, ActAdd 동일 ASR 기준 정확도 최대 +20%p.
- Table 1(θ∈[0°,180°] 평균): COAST가 4개 중 3개 모델에서 SLERP보다 높은 ASR(Llama-3.1-8B 52.13 vs 51.67, Llama-3.2-3B 54.76 vs 52.68, Gemma-2-9B 19.48 vs 19.28)이고 정확도도 대등/우위; Qwen2.5-14B만 ASR 근소 열위(6.93 vs 7.64, gradient descent 근사 오차로 저자 해석)하나 정확도는 대등.
- 연산 오버헤드: unsteered 대비 0.9-4.0%, SLERP 대비 1.5% 미만 — 실시간 배포에 무시 가능한 수준(Table 2).

## §5(산업)에서의 위치

이 논문은 §5의 "장벽=예측 불가한 부작용(collateral damage)"을 **정면으로 formalize하고 완화 알고리즘까지 제시**하는, 순수하게 "완화" 파트에 해당하는 사례다. 실배포 보고가 아니라 이론+벤치마크 논문이지만, Discussion에서 스스로 "can we steer?"에서 "can we steer responsibly?"로 논의를 전환한다고 명시하고, Impact Statement에서 "alignment, personalization, content moderation" 같은 실배포 시나리오를 직접 거론하며 예측 불가한 부작용 감소가 곧 신뢰 가능한 배포의 전제조건이라 주장한다. CAST(조건부 개입으로 "언제 개입할지"를 제어)와는 직교하는 축 — CAST는 "누구에게/언제" 개입할지를 조건화하고, 이 논문은 "개입이 결정된 뒤 그 개입을 얼마나 안전하게(비target 방향을 덜 건드리며) 실행할지"를 다룬다. 두 축을 합치면 "조건부 + 저손상" steering 파이프라인이 완성된다는 점에서 §5 산업 채택 논의에 상보적으로 기여.

## 우리 프로젝트 연결(β blend·projective로 파괴성↓)

- **구조적 유사성**: 이 논문의 Σ(collateral weighting matrix, 참조 코퍼스 2차모멘트로 추정)와 우리 conceptor C(succ/fail rollout 활성화 공분산으로 fit)는 둘 다 "공분산 기반 anisotropic 연산자"라는 수학 도구를 공유한다. 다만 목적이 반대다 — COAST의 Σ는 "일반적으로 자주/강하게 쓰이는 방향을 다치지 않게 보호"하는 방어적 penalty(task-agnostic)이고, 우리 C_steer=C_success∧¬C_failure는 "성공과 실패를 가르는 방향으로 밀어붙이는" 지향적 목표(task-specific, 대조적)다.
- **M=(1-β)I+β·C와 alignment budget의 대응**: 우리 identity-blend M=(1-β)I+β·C_steer(β로 개입 강도 조절)는 이 논문의 "고정 alignment budget α(목표 코사인 유사도)" 아이디어와 동일한 동기를 공유한다 — 둘 다 "steering을 얼마나 강하게 걸지"를 명시적 스칼라 노브로 노출해 과개입에 의한 collateral damage를 제어한다. 다만 COAST의 α는 기하학적으로 정확한 제약(구면 위 정확한 위도)이고, 우리 β는 conceptor 공간과 identity 사이 단순 선형 보간이라는 차이가 있다 — COAST의 "정확한 alignment budget 제약 하 최소 collateral damage 지점" 프레임을 빌리면, 우리 β 튜닝을 "목표 conceptor 정렬도 α를 고정하고 그 안에서 succ/fail과 무관한 다른 VLA 능력(지각/그라운딩)의 손상을 최소화하는 최적 지점 탐색"으로 정교화할 여지가 있다.
- **projective(구면 위 anisotropic) 선택 동기와 정확히 공명**: 이 논문이 SLERP(등방 가정)를 비판하고 anisotropic Σ를 도입하는 논증 구조 자체가, conceptor 계열(Postmus & Abreu, Miao COAST-VLA)이 rank-1 additive 벡터(ActAdd/CAA)를 비판하고 subspace projective 연산을 도입하는 논증 구조와 형식적으로 동일하다 — "비target/비판별 방향을 다 같이 취급하지 말고 실제 공분산 구조를 반영하라"는 동일 원리가 LLM 안전 steering과 VLA succ/fail steering 양쪽에서 독립적으로 재발견된 것.
- **Adaptive budget과 phase-matched steering의 접점**: C.7의 토큰별 adaptive alignment budget(α_{ℓ,t}=α·|<h,d>|, 이미 target feature를 강하게 드러내는 상태엔 더 큰 budget 허용)은 우리 phase-matched steering의 "phase마다 다른 개입 강도"라는 요구와 구조적으로 유사 — 판별 방향이 이미 salient한 rollout phase에서는 β를 키우고, 그렇지 않은 phase에서는 β를 낮추는 식의 phase-adaptive β 스케줄로 이식 가능.
- **못 가져오는 것**: 이 논문의 d는 여전히 rank-1 mean-diff 단일벡터다(multi-dim conceptor가 아님) — "d로 얼마나 안전하게 이동할지"를 푸는 논문이지 "d(혹은 우리의 다차원 C_steer)를 어떻게 더 잘 뽑을지"는 다루지 않는다. 우리 다차원 projective steering 문제와는 상보적이지 동일하지 않다.

## 면접 포인트(Q→A)

**Q1. 이 논문의 COAST와 우리가 이미 아는 COAST(Miao et al. VLA conceptor 논문)는 같은 것인가?**
A. 이름만 같고 완전히 다른 논문이다. 이 논문(Nguyen et al., arXiv:2605.01167)의 COAST=COllateral-damage Minimizing Activation STeering, LLM 안전 steering의 collateral damage를 최소화하는 geometry-aware 최적화. Miao et al.(arXiv:2605.17144)의 COAST=Contrastive Conceptor Activation Steering, VLA action expert에 conceptor 대수를 이식한 논문. 같은 달 공개된 무관한 두 논문의 우연한 약어 충돌이라 인용 시 arXiv ID로 명확히 구분해야 한다.

**Q2. collateral damage를 왜 "등방성 가정의 오류"로 진단했나?**
A. 표준 steering(ActAdd/SLERP)은 비target 방향으로의 변화 비용을 Σ∝I(등방)로 암묵 가정한다. 하지만 실제 LLM 표상은 feature가 클러스터링·상관되어 있어(anisotropic) target 방향과 얽힌 feature를 건드리면 훨씬 큰 피해가 난다. Fig.1에서 collateral damage 메트릭과 downstream accuracy 사이 r<-0.9의 강한 음의 상관으로 이 진단을 실증했다.

**Q3. COAST가 SLERP를 완전히 대체하는가?**
A. 아니다, strict generalization 관계다. Σ가 정확히/근사적으로 d에 직교하거나 구면 균등분포인 3가지 특수(등방) 케이스에서는 COAST가 정확히 SLERP로 환원된다는 것을 닫힌해로 증명했다. Anisotropy가 실재할 때만 COAST가 SLERP와 갈라져 더 안전한 지점을 찾는다 — 실험적으로 4개 모델 중 3개는 COAST가 우위, 1개(Qwen)는 근소 열위(최적화 근사 오차).

**Q4(우리 프로젝트). 이 논문의 Σ와 우리 conceptor C는 결합 가능한가?**
A. 원리적으로 가능하다. 둘 다 2차모멘트 기반 anisotropic 연산자이지만 역할이 반대(Σ=방어적 보호, C=지향적 판별)다. 우리 conceptor 방향(C_steer)으로 밀되, COAST식 Σ_ref(성공/실패와 무관한 일반 VLA 능력의 공분산)로 "이 방향 이동이 무관한 능력을 얼마나 건드리는지"를 정량화해 β를 자동으로 캡핑하는 constrained steering을 구성할 수 있다 — 다만 이는 다음 단계 설계이지 이 논문이 직접 제공하는 것은 아니다.

## 한계·비판

- Steering 방향 d 자체는 여전히 rank-1 mean-diff 단일벡터 — 논문의 기여는 "d로 얼마나/어떻게 안전하게 이동할지"이지 "d를 어떻게 더 잘 뽑을지"가 아니다. 다차원 conceptor/subspace 방향 선택(우리 프로젝트, Miao COAST-VLA) 문제와는 직교적이라 우리 핵심 난제(pathway/phase 식별)에는 직접 답을 주지 못한다.
- Σ(collateral weighting) 추정에 일반 코퍼스(C4) 하나만 사용 — "domain-specific 코퍼스도 쓸 수 있다"고 언급은 하지만(§3.2) 실험적으로 검증하지 않았다. Task-adaptive Σ의 효과는 미지수.
- Theorem 1의 전역수렴 증명이 "generic case"(β1≠0, λ1>0 등 non-degenerate 가정)에 의존하는데, 실제 활성화 Σ가 이 가정을 항상 만족하는지는 실증 검증이 없다 — T=1로 충분했다는 경험적 보고만 있을 뿐.
- 평가가 전부 텍스트 LLM 도메인의 harmful/harmless 이진 refusal steering 한 태스크에 국한 — 다른 semantic trait(진실성, persona)나 continuous action space(로봇 diffusion action head)로의 이전 가능성은 전혀 다루지 않는다. Anisotropic Σ 개념이 action head residual stream에서도 유효한지 미지수.
- SAE 사전 기반 방법을 "비용·독립성 가정의 한계"로 명시적 기각하지만, 대안(raw 2차모멘트)은 "왜 이 방향이 보호됐는지"에 대한 해석가능성을 SAE보다 낮춘다 — 이 trade-off를 논문이 정면으로 인정하지 않는다.
- T=1 iteration만으로 충분하다는 결과는 사실상 "단일 geodesic step ≈ closed-form 근사"라는 뜻 — 이 논문이 강조하는 "진짜 반복 최적화"의 실용적 이득이 제한적일 수 있음을 시사하며, 0.9-4% 오버헤드도 사실상 이 단일 step의 비용이다.
