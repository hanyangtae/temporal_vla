# Agent C — 중요성·장벽·전망 (조사결과, evidence/opinion 태그)

## 1. 중요성 찬성(Pro)
- 재학습 없는 test-time 제어 — RepE 2310.01405 (evidence, 방법론)
- 해석가능성 기반 안전 개입 — Golden Gate/Mapping the Mind (evidence 데모 + opinion)
- fine-grained·조합적 제어 — Conceptor Boolean; **CAST(Conditional Activation Steering, IBM, ICLR2025, 2409.05907)** "if 조건 then steer" (evidence, 산업연구소)
- 실시간/agentic 온라인 제어 — ASA(2602.04935), SPAR neural circuit breaker (evidence 프로토타입; 산업채택 미확인)
- 데이터/가중치 접근 불필요 — repeng, IBM lib (evidence; 단 API-only엔 못 씀=장벽 재등장)
- on-the-fly 개인화 — **Google "Steerable Chatbots: Preference-Based Activation Steering" 2505.04260** (evidence, Google)
- 거시 프레이밍 — Amodei "Urgency of Interpretability"(2025.04) (opinion, steering 직접 아님)

## 2. 장벽(왜 안 쓰이나)
- Brittleness/anti-steerable — Tan 2407.12404 (evidence)
- prompting > 모든 방법, SAE 경쟁력X — AxBench 2501.17148 (evidence)
- "해석가능성=안전" 전제 흔들림 — **Rogue Scalpel 2509.22067**(무작위 방향 steer도 유해순응 0→1~13%) (evidence)
- capability/coherence 손상 예측불가 — Capability-Behavior Trade-offs 2602.04903; Minimizing Collateral Damage 2605.01167 (evidence)
- 평가/QA 부실 — Towards Reliable Evaluation of Behavior Steering 2410.17245 (기존효과 과장 확인) (evidence)
- 기하학적 불확실성 원인 — Braun "Understanding (Un)Reliability of Steering Vectors" 2505.22637 (evidence)
- 대조데이터·튜닝비용·사실지식 무력·긴대화 감쇠·API-only 부적합 — Subhadip Mitra "Activation Steering in 2026 field guide" (opinion 실무자)
- always-on 위험/조건부 미성숙 — CAST가 vanilla steering 과잉거부 지적 (evidence)
- 툴링 위축 정황 — Goodfire Ember 공개API 폐기→select-partner (evidence; 원인 미확인)
- prompting/finetuning이 더 안정 — "A Sober Look at Steering Vectors"(alignmentforum) (opinion)

## 3. 전망
- Agentic 온라인 제어 — ASA 2602.04935, SPAR (evidence 프로토타입; 배포 미확인)
- Goodfire 로드맵 — **$150M Series B($1.25B), "model design environment" 비전** (evidence 투자 + opinion 비전)
- 규제·투명성 — EU AI Act Art.13(투명성 요구, 특정 메커니즘 미명시) (evidence 조문 + opinion 해석)
- 기법 성숙 — Gemma Scope, AxBench ReFT-r1(rank-1 학습, prompting 근접) (evidence)
- 낙관 opinion — Amodei 시급성/2027 목표
- 비관 opinion — Mitra("저평가+과대평가 동시"), Sober Look(이론·벤치·비용 미성숙)
- 현재 실배포 = 안전 gate(Circuit Breakers, Constitutional Classifiers) + interpretability API(Ember 축소, Gemma Scope 오픈) 두 틈새뿐 (evidence)

## 종합
찬성은 대부분 방법론/데모/프로토타입 evidence, 실배포는 "always-on 안전 gate"로 좁게 수렴. 장벽은 다수 최근 실증논문이 일관되게 brittleness·prompting우위·안전역설·평가부실 지적 + Goodfire 축소가 정황 뒷받침. 전망은 낙관/비관 팽팽(대부분 opinion). 2026 중반 = "안전 gate 틈새 좁은 성공 + 범용 확산은 신뢰성·평가·조건부 미해결". → 우리 phase-matched·조건부 steering 방향과 부합.

## 다운로드 후보(arXiv, 우리 프로젝트 연관 높은 순): CAST 2409.05907, Google Steerable Chatbots 2505.04260, Rogue Scalpel 2509.22067, Towards Reliable Evaluation 2410.17245
## web-only 링크: Golden Gate, Mapping the Mind, Amodei Urgency of Interpretability, Goodfire Series B(prnewswire), Mitra field guide, Sober Look(alignmentforum), EU AI Act Art.13
