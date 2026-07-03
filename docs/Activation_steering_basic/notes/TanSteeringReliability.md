# Analysing the Generalisation and Reliability of Steering Vectors (Tan et al. 2024)

- 출처: arXiv:2407.12404 (v8, 2025-05-04) · NeurIPS 2024 · Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso, Robert Kirk (UCL AI Centre + FAR AI)
- PDF: `docs/Activation_steering_basic/TanSteeringReliability_2407.12404.pdf`
- 정독 섹션: §3 Preliminaries(CAA 프로토콜 요약) + §5 In-distribution reliability + §6 OOD generalisation 중심, Appendix D(방법론 상세)·E(추가결과)·C(한계) 확인
- tier: must★★★ (steering 신뢰성 비판 논문)
- 한줄 역할: activation steering이 "듣는다"는 낙관적 서사에 정면 반박 — 같은 개념·같은 벡터도 입력별로 steerability가 양/음으로 요동치고(anti-steerable), 그 원인의 상당부분이 A/B·Yes/No 같은 spurious position/token bias임을 실증. 우리가 global steering 대신 phase/pathway 조건부를 택하는 핵심 근거 논문.

## 문제·동기
기존 연구(ActAdd, CAA, RepE, ITI 등)는 steering 효과를 **aggregate propensity**(데이터셋 평균)로만 보고했고, per-sample 신뢰성·OOD 일반화를 체계적으로 검증한 적이 없다. steering이 실제 배포 가능한 도구가 되려면 (1) in-distribution에서 모든 입력에 일관되게 작동해야 하고 (2) 시스템/유저 프롬프트가 바뀌는 실사용 상황(OOD)에서도 견고해야 한다. 이 논문은 이 두 조건을 40개 MWE persona 데이터셋 × 2모델(부록에서 4모델)로 최초로 대규모 체계 검증한다.

## 핵심 아이디어
CAA(Rimsky 2023) 프로토콜을 그대로 채택(대조 A/B MCQ, mean-difference 벡터, 마지막 토큰 위치에 λ·v 주입)하되 평가축을 aggregate propensity에서 **per-sample steerability 분포**로 바꾼다. "steerability"라는 새 지표를 정의: 여러 λ에서 logit-difference propensity(mLD)를 재고, λ에 대한 mLD 기울기(최소자승 직선 fit)로 요약한다 — 양수 기울기=의도대로 스티어됨, 음수=반대방향(**anti-steerable**). OOD는 system/user 프롬프트에 지시문을 주입해 4가지 분포이동을 설계하고, 한 세팅에서 뽑은 벡터를 다른 세팅에 적용하는 relative steerability로 일반화를 측정한다.

## 방법(steerability 지표, ID/OOD 평가)
- propensity: mLD = Logit(y+) − Logit(y−) (softmax 없는 logit-diff, Rimsky의 normalized-prob과 순서동일하지만 활성화에 더 선형적이라 분석에 유리).
- steerability s(v,D,Λ) = λ∈{−1.5,...,1.5} 7점에서 mLD의 기울기(직선 fit). s∈R, 음수=anti-steerable.
- 모델: Llama-2-7b-Chat(layer13), Qwen-1.5-14b-Chat(layer21) 주 실험 + 부록에서 Llama-2-70B(layer30), Gemma-2-2B-IT(layer14) 추가 검증. 레이어는 validation split 전 레이어 스윕으로 고정(저성능 데이터셋만 재스윕해도 최적 레이어 불변 — "레이어를 잘못 골라서 안 되는 것"은 배제).
- 40개 MWE persona 데이터셋(train40/val10/test50) + TruthfulQA + sycophancy(CAA 원 데이터셋 유지).
- relative steerability(식2): srel(vA,DB,Λ) = s(vA,DB,Λ)/s(vB,DB,Λ) — "그 데이터셋 자체 벡터" 대비 상대 성능으로 정규화해 OOD 일반화를 측정.

## 실험·결과(실패 조건·편차)
- **ID variance**: 13개 대표 데이터셋(전체 40개는 부록 Fig18) 모두 per-sample steerability가 넓게 분포. corrigible-neutral-HHH처럼 median 높은 것도 여전히 고분산 unimodal. myopic-reward류는 bimodal — 두 클러스터 중 하나가 음수(anti-steerable), **거의 절반 입력이 반대 방향으로 스티어**됨(Fig1).
- **spurious steerability bias(신규 발견)**: MCQ에서 정답이 A/B인지 Yes/No인지는 학습데이터에서 이미 무작위화했는데도, steerability 자체가 특정 위치/토큰 쪽으로 데이터셋마다 강하게(그러나 방향은 데이터셋마다 다르게) 편향됨(corrigible-neutral-HHH는 B쪽, self-aware-lm은 A쪽, Fig3·16). 표준 position/token bias(로짓 자체 편향)와 다른 종류이며, **데이터 재균형이나 logit calibration으로 못 고침**(그건 propensity를 바꾸지 steerability=변화량을 안 바꿈).
- 이 두 spurious factor(A/B, Yes/No)가 일부 데이터셋에서 per-sample steerability 분산의 상당 부분을 설명(Fig4) — 즉 뽑힌 벡터가 개념이 아니라 프롬프트 포맷 아티팩트를 인코딩했을 가능성.
- **OOD**: ID·OOD steerability는 상관(Llama ρ=0.891, Qwen ρ=0.694)하지만 완벽하지 않고 OOD가 평균적으로 더 낮음(특히 Qwen). ID에서 안 되면 OOD도 거의 안 됨(필요조건), 되더라도 불완전.
- steerability는 **모델보다 데이터셋(개념) 속성**(Llama vs Qwen ID ρ=0.769, OOD ρ=0.586, 아키텍처·크기·학습이 전혀 다른데도 상관) — "이 개념이 어렵다/쉽다"는 모델을 가로질러 전이.
- generalisation은 원본·타깃 세팅에서 **un-steered 모델 propensity의 유사도**와 상관(약~중간, Qwen ρ=-0.46, Llama ρ=-0.26) — 모델이 이미 그 행동을 자연스럽게 하는 세팅 간에는 전이가 잘 되지만, 정확히 steering이 필요한 상황(모델이 원래 안 하려는 행동을 유도)일수록 전이가 더 나쁨(용도와 신뢰성이 반비례하는 역설).
- 부록: multiplier range(±0.5~±1.5) 바꿔도 steerability 순위 강건. 4모델(7B~70B, Gemma 포함) layer sweep도 동일 패턴.

## activation-steering 흐름 위치(낙관론에 제동)
ActAdd(2308)→CAA(2312, aggregate 성공 보고)까지는 "steering이 듣는다"는 낙관적 서사가 확립되는 구간. 이 논문(2407, NeurIPS 2024)이 처음으로 그 서사에 **정량적 제동**을 건다 — 같은 CAA 프로토콜, 같은 벡터를 쓰되 측정 단위를 aggregate에서 per-sample로 낮추자 다수 개념에서 절반 가까운 입력이 반대로 스티어됨을 보임. Conceptor(2410)·COAST 등 이후 방법론 논문들은 여전히 aggregate 성공(accuracy 상승)만 보고 — "그 안에 anti-steerable 사례가 얼마나 숨어있는가"는 후속 논문 대부분에서 답해지지 않은 채 남는다. 서베이 흐름에서 이 논문 뒤에 오는 모든 방법론(conceptor, phase-matched steering 등)은 사실상 "이 brittleness를 어떻게 완화하려 하는가"로 재해석 가능하다.

## 우리 프로젝트 연결
- **조건부(pathway/phase-matched) steering의 직접 정당화**: 이 논문의 핵심 실패모드는 "하나의 전역 벡터를 모든 입력에 동일하게(prompt-agnostic) 적용"할 때 발생한다. 우리가 global steering 대신 phase-bin/pathway별로 별도 conceptor를 fit하는 것은 정확히 이 문제(입력 조건에 따라 steerability가 요동친다)에 대한 구조적 대응 — VLA rollout의 phase(초반 VL-편향/후반 DiT-편향)가 다르면 다른 벡터를 써야 한다는 우리 가설과 동형이다.
- **spurious bias ↔ length/instruction confound 유비**: "A/B, Yes/No 위치 편향"은 우리 프로젝트의 두 confound와 구조적으로 동일한 패턴이다. (1) seen18 length confound(실패=timeout=항상 긴 rollout, AUROC 0.998)는 우리 conceptor가 "성공/실패"가 아니라 "길이"라는 spurious feature를 인코딩했을 위험과 동형. (2) SlideDishwasherRack instruction confound(VL AUROC 0.93이 instruction in/out 쏠림 아티팩트)는 이 논문의 "steering vector가 개념이 아니라 프롬프트 포맷을 인코딩"과 정확히 같은 실패 패턴. 두 경우 모두 "데이터 재균형으로 안 고쳐진다"는 저자 경고가 그대로 적용될 위험 — 단순 conceptor 재fit이 아니라 confound 자체를 통제하는 실험 설계(길이-matched truncation, instruction 균형 확인)가 필요함을 재확인한다.
- **steerability가 데이터셋(=task) 속성**이라는 결과는 우리의 "task마다 phase-selective steering 효과가 다를 것"이라는 사다리식 ablation 전제와 부합 — 한 task에서 안 되면 다른 모델/체크포인트로 바꿔도 잘 안 될 가능성(모델보다 task 성질이 지배적).
- **"model propensity 유사도가 generalisation을 예측"**은 우리 online phase/failure-type 식별 문제와 연결된다: steering이 가장 필요한 순간(모델이 스스로는 실패로 가는 순간)이 바로 steering이 가장 안 통할 수 있는 순간이라는 역설 — phase-matched steering이 성공 분포 자체를 phase별로 다시 fit해야 하는 이유(전역 성공 분포가 실패 직전 phase의 activation과 너무 멀 수 있음).

## 면접 포인트(Q→A)
1. Q: "activation steering이 항상 잘 듣나?" A: "아니다. Tan et al.(2024, NeurIPS)이 CAA와 동일 프로토콜로 40개 MWE 데이터셋에 대해 aggregate가 아니라 per-sample steerability를 측정했더니, 다수 개념에서 절반 가까운 입력이 오히려 반대 방향으로 스티어되는(anti-steerable) 현상을 발견했다. steering vector가 개념이 아니라 spurious한 프롬프트 위치/토큰 편향을 인코딩했을 가능성이 크다는 것."
2. Q: "그 spurious bias는 정확히 뭔가?" A: "MCQ 대조 프롬프트에서 정답이 A인지 B인지, Yes인지 No인지는 학습 데이터에서 무작위화했는데도, steerability(효과의 변화량) 자체가 특정 위치/토큰 쪽으로 데이터셋마다 다르게 편향됐다. 로짓 자체의 표준 position bias와 다른 '스티어링 효과의' 편향(steerability bias)이라 데이터 재균형이나 logit calibration으로 고칠 수 없다."
3. Q: "OOD 일반화는 어떤가?" A: "ID·OOD steerability는 상관은 있지만(ρ 0.69~0.89) OOD가 체계적으로 낮고, 특히 모델이 원래 잘 안 하는 방향으로 유도해야 하는(=steering이 실제로 필요한) 상황일수록 일반화가 나쁘다는 역설적 결과를 보였다."
4. Q(우리 프로젝트): "그래서 왜 phase/pathway 조건부 steering을 택했나?" A: "이 논문이 보여준 건 '하나의 고정 벡터를 모든 입력에 동일 적용'하는 방식의 구조적 취약성이다. VLA rollout도 phase(초반 goal 파악/후반 motor 실행)마다 activation 분포가 다르므로, 전역 conceptor 하나로 全구간을 커버하면 이 논문이 지적한 anti-steerable/spurious bias 문제를 그대로 물려받는다. phase-bin별로 조건부 fit하는 것은 '입력(=phase) 조건에 맞는 국소 스티어링 벡터를 쓴다'는 이 논문의 함의를 직접 실천하는 것이다."

## 한계·비판
- MCQ 이지선다 포맷에 한정 — 저자 스스로 "beyond multiple-choice-question format"을 향후 과제로 명시. 자유생성·연속 행동공간(우리 VLA action)에서도 같은 anti-steerable 패턴이 나타나는지는 미검증.
- 왜 spurious bias가 생기는지, 어떻게 고치는지는 밝히지 않음(저자 스스로 "완화책 불명"이라 명시) — 진단(diagnosis) 논문이지 해법(solution) 논문이 아님.
- 모델 2종(주 실험 Llama-2-7b, Qwen-1.5-14b) + 부록 2종(Llama-2-70B, Gemma-2-2B-IT) 총 4개로 검증 폭이 넓어졌지만 여전히 소수 LLM. VLA(연속·시계열·멀티모달)로의 전이는 저자 범위 밖.
- steering vector 추출법은 CAA(mean-diff, 마지막 토큰, 단일 레이어)에 고정 — conceptor·PCA 등 다른 추출/적용법이 이 brittleness를 줄이는지는 다루지 않음(§D.4에서 PCA/LG가 MD보다 낫다는 근거 없다고만 서술, 직접 비교 실험 없음).
- λ 범위(±1.5)·레이어 고정 등은 부록에서 강건성 확인했지만, aperture류의 soft-projection 정규화는 아예 고려 대상이 아님(conceptor 계열과 직접 비교 불가 — 우리는 이 논문의 brittleness가 conceptor로 완화되는지 스스로 검증해야 함).
