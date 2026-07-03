# I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models (Grislain & Rahimi et al. 2026)

- 출처: arXiv:2509.16072v3 [cs.RO] (2026-02-19 개정) · ISIR, Sorbonne Université/CNRS, Paris · PDF: `docs/references/I-FailSense_2509.16072.pdf` · 정독 섹션: 전체(§I 서론 · §III 문제정의 · §IV 방법 · §VI 결과) · 서베이 배치: §7 VLA 방향 · tier: must · 한줄 역할: VLA policy 내부 activation이 아니라 별도의 3B VLM이 궤적 이미지+instruction을 보고 성공/실패를 이진 분류하는 "외부 관찰자(observer)" 실패 검출기 — 우리의 internal-latent 검출과 개입 층위가 근본적으로 다른 대조 사례.

## 문제·동기

Language-conditioned 로봇 조작에서 실패 검출은 크게 두 갈래다: (1) control error — grasp 실패, drop 등 시각적으로 바로 드러나는 저수준 실행 실패, (2) semantic misalignment error — 로봇이 "의미상 그럴듯한" 행동을 했지만 주어진 instruction과 다른 것(예: "파란 블록 들어" 지시에 빨간 블록을 드는 경우). 기존 VLM 기반 실패검출(AHA 등)은 거의 전부 (1)에 집중했고, (2)는 VLM의 근본적 한계 — 시각 관찰·모션·언어 사이의 grounding 부족 — 을 직접 드러내는데도 underexplored였다. Semantic misalignment는 control error보다 검출이 어렵다: 물체를 놓쳤는지는 시각 단서만으로 바로 보이지만, "옳은 물체를 옳은 방향으로 옳게 조작했는지"는 공간(어떤 물체)과 시간(어떤 모션) 차원을 instruction과 함께 추론해야 한다.

## 핵심 아이디어

VLA policy 자체를 건드리지 않고, 별도의 사전학습 VLM(PaliGemma2-mix-3B)을 "이 궤적이 이 instruction과 맞는가"를 판정하는 이진 분류기로 post-training한다. 두 단계로 나눈다: (1) LoRA로 VLM 전체를 실패검출 태스크에 맞게 가볍게 적응시키고, (2) VLM은 freeze한 채 그 LLM 내부의 서로 다른 깊이(K=3개 층)에 붙인 작은 분류 헤드(FS block)를 학습시켜, 여러 추상화 수준의 표현을 투표(voting)로 앙상블한다. 핵심 관찰: semantic misalignment 검출만으로 학습해도 control error·새 시뮬레이션 환경·실물(zero-shot/최소 파인튜닝)로 광범위하게 일반화된다 — "instruction과 관찰된 모션이 맞는지 검사"라는 능력이 실패 유형/도메인에 걸쳐 전이 가능한 범용 신호이기 때문이라는 주장.

## 방법(VLM 기반 실패 탐지 파이프라인)

- **입력 표현**: 궤적 τ의 여러 시점(T=4)·시점(N개 PoV) 관측을 하나의 큰 이미지 τ∈R^(3×(H·N)×(W·T))로 콜라주해 단일 VLM forward pass에 넣는다. 여기에 텍스트 goal g를 프롬프트로 결합.
- **Stage 1 (SFT/LoRA)**: 프로즌 SigLIP 비전 인코더와 LLM 사이 projection MLP를 finetune + LLM attention block의 KQV projection에 LoRA 모듈 삽입. `<success>`/`<fail>` 토큰에 대한 cross-entropy loss로 학습.
- **Stage 2 (grounded arbitration)**: VLM을 완전히 freeze하고, LLM 깊이 방향으로 균등 분포된 K=3개 층에서 feature f_k를 뽑아 FS block(hybrid attention pooling[MHA+MLP] → residual MLP+BatchNorm 블록 → 이진분류 MLP)을 각각 독립적으로 BCE로 학습.
- **투표(voting)**: 추론 시 각 FS block의 이진 예측 y_k(가중치 ω_k=1)와 VLM 자체의 free-form 텍스트 출력에서 뽑은 예측 y_vlm(가중치 ω_vlm=2, 동점 방지)을 가중합해 0.5 임계로 최종 판정.
- **데이터셋 구성**: 기존 expert demonstration {τ, g*}에서 positive={τ,g*,y=1}, negative는 같은 task 카테고리 내 다른 instruction으로 재라벨({τ,g≠g*,y=0}) — "의미는 통하지만 지시와 다른" 어려운 negative를 자동 생성. D_SMF-CALVIN(CALVIN 34 task), D_AHA(RLBench 79 task, OOD 평가 전용), D_SMF-DROID(실물, 6K train/276 test) 세 벤치.

## 실험·결과

- D_SMF-CALVIN(in-distribution): I-FailSense 90.6%(1 PoV)/88.2%(2 PoV) accuracy — zero-shot GPT-4o(58~63%), PaliGemma2-mix 베이스(47~53%, 거의 random), Qwen2.5-VL-7B(68~69%) 전부 상회. Ablation: LoRA만(Stage1) 85.7%→FS block 앙상블 추가(Stage1+2) 90.6%로 +5pt.
- D_AHA 일반화(Q2: 실패 유형 전이, Q3: OOD 시뮬 환경 전이): semantic misalignment만으로 학습했음에도 control error가 85%인 AHA 데이터셋에서 검출률 89.0% — AHA 논문 자체의 파인튜닝 baseline(7B 69.1%, 13B 70.2%)을 +19pt 상회, zero-shot VLM들은 7.5~50%로 저조.
- D_SMF-DROID(실물, Q4): 시뮬 전용 학습(D_SMF-CALVIN)만으로는 recall이 크게 떨어짐(accuracy는 오르나 F1 하락). FS block을 실물 데이터로 파인튜닝하면 71.0~74.3%까지 회복, Qwen2.5-VL-7B zero-shot과 동급 성능을 절반 파라미터로 달성.
- 저자 해석: semantic misalignment 검출은 "관찰된 모션이 instruction과 맞는가"를 언어-공간에서 정렬하는 학습이므로, control error(의미있는 모션 자체가 없음)에도 같은 정렬 능력이 전이된다는 가설.

## activation-steering 흐름 위치(검출 레이어; 외부 vs 내부)

I-FailSense는 활성화 스티어링 계보에 속하지 않는다 — 스티어링(개입) 자체를 하지 않고 순수 검출(분류)만 한다. 더 중요한 차이는 "무엇의 activation을 보는가"다: 이 논문이 앙상블하는 FS block들은 **행동을 생성하는 VLA policy의 hidden state가 아니라, 그 policy와 완전히 분리된 별도의 3B 관찰자 VLM의 내부 LLM 층**에 붙는다. 입력도 policy의 latent가 아니라 policy가 만들어낸 궤적을 렌더링한 픽셀(콜라주 이미지)이다. 즉 검출이 일어나는 층위가 "정책 내부 latent"가 아니라 "정책 출력(행동 결과)을 사후에 바라보는 별도 모델의 latent"다 — 우리 서베이의 SAFE(VLA 자신의 마지막 layer feature를 직접 probe)나 NOTALL/우리 pathway 검출(VLA 자신의 VL-SA/DiT block hidden state를 직접 hook)과는 개입/관측 대상 자체가 다르다. 비유하면 SAFE·우리 방법은 "환자 자신의 뇌파를 읽는다", I-FailSense는 "환자를 지켜보는 별도 관찰자의 판단을 신뢰한다"에 가깝다. 다층 앙상블(K개 층 투표)이라는 아이디어 자체는 우리의 "여러 depth에서 신호를 합친다"는 문제의식과 형태적으로 닮았지만, 대상이 관찰자 모델 내부라는 점에서 근본적으로 다른 층위다.

## 우리 프로젝트 연결(VLM출력 검출 vs latent 검출 대비)

- **층위 대비가 핵심**: 우리 프로젝트의 온라인 검출 난제는 "VLA 자신의 internal latent(VL-SA/DiT hidden state)만으로 phase/실패-type을 뽑아내 steering을 라우팅"하는 것이다. I-FailSense는 이 문제를 아예 다른 방식으로 우회한다 — VLA 내부를 보지 않고, 렌더링된 궤적 이미지를 별도 대형 VLM(3B, 우리 스티어링 개입보다 훨씬 무거운 forward pass)에 통째로 다시 넣어 "그럴듯해 보이는지"를 재추론한다. 이는 온라인 steering의 latency/온보드 제약에는 부적합하지만(3B VLM 추가 forward + 궤적 콜라주 대기 필요), 사후(post-hoc) 검출이나 offline 데이터 필터링/라벨링 용도로는 우리 latent probe보다 오히려 더 신뢰도 높은 신호일 수 있다.
- **semantic misalignment vs control error 구분이 VL/DiT pathway 구분과 개념적으로 겹친다**: semantic misalignment(잘못된 물체/목표 이해)는 우리 프레이밍의 goal(VL) pathway 실패에, control error(grasp 실패 등 실행 붕괴)는 motor(DiT) pathway 실패에 대응하는 것처럼 보인다. 그런데 이 논문의 핵심 결과는 "semantic misalignment만 학습해도 control error에 강하게 전이된다"는 것 — 외부 관찰자 관점에서는 두 실패 유형이 "관찰된 모션이 instruction과 맞는가"라는 동일한 검증으로 수렴한다는 뜻이다. 이는 우리가 내부 latent에서 VL-실패/DiT-실패를 별개 신호로 분리하려는 시도와 흥미로운 긴장 관계에 있다 — 출력 레벨에서는 두 실패 유형의 경계가 흐려질 수 있음을 시사한다(단, 이는 우리 latent-level 분리가 무의미하다는 뜻은 아니고, 관측 층위가 다르면 분리 가능성도 달라질 수 있다는 시사점 정도로 해석).
- **FS block의 다층 앙상블은 참고할 설계 패턴**: K개 층에서 독립적으로 학습한 분류 헤드를 투표로 합치는 방식은, 우리가 VL-SA/DiT 여러 depth에서 얻은 검출 신호(예: N16 pathway attribution의 block31 등)를 어떻게 통합할지 고민할 때 참고할 수 있는 단순하고 검증된 앙상블 전략이다.

## 면접 포인트(Q→A)

1. Q: "I-FailSense는 activation steering 논문인가?" A: "아니다. 순수 실패 검출(이진 분류) 논문이고 개입(steering)을 전혀 하지 않는다. 더 중요한 건 검출 대상 layer도 VLA policy의 activation이 아니라, policy가 만든 궤적을 이미지로 다시 렌더링해 별도의 3B 관찰자 VLM에 넣고, 그 관찰자 VLM 자신의 내부 LLM 층에서 뽑은 feature를 분류한다는 점이다 — '정책 내부 latent를 조작/probe'하는 우리 서베이의 다른 논문들과 관측 층위 자체가 다르다."
2. Q: "그럼 이 논문이 우리 프로젝트에 왜 필요한가?" A: "온라인 latent 기반 검출과 대비되는 '외부 관찰자' 접근의 상한선을 보여주기 때문이다. VLA 내부 latent를 전혀 안 보고도 궤적을 다시 렌더링해 대형 VLM으로 재추론하면 90%의 semantic misalignment 검출 정확도가 나온다 — 이는 latency/온보드 제약이 없는 상황(예: 오프라인 데이터 큐레이션, 사후 실패 리포트, 학습 데이터 필터링)에서는 우리 latent probe보다 강력한 대안일 수 있다는 걸 시사한다."
3. Q: "semantic misalignment 학습이 control error에 전이되는 이유는?" A: "저자 주장은, 두 실패 유형 모두 결국 '관찰된 모션이 instruction과 맞는가'를 언어-공간에서 정렬 검증하는 문제로 환원되기 때문이라는 것이다. Semantic misalignment는 의미있는 모션이 잘못된 대상을 향한 경우, control error는 의미있는 모션 자체가 실패한 경우인데, 둘 다 'instruction이 요구하는 모션 패턴과 관찰된 모션 패턴의 불일치'라는 동일한 신호로 검출된다."
4. Q: "SAFE와 I-FailSense를 층위 관점에서 비교하면?" A: "SAFE는 VLA 자신의 마지막 layer hidden state를 직접 probe한다 — '정책 내부 latent'가 검출 대상이자 (미래 연구로 제안하는) steering 대상이다. I-FailSense는 정책 출력(궤적 픽셀)을 별도 3B VLM에 다시 넣어 그 VLM의 내부 층을 probe한다 — 정책과 완전히 분리된 관찰자 모델의 latent가 검출 대상이며, steering으로 자연 확장되지 않는다(정책 activation에 접근하지 않으므로)."

## 한계·비판

- 3B 파라미터 VLM의 별도 forward pass(+궤적 콜라주 대기, T=4 프레임 필요) — 실시간 온라인 steering 라우팅에는 latency/연산 부담이 SAFE류 경량 probe(<1% 오버헤드) 대비 훨씬 크다. 저자도 실시간 제어 적용은 명시적으로 다루지 않는다.
- D_AHA는 공개되지 않아 저자가 자체 재구성(400개 negative pair)했다 — 원 AHA 벤치마크와 완전히 동일한 분포인지 검증 불가.
- Voting 가중치(FS block=1, VLM=2)가 경험적으로 고정되어 있고, 이 선택에 대한 ablation/민감도 분석이 없다.
- Qwen2.5-VL-7B의 zero-shot recall(0.92)은 높지만 precision(0.63)이 낮다는 관찰로 "우리 negative가 어렵게 설계됐다"는 저자 주장을 뒷받침하는데, 이는 동시에 데이터셋 설계(같은 task category 내 instruction 치환)가 특정 실패 패턴에 편향돼 있을 가능성도 열어둔다.
- 실물 전이(D_SMF-DROID)는 FS block만 재학습하고 LoRA된 VLM 백본은 그대로 재사용하는데, 시뮬 전용 LoRA가 실물 전이를 오히려 방해할 수 있다는 결과(D_SMF-CALVIN 단독 파인튜닝 시 recall -0.4)가 나왔음에도 이 개입점(LoRA 자체의 재학습 필요성)에 대한 추가 분석은 없다.
- Future work로 "실패 인식→회복(recovery)"을 명시했지만 본 논문 범위에서는 검출까지만 다루고, 검출 결과를 실제 정책 개입(steering이든 replanning이든)에 연결하는 실험은 없다.
