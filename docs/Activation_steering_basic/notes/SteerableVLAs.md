# InSight: Self-Guided Skill Acquisition via Steerable VLAs (Wang et al. 2026)

- 출처: arXiv:2606.24884v1 (2026-06-23, cs.RO) · Stanford University / Princeton University(Maggie Wang, Lars Osterberg, Stephen Tian, Ola Shorinwa, Jiajun Wu, Mac Schwager) · PDF: `docs/references/Steerable VLAs.pdf` · 정독 섹션: 전체(§1 Intro ~ §5 Conclusion + Appendix A~C 프롬프트) · 서베이 배치: §6 VLA · tier: must · 한줄 역할: VLA를 primitive-level 언어 라벨로 조건화해 "steerable"하게 만들고 VLM이 빠진 primitive를 찾아 데이터를 모아 LoRA로 재학습하는 skill-acquisition flywheel — activation이 아니라 언어/데이터 층위의 "steering"이라 우리 hidden-state steering과 개입 층위가 근본적으로 다른 대조 사례.
- **주의(제목 동명이인)**: 이 PDF는 Berkeley/Levine 그룹의 "Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control"(Chen et al., arXiv:2602.13193, 2026-02)이 **아니다**. 그 논문은 본 논문의 참고문헌 [9] "Steerable Policies"로 인용만 되는 별개 논문("language-conditioned primitives to guide behavior at test time"로 한 줄 요약됨). 실제 배정 PDF는 Stanford/Princeton 팀의 INSIGHT다.

## 문제·동기

VLA는 학습 데이터에 있는 스킬로 능력이 제한된다. 새 스킬 습득에는 human demo 수집+재학습 비용이 크고, real-world RL은 샘플효율·안전성 문제로 비실용적. 저자들의 핵심 통찰은 "조작 스킬은 compositional"이라는 것 — sweep과 scoop은 approach/lower primitive를 공유하고 다른 건 접촉 모션뿐이며, block flip은 pick-and-place의 grasp-lift를 재사용하고 rotate만 추가된다. 문제는 이런 primitive가 VLA 내부에 이미 있어도 전체 task instruction에 뒤엉켜(entangled) 있어 개별적으로 불러낼(steer) 수 없다는 것. Mars 로봇 예시(scoop만 학습했는데 dust storm 청소를 위해 sweep이 필요)로 동기를 제시한다.

## 핵심 아이디어

"Steerable"의 정의: VLA를 primitive-level 언어 라벨(예: "move gripper to the bowl", "lift upward", "twist", "pour")로 조건화해 개별 동작 단위를 선택적으로 유도할 수 있게 만드는 것 — 이는 hidden state 개입이 아니라 **conditioning input(언어 프롬프트) 세분화**다. 여기에 VLM을 "누락 primitive를 식별하고 시연을 만들어 정책을 확장하는 능동적 에이전트"로 재정의한다(SayCan/VoxPoser/Code-as-Policies류는 test-time에 기존 스킬만 조합하고 정책 자체는 갱신되지 않는 것과 대비). 두 축: (1) 무라벨 자동 primitive segmentation으로 steerability를 확보하고, (2) VLM-guided flywheel(gap 식별 → 저수준 컨트롤러로 시연 → VLM oracle 성공판정 → LoRA 재학습)로 그 steerable 어휘(vocabulary)를 지속적으로 넓힌다.

## 방법(steerable 메커니즘; 개입 층위)

- **Stage 1 자동 primitive segmentation**: VLM이 task instruction을 ordered primitive sequence로 분해(예시 제공, 필요시 신규 primitive 제안 허용). Demo는 gripper open/close 전이로 경계를 자르고, EE pose·모션크기(xy/z)·dominant-axis tag(xy/z/rxy/rz)를 VLM에 넘겨 frame별 primitive를 매칭. 각 segment를 독립 학습 episode로 삼아 π0.5(Gemma-2B backbone + Gemma-300M action expert)를 LoRA fine-tune, primitive label을 language prompt로 사용. Action space에 learned progress channel(구간 내 정규화 timestep으로 지도, ∈[0,1))을 추가해 termination 신호(threshold≈0.95 / EE motion 저하 / OOD "move to"에 한해 VLM completion check)로 쓴다.
- **Stage 2 VLM-guided skill acquisition**: 새 task를 VLM planner가 primitive sequence로 분해, 알려진 vocabulary V와 대조해 gap(V에 없는 primitive)을 표시. 각 gap은 VLM이 단일축 모션(translation dx/dy/dz 또는 gripper-local rotation drx/dry/drz, 부호있는 크기)으로 파라미터화 → **VLA가 아니라 저수준 스크립트 컨트롤러**가 실행. Known primitive는 steerable VLA가 언어조건화로 직접 실행. VLM oracle이 전/후 이미지 비교로 task 성공을 판정 → 성공한 gap rollout만 학습셋에 추가 → n rollout 후 V∪{new primitives}로 VLA를 LoRA 재학습. 이후 plan에서 그 primitive는 "gap"(컨트롤러 실행)이 아니라 "known"(VLA 실행)이 된다.
- 개입은 (a) inference 시 언어 prompt 세분화, (b) 데이터 누적 후 LoRA 파라미터 업데이트, 이 두 가지뿐. Hidden activation을 읽거나 쓰는 절차는 전혀 없다.

## 실험·결과

- LIBERO 시뮬 block flipping: pick-and-place demo만으로 학습된 VLA에 rotate-block gap을 채워 75% 성공(246 acquired rollout, 479 attempts) vs RL(SAC) baseline 0%(reach 23%/grasp 10%에 그치고 flip 미완).
- LIBERO drawer closing(OOD initial state): open-only demo라 close 부재. Approach primitive가 OOD(열린 서랍)에서 progress 신호가 열화 → VLM completion check로 강제 전환. 82 episode 중 70개 성공 수집 → 재학습 후 25회 평가 100% 성공, open 능력도 보존.
- 실물 xArm 6DoF: twist cap open 92%(vs CaP-X 32%, π0.5-only 0%), pour beans 96%(vs 16%, 0%), twist-then-pour 14-primitive 조합 80%(vs CaP-X 4%; π0.5 baseline은 twist/pour 자체가 0%라 조합 불가). CaP-X = zero-shot VLM 코드조합(Code-as-Policies류) baseline, 정책 자체는 갱신 안 함.
- 시간효율: trial당 wall-clock이 CaP-X보다 짧다(twist 55.1s vs 109.1s, pour 124.1s vs 206.9s) — VLM 호출을 gap에만 국한하고 known primitive는 VLA가 직접 실행하기 때문.
- Base skill retention: 신규 primitive 추가 후에도 원래 top/side pick-and-place 100% 유지(N=15). Sweeping from scooping demo: lateral-push gap 하나만 추가해 5/5 성공.

## activation-steering 흐름 위치

이 논문은 서베이의 activation-steering 계보(ActAdd/CAA/RepE/conceptor류 — forward pass 도중 hidden state h를 h+αv 또는 h·Mᵀ로 직접 조작)와 개입 층위가 전혀 다르다. "Steerable"이라는 단어를 쓰지만 실제 메커니즘은 (1) language-instruction 세분화(primitive label conditioning — 저자 스스로 Related Work에서 STEER[10]·Steerable Policies[9]·Hi Robot과 한 계열로 분류)와 (2) VLM 주도 데이터 flywheel + LoRA 재학습(파라미터 업데이트 기반 continual learning)이다. Hidden activation을 읽어 phase/실패를 판정하는 절차도 없다 — 대신 VLM이 이미지(전/후, wrist view)를 직접 보고 primitive 완료·task 성공을 명시적으로 판정한다(Appendix B의 PRIMITIVE_DONE, TASK_COMPLETION 프롬프트). 이 논문은 activation-steering 서베이에서 "steering"이라는 용어가 자연어/데이터 층위에서도 쓰인다는 것을 보여주는 **경계 사례(boundary case)**로 배치하는 것이 맞다.

## 우리 프로젝트 연결(개입 층위 비교; activation vs 명령/segmentation)

- **개입 층위가 완전히 다르다**: 우리는 DiT/VL의 residual/hidden state h에 conceptor projection(h'=h·Mᵀ)을 inference마다 적용해 파라미터 변경 없이 즉시 행동을 바꾼다. INSIGHT는 (a) 프롬프트 층위(primitive label 텍스트 교체)와 (b) 파라미터 층위(LoRA 재학습)만 조작한다 — 둘 다 우리가 지향하는 "재학습 없는 즉시 개입"이 아니라, 오히려 재학습(LoRA)이 개입의 본체다.
- **phase 개념의 피상적 유사성**: INSIGHT의 primitive 분해(twist-then-pour의 14-step plan 등)는 우리 "rollout phase"와 표면적으로 비슷해 보이지만, INSIGHT는 phase 경계를 VLM이 이미지+gripper transition으로 explicit하게 라벨링/판정하는 반면, 우리 핵심 난제는 explicit 라벨 없이 hidden activation만으로 phase를 online 추정하는 것이다. INSIGHT는 우리가 풀려는 문제를 VLM oracle로 우회한 사례 — "phase 신호가 주어지면 무엇이 가능한가"의 상한선 참고자료로는 유용하나 방법론 이식은 어렵다.
- **실패 처리 관점의 차이**: 우리는 succ/fail latent 분리+steering으로 실패를 성공으로 되돌리는 것이 목표인데, INSIGHT는 실패 rollout을 그냥 버린다(VLM oracle이 실패면 discard, 성공만 학습에 편입) — 실패 latent를 분석하거나 개입 대상으로 삼지 않는다. Limitation에서도 "failure analysis 통합은 future work"라고 명시(우리 문제의식과 정확히 겹치는 빈 공간이지만 이 논문은 손대지 않음).
- **VL/DiT pathway 분리와의 대응 없음**: INSIGHT는 π0.5 전체를 하나의 언어조건화 정책으로 다루고 내부 pathway(goal-VL vs motor-DiT) 구분을 하지 않는다 — 우리 pathway-resolved steering과 직접 비교할 내부 구조가 없다.
- 결론: "steering"의 다른 의미(behavior-level control interface + continual data acquisition)를 보여주는 대조 논문으로 배치. 방법론 이식보다는 activation steering의 스코프를 정의하는 경계 사례로 인용하는 것이 적절하다.

## 면접 포인트(Q→A)

**Q1. 이 논문 제목이 "Steerable VLAs"인데 activation steering 논문인가?**
A. 아니다. 실제 제목은 "InSight: Self-Guided Skill Acquisition via Steerable VLAs"(Wang et al., Stanford/Princeton, arXiv:2606.24884, 2026-06)이고, "steerable"은 VLA를 primitive-level 언어 라벨로 조건화해 개별 동작을 유도할 수 있다는 뜻이다. Hidden state를 조작하는 activation steering이 아니라 언어 프롬프트 세분화 + LoRA 재학습이 메커니즘이다. 제목이 유사한 "Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control"(Chen et al., Berkeley/Levine 그룹, arXiv:2602.13193)은 이 논문이 아니라 이 논문의 참고문헌 [9]로 인용되는 별개 논문이다.

**Q2. INSIGHT의 두 단계는 무엇인가?**
A. (1) 자동 primitive segmentation: VLM이 demo를 gripper transition+EE motion 축으로 primitive 단위로 분해해 VLA를 LoRA fine-tune(primitive label conditioning). (2) VLM-guided skill acquisition: 새 task에서 VLM이 vocabulary에 없는 primitive gap을 찾아 저수준 스크립트 컨트롤러(축+크기 파라미터)로 시연을 모으고, VLM oracle이 성공을 판정한 rollout만 모아 VLA를 재학습해 vocabulary를 확장한다.

**Q3. Primitive gap은 누가 실행하나? VLA인가?**
A. 아니다. Known primitive만 steerable VLA가 언어조건화로 실행하고, gap(새 primitive)은 VLM이 제안한 단일축 모션 파라미터(axis+signed magnitude)를 스크립트 저수준 컨트롤러가 그대로 실행한다. 성공하면 그 rollout이 학습 데이터가 되어 VLA가 재학습된 이후에야 그 primitive가 VLA로 실행 가능한 "known"이 된다.

**Q4(우리 프로젝트 관점). 이 논문이 우리 phase-matched steering 문제에 주는 시사점은?**
A. INSIGHT는 phase/completion 판정을 VLM oracle(이미지 비교)로 explicit하게 해결한다 — 즉 "phase를 알 수 있다면 무엇이 가능한가"의 참고 상한선을 보여준다. 하지만 우리 핵심 난제는 정확히 그 explicit 신호 없이 hidden activation만으로 phase/실패유형을 온라인 추정하는 것이라, 이 논문은 문제를 풀지 않고 VLM으로 우회했다는 점에서 방법론적으로 이식할 게 적다.

## 한계·비판

- 저자 스스로 명시: gap 실행이 single-axis motion으로 제한(복합 모션 표현 불가), 실패 rollout은 그냥 버려서 실패 분석/피드백을 활용하지 못함(future work), 매 rollout마다 사람이 환경을 리셋해야 함(진짜 autonomous가 아님), 낮은 DoF 로봇팔에서만 검증(mobile manipulator/humanoid 미검증).
- VLM(Gemini 3 Flash) 의존도가 매우 높다 — segmentation, planning, gap parameterization, oracle 판정 네 역할 전부를 VLM 프롬프트에 위임하는데, VLM의 공간추론 오류(예: PREANALYZE 프롬프트가 "wrist camera 축 오분류 방지"에 상당한 분량을 할애)가 파이프라인 전체 실패로 전파될 수 있다. 실제로 드로어 closing 실험에서 "incorrect axis selection이 dominant failure mode"라 인정.
- 평가 규모가 작다(실물 실험 25 trial/조건, sweeping 5 trial) — 통계적으로 얇다.
- arXiv 2606.24884, 2026-06-23의 매우 최신 프리프린트로 동료검토/외부 재현이 없다.
- "steerable"이라는 용어를 activation-steering 문헌과 다른 의미로 사용해 서베이 인용 시 혼동 위험 — 인용할 때 개입 층위 차이를 반드시 명시해야 한다.
