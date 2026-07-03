# Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models (Grant & Zhao 2026)

- 출처: arXiv 2603.19233v1 [cs.RO] (Case Western Reserve Univ., Grant/Zhao/Wang) · PDF: `docs/references/NOT ALL FEATURES ARE CREATED EQUAL_ICLR2026.pdf` · ICLR 2026 · 섹션=§6 VLA(서베이 배정, 선행토대) — 논문 전체(6개 아키텍처 cross-architecture study)가 대상, §4.5 Pathway Specialization·§4.6 SAE·Appendix E.1(subspace injection)·G.3-G.4(boosting/temporal ablation) 중심 정독 · tier=must · 한 줄 역할: **VL(goal)/DiT(motor) pathway 기능분리를 π0.5·SmolVLA·GR00T 세 아키텍처에서 cross-validation한 최대규모 연구** — 우리 프로젝트 pathway-resolved steering의 직접 근거이자 GR00T DiT feature fragility 수치의 출처.

## 문제·동기
VLA는 perception+language+motor를 한 아키텍처로 결합하지만 멀티모달 입력이 행동으로 어떻게 번역되는지 불투명하다. 실패 시 디버깅 수단이 행동 관찰뿐이고(classical robotics처럼 내부 kinematics/제어모델 검사 불가), LLM에서 성숙한 SAE·activation steering이 VLA로 확장 가능한지도 미검증이었다. VLA는 vision/language/proprioception이 뒤섞인 이종 토큰 시퀀스라 mean-pooling이 action-critical 정보를 파괴하는 등 LLM interp와 다른 함정이 있고, causal 검증도 rollout 기반 시뮬레이션이 필수라 LLM보다 비용이 크다.

## 핵심 아이디어 (VL/DiT 기능분리, feature fragility)
6개 아키텍처(80M~7B: π0.5, OpenVLA-OFT, X-VLA, SmolVLA, GR00T N1.5, ACT) × 4벤치마크 × 394,000+ rollout에 activation injection + per-token SAE + linear probe를 균일 적용한 cross-architecture mechanistic study.
- **visual pathway dominance**: null-prompt에도 baseline activation을 주입하면 near-identical 행동이 회복(π0.5 cosine 0.999, SR 73~77% 회복) — 언어보다 시각/scene 맥락이 행동을 지배.
- **pathway specialization (핵심)**: multi-pathway 아키텍처(π0.5, SmolVLA, GR00T) 전부에서 **expert/action(DiT류) pathway = motor program encode, VLM pathway = goal semantics encode**가 재현. expert-layer injection이 VLM-layer injection보다 ~2배 큰 행동 displacement(SmolVLA: 15.8% vs 9.0%).
- **GR00T N1.5 구조**: 12 Eagle LM + 4 VL-SA + 16 DiT = 32 layers. DiT가 ablation에 가장 민감(40~80% SR drop), Eagle은 중간, VL-SA는 per-token SAE 품질이 낮음(83~89% EV)에도 가장 resilient.
- **feature fragility**: GR00T DiT는 9배 증폭만으로 ΔSR=−68pp — π0.5 expert(−84pp@−3x suppression)에 버금가는 catastrophic sensitivity. OFT(−6pp@−3x)·SmolVLA(−3pp@5x)는 상대적으로 강건 — fragility는 아키텍처·훈련 레짐 의존적이지 VLA 보편속성이 아니다.
- **runtime 진단 가능성 주장**: expert-pathway injection → active misdirection(엉뚱한 목표로 적극 이동), VLM-pathway injection → passive stalling(정지) — pathway별 activation norm 모니터링으로 motor error/goal error 구분 가능하다고 제안(단, offline injection 관찰이며 online classifier 검증은 없음).

## 방법 (per-token SAE, frequency-weighted contrastive selection, ablation/boosting 비대칭)
- **per-token SAE**: TopK(k=64) sparsity, tied encoder-decoder weight, 4x/8x expansion. action token별(50개) 독립 처리가 필수 — mean-pooling은 approach/manipulation/terminal의 이종 시간구조를 파괴해 R2>0.95인데도 task 완전 실패(π0.5 mean-pool 0.4% vs per-token 70% SR). 단, X-VLA와 GR00T VL-SA는 예외적으로 mean-pooling이 EV를 개선(GR00T VL-SA 83~89%→99%) — pooling 전략의 최적값이 아키텍처·pathway마다 다르다.
- **frequency-weighted contrastive selection**: score_f = d_f × freq_f (d_f=Cohen's d, concept-present vs concept-absent 활성화차; freq_f=top-k active에 등장하는 샘플 비율). TopK sparsity 하에서는 평균활성이 높아도 개별 샘플의 top-64에 항상 뽑히지 않는 feature가 있어, 빈도 가중 없이는 causal relevance를 과대평가한다.
- **ablation/boosting 비대칭**: 2~5개 concept feature ablation은 유의미한 효과 없음(p=0.975, mean Δ=+3.3% — redundant encoding으로 모델이 보상). 반대로 feature boosting은 7배에서 −14%(p=2.27e-4), 15배에서 −50.7% — "제거엔 강건, 첨가엔 취약"한 비대칭.
- **boosting은 양방향 파괴적**: dampening(α=−0.5)과 boosting(α=+1.0) 모두 baseline 97.1%→5.7%/13.3%로 근붕괴. ablation 이후 boosting으로 복구도 불가(한 번 오염되면 에러가 누적). concept substitution(OPEN ablate + PUT boost)도 실패 → 각 concept이 분리된 subspace를 점유.

## 실험·결과 (fragility 수치, subspace injection E.1 등)
- **fragility**: GR00T DiT −68pp@9x amp; π0.5 expert −84pp@−3x suppression; OFT −6pp@−3x; SmolVLA −3pp@5x(non-monotonic: 0.5~2x는 완만한 증폭, 5x에서만 급락).
- **temporal ablation (Table 15, GR00T, 160 조건×32층, MLP-targeted hook)**: DiT early-window ablation(−50.3pp, 58% destructive) ≈ full-episode(−50.8pp) ≫ mid/late(−12pp) — motor program이 approach phase에 조기 커밋되고 이후 DiT feature는 expendable. Eagle LM은 완만한 프로파일(early −15.1pp vs late −11.8pp, sustained task-context 역할과 부합). task 난도와 상관: libero long(303-step) early-DiT drop −62pp vs libero goal(108-step) −44pp.
- **subspace injection (E.1, π0.5 layer17 LDA)**: goal-discriminant 20/1024차원(2%)만 주입해도 goal 식별 유지(3/5 pair 100% success), full injection(1024차원)은 완전 붕괴(4/5 pair 0%). goal-판별 차원(417,909,934)과 action-변조 차원(62,618,14)이 거의 겹치지 않음 — 같은 layer 안에서도 goal identity와 motor execution이 분리된 subspace에 산다는 **causal 증거**.
- GR00T probe: 32층 전체 100% task ID, 96.4% success prediction(DiT L14는 97.7%). GR00T cross-task override rate 57.0%(suite-dependent: goal 85.6% vs long 33.3%).

## activation-steering 흐름 위치
Häon et al. 2025(CoRL, π0/OpenVLA FFN-neuron steering, 우리가 이미 재현) → Molinari et al.(world-model probing) → Khan et al.(Magma SAE steering, generality 미평가)로 이어지는 **단일 아키텍처** VLA mech-interp을 6개 아키텍처 cross-validation으로 확장한 첫 사례. 같은 notes 폴더의 `SAE_VLA_pi05.md`(Swann 2026)와 자매 관계: 그쪽은 π0.5 단일 아키텍처에 SAE 학습+general/memorized 분류+steering까지 완결하고, NOTALL은 pathway specialization의 cross-architecture 재현(π0.5/SmolVLA/GR00T)에 특화. 우리 pathway 분리 근거는 NOTALL, memorized-direction 오염 경고는 SAE_VLA_pi05에서 가져온다.

## 우리 프로젝트 연결
- **빌리는 것**: (1) VL(Eagle+VL-SA)=goal/what, DiT=motor/how 기능분리의 cross-architecture 인과 증거(expert injection이 VLM보다 2배 displacement, 세 아키텍처 재현) — pathway-resolved steering의 직접 근거. (2) fragility 수치(GR00T DiT −68pp@9x, 양방향 파괴적 boosting)는 우리 conceptor steering을 **projective(곱셈형 soft projection, 하드 클리핑 아닌 [0,1] 고유값) + suppressive(첨가보다 억제 위주) + early(초기 phase 개입) + 소강도(저배율)**로 설계해야 하는 직접적 사전 경고 — additive 고배율은 반드시 붕괴한다. (3) temporal ablation(early commit, Table 15)이 phase-matched steering의 phase-bin 설계 근거: DiT 개입은 approach/early phase에 집중해야 효과가 크고 late phase는 expendable.
- **메우는 곳(NOTALL의 한계)**: (1) 전부 offline injection/ablation — online(추론 중 실시간) phase/failure-type 식별은 하지 않음, 우리 핵심 미해결 문제. (2) "pathway별 activation norm으로 motor/goal error 구분 가능"은 제안·관찰 수준이고 causal online detector로 검증되지 않음 — 우리 online phase/type 식별이 이 공백을 메운다. (3) succ/fail 대조 실험이 없음(baseline vs null/cross-task 뿐) — 우리 C_success ∧ ¬C_failure contrastive conceptor가 이 축을 새로 도입. (4) phase별 성공 분포를 타깃한 steering(phase-matched)은 없음 — temporal ablation(파괴 실험)만 있고 phase-conditional positive steering 실험은 없음.

## 면접 포인트 (Q→A)
1. Q: "VL/DiT 기능분리가 GR00T에서 구체적으로 어떻게 확인되나?" A: "32층(Eagle 12 + VL-SA 4 + DiT 16) 전체 ablation에서 DiT는 40~80% SR 하락으로 가장 민감, Eagle은 중간, VL-SA는 per-token SAE 품질이 낮음(83~89% EV)에도 가장 resilient하다. probe는 32층 전체에서 100% task ID·96.4% success prediction을 달성해 두 pathway가 서로 다른 정보(goal vs motor)를 encode함을 뒷받침한다."
2. Q: "이 논문의 fragility 결과가 우리 steering 설계에 왜 중요한가?" A: "GR00T DiT는 9배 증폭만으로 SR이 68pp 빠지고, dampening·boosting 모두 양방향으로 파괴적이다. 즉 단순 additive 고배율 벡터 스티어링은 GR00T DiT에서 거의 작동 불가에 가깝다 — 우리가 conceptor 기반 soft projection(곱셈형)과 저배율·억제 위주 개입을 택한 이유의 직접 근거다."
3. Q: "온라인으로 phase나 실패 유형을 식별할 수 있다는 근거가 이 논문에 있나?" A: "직접적으로는 없다. offline injection/ablation으로 pathway별 activation norm이 motor error(active misdirection)와 goal error(passive stalling)를 사후적으로 구분한다고 관찰했을 뿐, 실시간 online classifier로 검증하지 않았다. 이게 우리가 메우려는 핵심 공백(online phase/failure-type 식별)이다."
4. Q: "temporal ablation 결과가 phase-matched steering에 주는 시사점은?" A: "GR00T DiT는 early-window(approach phase) ablation이 full-episode 파괴력(−50.3pp vs −50.8pp)에 거의 근접하고 mid/late는 −12pp에 불과하다 — motor program이 trajectory 초기에 커밋됨을 뜻한다. 우리 phase-matched DiT steering도 초기 phase에 개입을 집중해야 효과가 클 것이라는 사전 증거다."
5. Q: "이 논문과 SAE_VLA_pi05(Swann 2026)의 관계는?" A: "SAE_VLA_pi05는 π0.5 단일 아키텍처에 SAE를 학습시켜 general/memorized 분류+steering까지 완결한 논문이고, NOTALL은 6개 아키텍처를 cross-validation해 pathway specialization(VL=goal/DiT=motor)이 아키텍처와 무관하게 반복됨을 보인 논문이다. 우리는 pathway 분리 근거는 NOTALL에서, memorized-direction 오염 경고는 SAE_VLA_pi05에서 가져온다."

## 한계·비판
- 시뮬레이션 전용(LIBERO/MetaWorld/SimplerEnv/ALOHA) — real-world fine-tuning 후에도 pathway specialization·fragility가 유지되는지 미검증.
- 다수 ablation 수치(Table 9,14,16,18 등)가 **full-layer residual stream hook**(과교정된 개입)으로 수집돼 효과가 인플레이트됨. 저자가 MLP-targeted hook으로 재확인하니 concept ablation은 무효과(p=0.975)로 뒤집혔고, π0.5 temporal ablation(Fig.9, Table 14)의 phase-dependence는 **아티팩트**로 판정됨 — cross-architecture temporal 결론은 GR00T Table 15만 유효.
- **steering→SR 상승 positive 실험이 없음**: boosting은 전부 파괴적 결과만 보고하고, "특정 방향을 조종해 SR을 올렸다"는 사례가 없음 — 우리 목표(ΔSR 상승)와 반대 방향(파괴)의 증거만 제공한다는 점에 유의.
- cross-task injection의 temporal misalignment confound 가능성을 저자도 인정(반박 근거로 99.8% source-trajectory 유사성 제시하나 완전 배제는 아님).
- concept ablation은 15,096+ pair 규모지만 kill-switch의 70%가 object concept(bowl/cabinet 등)이고 motion primitive는 소수 — "실패 유형(failure-type, goal vs motor)"과 직결되는 개념 분해는 아님.
