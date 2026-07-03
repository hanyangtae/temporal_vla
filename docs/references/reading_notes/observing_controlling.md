# Observing and Controlling Features in Vision-Language-Action Models — 정독 노트

- **저자/출처**: Hugo Buurmeijer*, Carmen Amo Alonso*, Aiden Swann†, Marco Pavone*‡
  (Stanford Aero/Astro, Mechanical Eng, NVIDIA Research). arXiv:2603.05487v1 [cs.RO], 2026-03-05.
- **PDF**: `docs/references/Observing and Controlling Features in Vision-Language-Action Models.pdf` (9쪽).
- **한 줄 요약**: **SAE를 쓰지 않는다.** 순수하게 **선형 observer(probe) Wx+b** 로 로봇 state/action을
  layer마다 관측하고, **최소 노름 additive control u=(ζ_target−ζ)·W/‖W‖²** 로 표현공간을 밀어
  출력(gripper/height/speed)을 제약구간 안으로 steer한다. VLA 백본 재학습 없음, 추론 시간 개입.

> 우리 프로젝트 메모("NOT an autoencoder" 논문)와 정확히 일치함을 확인했다. 프롬프트에 적힌
> observer 식과 control 식이 논문 원문과 그대로 맞는다(아래 §2, §3에서 원식 인용).

---

## 1. 문제·주장 (무엇을, 어떤 모델/벤치에서, SAE 여부)

- **문제의식**: LLM의 mechanistic interpretability / activation steering 도구가 VLA로 자명하게
  이식되지 않는다. VLA는 (i) multimodal 입출력, (ii) transformer+diffusion/flow-matching 하이브리드,
  (iii) 물리 세계와의 **closed-loop** 상호작용 때문에 LLM(open-loop 텍스트 생성)과 다르다.
- **핵심 주장 두 개(새 개념 정의)**:
  - **Feature-Observability (Def 1)**: layer ℓ의 activation xℓ ∈ R^d 에서 feature ζ ∈ R^n 을
    복원하는 map(observer) fℓ: R^d→R^n, fℓ(xℓ)=ζ 가 존재하면 그 feature는 layer ℓ에서 관측가능.
  - **Feature-Controllability (Def 2)**: 원하는 집합 D⊂R^d 가 주어질 때, 개입 ˜xℓ=gℓ(xℓ)를
    ℓ..T로 전파했을 때 ζ∈D 가 되게 하는 map(controller) gℓ 이 존재하면 controllable.
  - 두 성질은 **독립**이다(관측가능하지만 제어불가, 또는 그 반대가 가능 — 고전 제어이론 표준,
    Kalman[7] 인용). 단, 제안한 controller가 observer를 쓰기 때문에 실무적으로는 LC ⊆ LO.
- **모델/벤치(두 아키텍처 커버)**:
  - **π0.5** (transformer + flow-matching "action expert" 하이브리드) — **Libero** 데이터셋/시뮬.
  - **OpenVLA** (autoregressive transformer, DINOv2+SigLIP 비전, Llama2 백본, action token) —
    **BridgeData V2** 데이터셋.
  - 평가 시뮬은 **Libero spatial suite 10개 task, task당 10 rollout/method**, single NVIDIA 5090 GPU.
- **SAE 사용 여부 — 명확화**: **본문 방법에는 SAE가 전혀 없다.** SAE는 오직 **Limitations & Future
  Work** 에서 "observer 학습에 라벨이 필요하다"는 한계를 언급하며 *"향후 라벨 없이 feature를
  발견하려면 SAE 같은 self-/unsupervised 방법을 쓸 수 있다"* 는 **미래 방향**으로만 등장한다.
  즉 이 논문은 우리가 검토 중인 세 논문 중 **비-SAE(선형 observer/probe) 대조군**이 맞다.
- **작업 범위 한정(중요)**: 저자들은 명시적으로 **transformer 부분만** 다룬다. flow-matching /
  diffusion head로의 확장은 **future work로 미뤄둔다.** 즉 π0.5에서도 개입 지점은 VLM/transformer
  쪽이지 action-expert(flow) 내부가 아니다. — 우리 관점에서 결정적 제약(아래 §7에서 상술).

---

## 2. Linear Observer 수식 (정확한 형태·타깃·학습·정확도)

- **형태 (Eq. 3)**:
  ```
  fℓ(x) := Wℓ x + bℓ ,   Wℓ ∈ R^{d×n},  bℓ ∈ R^n
  ```
  프롬프트에 적힌 Wx+b 와 동일. 일반화판 fℓ(x)=ν(Wℓx+bℓ) (ν = 알려진 단조 비선형)도 성립한다고 언급.
- **타깃 feature ζ (로봇 state/action으로 한정)**:
  - state s = (x, y, z, ϕ, θ, ψ, g):
    - (x,y,z) ∈ R³ Cartesian **위치(position)**,
    - (ϕ, θ, ψ) ∈ [0,2π)³ **자세(roll/pitch/yaw, orientation)**,
    - g ∈ [0,1] **정규화 gripper 개폐(aperture)**.
  - action a = Δs (state 공간에서의 상대 변위). → position/orientation은 **연속 회귀**, gripper는 **이진**.
  - 더 추상적인 semantic feature(affordance, 관계 술어, task-level goal)는 **명시적으로 future work로 유보.**
- **학습 방식 (Algorithm 1)**: 입력 s^(i) 를 layer ℓ까지 forward 해 activation x_ℓ^(i) 수집 →
  {x_ℓ^(i), ζ^(i)} 로 Wℓ, bℓ 학습. **모든 layer ℓ∈[1..T]에서 각각** probe를 학습.
  - position/orientation = **연속 라벨 regression probe**, gripper = **이진화 라벨 binary probe**.
  - **주의(원문 불일치)**: 본문은 "regression task"라고 하면서 Eq.(4)에는 **cross-entropy(BCE) 형태**
    (ζ·log(...) + (1−ζ)·log(1−...))를 적어 두었다. 연속 회귀에 BCE는 어색하다. 해석하면 Eq.(4)는
    gripper(이진) 케이스의 손실이고, 연속 position/orientation은 최소제곱 회귀로 학습한 것으로 읽힌다.
    **논문 표기가 다소 모호**하므로 그대로 옮겨 두되, 실제로는 "이진=BCE / 연속=회귀"의 혼용으로 본다.
- **robustness 보증 (Remark 1)**: layer마다 fℓ 학습이 관측의 안정성을 보장하지 않는다(ε 섭동이 ζ를
  임의로 크게 흔들 수 있음). 그래서 학습 후 **경험적으로** ‖fℓ(x+ε) − fℓ(x)‖ < δ 를 검증한다.
- **보고된 정확도 (Fig. 3)**: π0.5(Libero) / OpenVLA(BridgeData V2)에서
  - 좌: 학습된 probe의 **MAE** vs "train mean 예측" baseline,
  - 우: **accuracy** vs "majority class" baseline,
  - **단, "best performant layer"만** 표시한다. → **layer별 정확도 수치 표는 없다.** 정량치는 그림 안에만
    있고 본문 산문에는 구체 숫자가 명시되지 않는다. (헤드라인은 "state/action이 선형으로 잘 관측된다"
    는 질적 주장이며, layer-by-layer 정확도 곡선의 정확한 값은 추출 텍스트로 확인 불가.)

---

## 3. Control 수식·메커니즘 (최소 additive control, depth 발견)

- **controller 형태 (Eq. 5)**: gℓ(x) := x + uℓ (단순 additive 개입, uℓ = 표현공간 섭동).
- **최적화 (Eq. 6)**: uℓ = argmin_u ‖u‖²₂  s.t.  fℓ(xℓ+u) ∈ D.
  → "관측된 feature를 목표영역 D로 넣는 **최소 노름** 개입". D는 observer의 preimage로 들어감.
- **닫힌 해 (Eq. 7)** — D=[ζmin, ζmax] ⊂ R (1차원 scalar feature)로 두고 선형 observer 사용 시,
  Cheng & Amo Alonso 2024([3], Thm 4.1)에 따라 **closed-form**:
  ```
  uℓ = (ζmax − ζℓ) · Wℓ/‖Wℓ‖²₂     if ζℓ > ζmax
  uℓ = (ζmin − ζℓ) · Wℓ/‖Wℓ‖²₂     if ζℓ < ζmin
  uℓ = 0                             otherwise
  where  ζℓ = f(xℓ) = Wℓᵀ xℓ + bℓ
  ```
  → 프롬프트가 적은 **u=(ζ_target−ζ)·W/‖W‖² 형태를 그대로 확인**. probe 방향 W를 따라가는
    **rank-1** 밀기이며, 목표를 벗어났을 때만 딱 경계까지 밀고(하드 제약), 안에 있으면 개입 0.
  - **한 번에 하나의 scalar feature** (1차원 D)에 대해 적용. 즉 다차원 subspace 연산이 아니다.
- **추론 적용 (Algorithm 2)**: 정상 forward pass에 두 줄만 추가. LO(관측 layer집합)에서 ζℓ 계산,
  그중 LC(제어 layer집합, LC⊆LO)에서 uℓ 계산·적용. 선형+closed-form 이라 **오버헤드 거의 0**.
  closed-loop 캐비앳(Remark 2): observability/controllability는 "closed-loop이 입력을 probe 학습
  분포 밖(OOD)으로 밀지 않는 한" LLM→VLA로 넘어간다.
- **"shallower = more controllable" (depth 발견, Fig. 4)**:
  - 단일 layer ℓ에만 xℓ+α 섭동을 주고, 여러 episode에서 **delta-yaw action** 과 **delta-gripper
    action** 의 평균 변화량을 layer 함수로 측정. 강도 α별 곡선.
  - 결과: **얕은(earlier) layer일수록 개입 효과가 크고, 깊어질수록 효과가 감소.**
  - **원인(중요, 다소 mundane)**: 표현 벡터의 **L2 노름이 depth에 따라 증가**하기 때문(Fig.4 하단에
    ‖xℓ‖ vs depth 표시). 고정 크기 α의 **상대적** 효과가 깊은 layer에서 작아진다. π0.5와 OpenVLA가
    같은 변화를 내는 데 필요한 α 절대값이 다른 것도 이 노름 차이로 설명.
  - → 즉 "얕을수록 제어가 잘 된다"는 **근본 인과 도달성**이라기보다 **노름 성장에 따른 스케일
    정규화 아티팩트** 성격이 크다. (우리 observe≠steer 논의에 시사점, §7.)
  - **구체 layer 숫자**: 본문 산문에 "가장 잘 되는 layer 인덱스" 표는 없다. 단 **Fig. 5의
    개입 시각화는 π0.5·OpenVLA 모두 transformer layer 9** 를 운영 지점으로 사용.
- **robustness의 모델 의존성 (Fig. 4)**: π0.5는 α↑에 따라 관측 feature 변화가 **매끄럽게 단조 증가**
  (관측이 개입에 robust). 반면 **OpenVLA의 delta-yaw는 robust하지 않음**(순서가 깨짐); delta-gripper는
  순서는 있으나 π0.5만큼 깔끔하지 않음. → "대부분 관측은 robust하나 일부 feature/모델은 아니다".

---

## 4. 관찰 vs 제어 gap (best-observed layer ≠ best-controlled layer?)

- 저자는 observability와 controllability를 **개념적으로 독립**이라고 못 박고(Def 1/2, Kalman[7]),
  실무적으로만 LC⊆LO(제어가 관측을 재사용)로 묶는다.
- **정량적 gap 표는 제시하지 않는다.** 다만 정황상:
  - **관측(Fig.3)**: feature별 "best performant layer"가 따로 있고(연속/이진마다 다를 수 있음),
    반드시 가장 얕은 layer가 아니다.
  - **제어(Fig.4)**: 효과는 **얕은 layer로 갈수록 커진다**(노름 때문). 운영 개입은 layer 9.
  - → 즉 **"가장 잘 관측되는 layer"와 "가장 잘 제어되는(효과 큰) layer"가 일치한다는 보장은 없고**,
    오히려 제어 효율은 얕은 쪽으로 편향된다. 논문은 이 gap을 명시적 실험으로 분리·수치화하지 않았다.
  - 추가로 OpenVLA delta-yaw처럼 **관측은 되지만 개입에 robust하지 않은**(제어 신뢰도 낮은) 사례를
    보여 "관측된다≠제어된다"를 정황적으로 뒷받침.
- **우리 관심(observe≠steer)과의 접점**: 이 논문은 우리처럼 "관측 분리도(AUROC)가 곧 steer 성능"이라고
  가정하지 않는다. 오히려 **관측(observer)은 제어의 방향 W를 공급하는 부품**이고, 실제 성능은
  **closed-loop success rate + constraint satisfaction 트레이드오프**(=인과 측정)로 판정한다.
  이 태도는 우리의 "진짜 arbiter는 관측 분리가 아니라 ΔSR"과 정확히 같은 철학이다.

---

## 5. 평가·결과·한계·confound

- **metric**: (a) **constraint satisfaction ratio** (개입이 목표 제약구간을 지키는 비율),
  (b) **closed-loop task success rate**. 저자는 "steer 성능은 **success rate와 함께** 봐야 의미"라고
  강하게 주장(LLM steering이 naturalness/coherence를 보존해야 하듯). baseline = no-intervention, prompting.
- **제어 대상 3가지**: gripper state, end-effector **height**, end-effector **speed**
  (speed는 모델 직접 출력이 아니라 v=‖Δx,Δy,Δz‖/dt 로 유도된 파생량).
- **헤드라인 결과(질적, 수치 상세는 그림)**:
  - **Gripper (Fig.6)**: open/close 모두 near-perfect constraint 만족 + **success rate 90%+ 유지**.
  - **Height (Fig.7,8)**: near-perfect 제약 만족, 단 제약 task가 본질적으로 더 어려워 **modest SR 하락**.
    "충분히 robust한 base + 좋은 recovery면 이 하락은 없앨 수 있을 것"이라 가설.
  - **Speed (Fig.9,10)**: **감속은 신뢰성 있게** 되나 **가속은 덜 정확**(fast 영역 학습데이터 부족 탓,
    Häon[5]와 일치). speed 제어 시 **SR은 거의 완벽 유지**.
- **한계(저자 명시)**:
  1. observer 학습에 **라벨 필요** → 향후 SAE 등 unsupervised feature discovery.
  2. **transformer 부분만** 다룸 → diffusion/flow-matching head 확장이 과제.
  3. **저수준 feature(state/action)** 만 → task goal/affordance/spatial relation 같은 고수준은 open.
  4. 개입의 **safety 보증/bound** 필요.
- **confound 논의 — 우리 것과 대비(중요)**: 이 논문은 우리가 겪는 **scene/task/rollout-length confound를
  전혀 다루지 않는다.** 이유가 구조적이다 — 이들의 타깃은 **매 timestep의 ground-truth 연속 kinematic
  량**(pose/gripper/speed)이라 "성공/실패" 같은 결과 라벨을 관측하지 않는다. 따라서 outcome-vs-scene
  혼동 문제 자체가 발생하지 않는다. success rate는 오직 **"steer가 정책을 망가뜨렸나"의 가드레일**로만
  쓰이고, **관측 대상이 아니다.** — 즉 confound-robustness에 대한 직접 교훈은 이 논문에서 얻을 수 없다.

---

## 6. SAE와의 대비 (선형 observer가 주는 것 / SAE가 주는 것)

| 축 | 선형 observer(이 논문) | SAE(대조 논문들) |
|---|---|---|
| 라벨 | **필요**(supervised, feature 이름을 미리 지정) | 불필요(self-/unsupervised) |
| 샘플효율 | **연속·dense 라벨에 매우 높음**(episode×timestep = 대량, 저분산) | dictionary 학습에 대량 데이터, 분산 큼 |
| 발견성 | 지정한 feature만 관측(미지의 "실패모드" feature는 못 찾음) | **모르던 feature/조합을 발견 가능** |
| steer 직접성 | **W가 곧 제어방향**, closed-form u로 즉시 개입, 오버헤드 0 | latent→출력 인과 연결을 별도로 세워야 함 |
| 해석성 | 명명된 물리량(pose/gripper) 축으로 **직관적** | dict atom이 해석가능하나 명명·검증 필요 |
| 제어 표적 | scalar target ζ_target 이 **있어야** 함(box 제약) | target 없이 atom 활성/억제로 조작 |
| confound 견고성 | **선형이라고 confound가 사라지지 않음**(혼동 라벨에 회귀하면 그대로 재인코딩) | 마찬가지로 자동해결 안 됨(atom↔outcome 연결에서 소표본 문제 동일) |

- 요지: 선형 observer의 강점은 **"이름이 있고 매 스텝 dense·연속 라벨이 있는 물리량"** 을 **싸고
  저분산**으로 관측하고 그 방향으로 **바로 최소개입 steer**하는 것. SAE의 강점은 **라벨 없이 미지의
  feature를 발견**하는 것. **둘 다 scene/length confound를 자동으로 없애 주지는 않는다.**

---

## 7. 우리 프로젝트 적용성 (핵심 판단)

우리 상황 재확인: GR00T N1.5, DiT block residual [L=7, K=4, D=1536] + VL [2048], 직렬
Eagle-LM→VL-SA→DiT. 문제 = latent의 성공/실패 분리가 **scene/task(AUROC=1.0)·length/phase confound**
에 지배되고, 순수 outcome 신호는 permutation-null(~0.9) 근처, scene당 실패 3–4개(소표본). 진짜 arbiter는 ΔSR.

**(A) "outcome observer" = pose observer의 유비로 직접 쓸 수 있나? → 대체로 NO, 이유가 구조적.**
- 이 논문 observer가 저분산·샘플효율이 좋았던 근본 이유는 타깃이 **매 timestep 연속·ground-truth**
  라벨이었기 때문(episode×timestep 규모). 우리의 outcome은 **episode당 1비트, scene당 실패 3–4개** —
  **정확히 반대 regime.** 선형이라는 사실이 소표본·이진 문제를 완화해 주지 않는다.
- 더 근본적으로 **선형 probe는 confound를 없애지 못한다.** outcome 라벨이 scene과 얽혀 있으면 선형
  probe는 scene 방향으로 회귀할 뿐이다(=우리의 AUROC=1.0 재현). SAE든 선형이든 여기선 같은 함정.
- 제어식도 안 맞는다: u=(ζ_target−ζ)W/‖W‖² 는 **scalar target ζ_target 이 있어야** 작동한다.
  pose/gripper/speed엔 목표값이 있지만 **"success"는 servo할 좌표가 아니다.** outcome을 직접
  minimal-control로 밀 수 없다.

**(B) 그러나 진짜로 유용한 이식 두 갈래 (권장):**
1. **선형 progress/phase observer** — 이게 이 논문의 regime에 정확히 들어맞는 저비용·저분산 이식.
   - normalized progress ζ∈[0,1] 또는 phase-bin을 **매 스텝 연속 라벨**로 두고 Wx+b 회귀. dense·저분산.
   - 우리 method의 **핵심 난제(online phase/type 식별)** 와 phase-matched DiT steering의 **online phase
     신호 공급원** 문제에 직결. 메모리의 "VITA progress predictor를 보조 부품으로 복귀 검토"와 부합 —
     무거운 VITA 대신 **layer별 선형 progress probe**가 더 싸고 분산이 작은 대안이 될 수 있다.
   - 소표본 문제를 우회하는 것도 여기 포인트: outcome(1비트/ep)이 아니라 progress(연속/스텝)를
     관측하면 표본이 폭증한다.
2. **최소노름 additive control = 우리 conceptor의 값싼 ablation 사다리 rung**.
   - 우리 C_steer = C_success ∧ ¬C_failure, h'=h·Mᵀ 는 **다차원 subspace projection**.
     이 논문 control은 **rank-1, scalar-target, closed-form** — 정확히 우리 사다리(global→pathway-split
     →+phase-bin)의 **가장 단순한 baseline**으로 넣기 좋다. "target이 있는 축(예: 특정 kinematic feature,
     또는 progress 좌표)"에 대해 rank-1 minimal control의 ΔSR을 재고, conceptor가 그보다 나은지 비교.
   - 즉 관측 분리도가 아니라 **ΔSR로 conceptor vs rank-1-control**을 붙이는 저분산 대조군.

**(C) 어디에 tap 하나? (depth 발견의 우리 매핑 — 캐비앳 큼):**
- 논문의 "얕을수록 제어 잘 됨"은 **VLM/transformer layer** 이야기이고 **flow-matching/diffusion head는
  명시적으로 future work로 제외**했다. 그런데 **우리가 실제로 steer하는 지점(DiT block residual)이 바로
  그들이 손대지 않은 부분**이다. → depth 발견의 직접 이식은 **VL 경로(Eagle-LM→VL-SA)** 쪽에 더 맞고,
  DiT block으로의 전이는 보장되지 않는다.
- 게다가 depth 발견 자체가 **‖x‖ 노름 성장 아티팩트**라, 우리 DiT 7 layer에도 적용하려면 개입 강도를
  **layer 노름으로 정규화**(u를 ‖xℓ‖에 스케일)해야 층간 비교가 공정하다 — 실무 팁으로 유용.
- 결론적 tap 제안: **online phase 신호는 얕은 VL/초기 DiT layer의 선형 progress probe**로 뽑고(관측),
  **steer 개입은 여전히 우리 DiT 표적에서 ΔSR로 검증**. 관측 layer와 개입 layer를 분리해도 무방(LC⊆LO 완화).

**(D) 두 SAE 논문 대비 pros/cons 요약(우리 소표본 outcome-isolation 관점):**
- **선형 route pros**: 라벨만 dense·연속이면 압도적 샘플효율/저분산; 오버헤드 0; W가 곧 steer 방향이라
  관측→개입이 매끄럽게 이어짐; rank-1 minimal control이 conceptor의 명확한 저비용 baseline.
- **선형 route cons**: **outcome/실패모드처럼 (i) 이진·소표본이고 (ii) 우리가 미리 이름 붙이지 못한**
  feature엔 부적합; confound를 자동으로 걷어내지 못함; scalar target 없는 "성공"엔 control식이 안 맞음.
- **SAE route가 이길 지점**: 라벨 없이 **미지의 outcome/failure-mode feature를 발견**해야 할 때. 단
  이때도 "atom↔실패결과" 연결은 **scene당 3–4 실패의 같은 소표본 문제**에 부딪힌다 → SAE도 만능 아님.
- **공통 결론**: confound(scene/length)는 어떤 표현학습(선형·SAE)으로도 자동 해소되지 않는다.
  해법은 방법론이 아니라 **설계**(within-task 고정-t 분석, length 통제, 그리고 최종 판정은 ΔSR 인과).
  이 논문은 그 "인과 판정 우선" 철학의 강한 선례(constraint-sat + SR 트레이드오프)를 제공한다.

**최종 판정 (정직하게)**: 선형-observer route는 **outcome 자체의 소표본 분리**에는 SAE보다 낫지 않다
(둘 다 confound·소표본을 못 푼다; 오히려 control식이 outcome에 안 맞음). **그러나** 이 논문의 진짜
가치는 다른 데 있다 — **(1) dense·연속 progress/phase를 값싸고 저분산으로 관측하는 선형 probe**로
우리 method의 병목인 **online phase 신호**를 공급하는 것, **(2) rank-1 minimal control을 conceptor의
ΔSR baseline**으로 세우는 것. 이 두 가지는 소표본·저분산이 필요한 우리 상황에서 SAE보다 확실히
싸고 분산이 작은 실전 이식처다. outcome-isolation 자체는 표현학습이 아니라 confound 통제 설계로 풀 것.
