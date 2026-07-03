# Mechanistic Interpretability for Steering Vision-Language-Action Models (Häon 2025)

- 출처: Bear Häon*, Kaylene Stocking*(공동1저자), Ian Chuang, Claire Tomlin (UC Berkeley EECS) · arXiv:2509.00328v1 [cs.RO] (2025-08-30) · 9th CoRL 2025 (Seoul) · PDF: `docs/references/Mechanistic Interpretability for Steering.pdf` · 섹션=§6 VLA로 배정됐으나 **실물 논문엔 그런 절이 없음**(논문 전체가 VLA 대상: §1 Intro–§2 Related Work–§3 Interpreting VLAs–§4 Steering VLAs–§5 Discussion/Conclusion–§6 Limitations + Appendix A/B/C; 실제 §6은 "Limitations") — 전체 정독. tier=must. 한줄역할: LLM 계열 FFN value-vector 해석 기법을 VLA에 최초 이식해 "단일 뉴런 override"만으로 fast/slow·low/high 같은 행동 축을 zero-shot 제어한 시작점 논문 — **우리 팀이 Phase A(해석+override)를 저자 코드로 재현 완료**한 직접 baseline.

## 문제·동기
VLA는 kinematics/dynamics/control이 명시적으로 모델링된 classical robotics 파이프라인과 달리 내부가 블랙박스라, 실패를 진단·수정할 방법이 없다. LLM 쪽 mechanistic interpretability(induction heads, SAE monosemanticity, attribution graphs)가 유사한 문제에서 성과를 냈다는 데 착안해 "VLA에도 같은 도구가 통하는가"를 묻는다. SAE는 학습 데이터가 많이 필요한데 로보틱스 오픈데이터는 자연어 대비 훨씬 작다는 제약 때문에, 저자들은 **추가 학습 없이(gradient/보상/환경 상호작용 없이) 기존 가중치만으로 해석 가능한** 기법을 택한다.

## 핵심 아이디어
Geva et al. 2022(ref[8], arXiv:2203.14680)의 FFN value-vector 해석을 그대로 채택: FFN(x) = f_θ(x)ᵀW_θ = Σ_i [f_θ(x)]_i · w_θ^(i). 각 행 w_θ^(i)("value vector")는 입력과 무관한 고정 basis이고 최종 출력 토큰과 같은 선형공간에 있으므로, language-model head로 vocab에 projection하면 상위-N 토큰으로 그 뉴런의 "의미"를 읽을 수 있다(logit-lens 식). linear representation hypothesis(개념=latent space의 방향)에 기대어, "fast"/"up" 같은 개념에 정렬된 뉴런 집합을 찾아 그 활성값을 강제로 올리면 행동이 그 개념 쪽으로 움직일 것이라 가정한다.

## 방법(FFN 뉴런 override, 백본 전 층 대상)
- **해석 파이프라인**: 각 value vector를 top-5(클러스터링용)/top-30(패턴판정용) 토큰의 softmax 가중 임베딩으로 사상 → cosine kNN(k∈{10,20,40}, cuML)으로 클러스터링 → 목표 개념 단어(예 "fast")와 코사인 유사도가 가장 큰 클러스터를 선택(수동 또는 자동).
- **개입 연산**(식3~4): 선택된 뉴런 인덱스 집합 S에 대해 f~_θ(x)_i = α (i∈S일 때) / 원래값 (아니면). FFN_steered(x) = Σ_i f~_θ(x)_i · w_θ^(i). **더하는 게(additive) 아니라 활성값 자체를 고정 스칼라 α로 클램프(override)** — ActAdd/CAA류 "방향 벡터를 계수만큼 더한다"와는 다른 연산.
- **적용 지점**: OpenVLA(PyTorch, Llama2-7B 백본, 32층 추정) — FFN down_proj에 forward hook(대안: gate_proj, GEGLU 구조라 이 경우 α는 GELU(α)로 스케일되는 식7). π0-FAST(JAX, PaliGemma-3B 백본, 18층) — S·α를 리팩터한 FFN 코드에 직접 주입. **개입 대상은 항상 VLA의 base VLM(백본) 내부 FFN이고, 별도 flow-matching/DiT 액션 헤드는 건드리지 않는다** — 애초에 이 논문이 쓰는 OpenVLA·π0-FAST 둘 다 액션을 이산 토큰으로 백본이 직접 뱉는 단일-네트워크 구조라(separate action expert 없음), "VL 백본 vs DiT 액션헤드"라는 분리 자체가 이 아키텍처엔 존재하지 않는다.
- **깊이(temporal) 개입**: "early/late/full" 3조건 — OpenVLA 전체 352,255개 value vector를 절반씩 나눠 early-layer/late-layer 서브셋에서 각각 독립적으로 클러스터링(full은 전체에서). 즉 특정 레이어 1개가 아니라 **여러/전체 레이어에 걸친 뉴런들을 동시에 override**하는 방식 — COAST(단일 layer의 hidden state 전체를 곱셈 게이트로 미는 것)와 개입 단위(뉴런 vs hidden-state 전체) 자체가 다르다.
- **하이퍼파라미터**: LIBERO — cluster size {10,20} × α {2,4,6,8,10,20}. 로봇 — 수작업으로 top-6 value vector(task 키워드 빈도 최고) 선택, α=10 고정.

## 실험·결과(SR이 아니라 displacement/height — 아래 주의 참고)
**LIBERO-Long sim**(OpenVLA-7B, H100, LIBERO fine-tuned ckpt, 결정적 seed, 10 task × 10 rollout):
- fast vs slow 클러스터: end-effector displacement 평균 +27.73%(fast가 큼), 최대 +148.54%, 전 10 task paired t-test p<0.001, effect size d=-0.091~1.419.
- early/late/full-layer up-클러스터: 평균 Y-displacement full 0.098 > late 0.086 > early 0.007(early-vs-late d=-0.376) — 깊은/전체 레이어 개입일수록 효과 큼.

**UR5 실물 로봇**(π0-FAST-3B, LoRA fine-tune, task별 75/120 episode):
- Low/High Transport: low 개입 → 최저 높이 뚜렷히 낮춤. high 개입 ≈ baseline(모델이 이미 baseline을 "high"로 여기는 듯).
- Slow/Fast Transport: slow 개입 → 가장 느림. fast 개입 ≈ baseline(같은 비대칭).
- baseline 대조: random-vector 개입 ≈ no-intervention(거의 차이 없음) → 의미기반 벡터 선택이 실제로 유효함을 통제. prompt에 키워드만 추가하는 baseline보다 activation steering이 더 강함.
- **주의(우리 관찰, 논문 원문 검색 결과)**: 이 논문은 "success rate"라는 표현이 본문에 단 한 번도 나오지 않는다(전문 검색 0회). 지표는 전부 displacement(mm)/height(cm) 같은 연속 행동량이지 task 성공률이 아니다. "steer가 행동을 바꾼다"는 보였지만 "성공률을 올리는지"는 이 논문 범위 밖 — 우리 프로젝트의 핵심 metric(ΔSR)과 직접 비교 불가능한 논문이라는 점을 명확히 해야 한다.

## activation-steering 흐름 위치(LLM steering의 VLA 이식)
LLM 계열 계보(induction heads Olsson 2022 → SAE monosemanticity Bricken/Cunningham 2023 → FFN key-value memories Geva 2021 → concept-promoting value vectors Geva 2022)의 **해석 기법**을 그대로 가져와 VLA에 최초로 적용한 논문. 다만 개입 연산 자체는 ActAdd/CAA(방향 벡터를 계수만큼 더함, additive)가 아니라 **선택된 뉴런의 활성값을 고정 스칼라로 클램프(override, 하드 게이팅)**하는 방식이라 계보 안에서도 다소 이질적이다. 이후 COAST(`docs/Activation_steering_basic/notes/COAST.md`)가 이 논문을 "backbone feature 단위 개입은 종합 task-SR 개선을 보고하지 못했다"는 선행연구로 명시적으로 인용하며, FFN 단일-뉴런 override를 conceptor 기반 multi-dim contrastive steering(action expert 내부, closed-form fit)으로 발전시킨다.

## 우리 프로젝트 연결(직접 인접·재현 경험·차이)
- **재현 경험(Phase A, 완료)** — 기록: `docs/steering/16_mechinterp_reproduction.md`. 저자 공개 코드(third-party라 vendoring 안 함, 별도 클론 `~/pkt_ws/mechanistic-steering-vlas`, env `openvla-interp`)로 OpenVLA-7B + LIBERO-10에서 해석(logit-lens)과 override 둘 다 재현.
  - 해석: 우리 CPU projection이 저자 배포 `up_10` 클러스터와 10/10 일치, action-token이 후반 레이어(L31 86%)에 집중되는 논문 Fig2b 패턴도 재현.
  - 스티어링(coef=6, 5 trial, n=50/조건): **비교 방식에 따라 fast/slow 효과 크기가 완전히 달라짐** — 길이 미통제(전체 episode) +26.3%(논문 27.73%와 거의 일치) → 길이통제(task별 최단길이 truncate) +15.7%(**효과 ~반감**) → 논문 방식 그대로(paired ratio 평균) +37.6%(p=0.029, dz=0.82). fast·slow 둘 다 baseline보다는 느림(fast −19%, slow −36%) — 로봇 섹션의 "high/fast resembled baseline" 관찰과 정합.
  - **길이 confound가 외부 모델·외부 논문에서도 재확인됨** — seen18 길이 confound(`docs/insight` 계열)가 우리 데이터셋만의 아티팩트가 아니라 일반적인 함정임을 교차검증한 셈.
- **차이 ① 단일벡터/뉴런 override vs conceptor.** 이 논문은 뉴런 몇 개(hand-selected 6개 or kNN cluster 10~20개)의 활성값을 스칼라로 클램프하는 저차원 개입. 우리(및 COAST)는 h' = h·Mᵀ, M=(1−β)I+β·C_steer로 hidden state 전체를 succ/fail 대조로 얻은 multi-dim conceptor subspace에 곱셈 게이팅 — 판별 방향을 데이터(성공/실패 rollout)에서 fit하지, 사람이 vocab 의미로 골라내지 않는다.
- **차이 ② phase 없음 vs phase-matched.** 이 논문의 개입은 매 스텝 동일한 정적(static) override — rollout이 어느 단계인지 전혀 조건화하지 않는다. "early/late/full" 실험조차 모델 **깊이**(layer index) 축이지 rollout **진행 시점**(task-phase) 축이 아니다(COAST의 denoising-step과도 다른 축, 셋 다 phase-matched 개념이 없음). 우리 (2)phase-matched DiT steering이 정확히 이 빈틈.
- **차이 ③ pathway 프레이밍이 원천적으로 성립 안 함.** 이 논문이 쓰는 OpenVLA·π0-FAST는 액션을 이산 토큰으로 백본이 직접 생성하는 **단일 네트워크**(별도 flow-matching/DiT 액션 헤드 없음) — 그래서 "VL 백본만 개입 vs DiT 액션헤드만 개입"이라는 우리 pathway 분리 질문 자체를 이 아키텍처에 던질 수 없다. GR00T(Eagle-VL-SA + 별도 DiT 액션 헤드)에서만 성립하는 질문이라는 걸 이 논문과의 대조가 오히려 명확히 해준다.
- **작동하는 positive control 역할.** COAST positive control(N1.5 global steering)은 우리 재현에서 ΔSR≈0으로 실패했지만, 이 논문 방법은 "개입이 행동을 실제로 바꾼다"는 것 자체는 재현 성공 — 우리 방법이 최소한 이겨야 할 "steering이 아예 안 먹히는 건 아니다"라는 baseline을 제공한다.

## 면접 포인트(Q→A)
1. Q: "이 논문의 개입이 ActAdd 같은 additive steering과 어떻게 다른가?" A: "ActAdd/CAA는 활성화에 방향 벡터×계수를 더한다(h+α·v). 이 논문 식(3)은 선택된 FFN 뉴런의 활성값 f_θ(x)_i 자체를 원래 값과 무관하게 고정 스칼라 α로 덮어쓴다(override/클램프) — 입력에 따라 달라지는 원래 활성값 정보를 그 뉴런에 한해 완전히 버리는 하드 게이팅이라, 더하기보다 스위치에 가깝다."
2. Q: "왜 SAE 대신 단일 뉴런(FFN value vector) 해석을 썼나?" A: "SAE는 대량의 학습 데이터가 필요한데 로보틱스 오픈데이터(Open X-Embodiment 등)는 자연어 대비 훨씬 작고 편향돼 있다. FFN value vector는 이미 학습된 가중치 행 자체를 vocab에 projection만 하면 되므로 추가 학습이 전혀 필요 없다 — 대신 여러 개념이 한 뉴런에 뒤섞여 있을 수 있어 SAE보다 disentangle 능력이 떨어진다는 걸 저자도 인정한다."
3. Q(우리 재현): "우리가 재현한 수치와 논문 수치가 왜 비교 방식에 따라 26%~38%로 갈리나?" A: "논문 방식(paired ratio 평균, task별 fast/slow 쌍의 비율을 10개 task에 걸쳐 평균)이 가장 크게 나온다(+37.6%, 우리 재현). 길이(episode length)를 통제하지 않은 전체-episode 비교가 그다음(+26.3%, 논문 원 수치 27.73%와 거의 일치). task별 최단 성공 길이로 truncate해 길이 confound를 제거하면 +15.7%로 절반 가까이 줄어든다 — 즉 논문이 보고하는 효과 크기의 상당 부분이 길이 미통제 프레이밍에 기인하며, 이는 우리 프로젝트의 seen18 길이 confound 문제의식이 외부 모델에서도 재현됨을 보여준다."
4. Q: "이 논문이 VL/DiT pathway 분리 실험이 아닌 이유는?" A: "OpenVLA와 π0-FAST 둘 다 액션을 이산 토큰으로 백본 VLM이 직접 출력하는 단일 네트워크다 — GR00T처럼 별도 flow-matching/DiT 액션 헤드가 없다. 그래서 이 논문의 '전 layer FFN 개입'은 우리 식으로 보면 VL과 DiT가 아직 물리적으로 분리되지 않은(pathway 개념이 성립 안 하는) 아키텍처에 대한 개입이고, 실제로 논문도 'action token이 모든 layer에 몇 %씩 섞여 있다(하드한 사고→제어 전환 없음)'는 걸 발견으로 보고한다 — 이게 바로 pathway 분리가 이 아키텍처에서는 애초에 깔끔하지 않다는 방증이다."

## 한계·비판
- Semantic ambiguity/representational drift(저자 자인, §6 Limitations): kNN 클러스터가 서로 다른 행동을 뒤섞을 수 있음(예 "slow and careful" vs "slow and stuck"). 같은 방향이 모델/태스크/시간에 따라 다르게 작동할 수 있다는 것도 저자가 명시적으로 인정.
- fine-tuning이 steerability를 어떻게 바꾸는지 불명(§6) — VLM pretrain 개념이 fine-tuning 후에도 행동과 정렬된 채 유지되는지는 미해결.
- 평가 범위가 pick-and-place·단일 팔에 국한(§6) — mobile/bimanual/unstructured 환경 미검증.
- **success rate를 전혀 보고하지 않음**(전문 검색 결과 0회, 우리 관찰) — "개입이 행동(속도/높이)을 바꾼다"는 causal 증거는 강하지만, "task 성공에 도움이 되는지/해가 되는지"는 이 논문 프레임 밖. 우리 프로젝트가 요구하는 ΔSR 인과 근거로는 직접 쓸 수 없다.
- fast/high 방향 개입이 baseline과 거의 차이가 없는 비대칭성(저자도 "모델이 이미 baseline을 fast/high로 여기는 듯"이라 추측만 하고 원인 미규명) — 개입 강도가 방향에 따라 비대칭적으로 작동한다는 건 override 방식(하드 클램프)의 fragility를 시사.
- 정적·비-phase-conditioned 개입 — rollout이 어느 시점인지 전혀 반영하지 않아, 태스크 초반에 필요한 개입과 후반에 필요한 개입을 구분 못 한다(우리 프로젝트가 메우는 gap).
- 재현 시 확인된 길이 confound(위 "재현 경험" 참고) — 논문이 보고하는 효과 크기(27.73%)의 상당 부분이 길이 미통제 비교에서 온다는 게 우리 재현으로 드러남(길이통제 시 ~반감).
