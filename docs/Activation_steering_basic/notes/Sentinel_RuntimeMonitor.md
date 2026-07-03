# Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress (Agia et al. 2024, Sentinel)

- 출처: arXiv 2410.04640v2 [cs.RO] (Stanford + NVIDIA, Agia/Sinha/Yang/Cao/Antonova/Pavone/Bohg), CoRL 2024 · PDF: docs/references/Sentinel_RuntimeMonitor_2410.04640.pdf · 섹션=§7(결론이지만 우리 문맥상 핵심은 §3~4 방법론 전체, §5-6 실험, Appendix A/D 상세) · tier=must · 한 줄 역할: 생성형 정책(diffusion policy) 실패를 온라인에 두 상보적 유형(행동 불일치 vs 진행 정체)으로 나눠 검출하는 완전 블랙박스 runtime monitor — 우리의 "online phase/failure-type 식별" 문제와 문제 정의가 원형적으로 같지만, latent가 아니라 출력(action 분포·픽셀)만 사용한다는 점이 결정적으로 다르다.

## 문제·동기
모방학습 기반 생성형 정책(diffusion policy 등)은 학습분포를 벗어나면(OOD) 예측불가능하게 행동한다. 기존 OOD 검출(재구성 오차, 임베딩 유사도, epistemic uncertainty)은 개별 input-output 샘플의 이상치 여부만 보고, trajectory를 따라 발생하는 다중·시간상관 예측오차가 누적되는 closed-loop 실패를 포착하지 못한다. 여기에 생성형 정책 특유의 난점이 겹친다 — action이 multimodal 분포에서 샘플링되므로 스텝마다 크게 달라 보이는 것이 정상적 무작위성인지 진짜 실패 신호인지 구별하기 어렵다(기존 model-free 정책보다 실패 양상이 다양해짐). 저자들은 "실패 유형이 이질적이라 단일 검출기로 다 잡기 어렵다"는 전제에서 출발해, 실패를 성질이 다른 두 카테고리로 명시적으로 쪼갠다.

## 핵심 아이디어
실패를 (1) erratic failure(정책 출력이 시간에 따라 요동·모드 충돌 — 예: 장애물 회피를 좌/우 중 하나로 커밋 못하고 jitter) 와 (2) task progression failure(출력은 시간적으로 일관되지만 과제 진행이 정체 — 예: 물건을 자신있게 엉뚱한 곳에 놓음) 로 정의하고, 각 카테고리를 그 특성에 맞는 전용 검출기에 배정한다: erratic은 빠르게 잡아야 하므로 계산비용이 거의 0인 통계적 방법(STAC)으로, task progression은 즉각 개입이 필요 없으므로 느리지만 의미이해가 강한 VLM으로. 핵심 통찰은 "한 카테고리를 다른 카테고리의 여집합으로 정의하면 두 검출기를 단순 논리합(OR)으로 합쳐도 커버리지가 trivial하게 완성된다"는 것 — 복잡한 앙상블·가중치 학습 없이 상보적 검출기 결합이 성립한다.

## 방법 (STAC=temporal action consistency, progress 정체; 두 통계로 runtime monitor)
- **정책 정형화**: receding-horizon으로 매 t 시점에 길이 h의 action chunk를 샘플링하고 앞 k개만 실행 후 t+k에서 재계획. t와 t+k에서 각각 예측한 action sequence는 h−k 스텝 구간에서 시간적으로 겹친다(Fig.2).
- **STAC(§4.1)**: 이 겹치는 구간의 두 marginal action 분포 π̄_t(t 시점에 예측한 미래분포)와 π̃_{t+k}(t+k 시점에 재예측한 같은 구간 분포) 사이의 통계적 거리 D̂(MMD-RBF, forward/reverse KL via KDE — 개별 action point가 아니라 batch 샘플링한 분포 간 거리를 씀, action multimodality 때문에 point-wise 비교는 노이즈가 큼)를 계산하고, 누적합 η_t = Σ D̂(식1)를 매 재계획 시점마다 갱신. η_t는 D̂≥0이라 단조증가 — 이 성질이 뒤의 FPR 보장 증명에 핵심적으로 쓰인다. 임계값 γ는 성공 rollout M개(calibration set Dτ)의 종단 η 값의 (1−δ) 분위수로 오프라인 보정. 가설: 정책이 내재적 world model을 갖고 있어 in-distribution이면 t 시점 예측과 t+k 시점 재예측이 잘 맞아 거리가 작고, OOD면 어긋나 거리가 커진다.
- **conformal FPR 보장(Prop.1/2, Appendix D)**: η_t가 단조증가하므로 "trajectory 중 한 번이라도 경보"라는 이벤트가 "종단 시점 η_H'"라는 단일 iid 통계량으로 환원되어, split conformal prediction(distribution-free, 분포가정 불필요)을 그대로 적용해 FPR ≤ δ를 증명. 실패 데이터 없이 성공 데이터만으로 이 보장을 얻는다는 게 강점.
- **VLM 진행-정체 검출(§4.2)**: 로봇 이미지 시퀀스(비디오) I_0:t + task 자연어 설명 + 경과시간/제한시간을 프롬프트에 넣고, 4단계 CoT(질문 생성→답변→진행상황 분석→{ok,failure} 분류)로 zero-shot 판정. 단일 이미지(Image QA)는 TNR이 거의 0으로 참패 — 반드시 비디오(시간에 걸친 변화)를 봐야 진행 여부를 판단 가능. VLM은 STAC보다 훨씬 느리게(에피소드당 2회 정도) 비동기 호출, task progression은 급박하지 않다는 전제로 정당화. 도메인에 따라 최적 VLM이 다름(Close Box=GPT-4o, Cover Object=Claude 3.5 Sonnet) — 신뢰도를 올리려 prompt ensembling(성공 비디오/goal 이미지 등 변형 프롬프트의 다수결)을 추가.
- **결합**: 두 검출기를 논리합(OR)으로 병렬 실행(Fig.3) — 하나라도 경보하면 즉시 정지.

## 실험·결과
4개 도메인(PushT=action multimodality, Close Box=14-DoF bimanual erratic failure, Cover Object=task progression failure, 실물 Push Chair=복합) × object scale/pose/dynamics shift로 OOD 생성. STAC은 PushT에서 balanced accuracy 90%+ 로 임베딩/재구성/출력분산 baseline을 압도(Fig.5), Close Box에서 aggregate accuracy 96%이며 특히 임베딩 기반 방법과 달리 "OOD인데 성공"과 "실패"를 잘 구별(임베딩 방법은 OOD-success에서 TNR 0으로 폭락). VLM은 Cover Object 같은 progression-failure 도메인에서 STAC(TPR 44%)이 놓치는 실패를 잡아, 결합 시 TPR 93%·FPR 증가는 +7%뿐(Fig.6). 전체적으로 Sentinel은 미지 실패의 97%+를 검출, 단일 검출기 대비 +18% — 실물 Push Chair에서도 95% accuracy. Image QA→Video QA 전환이 TNR을 극적으로 올린다는 것과, STAC의 통계적 거리(MMD/KL) vs 비통계적 최소거리 baseline 비교(Fig.5 right, multimodality 무시 시 성능 급락)가 두 핵심 ablation 결론.

## activation-steering 흐름에서의 위치 (개입 전 '검출' 레이어)
이 논문은 steering이 아니다 — activation(hidden state)에 전혀 접근하지 않는 완전 블랙박스 runtime monitor로, "언제/무엇을 개입할지"를 정하기 위한 검출 레이어에 해당한다. STAC은 정책이 샘플링한 action-chunk 출력 분포만, VLM은 카메라 픽셀(비디오)만 본다 — 어떤 generative policy 아키텍처에도 이식 가능(model-agnostic)하다는 게 설계 철학이다. 우리 파이프라인에서 이 논문이 대응하는 지점은 conceptor/steering 벡터를 "언제 적용할지" 트리거하는 online detector 자리이지, steering 개입 자체(h' = h·Mᵀ)와는 층위가 다르다.

## 우리 프로젝트 연결 (실패 유형 2분법·online 검출의 직접 선례; 신호 차이[통계 vs latent])
- Sentinel의 "실패를 상보적 두 유형으로 나눠 각각 전용 검출기 배정" 프레임은 우리의 VL(goal)/DiT(motor) 실패-유형 분리·phase 식별 문제와 문제 정의 층위에서 원형적으로 같다. 다만 그들의 두 축(행동 통계적 일관성 vs 의미적 과제진행)은 우리 두 축(어느 pathway가 원인인가)과 정확히 같은 분할이 아니다 — 굳이 대응시키면 erratic(모드 충돌·jitter)은 실행 붕괴이니 motor/DiT 실패에, task progression(자신있게 틀린 계획 수행)은 목표 오인식이니 goal/VL 실패에 느슨하게 대응될 수 있지만, 논문은 내부 서킷을 전혀 다루지 않으므로 이 대응은 우리의 해석이지 저자의 주장이 아니다.
- **결정적 신호 차이**: STAC은 action-chunk 출력 분포의 통계량(MMD/KL), VLM은 픽셀 비디오 — 둘 다 latent/hidden activation을 쓰지 않는다. 반대로 우리(N16 pathway_online_detection, DiT block31 t_d=11 AUROC 0.92)는 모델 내부 hidden state에 hook을 걸어 pathway별 신호를 직접 읽는다. 이는 범용성-특이성 트레이드오프다: Sentinel은 모델 내부에 접근할 필요가 없어 어떤 정책에도 이식되지만, 우리 activation 기반 신호는 더 이른 시점에 더 causal한 원인(어느 pathway)까지 특정 가능한 잠재력이 있는 대신 모델별 hook 인프라가 필수다. STAC의 "출력 action-chunk 겹침 구간의 분포 거리"라는 아이디어는 우리도 활동(hidden state) 없이 action 출력만으로 값싸게 계산 가능한 상보적 detector로 그대로 이식할 수 있다 — GR00T DiT도 diffusion 기반 chunk-predict/partial-execute 구조(chunk 16 예측/5 실행)라 STAC의 전제(receding horizon, 겹치는 window)가 구조적으로 맞아떨어진다.
- VLM 진행-정체 모니터는 우리가 "TTA/VITA progress predictor"로 불러온 것과 목적이 같다(온라인 progress 신호 공급). 다만 zero-shot VLM 호출은 14~23초 지연(GPT-4o 평균 14.0초)이라, 우리가 요구하는 phase-matched 실시간 steering 라우팅에는 그대로 쓸 수 없다 — "학습된 회귀 헤드(ProgressHead)로 대체해야 한다"는 근거를 이 논문의 latency 수치가 뒷받침한다.
- conformal 기반 FPR 보장(Prop.2, 성공 데이터만으로 distribution-free δ 보장)은 우리가 현재 AUROC만 보고하는 것과 대비되는 실무적 엄밀함의 기준점 — η_t 단조누적+종단값 conformal이라는 트릭은 threshold 기반인 우리 online detector에도 적용 가능한 패턴이다.

## 면접 포인트 (Q→A)
1. Q: "STAC이 왜 개별 action 샘플이 아니라 분포 간 통계적 거리를 쓰나?" A: "diffusion policy는 multimodal이라 같은 상태에서도 매 재계획마다 다른 mode를 샘플링할 수 있다. 개별 action을 직접 비교하면 이 정상적 무작위성과 진짜 실패를 구별할 수 없어서, action-chunk의 겹치는 구간에 대해 batch로 샘플링한 분포 간 MMD/KL 거리로 '정책의 내부 world model이 시간에 걸쳐 일관되는가'를 측정한다. 실제로 비통계적(최소거리) baseline은 multimodality를 무시해 false alarm이 폭증한다(Fig.5 right)."
2. Q: "STAC과 VLM을 왜 단순 논리합(OR)으로 합치나, 더 정교한 결합이 낫지 않나?" A: "한 카테고리를 다른 카테고리의 여집합으로 정의했기 때문에 OR만으로 커버리지가 완성된다. 저자는 세 가지로 정당화한다 — importance weighting이 불필요(실패 데이터 없이 계산), 해석가능성(두 검출기가 각각 저FPR이면 결합 FPR도 낮다고 직관적으로 보장), 런타임 제약(STAC은 빠르고 VLM은 느려 서로 다른 timescale에서 독립적으로 동작해야 함). 한계로는 worst-case에서 union bound로 FPR이 합산돼 실제로 결합 후 TNR이 개별보다 낮아지는 경우가 관찰된다(예: Table1 Sentinel TNR 87% vs STAC 단독 94%)."
3. Q(우리 프로젝트 관점): "Sentinel의 실패 2분법이 우리 VL/DiT 2분법과 같은 것인가?" A: "아니다. Sentinel은 '행동 통계적 일관성(STAC)' 대 '의미적 과제진행(VLM)'이라는, 검출 신호의 출처가 다른 축으로 나뉜다. 우리는 '어느 pathway(goal-VL vs motor-DiT)가 원인인가'라는 축이다. 느슨하게는 erratic~motor 실행 붕괴, progression~goal 오인식에 대응시킬 수 있지만 논문은 내부 서킷을 전혀 다루지 않는 완전 블랙박스라 이건 우리의 해석일 뿐이다. 반면 '온라인에 두 상보적 실패유형을 각각 전용 검출기로 잡고 OR/사다리로 결합한다'는 시스템 설계 패턴 자체는 직접 참고할 선례다."
4. Q: "이 논문의 온라인 검출과 우리 프로젝트의 온라인 검출은 근본적으로 뭐가 다른가?" A: "Sentinel은 model-agnostic 블랙박스 — 정책 내부 hidden state에 전혀 접근하지 않고 출력(action 분포)과 픽셀(비디오)만 본다. 우리는 activation 자체를 훅으로 읽어(VL-SA, DiT block) pathway별 신호를 뽑는다. 우리 방식은 더 이른 시점에 더 causal한 원인 특정(어느 pathway 탓인가)까지 가능할 잠재력이 있지만 모델별 hook 인프라가 필수라 범용성이 떨어진다. STAC은 우리가 놓치고 있던 '출력만으로 얻는 값싼 online consistency 신호'라는 보완책을 보여준다 — GR00T의 diffusion DiT도 chunk-predict/partial-execute 구조라 STAC 수식이 거의 그대로 이식 가능하다."
5. Q: "STAC의 FPR 보장(conformal, Prop.2)이 우리 방법에 어떻게 적용될 수 있나?" A: "η_t가 단조증가한다는 성질을 이용해 'trajectory 중 한 번이라도 경보'라는 사건을 '종단 시점 η 값'이라는 단일 iid 통계량으로 환원하면, split conformal prediction을 그대로 적용해 임의의 conformity score에 대해 성공 데이터만으로 distribution-free FPR≤δ를 증명할 수 있다. 우리 threshold 기반 online detector에도 이 '단조 누적통계 + 종단값 conformal' 트릭을 적용해 지금의 AUROC 보고를 넘어 형식적 FPR 보장으로 강화할 수 있다."

## 한계·비판
- 두 실패 카테고리가 완전한 것이 아님을 저자도 인정 — sensor fault, hardware 고장, contact-rich 실패 등은 erratic도 progression도 아닐 수 있고, future work로 카테고리 확장을 남겨둠(§7 Limitations).
- STAC의 conformal FPR 보장(Prop.2)은 STAC 단독에만 성립하고, VLM은 블랙박스라 결합(Sentinel) 전체에는 형식적 보장이 없다 — 실제로 결합 시 FPR이 개별 대비 상승(Table1/5/7에서 TNR 하락 관찰)하는 worst-case union bound 문제가 실증적으로도 나타난다.
- VLM 성능이 모델·도메인마다 크게 갈린다(Close Box=GPT-4o 최적, Cover Object=Claude 3.5 Sonnet 최적, 원인 불명) — 재현성·신뢰성이 약하고, cloud VLM 호출 지연(14~23초)은 즉시 개입이 필요한 상황엔 원천적으로 부적합(저자도 "task progression은 급박하지 않다"는 전제로만 정당화).
- task_descriptions 프롬프트를 각 과제마다 매우 상세히 수동 작성해야 함 — 새 과제로 확장할 때 프롬프트 엔지니어링 비용이 크고 스케일링에 불리.
- 실패를 사전에 "예측"하는 게 아니라 발생 후(또는 발생 중) "감지"만 한다는 점을 저자도 명시적으로 한계로 인정 — 우리가 원하는 "steering 개입을 위한 선제적 판단"과는 목적이 다르다.
- calibration을 demonstration data로 대체하면 covariate shift 때문에 FPR이 폭증한다는 것을 저자가 실험으로 확인(§B.4.2) — 반드시 정책 rollout(성공만)으로 calibration해야 함, 우리 conceptor fit에도 유사한 함정(학습 데이터가 아니라 rollout 분포로 fit해야 함) 가능성을 시사.
- STAC의 수학(MMD/KDE on action-chunk overlap)은 diffusion-policy의 receding-horizon chunk 구조에 강하게 의존하는 설계라, chunk 구조가 없는 정책이나 우리 DiT와 다른 action-head 설계에는 재도출이 필요할 수 있다.
