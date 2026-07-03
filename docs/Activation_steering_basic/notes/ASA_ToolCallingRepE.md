# ASA: Backbone-Training-Free Representation Engineering for Tool-Calling Agents (Wang et al. 2026)

- 출처: arXiv:2602.04935v3 [cs.SE] 9 Jun 2026 (Renmin University of China 주저자 그룹 + Univ. of Macau + Central South Univ. + Jiangxi Normal Univ. + Macau Millennium College + Peking Univ.; Wang/Zhou/Ma/Fu/Liang/Cao/Huang/Fang/Pan) · PDF: `docs/Activation_steering_basic/ASA_ToolCallingRepE_2602.04935.pdf` · §5 파트 = 전망(agentic 온라인 제어) · 3축 = 쓰기✗(백본 동결, gradient 업데이트 없음) · inference✓(추론 시 forward hook으로 hidden state gated update) · 연구·프로토타입(NESTFUL/BFCL 벤치마크 전용, 단일 RTX 4090 스케일, 실 서비스 agent 프레임워크 미배포) · 한줄역할: tool-calling LLM agent가 매 생성 스텝마다 hidden state에서 "지금 tool 경계를 열지/막을지/가만둘지"를 gated steering으로 실시간 결정하는 training-free RepE 컨트롤러 — steering이 agentic 온라인 제어로 갈 수 있는 구체적 청사진.

## 문제·동기
Tool-calling agent는 스키마가 바뀌거나(신규 API, 인자 변경) 멀티턴·중첩호출·긴 컨텍스트 상황에서 tool 호출을 놓치거나(missing call) 파서가 못 읽는 출력을 낸다. 저자들은 "Intent-Execution Gap"을 제기한다: mid-layer hidden state에 tool-use 의도가 선형적으로 읽힌다(probe AUC 약 0.84, shuffle control은 chance)고 해서 실행이 보장되지 않는다 — probe 상위 10% 버킷에서 trigger recall은 100%인데 strict success(형식+tool명+인자+시퀀스 전부 통과)는 25%에 불과(Table 2). 즉 "tool 모드로 들어갔다"(boundary entry)와 "스키마를 정확히 실현했다"(schema realization)는 별개 사건이며, 기존 대응(prompting/RAG, LoRA/SFT/RL 파인튜닝, grammar-constrained decoding)은 각각 해석성·재학습비용·surface-level 한계를 갖는다.

## 핵심 아이디어
tool-use 방향을 두 성분으로 분해한다. (1) shared boundary 방향 vb = E[h|tool 필요]-E[h|불필요] — direct-answer↔tool-call 경계를 여는 공통 변위. (2) domain(tool-schema family)별 raw 방향 vd에서 vb 성분을 제거한 domain-local residual v_s^(d) = vd - (vd·vb/|vb|^2)·vb — 스키마 국소 편향. 도메인 방향들의 pairwise cosine이 낮음(Table 10, 예: Search-Translation 거의 직교)을 근거로 "공유 성분 vs 국소 잔차" 분해가 정당화된다. 여기에 signed abstention gate를 얹어 "언제 boundary를 열지/닫지/가만둘지"를 조건화한다 — unconditional amplification이 아니라 확률적으로 신뢰할 때만 개입.

## 방법(training-free RepE, tool-calling 중 개입)
- 신호: (i) 도메인 예측기 R이 예측 도메인 d̂과 신뢰도 sdom(x); (ii) 도메인별 logistic probe p(x)=sigma(w_d̂^T h_L(x)+b_d̂) = tool-use evidence; (iii) boundary-readiness 스칼라 rb(x)=calib(vb^T h̃_L(x)) (validation에서 적합한 calibration map).
- Gate: boundary action ab(x)∈{-1,0,+1} — p(x)>τp 이고 rb(x)<τb면 +1(경계 열기/rescue), p(x)<1-τp 이고 rb(x)>γ면 -1(과triggering 억제), 아니면 0(관여 안 함). schema-local action as(x)∈{0,1} — boundary-ready하고 도메인 신뢰도가 충분할 때만 1.
- 개입: h'_L(x) = h_L(x) + ηb·ab(x)·v̂b + ηs·as(x)·v̂s^(d̂) — 두 계수 모두 validation에서 선택, gate가 0이면 hidden state 무변경(원 backbone 그대로).
- 두 인스턴스화: ASA-Prefill(디코딩 전 딱 1회, 진단·ablation용) vs ASA-StateCascade(멀티턴/중첩호출 trajectory 동안 여러 생성 state에서 반복 재평가·재개입 — 실질적 "온라인" 버전). StateCascade는 BFCL에서 rescue/stop-avoid/hurt-avoid 3개 전용 logistic classifier(AUC 0.96대)를 층 20/24/28에 배치해 스텝별로 gate를 재계산한다. 생성 스텝 1859개 중 53.5%는 abstain, 26.7% stop-avoid, 16.5% rescue, 3.3% hurt-avoid — 대부분 스텝에서 아무 것도 안 하고 선택적으로만 개입.

## 실험·결과
- NESTFUL(중첩 API 호출 실행 평가): missing-tool rate 59.14%→6.72%, first-call acc 24.46%→41.94%, sequence acc 16.94%→25.00% (Table 3). Probe&Prefill·Tool-identity steering(ungated mean-diff) 등 근접 baseline보다 first-call/sequence trade-off 전체가 우월.
- BFCL: multi-turn Prompt-mode 32.50%→38.75%(Base/Miss-Func/Miss-Param/Long-Ctx 전 카테고리 개선), single-turn non-live AST 89.40%→95.60%, live single-turn AST 72.50%→77.00% — LoRA/QLoRA/Prefix-tuning/Parser-reward RL 등 학습형 baseline과 견줄 성능을 무학습으로 달성(Table 4).
- Ablation(Table 6/29): no-gate steering은 FPR 0.052→0.500, success precision 0.692→0.337로 붕괴 — gating이 성능이 아니라 "폭주 방지"의 핵심. Random direction은 개선 거의 없음(대조군). Global-only(도메인 미분리)는 full ASA보다 열등 — shared+local 분해 유효성 지지.
- Overhead: NESTFUL 기준 latency +10.9%(6.4s→7.1s), 저장 자산 약 20KB(steering vector+probe weight)(Table 17). LLaMA-3.1 cross-backbone sanity check에서도 유사 패턴 재현(Table 26).

## §5(산업)에서의 위치(agentic 제어 전망)
§5 현재 지형은 "안전 gate로 상시 켜두는" Circuit Breakers류(RepE를 LoRA로 가중치에 새김, 쓰기축)와 "interpretability API"(Ember/Gemma Scope) 두 틈새뿐이라고 정리했다. ASA는 세 번째 유력 경로 — agentic 실행(도구 호출) 중 표현공간에서 실시간 조종·차단하는 순수 inference-time RepE — 를 구체적 수치(BFCL/NESTFUL, latency/저장 비용)로 보여준다. 3축으로 보면 쓰기✗(백본 완전 동결) / inference✓(forward hook, gated update, disable하면 원 모델과 동일) / 연구·프로토타입 단계(벤치마크 논문, 실 프로덕션 agent에 배치된 사례는 아님) — Circuit Breakers(쓰기축, 실배치)와 대비되는 "순수 추론시 개입이 실전급 성능까지 갈 수 있다"는 증거 사례로 §5 전망 문단에 넣기 적합하다.

## 우리 프로젝트 연결(online 개입 문제구조 유사)
- 문제 형태가 원형적으로 같다: "실행 중(온라인) 표현공간에서 현재 상태를 읽고, 조건부로 개입 여부·방향·강도를 결정한다"는 뼈대는 우리 phase/failure-type 온라인 검출→라우팅 문제와 동형. shared(vb, 공통 tool-mode 방향) + domain-local(v_s^(d), 스키마 국소 잔차) 분해는 우리 pathway 분리(VL=공유 goal 방향, DiT=phase-conditioned 국소 방향) 아이디어와 구조적으로 유사하다 — 단 그들의 "domain"은 지도학습 가능한 discrete 카테고리(코드/수학/검색/번역)이고, 우리 phase/failure-type은 라벨 없는 continuous latent라는 점이 핵심 차이.
- signed gate(ab∈{-1,0,+1}, as∈{0,1})는 "언제 steering을 켤지"라는 우리 conditional(phase-matched) steering 요구를 실제로 구현한 사례 — no-gate ablation(FPR 0.500, success precision 0.337로 붕괴, Table 6)은 "unconditional steering이 위험하다"는 우리 핵심 전제를 다른 도메인에서 독립적으로 재확인해준다.
- ASA-StateCascade의 "trajectory 동안 여러 생성 state에서 반복 재평가"는 우리 rollout 중 매 5-step마다 재계획하는 구조(chunk 16 예측/5 실행, groot_chunk_predict16_execute5)와 결이 같은 "반복적 online 재평가" 패턴 — 단 그들은 discrete parser event(트리거 토큰 등장)라는 깨끗한 지도 신호가 있어 gate classifier AUC 0.96대가 나오는 반면, 우리는 그런 명확한 boundary event가 없다(이게 우리 핵심 난제).
- 저비용성(latency +10.9%, 저장 ~20KB)은 "training-free online steering이 실무적으로 저비용"이라는 실현가능성 참고 사례 — 우리 VLA steering도 5-step 해상도 내 conceptor projection 연산이 실시간성을 해치지 않아야 한다는 목표와 부합.

## 면접 포인트(Q→A)
1. Q: "tool-calling 논문인데 VLA steering이랑 무슨 관계인가?" A: "문제 구조가 같다 — 실행 중 hidden state를 읽고 조건부로 실시간 개입한다. 다만 그들은 discrete parser 라벨(스키마 도메인, `<functioncall>` 트리거 여부)이 있어 gate classifier가 AUC 0.96까지 나오는 반면, 우리는 라벨 없는 continuous phase/failure-type을 온라인에 추론해야 한다 — 그게 우리 핵심 난제다."
2. Q: "shared+local 분해가 너희 pathway 분리와 뭐가 다른가?" A: "그들은 tool-schema 도메인(코드/수학 등)별 잔차를 분리하고, 우리는 아키텍처 pathway(VL=goal/DiT=motor) 자체의 hidden state를 분리해서 각각 steer한다. '공유 성분 제거 후 잔차'라는 분해 연산은 유사하지만 분해 기준(카테고리 vs 아키텍처)이 다르다."
3. Q: "no-gate ablation이 왜 우리한테 중요한 증거인가?" A: "unconditional steering이 FPR을 0.052에서 0.500으로, success precision을 0.692에서 0.337로 붕괴시킨다는 정량 증거다. 우리가 항상 주장하는 'steering은 조건부(phase-matched)여야 한다'는 전제를 완전히 다른 도메인(LLM tool-calling)에서 독립적으로 재확인해준다."
4. Q: "이 논문의 online 개입이 진짜 우리가 말하는 online인가?" A: "부분적으로만 같다. StateCascade는 생성 스텝마다 재평가하지만 gate feature가 discrete parser event에 강하게 의존한다 — 명확한 경계 신호가 이미 존재하는 쉬운 세팅이다. 우리는 '아직 실패가 일어나지 않았는데 phase/type을 latent에서 조기에 읽어야' 하는, 명확한 경계 이벤트가 없는 더 어려운 문제다."
5. Q: "latency/저장 오버헤드 수치가 우리한테 왜 유의미한가?" A: "training-free online steering이 +10.9% latency, ~20KB 저장으로 실무 배치 가능함을 보여준다. 우리 VLA steering도 5-step 재계획 주기 안에서 conceptor projection이 이 정도 저비용이어야 실전 채택 가능성이 있다는 실현가능성 참고선이 된다."

## 한계·비판
- "training-free"는 정확히는 "backbone-frozen"이다 — steering 방향(vb, v_s^(d)), 여러 개 logistic gate classifier, 도메인 예측기, 카테고리별 threshold/강도(η)는 모두 calibration split에서 지도학습으로 적합되며 layer sweep·grid search 등 상당한 offline 준비 비용이 든다.
- 저자 스스로 인정: NESTFUL 성능 향상은 boundary entry·state-dependent gating·schema-local residual 세 채널의 결합 효과이며, 컴포넌트별 causal 기여도를 완전히 분리하지 못한다(Limitations).
- 근본적인 post-entry 실행 정확도(중첩 인자 바인딩, 세밀한 스키마 실현)는 여전히 병목으로 남는다 — ASA가 고치는 것은 주로 "boundary entry"와 "trajectory continuation"이지, "정확한 tool 호출 구성" 자체는 아니다(§7.1).
- 단일 백본(Qwen3-8B) 중심 실험. LLaMA-3.1 cross-backbone 체크는 NESTFUL 스타일에 한정되고, BFCL은 baseline strict success가 10% 미만이라는 이유로 비교 자체를 생략 — 일반화 근거가 제한적이다.
- domain(tool-schema family)이 사전 정의·라벨된 상태에서 도메인 예측기를 지도학습한다 — 완전 unsupervised가 아니다. 라벨 없는 continuous 신호(우리의 phase/failure-type)로의 이식 가능성은 미검증.
- 비교 baseline 다수(Tool-use SFT, ToolZero-style RL, ToolRL-style reward RL 등)가 저자들이 동일 backbone·evaluator 하에서 직접 재구현한 "controlled analogue"라서, 원 논문이 보고한 성능과 정확히 같지 않을 수 있다(재현성 유보 필요).
- Table 29에서 domain_only(shared 방향 없이 도메인 잔차만 사용) precision이 0.364로 오히려 낮음 — shared 성분이 실질적 성능의 대부분을 담당한다는 의심을 남기며, "도메인 분해"가 기대만큼 독립적 기여를 하는지는 불분명하다.
