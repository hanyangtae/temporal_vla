# VLA test-time sample-and-select 채택 현황 조사 (2026-08-04)

**질문**: 최신 VLA/robot foundation model들이 "후보 여러 개 생성 + 채점 선택"(sample-and-select)을
기본 추론 경로에 내장했는가? — opus agent 3계열 병렬 조사 (플래그십 코드 실독 / world-model 계열 /
산업 채택·서베이). 근거 등급: [코드실측]=추론 코드 직접 확인, [검증]=1차 출처(논문 전문·공식 문서)
fetch, [미확정]=공개 근거 없음.

## 총평

1. **플래그십 전원 내장 안 함** — GR00T N1.6/N1.7, π*0.6/π0.7, Gemini Robotics 1.5/2,
   Figure Helix 02, 1X Redwood, GR-3, RDT-2, Wall-X, OpenVLA-OFT, SmolVLA, AgiBot GO-1 전부
   단일 forward/단일 denoise. 코드 있는 곳은 코드로 확인 (`best_of|verifier|rerank|candidate` 0 hit).
2. **test-time compute는 쓰되 선택에 안 씀** — 3갈래: (a) CoT thinking token(Gemini ER, 토큰 예산
   스케일 곡선 공개), (b) 생성형 conditioning(π0.7: 14B world model이 subgoal 이미지 **1장** 생성,
   순위 없음; AgiBot GE-Act; Unitree — 마케팅은 "world model 최적화"라지만 코드는 `assert bs==1`),
   (c) 지연 은폐(RTC 비동기 chunking). 정확한 프레이밍: "**parallel candidate generation +
   reranking을 안 한다**" (GR00T는 추가 test-time compute 자체가 0이라 "reasoning은 하되 선택 안 함"
   프레이밍의 반례).
3. **명시적 반대 베팅** — 1X: "best-of-N은 future work" 명문화. Generalist GEN-0: "without
   inference-time guidance... physics doesn't stop". GR00T N2 기반 DreamZero(2602.15922):
   "without test-time optimization — 7Hz는 search 기반으론 불가".
4. **예외는 실시간 제약 느슨한 니치 둘** — ★Sereact Cortex 2.0(2604.20246, 산업 pick-place):
   world model 후보 미래 + Process-Reward 채점, 기본 내장. k=1→30에서 SR 0.962→0.996, 지연
   310ms→9.2s (action-space test-time compute 스케일 곡선의 유일 공개 사례). τ0-WM(2606.01027):
   opt-in 모드만. Cosmos-Policy: planning 지원하나 NVIDIA cookbook이 "복잡도·5s/chunk 비용으로
   생략" — 기본 경로에 플래그 없음.
5. **부품은 출하되기 시작, 배선은 안 됨** — LeRobot v0.6 SARM(`select_action`이
   `NotImplementedError` — 명시적으로 비워둠)·Robometer, Stanford RoboReward(2601.00675, 4B/8B
   공개 — eval·RL학습 포지셔닝), τ0-VLA(아키텍처엔 value-model search 있으나 모델카드 "not part
   of this checkpoint"). 근시일 형태 = 정책 옆에 value/reward head 동봉이지 제어 루프 내 search 아님.
6. **막는 건 latency, 저자들 자인** — RoboMonkey "less suitable for high-frequency control"
   (H100 1장 1.5Hz), 독립 측정 1.94×(MG-Select Table 14). 자율주행에선 candidate+scorer가 표준인
   이유 = scorer가 ms급(trajectory vocabulary); VLA는 후보마다 대형 forward라 구조적으로 다름.
   ⚠ Spec-VLA류 "speculative decoding verify"는 무손실 가속이지 품질 선택 아님 — 용어 혼동 주의.
7. **패러다임은 "상시→조건부"로 형질 변경 중 (2026 수렴 처방)** — Gated GeoBoN(2607.17454:
   26.2% 발동으로 이득 74.8% 회수), τ0-VLA confidence routing, RL2-VLA SAFE+CP 게이트가
   독립적으로 같은 처방에 수렴. **"실패 예측 시에만 개입" 게이팅 축이 2026년의 실질 전선.**
8. **world model의 4역할 분화** — 학습데이터 생성(DreamGen; Genie 3는 목표-무지라 채점 구조적
   불가) / 추론 conditioner(π0.7 — 산업 대세) / 오프라인 평가자(1X checkpoint 랭킹, WorldEval) /
   추론 selector(최희소). Fast-WAM: "test-time 상상 필요 없음 — 이득은 video co-training에서"
   (190ms vs 590–800ms).

## 우리 연구에의 함의

- **니치 무침범 재확인**: 예외 사례들(Cortex 2.0, τ0-WM)조차 선택 기준이 무차별 전역
  progress/value 스칼라 — 실패 TYPE(goal/motor)·phase 해상도는 전무. 내부 latent write-in도 전무.
- **게이팅 수렴(§7)은 우리 "언제" 축의 외부 검증**이자, RL2가 선점한 축이 필드 전반의 전선이라는
  뜻 — 우리 기여 주장은 type/phase 해상도 + 무verifier 단일-forward 개입으로 더 좁고 깊게.
- **latency 논거(§6)는 우리 차별점 강화**: 우리 activation write-in은 후보 N개·외부 채점 없이
  단일 forward에 얹힘 — sample-and-select의 구조적 부채(1.5Hz)가 우리에겐 없음.
- **phase/progress 신호 공급원 후보 발견**: LeRobot Robometer(Qwen3-VL progress 모델),
  BAAI RoboBrain 2.5 General Reward Model(step-aware dense progress), SARM(stage-aware) —
  VITA 대체재로 검토 가치. 단 이들은 경쟁자의 미싱피스이기도 함.
- **Cosmos-Policy가 RoboCasa 사용** (`Cosmos-Policy-RoboCasa-Predict2-2B`, SR 67.1%, planning
  미사용) — 같은 벤치의 test-time-compute 이웃. exp4-3 Cosmos 분리도 지도와 접점.
- **"배치됐나"는 문헌에서 아직 안 물어진 질문** — VLA TTS 전용 서베이 부재, 어떤 목록도 배치
  상태 미추적. 포지셔닝/서론 소재.

## 미확정 (주장 금지)

Tesla Optimus(1차 자료 전무 — 인용 자체 비권장; FSD 2022 interaction search를 Optimus로 전이
금지), Skild, Dyna DYNA-1(reward model 공개하나 온라인 사용 미공개), Generalist GEN-1의 비공개
"new inference-time techniques"(유일하게 열린 문), Gemini Robotics 2(테크리포트 없음 —
블로그·모델카드뿐), GR00T N2(미출시 — DreamZero 논문 기반 추론).
