# Related works 지도 (2026-07-23 갱신)

용도: 랩미팅 덱의 related-works 슬라이드(구 S6 표 / S7 niche) 재작성 근거 + 논문 쓸 때 포지셔닝 단일 출처.
개별 정독 노트는 `docs/references/reading_notes/`, 방법론 후보 비교는 `docs/steering/07_steering_methods_survey.md`.

---

## 1. 06-23 덱 대비 무엇이 달라졌나

세 가지가 바뀌었고, 셋 다 표의 **행 추가가 아니라 축 변경**을 요구한다.

1. **COAST의 위상 강등.** 06-23 덱은 COAST를 "메인 메서드 · 우리와 문제설정 동일"로 세웠다. 이후 exp3 900판 + fit30 1800판(둘 다 사전등록 6-Holm, 위약 포함)이 전면 null이고, 사후 산술 진단으로 배포 C_steer가 데이터 위 ≈영행렬(R-가중 이득 0.006~0.007)이라 M=(1−β)I+βC ≈ (1−β)I 균일 감쇠였음이 확정됐다. → **"우리가 따라가는 방법"이 아니라 "우리가 재현 실패를 보고한 선행연구"**로 기술해야 한다. conceptor 수학 자체(2차 모멘트 Boolean 대수)는 토대로 유지.
2. **WA-LQR 등장 (신규).** 같은 DiT residual stream에 diff-of-means를 쓰는 논문이 나왔다. 우리 exp4-2의 직접 선행이자, "니치 0건" 주장의 부분 반례를 포함한다(아래 §5).
3. **SAE 3편 정독 완료.** Dr.VLA / Event-Grounded SAE / Observing&Controlling. 셋 다 **outcome을 scene·길이 confound에서 분리하는 문제를 직접 풀지 않는다**는 게 공통 결론이고, 이게 우리 니치의 새 방어선이 된다.

---

## 2. 배치 축 (표를 이 축으로 읽는다)

기존 표는 "탐지 / steering / 분석"이라는 **목적** 축 하나였다. 그 축으로는 COAST와 WA-LQR이 같은 칸에 들어가 버려서 우리가 왜 다른 일을 하는지 안 보인다. 축을 넷으로 늘린다.

- **축① 대조축 (contrast pair)** — 방향을 무엇과 무엇의 차이에서 뽑나
  자연 outcome(succ/fail) · 유도 교란(clean/perturbed) · 의미 개념(concept) · 물리량 라벨(kinematic) · pathway 절제(ablation)
- **축② 연산자 형태** — additive 단일벡터 · affine setpoint · projective 다차원 · closed-loop 피드백 · SAE feature edit
- **축③ 시간 조건** — 없음(전 구간 pool) · denoising-t 조건부 · **rollout-phase 조건부**
- **축④ 증거 품질** — 위약 대조 유무 · 유의성 검정 유무 · 표본 수

축③과 축④가 우리 자리다. 축③에서 rollout-phase 조건부는 아직 아무도 안 했고, 축④에서 위약 대조를 표준으로 도는 곳도 없다.

---

## 3. 전체 표

### 3.1 개입(steering) 계열

| 논문 | 대조축 | 연산자 | 시간 조건 | 보고 성과 | 증거 품질 | 우리와의 관계 |
|---|---|---|---|---|---|---|
| **COAST** (2605.17144) | 자연 outcome | projective 다차원 conceptor | 없음 (전 timestep pool) | π0.5 RoboCasa +0.16 / LIBERO-10 +0.37 / MetaWorld +0.25 / GR00T N1.5 RoboCasa +0.16 / 실로봇 +0.40 | 위약 대조 없음 | **재현 실패 보고 대상.** 2700판(exp3+fit30) 전면 null. 사후 진단: 자연 실패에선 succ/fail 2차 모멘트 부분공간이 겹쳐 C_steer가 퇴화 |
| **WA-LQR** (2607.14943, RSS'26 워크샵) | 유도 교란 (gripper arm만 outcome 혼입) | diff-of-means 벡터 + 층간 LQR closed-loop | denoising-t 조건부 O, rollout-phase X | 카메라 46.0→59.3 (+13.3) / gripper 61.3→72.7 (+11.4) / 노이즈 26.7→58.7, ActAdd는 67.3 | **위약·유의성 검정 전무** (코드 grep 확인), 30 trial/task, 이득은 전부 *교란된 입력*에서 | **exp4-2 직접 선행.** 같은 DiT residual + 같은 diff-of-means. 우리 null과 저들의 +13~40pp 차이는 연산자가 아니라 **대조축 신호 강도** 탓이라는 해석을 지지 |
| **Dr.VLA** (2603.19183) | 의미 개념 (SAE feature) | 단일 decoder-column 덧셈 | 없음 | 정성 인과만, **ΔSR 없음** | 폐루프 SR 판정 없음 | generality-vs-memorization metric이 우리 scene confound와 정확히 대응(암기 feature = 우리 scene 누출). SAE로 scene을 떼어낸 뒤 conceptor를 얹는 경로의 근거 |
| **Event-Grounded SAE** (2605.17204) | 의미 개념 (event 정렬 feature) | SAE feature zero-out / 편집 | event window 조건부 (post-hoc 랭킹) | OpenVLA L31 event-정렬 zero-out ΔSR −21 유의 | 폐루프 SR로 판정(좋음), 위약 없음 | **경고 신호**: π0.5 action-expert는 아무 랭킹에나 붕괴 = 비선택적. GR00T DiT가 바로 그 유형 → 단일 feature steer 금지, 다차원 필요. 이름과 달리 event는 SAE 학습이 아니라 feature 선택 단계에만 들어감. **동료가 재현 중** |
| **Observing & Controlling** (2603.05487) | 물리량 라벨 (pose/gripper) | 선형 observer + 최소노름 additive control | 없음 | Libero spatial 10 task × 10 rollout | 폐루프 SR 판정 | **SAE 아님**(선형 probe 대조군). flow/diffusion head steering은 **명시적으로 보류** → 우리 DiT 타깃이 비어 있음을 저자들이 스스로 확인해 준 셈. 선형 progress observer는 online phase 신호원 후보 |
| **CAA / ActAdd** (LLM) | 자연 대조 라벨 | additive 단일벡터 | 없음 | — | — | 우리 다차원 연산자의 baseline. exp4의 setM(setpoint mean-diff)이 이 계열의 affine 확장 |
| **SteerVLM** (2510.26769) | 학습 기반 | 경량 학습 steering 모듈 | 없음 | — | — | "백본 추가학습 없음" 원칙과 긴장. 후순위 |
| **LAE** (2509.20623) | — | latent activation editing | — | — | — | 미독 |

### 3.2 탐지 계열

| 논문 | 하는 일 | 우리와의 관계 |
|---|---|---|
| **SAFE** (2506.09937) | action-expert 마지막 layer feature로 실패 탐지, per-step LSTM | 우리 재현 완료: 공정 metric(min-length T)에서 seen 0.683 / **unseen 0.434 ≈ chance**. SAFE는 길이 confound를 직접 통제하는 유일한 선행 |
| **Path-Deviation-Heads** (2603.13782) | 경로 이탈 탐지 head | 온라인 검출 니치의 직접 경쟁자 |
| **FIPER** (2510.09459) / **Sentinel** (2410.04640) / **I-FailSense** (2509.16072) | 실패 예측 · 런타임 모니터 | 인접. 외부 신호 기반이라 internal-latent 니치와 안 겹침 |
| **VITA** | progress predictor | 메인 아님. phase-matched steering이 online phase 신호를 요구하므로 **보조 부품**으로 복귀 가능 |
| **KnowNo** (2307.01928) | 불확실할 때 도움 요청 | 개입 대신 위임. 대조 프레이밍용 |

### 3.3 분석·수학 토대

| 논문 | 하는 일 | 우리와의 관계 |
|---|---|---|
| **NOTALL** (ICLR 2026) | VL/DiT pathway를 인과 절제로 분해 (goal vs motor) | pathway-resolved steering의 동기. GR00T DiT fragility 경고(−68pp @9× 증폭, ablation은 관대 p=0.975), goal-subspace 20/1024 주입 시 task 보존 |
| **Conceptors** (Jaeger 2014) | soft-projector Boolean 대수 (AND/OR/NOT) | C_steer = C_succ ∧ ¬C_fail의 수학 토대. **COAST 재현 실패와 별개로 유지** |

### 3.4 미독 (표에 넣기 전 확인 필요)

- `docs/references/FindingNeMo_2406.02366.pdf` — 정독 노트 없음
- `docs/references/LA-LQR.pdf` — WA-LQR의 제어이론 참고문헌으로 추정되나 미확인
- `docs/references/Scaling World Model.pdf`, `pi07.pdf`, `CoT-VLA.pdf` — 배경 자료, 포지셔닝에 미사용

---

## 4. 축별로 본 지도 (슬라이드용 한 장 요약)

```
대조축(무엇의 차이에서 방향을 뽑나)
  자연 outcome  ──  COAST(재현X) · SAFE(탐지) · 우리 exp1-3(null)
  유도 교란     ──  WA-LQR(+13~40pp, 위약 없음) · 우리 exp4-2(진행중)
  의미 개념     ──  Dr.VLA · Event-SAE
  물리량 라벨   ──  Observing&Controlling
  pathway 절제  ──  NOTALL

시간 조건
  없음(pool)         ──  COAST · Dr.VLA · Observing
  denoising-t 조건부 ──  WA-LQR
  event window       ──  Event-SAE (post-hoc 선택)
  rollout-phase 조건부 ──  ★ 우리만 (미점유)

증거 품질
  위약 대조 있음  ──  ★ 우리만
  폐루프 SR 판정  ──  COAST · WA-LQR · Event-SAE · Observing · 우리
  정성 인과만     ──  Dr.VLA
```

---

## 5. 니치 재기술 (06-23 대비 정직하게 축소·이동)

**06-23 주장**: 미점유 영역 = internal-latent × online × failure-TYPE(goal vs motor) × phase-matched steering.

**지금 정확한 기술** — 네 개 방어선으로 쪼개고, 각각의 강도를 다르게 매긴다.

| 방어선 | 강도 | 근거 / 반례 |
|---|---|---|
| ② **rollout-phase 조건부 개입** | **강함** | 아무도 안 함. WA-LQR은 denoising-t 조건부 + chunk 단조 감쇠일 뿐이고, Event-SAE의 event는 개입 조건이 아니라 feature 선택 기준 |
| ③ **위약 대조 포함 엄밀 평가** | **강함** | COAST·WA-LQR 둘 다 위약 없음. 우리는 위약이 실제로 효과를 기각시킨 전례(exp2 s300033 +10 vs 위약 +8)가 있음 |
| ④ **nominal 입력에서의 SR 개선** | 중간 | WA-LQR의 이득은 전부 교란된 입력에서 측정 — 더 쉬운 타깃. 단 "nominal이 더 어렵다"는 건 우리 null의 변명으로도 쓰이므로 조심 |
| ① **자연 outcome 대조** | **약해짐** | 2700판 null. 그리고 WA-LQR의 gripper arm은 성공/실패 rollout을 버킷팅해 짝지으므로 **사실상 outcome 대조이고 +11.4pp를 얻었다** → "outcome 대조 × VLA는 0건"이라는 06-23 주장의 부분 반례. 인용 시 정확히: 그쪽은 *교란으로 유도한* 실패이고 state-matched가 아니라 여전히 자연 실패 대조와는 다르다 |

**추가 니치 (SAE 축, 3편 통합 결론)**: 세 논문 중 어느 것도 **outcome을 scene·길이 confound에서 통제한 채 feature를 선택**하지 않는다. Event-SAE는 오히려 자기 Table 10에서 같은 누출을 드러낸다(success probe: SAE code 0.79–0.93 vs task-id만으로도 0.54–0.64). Dr.VLA의 metric은 generality용, Event-SAE의 랭킹은 event용. → **confound 통제 하의 outcome-특이 SAE feature 선택**이 빈자리.

**한 줄 발표 멘트 초안**:
> "탐지도 있고 steering도 있는데, **언제 개입할지를 rollout 시간축에 조건부로 정하는 연구**와 **위약 대조로 효과를 검증한 연구**는 아직 없습니다. 그리고 저희는 그 검증을 실제로 돌려서, 저희 자신의 이전 결과를 기각시켰습니다."

---

## 6. 덱 슬라이드 초안 (구 S6/S7 대체)

**S6 — 관련 연구 (표 1장)**: 위 §3의 개입 계열 5행 + 탐지 2행으로 압축. 열은 `논문 / 대조축 / 시간조건 / 성과 / 증거품질`. 기존 표의 "우리와의 관계" 열은 발표 멘트로 빼서 슬라이드를 가볍게.

**S7 — 우리 자리 (§4 다이어그램 1장)**: 대조축 × 시간조건 2차원 격자에 논문들을 점으로 찍고, "rollout-phase 조건부" 행이 비어 있음을 보임. 우리 exp4-1/2/3이 그 행에 들어가는 그림.

**S6′ (신규 1장) — COAST 재현 실패**: 이건 related works가 아니라 결과라 Part 1로 옮기는 게 맞지만, 표에서 COAST를 "메인 메서드"로 소개하지 않는 이유는 S6에서 한 줄 언급 필요.

---

## 7. 남은 작업

- [ ] FindingNeMo / LA-LQR 정독 후 표 편입 여부 결정
- [ ] Event-SAE 동료 재현 결과가 나오면 §3.1 행에 "우리 재현" 열 추가 (COAST·SAFE와 같은 형식)
- [ ] WA-LQR gripper arm 반례를 논문 초고의 related works에 어떻게 쓸지 문장 확정 (지금은 각주 수준)
