# RL2-VLA: Adaptive RL Latent Compositional Steering with Test-Time Scaling

- arXiv:2607.26991v2 (2026-07-30), Tan*·Shailesh*·Iyer·Teo·Ju·Gu·Sartoretti (NUS + Toronto + ST Engineering)
- 프로젝트: https://rl2-vla.github.io (코드 "final paper와 함께 공개" 예정, 분석 시점 미공개)
- PDF: `docs/references/1RL2-VLA_*.pdf`. 분석일 2026-08-03.
- 우리 "**언제 steer할 것인가**"(online 실패검출 → 개입 게이팅) 축의 **직접 선행연구**.
  단 개입 층위가 다름 — activation write-in이 아니라 action-space 합성 + verifier 선별.

## 한 줄 요약

frozen VLA(π0/π0.5/OpenVLA)에 대해, VLA action-expert latent를 조건으로 학습한 경량 offline RL
policy(QAM)의 flow velocity를 VLA velocity와 가중합성해 다양한 action 후보를 만들고,
**SAFE 실패검출 + conformal prediction threshold가 실패를 감지한 timestep에만** 이 합성을 켠 뒤,
VLM verifier(CoVer/RoboMonkey)가 후보 중 최선을 골라 실행 → OOD에서 SR +8~17pp.

## 파이프라인 (한 timestep)

```
SAFE-LSTM(latent) → 실패점수 s_t > CP threshold δ_t ?
  ├─ 아니오: base VLA 그대로 N개 샘플
  └─ 예:    매 flow step에서 v_comp = w·v_VLA + (1−w)·v_RL (w~N(0.5,0.25), sample별)
→ verifier가 N(=40)개 점수화 → 1개 실행
```

- **QAM** (Q-learning with Adjoint Matching): flow-matching policy를 offline Q-learning으로
  학습하는 별도 논문[37]의 알고리즘. 여기서는 512×4층 소형 flow policy π_RL(a|e_t)를
  VLA finetuning 데이터셋(BridgeV2/DROID) replay로 학습. 보상 = 데모 끝 H=3 스텝 positive
  (relabeling — 환경 보상 아님). autoregressive(OpenVLA)는 velocity가 없어 CQL + Gaussian
  perturbation으로 대체.
- **latent 추출**: flow-matching VLA는 velocity 생성 직전 action-expert hidden state,
  집계 = horizon Mean × diffusion-step First (SAFE 레시피 4종 중 val 선택). OpenVLA는
  마지막 block의 decode 직전 벡터, token Last. 추론 시엔 VLA forward 부산물이라 추가 비용 ≈ 0
  (SAFE 1–2ms, QAM 12–48ms vs VLA 232ms+).

## "언제" 파트 상세 (우리가 볼 핵심)

- **검출기**: SAFE-LSTM (per-timestep causal, latent 조건, BCE). **task별 online rollout
  수집 필수** — seed당 100 × 3 seed = 300판/task, 성공·실패 자연 혼합, 60/40 분할.
  QAM과 달리 여기만 환경 상호작용 비용이 든다.
- **threshold**: 시변 conformal prediction band. D_val의 **성공 rollout**으로 보정 —
  "새 성공 rollout이 확률 1−α로 band 아래" 보장. 단일 검출기를 전 task에 학습하되
  CP band는 per-task(+OOD용 combined) 보정.
- **α 선택 휴리스틱**: α가 개입 빈도를 직접 결정하는데 최적값이 task마다 다름 →
  val에서 balanced accuracy로 top-3 α 후보를 좁힌 뒤 소규모 sweep. tie면 RL/VLA 중
  강한 쪽으로 개입 빈도를 치우침. task 추가마다 1회 반복 필요.
- **인과 근거 (이 논문의 최대 기여)**: success/failure 상태 **분리 scaling law** —
  BridgeV2 val에서 NRMSE 상·하위 1,024 튜플로 성공/실패 set을 만들고 best-of-N 오차를
  측정하면, 다양성 주입(합성)은 **실패 상태에서만 오차를 크게 줄이고 성공 상태에선
  오히려 늘린다**. downstream ablation: adaptive vs always **최대 +8.9pp**,
  SAFE trigger가 VLM(CoVer) trigger보다 +3.5pp.

## Ablation 수치 (기여 분해)

| 요소 | 비교 | 기여 |
|---|---|---|
| latent 조건 (두 번째 policy의 유능함) | latent vs raw obs | +38.8pp (raw는 0.5%로 붕괴) — 필수 전제 |
| 실패 시에만 켜기 | adaptive vs always | 최대 +8.9pp |
| 두 번째 policy가 하필 RL | RL vs BC | **+4.5pp뿐** |

→ 개선의 주성분은 "RL이 데모 너머 행동을 발견"(논문 서사)이 아니라 **비상관 두 번째
policy가 후보 집합의 다양성을 키워 best-of-N의 max를 올리는 것** + 실패 게이팅.
verifier 없는 조건은 실험에 없음 (verifier 의존은 상쇄 안 된 구조적 전제).

## 비용 구조 — "modular"의 실체

레시피가 모듈식일 뿐 산출물 재사용 불가. **VLA마다 전부 재학습**: QAM은 latent 공간에
결합(π0 1024d ≠ OpenVLA 4096d, 500k~1M step)·모델 패밀리 바뀌면 개입 메커니즘 자체 교체,
SAFE도 per-VLA + per-domain (sim 학습 SAFE가 real 일반화 실패 → 실기 rollout 재수집,
limitations에 자인). CP·α는 per-task.

## 우리와의 비교

| 축 | RL2-VLA | 우리 |
|---|---|---|
| 개입 층위 | **action-space** (velocity 합성 + best-of-N verifier 선별) | activation write-in (residual stream h′=h·M) |
| latent의 역할 | 조건 입력만 (제목의 "latent steering"은 오해 소지) | 개입 대상 그 자체 |
| "언제" | SAFE+CP **binary** 게이트, per-task α | online pathway(type)·phase 식별 (미해결, RQ3/4) |
| 실패 유형/phase | 구분 없음 — 켜지면 항상 같은 개입 | goal/motor 라우팅 + phase-matched가 핵심 기여 목표 |
| 추가 학습 | QAM RL + SAFE + (기성) verifier | 없음 (rollout에서 fit만) |
| read≠write | **우회** — 방향을 쓰지 않고 후보를 흩어 verifier가 선별 | 정면 돌파 필요 (exp5-2 직접증거) |
| 실패 세팅 | OOD(unseen instruction/object) 유발 실패 | in-domain 정책 실패 |

**니치 영향**: "online 실패검출로 steering 게이팅" 축은 이제 선점됨(+인과 증거까지).
우리 기여 주장은 **type(goal/motor)·phase 해상도 + activation-level 개입 + 무학습/무verifier**
쪽으로 무게 이동. 미점유 niche(내부 latent × online × 실패 TYPE × phase-matched **write-in**)는
여전히 비어 있음 — RL2는 TYPE·phase·write-in 세 칸 모두 없음.

## 가져갈 것 / 경계할 것

- (+) **게이팅 축의 외부 인과 검증**: "성공 상태 개입은 해악, 실패 상태만 이득"이 우리
  관측(apple 유의 해악, drawer β=1.0 파괴, gated 처방)과 독립 재현. 시변 CP threshold는
  우리 online 검출기(DiT block31 AUROC 0.92)에 이식 가능한 부품.
- (+) SAFE의 real 일반화 실패 = 실패검출기의 scene/domain 특이성 — 우리 scene 암기
  관측(unseen 전이 불가)과 같은 현상의 독립 증거.
- (−) length confound 논의 전무 (단 per-timestep causal 게이트 용도라 time-pooled보다는 덜 취약).
- (−) best-of-N+verifier 논리는 단일-forward 개입에 이식 불가 — 우리 read≠write 난제의
  해답이 아님.
- 관련 문헌 (이 논문 Related Works에서): SCALE(실패 예상 시 다양성 확대 — "언제" 축 최근접),
  VLA-ATTC·OneTwoVLA·Recurrent-Depth VLA(adaptive inference 트리거 대안),
  DSRL(diffusion noise RL steering — 제3의 개입 지점), DynaGuide/VLS/VLA-Pilot(differentiable
  steering), RoboMonkey/CoVer(verifier 계열 원류).
