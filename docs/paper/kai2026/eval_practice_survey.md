# 세밀 phase detect의 용도와, 분절/검출 논문들의 평가 관행

작성 2026-08-19. 대상 질문(사용자 원문):
> "action phase를 detect하고, 이게 GT(사람 라벨)보다 촘촘하면 **그래서 뭘 할 수 있는가**?
> 다른 detect/segmentation 논문들은 **뭘로 eval하고 있는가**?"

용도: `manuscript_draft.md` §3 마지막 문단·§4 결론의 근거 보강, 그리고 "우리 원고에
어떤 eval을 하나 더 붙일까"의 판단 재료.

**신뢰도 표기**: `[레포노트]` = 이 레포 정독 노트(`docs/references/reading_notes/`)에서
확인된 서지·수치. `[레포실측]` = 우리 실험 산출물(`docs/steering/*`). `[웹]` = 이번
조사에서 arXiv/CVF 원문으로 확인. `[미확인]` = 근거 부족 — 원고 인용 금지.
서지가 확인 안 된 논문은 아예 넣지 않았다.

---

## 1. "세밀한 phase를 읽으면 뭘 할 수 있는가"

### 1.1 문헌이 실제로 보여준 용도

| # | 용도 | 근거 논문 | 무엇으로 입증했나 |
|---|---|---|---|
| U1 | **전문가 라우팅** — phase마다 다른 action expert | PAMAE, arXiv:2606.27144 `[레포노트]` | phase 판독 정확도는 미보고, **하류 SR**로만: π0 73.8→83.0%, π0.5 85.8→91.4% |
| U2 | **이중 전문가 스위칭**(move/operate) | Move-Then-Operate, arXiv:2604.23620, ICML 2026 `[레포노트]` | 라우터 분류 accuracy 미보고, RoboTwin2 SR π0 대비 **+24.1%p**, 데이터 10× 절감 |
| U3 | **재계획 시점 결정** — subtask 경계에서 action chunk 재생성 | PACE, arXiv:2606.00537 `[레포노트]` | 경계 검출기 정확도 없음(결정론적 valley 검출), SR 57.8→64.2%, 실기 50.7→70.4% |
| U4 | **phase별로 감시 대상을 바꾸는 실패 감시** | ConditionNET, arXiv:2502.01167, RA-L 2025 `[레포노트]` | pre/core/effect 3단계마다 판정 기준을 바꾸고 **core phase에선 판정을 끔**. 지표는 프레임 판정 Acc 0.97 |
| U5 | **phase별 dense reward / MoE gate** | SARM2, arXiv:2606.10305 `[레포노트]` | stage MSE(0.006/0.031) + rollout 분류점수 ρ(0.833/0.667) |
| U6 | **test-time action 후보 선택**(progress/value 판독) | What Frozen VLAs Already Know About Success, arXiv:2605.28527 `[레포노트]` | 내재 지표(R² 0.55, matched-pair 정렬 92~94% vs shuffle 50%) **와** 하류 SR(26.7→44.3%) 둘 다 보고 — 이 리스트에서 가장 엄격 |
| U7 | **skill library 구성 → 정책 학습** | BUDS(RA-L 2022)·LOTUS(ICRA 2024)·XSkill(CoRL 2023)·UVD(ICRA 2024) `[웹]` | 전부 하류 정책 성공률로만 (§2.2) |

세 갈래로 수렴한다: **(a) 계산 배분**(라우팅·전문가·재계획), **(b) 감시 기준 전환**
(ConditionNET), **(c) test-time 선택·개입의 조건 신호**.

**중요**: "GT보다 촘촘함" 자체를 장점으로 내세운 논문은 이번 조사에서 확인되지 않았다.
비전 TAS 쪽에선 촘촘함이 대개 over-segmentation = **벌점** 취급이고(§2.1), 로봇 쪽은
분절 해상도를 아예 평가하지 않는다(§2.2). "촘촘한 단위가 GT 단위보다 하류에서 낫다"는
주장을 정면으로 한 사례는 없다 — 우리 원고가 들어갈 빈자리가 여기다.

### 1.2 우리 레포 실측이 이미 보여준 용도 (원고에 직접 쓸 수 있는 것)

| # | 실측 | 출처 | 요지 |
|---|---|---|---|
| E1 | **intrinsic 단위가 GT 단위보다 succ/fail을 더 잘 층화하는 task 4/9**(동급 1, 열세 1) | 41 §5 `[레포실측]` | drawer-left 0.93/z4.5 vs GT 0.94/z2.3; bread 0.77/z3.4 vs 0.67/z2.3; candle 0.73/z2.6 vs 0.62/z1.8. **탐색 라운드**(다중비교 보정·위약 없음) |
| E1′ | 그 우위의 상당 부분은 **후기(사후 판독) 클러스터 기여** — 조기(relpos<0.5) 한정 비교에선 intrinsic k8 +1.06 ≈ GT +0.99 동급. 예외 drawer-left(+2.1 vs +0.5) | 41 §8.2 `[레포실측]` | 원고에 E1을 쓸 땐 이 유보를 같이 써야 정직하다 |
| E2 | **k=8 근방이 최적** — 전 셀 mean z: k6 +0.69 / **k8 +1.05** / k12 +0.75 / k16 +0.61 / k24 −0.30 | 41 §8.3 `[레포실측]` | "촘촘할수록 좋다"가 아니라 **GT(3~6단계)보다 촘촘한 특정 대역**에 최적이 있다 |
| E3 | **phase 단위 길이 절제가 failure detector 조기성을 개선** — TPR/FPR 불변인데 W 이전 발화율 preW가 drawer-left 0.17→1.00, bread 0.48→0.92; phase-gt ≥ rollout 절제, 일부 task FPR도 감소(marshmallow 0.11, bread 0.23) | 43 §2 `[레포실측]` | 단 여기서 쓴 phase는 **GT phase**(dwell cap) — intrinsic 단위로는 미검증 |
| E3′ | 조기 발화는 dwell 초과(지지집합 이탈)가 아니라 **내용 신호** — 발화 실패판의 69%가 dwell 정상 범위에서 발화 | 43 §5 `[레포실측]` | 이득이 아티팩트가 아니라는 기전 분해 |
| E4 | phase 단위가 온라인 게이팅 파이프에 실배선(감지→phase 판정→phase별 연산자 스위칭). **감지·판정·스위칭 사슬 검증 통과**, 병목은 개입 연산자(구제 0, 위약 동급) | 42 §2·§4 `[레포실측]` | read≠write. 용도 주장은 **검출·조건화까지만**, SR 개선은 금지 |
| E5 | kNN(k=15) 라벨 전이 → 재군집화 없이 **스텝 단위 온라인 판독 가능**, serve 배선 존재 | 40 §2, `numbers.md` `[레포실측]` | 용도 주장의 실행가능성 전제 |

**원고용 한 줄**: 우리가 근거를 갖고 말할 수 있는 용도는 (i) **실패 감지의 조건 단위**
(E1 층화도 + E3 조기성), (ii) **개입 게이팅의 좌표**(E4 — 배선·판정까지 검증, 효과는
미확립)이다. ΔSR 개선은 주장하지 않는다.

---

## 2. 타 논문 eval 관행

### 2.1 비디오 Temporal Action Segmentation — 내재적 지표가 완비된 쪽

정본: Ding et al., *Temporal Action Segmentation: An Analysis of Modern Techniques*,
TPAMI 2023, arXiv:2210.10352 `[웹]`.

| 지표 | 정의 | 잡는 것 / 못 잡는 것 |
|---|---|---|
| **MoF** (frame-wise accuracy) | 맞춘 프레임 비율 | 프레임 라벨 품질만. 서베이 명시: **"분절이 잘게 쪼개져도 점수는 높을 수 있다"** + 클래스 불균형 취약 |
| **Edit score** | 예측 segment 시퀀스 vs GT segment 시퀀스 Levenshtein 유사도 | 행동 **순서** 품질. over-segmentation이 insertion으로 직접 페널티 |
| **F1@{10,25,50}** | segment tIoU>τ & 라벨 일치면 TP, 각 GT는 1회만 매칭 | 중복 검출이 FP로 잡혀 **over-segmentation이 precision을 깎음** |

벤치마크: GTEA / 50Salads(mid·eval 두 granularity) / Breakfast / YTI / Assembly101.

**Unsupervised의 추가 관행** — 클러스터↔라벨 대응이 없으므로 **Hungarian matching 후**
MoF/F1/IoU를 잰다. 매칭 수준(per-video / activity / global full)에 따라 점수가 크게
달라져 최근 논문(ASOT)은 per·full **양쪽 보고**가 관행.

| 논문 | 지표 | 비고 |
|---|---|---|
| CTE, CVPR 2019 `[웹]` | MoF + F1(IoU>50%), activity-level Hungarian | **K를 GT의 최대 행동 수로 고정** — 이후 표준 프로토콜 |
| TW-FINCH, CVPR 2021, arXiv:2103.11264 `[웹]` | MoF + IoU, **per-video** Hungarian. Breakfast 62.7/42.3, 50Salads(mid) 66.5/48.4 | ★ over-segmentation을 직접 논함: K를 키우면 MoF 57.8%로 하락하지만 **weighted cluster purity 83.8%** → "촘촘해도 클러스터는 순수"를 purity로 방어 |
| TOT, CVPR 2022, arXiv:2105.13353 `[웹]` | MoF + F1, activity-level Hungarian(per-video보다 엄격함을 명시) | 50Salads(eval) 44.5/48.2, YTI 45.3/32.9, Breakfast 39.0/30.3 |
| ABD, CVPR 2022 `[웹]` | 경계 검출 방식이나 평가는 표준 MoF/F1/mIoU | 원문에 boundary-level PR이 따로 있는지는 **미확인** |
| ASOT, CVPR 2024, arXiv:2404.01518 `[웹]` | MoF + F1 + **mIoU**(클래스 평균 → 불균형 처리), per-video·full 양쪽 | GT action order 사전지식 불필요 강조 |
| HVQ, AAAI 2025, arXiv:2412.17640 `[웹]` | MoF/F1/Recall + **JSD** — 예측 segment **길이 분포** vs GT 길이 분포 | over/under-segmentation 편향을 분포 수준에서 직접 정량화 |

**"GT보다 촘촘할 때" 처리 관행 4가지** `[웹]`:
1. K를 GT에서 가져와 고정(CTE 이후 표준) — 즉 대부분은 "몇 개로 자를지"를 **평가하지 않는다**.
2. K를 키워 촘촘해지면 MoF/F1은 떨어지고, **purity(many-to-one)로 유용성을 별도 방어**(TW-FINCH).
3. Edit/F1@τ가 over-segmentation의 표준 페널티 장치. MoF만 쓰면 이 축이 통째로 빠진다.
4. 최신 흐름은 **분포 지표**(HVQ의 JSD)로 촘촘함 자체를 편향으로 측정.

### 2.2 로봇 skill discovery / 시연 분절 — 내재적 지표를 사실상 아무도 안 잰다

| 논문 | 분절 품질 지표(내재적) | 하류 지표(외재적) | 정성 비중 |
|---|---|---|---|
| **BUDS**, Zhu·Stone·Zhu, RA-L 7(2):4126–4133, 2022, arXiv:2109.13841 `[웹]` | **없음** (GT 경계 대조 없음). skill 수 K ablation이 대리: K=1→0%, K=3→24.2%, K=6 최고, K=9→60.6%, K=11→44.6% | 정책 SR: Kitchen 72.0±4.0, Tool-Use 58.6, Hammer-Place 68.6, Real-Kitchen 56%; multi-task 시연 +8%p | **높음** — Fig.4 색분할 타임라인 + 의미 주석이 "semantically meaningful" 주장의 전부 |
| **LOTUS**, Wan·Zhu·Shah·Zhu, ICRA 2024, arXiv:2311.02058 `[웹]` | **없음** (boundary PR·사람 평가 전무) | LIBERO lifelong 3종 **FWT / NBT / AUC**(전부 SR 기반). LIBERO-OBJECT FWT 74.0, NBT 11.0, AUC 65.0; 평균 +11%p | 중 (skill 재사용 색분할) |
| **UVD**, Zhang et al., ICRA 2024, arXiv:2310.08581 `[웹]` | **없음** (사람 주석 일치도·GT subtask 대조·user study 전부 부재) | goal-conditioned IL SR(FrankaKitchen OoD 0.014–0.035→0.084–0.188), RL full-stage 65–100%, 실로봇 OoD 0.15–0.25(baseline 0.0) | **매우 높음** — 본문+부록 그림 다수, t-SNE |
| **XSkill**, Xu et al., CoRL 2023(PMLR 229), arXiv:2307.09955 `[웹]` | **없음** (purity·NMI·GT 라벨 대조 없음) | cross-embodiment SR: sim same 95.8%, cross 89.4%(1×)/70.2%(1.5×); 실로봇 UR5 평균 77% | **높음** — t-SNE, prototype projection, skill timeline |
| **AWE**, Shi·Sharma·Zhao·Finn, **CoRL 2023**(PMLR 229:2195–2209), arXiv:2307.14326 `[웹]` | **내적 기준만**: error budget η에 따른 재구성 오차 vs waypoint 수 trade-off, 등간격·속도 기반 heuristic과 비교. **GT keyframe 대조는 없음(저자도 한계로 인정)** | SR: sim 최대 +25%p, 실기 bimanual +4–28%p, decision horizon 1/10 (Cube Transfer 86→99%, Coffee Making 36→64%) | 중 |
| SPRINT, ICRA 2024, arXiv:2306.11886 `[웹]` | 해당 없음(시간 분절이 아니라 LLM relabeling + offline RL chaining) | ALFRED-RL unseen long-horizon 성능·학습 속도 | — |
| SkillDiffuser, CVPR 2024, arXiv:2312.11598 `[웹]` | GT skill 분절 대조 유무 **미확인** | Meta-World·LOReL SR + 정성 해석 주장 | — |

> ※ 사용자가 언급한 "AWE = RSS 2023"은 **CoRL 2023**이 맞다(`[웹]` 확인). "TAPS"는 시연
> 분절 계열 논문으로 특정하지 못해 **제외**.

### 2.3 실패 검출 계열 — 정확도축 + 시간축

`[레포노트]` `phase_detection/F_evaluation_protocol.md` 요약:

| 지표 | 정의 | 쓰는 논문 |
|---|---|---|
| **episode ROC-AUC (길이통제 s_T)** | task별 min-length T로 잘라 그 시점 점수로 AUROC | SAFE(NeurIPS 2025) 헤드라인 |
| **TPR/FPR/bal-acc @ α** | conformal α로 임계값 확정 후 이진 판정 | SAFE §6.1 |
| **T-det / detection time** | 실패 판의 경보 시점(정규화 여부는 논문마다) | SAFE(raw step), Hide-and-Seek(궤적 길이로 정규화) |
| **정확도-적시성 결합 스칼라** | TWA(FIPER·Hide-and-Seek), AUCPDT(VLA-FAIL) | 적시성을 단일 숫자로 접는 최근 흐름 |
| **길이 confound 통제** | 실패=timeout이라 시간만 세도 AUROC≈1 → min-length truncation | SAFE만 명시적, 후속들은 계승 안 한 사례 확인됨 |

우리 §3 층화 수치는 이미 이 관행(길이 AUROC 병기·scene LOSO·순열 z, 41 §0)을 따르므로,
"탐색적"이라는 라벨만 유지하면 관행에 어긋나지 않는다.

### 2.4 한 장 요약 — 두 축

| 분야 | 분절 품질(내재적) | 하류(외재적) |
|---|---|---|
| 비디오 TAS(지도) | MoF + Edit + F1@{10,25,50} — **논문의 전부** | 없음 |
| 비디오 TAS(비지도) | 위 + Hungarian 수준 명시 + purity(TW-FINCH) / mIoU(ASOT) / JSD(HVQ) | 없음 |
| 로봇 skill discovery | **거의 없음** (AWE의 내적 재구성 오차가 유일한 예외) | 정책 SR, lifelong FWT/NBT/AUC, cross-embodiment SR, sample efficiency |
| VLA phase 라우팅(PAMAE·MTO) | **없음** (라우터 accuracy 미보고) | 태스크 SR |
| VLA 실패 검출 | (분절 아님) | AUROC·TPR/FPR·T-det·TWA/AUCPDT |
| **우리 원고** | MI·purity·**시간 대조군 margin**·boundary z·off-phase rate | succ/fail 층화 AUROC/z (보조·탐색적) |

즉 우리는 **로봇 쪽에서 아무도 안 재는 내재적 지표를 재고 있고**, 그 지표가 비전 TAS
관행(purity·many-to-one)과 정확히 대응한다. 이건 원고의 약점이 아니라 포지션이다.

---

## 3. 우리 원고에 추가 권고 (기존 데이터로 가능한 것 우선)

### R1 (강력 권고, 추가 계산 0) — over-segmentation 프레임을 명시적으로 채택

지금 §3은 "3~4배 촘촘하다 + 경계는 GT와 안 맞는다(z −1.0~+0.5) + 그래도 purity 0.80"을
사실로만 나열한다. 심사자 입장에서 이건 **over-segmentation을 자백한 것**으로 읽힐 수
있다 — 비전 TAS에서 촘촘함은 edit/F1@τ로 벌점이 붙는 축이기 때문이다(§2.1).

처방: TW-FINCH가 쓴 것과 **같은 방어 구조**를 한 문장으로 명시한다 —
"K를 늘리면 프레임 정확도는 떨어지지만 클러스터 purity는 유지된다(83.8%)"의 우리 판이
이미 있다(purity 0.796, off-phase 중앙값 0.22, phase당 cluster 1.33~2.67 = **many-to-one
대응**). 즉 "경계가 GT와 안 맞는다"를 결함이 아니라 **many-to-one 세분(over-segmentation
이되 phase-순수)**으로 정의하고, 그 판정 근거가 purity/MI/margin이지 boundary F1이 아님을
명시. 비용 0(수치 전부 `numbers.md`에 있음), 이득은 프레이밍 방어.

### R2 (권고, 대부분 기존 산출물) — granularity–유용성 곡선을 하나의 도표로

로봇 분절 논문 중 해상도를 평가한 사례는 BUDS의 K ablation(K=1/3/6/9/11 → SR
0/24.2/최고/60.6/44.6%)이 사실상 유일하다 `[웹]`. 우리는 같은 형태의 곡선을 **두 축**으로
이미 갖고 있다:

- 내재적: k별 align margin·purity (41 §5, `ref align_pertask.json`)
- 하류: k별 succ/fail 층화 mean z (41 §8.3 — k6 +0.69 / k8 +1.05 / k12 +0.75 / k16 +0.61 / k24 −0.30)

이 둘을 **같은 x축(k, GT 단계수 3~6을 세로선으로 표시)** 위에 겹치면
"GT 해상도보다 촘촘한 k≈8에서 하류 유용성이 최대"라는 원고의 핵심 주장이 **한 장으로**
선다. 지금 §3 마지막 문단의 "4/9 task" 서술보다 강하고, E1′(조기 한정 시 동급) 유보도
곡선 위 각주로 자연스럽게 들어간다. 필요한 추가 작업은 기존 JSON 재플롯 수준.

### R3 (선택, 추가 계산 소량) — segment 길이 분포 지표

HVQ(AAAI 2025)가 도입한 JSD(예측 길이 분포 vs GT 길이 분포)는 우리 "평균 4.3 vs 16.1
스텝" 서술을 분포 수준으로 승격시킨다. 특히 **길이 1 깜빡임이 구간의 31~37%**라는 우리
한계(40 §5)를 숫자 하나로 정직하게 표현할 수 있다. 다만 1~2쪽 분량에선 우선순위 낮음 —
R1·R2 먼저.

### 하지 말 것

- **하류 SR(ΔSR) 추가 주장 금지.** 로봇 분절 논문 관행이 SR이라 유혹이 크지만, 우리 실측
  (42 §4)은 개입 arm이 위약과 동급·구제 0이다. SR을 붙이는 순간 원고가 무너진다.
- E3(조기성)를 "intrinsic phase의 이득"으로 쓰지 말 것 — 43의 phase 절제는 **GT phase**
  dwell 기준이다. 쓰려면 "phase 단위 일반의 유용성"으로만.
- E1의 "4/9 우세"를 단독 인용 금지 — 41은 탐색 라운드(위약·다중비교 보정 없음)이고
  조기 한정 시 GT와 동급(41 §8.2)이다. 원고 현재 문구("탐색적 관찰")를 유지할 것.

---

## 부록 — 참조 경로

- `docs/references/reading_notes/phase_detection/{A,B,F,G}_*.md`
- `git show exp/grid-phase-sep:docs/steering/41_grid_phase_separation.md`
- `git show feat/online-gated-pipe:docs/steering/42_online_gated_pipeline.md`
- `.claude/worktrees/safe-length-ablation/docs/steering/43_safe_truncation_ablation.md`
- `git show exp/grid-phase-sep:docs/steering/40_action_phase_readout_review.md`
- `docs/paper/kai2026/{manuscript_draft,numbers,references}.md`
