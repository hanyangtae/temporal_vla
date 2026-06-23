# 로컬 Claude 핸드오프 프롬프트 — 랩미팅 PPT 보강

> 아래 블록을 그대로 복사해 새 Claude 세션에 붙여넣으면 된다. 논문 PDF(특히 Dr.VLA)는
> 붙여넣은 뒤 사용자가 직접 첨부한다.
>
> **중요**: 대상 PPT(`랩미팅0618.pptx`, 23장)는 이미 atomic-action 프레이밍이 들어가 있다
> (S15 Core Hypothesis = atomic action pick/moving/place/push, S16 phase segmentation,
> S17 VLM-vs-expert + Sangwoo probe, S11 VLM-detector-underperforms). 따라서 이 작업은
> "새 축 추가"가 아니라 **이미 있는 덱을 보강·갱신·갭 메우기**다.

---

너는 VLA latent steering 연구의 **이미 거의 완성된 랩미팅 PPT를 "보강·수정"**하는 작업을
맡는다. 처음부터 만드는 게 아니고, 새 축을 새로 넣는 것도 아니다. 기존 23장의 구조·스타일을
보존하면서 **갭을 메우고 최신 결과로 갱신**하는 최소 침습 편집이다.

## 0. 대상 파일 & 환경

- **PPT**: `/home/dongkyu/pkt_ws/temporal_vla/랩미팅0618.pptx` (PPTX, **23장**, ~10MB).
- 편집 도구: **python-pptx 미설치** → 먼저 `pip install python-pptx` 하거나, 정 안 되면
  PPTX(zip 안 `ppt/slides/slideN.xml`)를 직접 편집. 텍스트 추출은 zipfile+ElementTree로 가능.
- repo 루트: `/home/dongkyu/pkt_ws/temporal_vla/`.
- **사용자에게 받을 것**: 논문 PDF, 특히 **Dr.VLA** (repo에 없음). 받기 전엔 Dr.VLA 인용 자리를
  `[Dr.VLA: 의미론적 단위 steering — PDF 확인 후 채움]` placeholder로 둔다.

## 1. 현재 슬라이드 지도 (2026-06-18 시점, 추출본)

```
S01 타이틀 — VLA Activation Steering (Park kyungtae / Park dongkyu)
S02 RECAP: 'VLA Loop Break' (success-only, no temporal history → loop)
S03 Looking Inside: Reframing (loop는 증상; online으로 뭔가 시도)
S04 What Is an Activation? (Eagle-VLM→VL-SA→DiT; VL/DiT 두 tap)
S05 What Is Steering? (h' = h·Mᵀ, multi-dim subspace)
S06 Related Work (COAST/SAFE/… 표)            ← Dr.VLA 추가 필요
S07 Our Niche (internal-latent × online × failure-TYPE)
S08 Experiment (seen18 1800 rollouts, N1.6, SR 0.537)
S09 Shared Failure Zone (SAFE replication)
S10 Real Failure Signal (length-controlled, AUROC ~0.6)
S11 VLM-Side Detector Underperforms (SAFE-LSTM seen 0.683 / unseen 0.434)
S12 Two Failure Regimes (initial-condition / execution-drift)
S13 Timing & Pathway (VL early t≤8 / DiT late t≥12)
S14 Steering Attempts — Status (N1.6 DiT-only ΔSR≤0)   ← COAST repro 최신화 검토
S15 Core Hypothesis (atomic action pick/moving/place/push 단위 steer)  ← 이미 새 축
S16 Action-Phase Segmentation (How) (offline heuristic: proprio+gripper+EE)
S17 Where Is Phase Decided? VLM vs Expert (Sangwoo probe)
S18 Next Steps (faithful N1.5 → fixed-instruction → phase-matched → cross-bench)
S19 Limitations & Caveats (signal weak, length/instruction confound)
S20 Thank you
S21 Appendix A — Conceptor Math
S22 Appendix B — Experimental Standards
S23 Appendix C — Number Sources
```

→ 먼저 PPT를 직접 열어 이 지도가 맞는지 확인하고(텍스트/그림 모두), 어긋나면 실제 내용 기준으로
다시 매핑하라.

## 2. 이 덱의 메시지 (유지할 축 — 새로 만들지 말 것)

- task/instruction 단위로 뭉뚱그리지 말고 **pick / moving / place / push 같은 atomic action
  (=subtask phase) 단위로 그 phase에 맞춰 steer**하는 게 더 낫다 (S15).
- 논문 목적: "phase 단위 steering이 기존 global/time-pooled steering보다 더 잘 먹힌다"를 보이는 것.
- 떠오르는 미해결 문제 2개 (S16, S17): (a) rollout/inference에서 atomic action phase를 **어떻게
  구분?** → conventional 후보(offline heuristic / online 검출기). (b) phase는 action expert(DiT)에서
  실행되는데 **앞단 VLM이 phase 정보를 토큰으로 넘기는지** 어떻게 확인? (상우 공동).
- 선행연구(S06): SAFE(검출)/COAST(steer→SR↑)/NOTALL(VLM·DiT 훼손 결과 다름) **+ Dr.VLA(추가 예정)**.
- 내 실험: VLM detector 잘 안 됨(S11), task 단위 latent 구분은 잘 됨, COAST 재현 안 됨(S14).

## 3. 우선 처리할 갭/갱신 (이번 보강의 실제 작업)

1. **S06 Related Work에 Dr.VLA 추가** — "실제 steer 시 의미론적 단위로도 steer 가능 + 경고: 어떤
   feature가 최상위 활성화를 보여도 신뢰할 방향으로 조정을 유도한다는 보장은 없다; 정렬돼 보이는
   feature조차 조작 시 제한적·예측 불가능한 인과 영향." → 우리와의 관계 = **atomic-action(의미론적
   단위) steering의 직접 선행이자, "feature 정렬 ≠ 인과적 steerability"라는 경계.** 정확한 문구·인용은
   PDF 받은 뒤 대조. (지금은 placeholder.)
2. **S14 / S18 COAST 재현 상태 최신화** — 현 덱은 "N1.6 DiT-only ΔSR≤0 → faithful N1.5 재현 예정".
   최신 사실(`16_coast_pi05_libero_repro.md`): GR00T-N1.5×RoboCasa 충실 재현도 mean ΔSR≈+0.014로
   **positive control 실패(n=30)** → control을 **π0.5×LIBERO-10(논문 +0.33로 최대 gain)**로 옮겨 재시도
   중. S14/S18에 이 pivot을 반영할지 사용자에게 확인.
3. **S11 ↔ instruction confound 연결** — "VLM detector 잘 안 됨"을 instruction confound와 묶어 설명
   (`11_instruction_confound.md`: 헤드라인 VL AUROC 0.93은 instruction in/out 쏠림 아티팩트). VL 신호가
   약해 보이는 게 confound 때문일 수 있다는 caveat를 S11/S13에 한 줄.
4. **S16/S17 보강** — atomic-action phase 검출 후보를 `14_pathway_phase_online_steering.md`의
   "phase를 online에 어떻게 아나"(절대 t-bin / progress-normalized=online 불가 / subtask 검출기)와
   정합시키고, "VITA식 progress predictor가 보조 phase 신호로 부활 가능"을 옵션으로.
5. **전체 일관성 점검** — 수치/그림이 부록 C(S23) 출처와 맞는지, 갱신된 수치가 있으면 동기화.

## 4. 읽을 자료 (repo 루트 기준)

### 방향·thesis 단일 출처
- `docs/steering/14_pathway_phase_online_steering.md` — pathway(VL/DiT) 분리 + phase-matched,
  "online에 phase/type 읽을 수 있나"가 중심 미해결 문제. S15~S17의 논거.
- `docs/steering/15_research_structure.md` — RQ1~4 / 가설 C1~C4 / crossover 검증 설계.
- `docs/steering/13_lab_meeting_2026-06_outline.md` — 이 덱의 **원본 아웃라인**(슬라이드↔그림 경로↔수치
   출처 매핑). 현 PPT와 대조용. 단 13은 atomic-action 축이 약하니 덱이 더 최신임에 유의.

### 슬라이드 근거(수치·그림)
- `docs/steering/01_seen18_latent_analysis.md` — task 분리·길이 confound·실패 onset 두 regime.
- `docs/steering/06_coast_groot_n16_summary.md` — COAST 재현 ΔSR≤0(N1.6) (S14).
- `docs/steering/16_coast_pi05_libero_repro.md` — COAST positive-control 재시도(π0.5×LIBERO) (S14/S18).
- `docs/steering/11_instruction_confound.md` — VL 신호 instruction 아티팩트 경고 (S11/S13).
- `docs/seen18_safe_detector_verification.md` — SAFE-LSTM 수치(seen 0.683 / unseen 0.434) (S11).

### related works / 방법 메뉴
- `docs/steering/07_steering_methods_survey.md` — COAST/CAA/SAE/NOTALL/subspace 비교 (S06/S07).
- `docs/steering/README.md` — 문서 지도.

### 산문/그림 재사용 1순위
- `outputs/weekly_report/2026-06-w2.md` — related-works·N1.6 재현·confound가 figure 경로·판정까지 정리.
- 논문 원문: `docs/references/` (COAST/SAFE/NOTALL/PathDeviationHeads/VITA PDF·txt).
  **Dr.VLA는 여기 없음 → 사용자 제공 대기.**

## 5. 규칙 (엄수)

- **단정 금지** — 분석 진행 중. 모든 정량 수치는 "현재 스냅샷, 갱신될 수 있음". confound·한계는 약점이
  아니라 *엄밀성의 성과*로.
- **그림은 경로를 `ls`로 실존 확인한 것만** 임베드/교체. 없으면 표로 대체(추정 경로 금지).
- **수식에 달러($) LaTeX 구분자 미사용** — plain/unicode.
- **모든 산출물은 한글.**
- **기존 PPT의 스타일·레이아웃·슬라이드 순서를 보존.** 새로 갈아엎지 말고 최소 침습으로.
- PPTX 직접 편집은 **백업 먼저**(`cp 랩미팅0618.pptx 랩미팅0618.bak.pptx`).

## 6. 작업 순서

1. PPTX를 열어(또는 텍스트 추출) §1 지도가 현재와 맞는지 확인. 어긋나면 실제 기준으로 재매핑.
2. §3 갭 목록을 **편집 계획(슬라이드별 무엇을 어떻게)**으로 정리해 사용자에게 보여주고 **승인받는다.**
   바로 편집하지 마라.
3. 승인 후 백업 → python-pptx(or XML)로 수정. 수정마다 출처(어느 md/figure/수치)를 슬라이드 노트에.
4. Dr.VLA PDF를 받으면 placeholder를 실제 인용(문구·페이지)으로 교체.
5. 변경 요약(무엇을 왜 바꿨는지)을 한글로 보고.
