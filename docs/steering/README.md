# Latent Steering (메인 연구 라인)

이 디렉터리는 프로젝트의 **메인 method — pathway-resolved + phase-matched activation
steering** 문서 기준 위치다. 목표는 GR00T RoboCasa rollout에서 성공/실패 latent 신호를
pathway별(VL goal / DiT motor)·phase별로 찾고(길이 confound 통제 필수), 추론 중 그 신호로
**어느 pathway·어느 phase에서 실패하는지 식별**한 뒤 phase-matched conceptor steering으로
개입해 Success Rate를 올리는 것이다. **중심 미해결 문제: online phase/failure-type 식별이
가능한가** — 안 되면 steering을 라우팅할 수 없다. 메인스트림 thesis는 `14_*` 참고.

번호 prefix(`01_`, `05_`, `07_`, ...)는 대략의 읽기 순서다(아래 Reading Order 참조). TTA/VITA
progress predictor 방향은 현재 메인이 아니며 [`../ttt/`](../ttt/README.md)에 보존 문서로 둔다.

## 현재 결론

- 초기 SAFE detector와 DiT-only COAST steering은 평균 SR 개선을 만들지 못했다.
- 실패 원인 해석은 NOTALL 가설과 맞는다: GR00T pathway는 goal semantics(VL)와 motor
  program(DiT)이 나뉘며, DiT-only feature는 goal-type failure를 충분히 잡지 못한다.
- Phase 3 결과에서는 VL pathway가 t≤8에서 더 이른 실패 신호를 내고, DiT는 t≥12에서 강해진다.
- **현재 main stream = pathway-resolved + phase-matched steering** (VL/DiT 분리 + DiT는 rollout
  phase 조건부). 중심 미해결 문제는 **online phase/failure-type 식별**. 상세: `14_pathway_phase_online_steering.md`.
- 검증은 사다리식 ablation(global → pathway-split → +phase-bin)으로 각 단계 ΔSR 비교.
- 다음 실험 축: fixed-instruction confound 제거 → VL conceptor fit + phase-matched steering →
  ΔSR 인과 재측정. (faithful N1.5 COAST positive control 은 재현 실패—원인 미상—로 중단.)

## 수집 계약

- 기본 SAFE/multilayer 수집은 DiT `hidden_states`만 저장한다.
- N1.6 full block residual pathway의 model-token 축은 `T=51 = state(1)+action(50)`이다.
  N1.6 SAFE 기본 export는 full 51-token residual이 아니라 마지막 action-50 block 중
  RoboCasa valid horizon 16만 저장한 `[K,16,1024]` action-token feature다.
- N1.5 aligned block residual pathway의 model-token 축은
  `T=49 = state(1)+future_tokens(32)+action(16)`이다. N1.6과 shape를 맞추려고
  padding/truncation하지 않고, verifier는 `token_count=49`를 기대한다.
- VL+DiT pathway 수집은 명시적으로 `feature_server.py --capture-vl`
  또는 LeRobot `scripts/serve/lerobot.py --capture-vl`을 켤 때만 동작한다.
  이 경우 pkl에는 DiT `hidden_states`와 VL `vl_hidden_states`가 step 단위로 함께 저장된다.
- steering용 pathway run은 `verify_rollout_collection.py --require-vl-hidden-states`로
  `vl_hidden_states` 존재, step count, shape, VL metadata를 검증한다.
- `collect_pathway_parallel.sh`는 `--capture-vl`로 서버를 띄우고 수집 종료 후 위 verifier를
  기본 실행한다. 필요할 때만 `VERIFY_AFTER_COLLECT=0`으로 끈다.

## 사용 절차

1. 방향·가설·open problem·ablation 사다리는 `14`(메인스트림 단일 출처) → `15`(연구 구조 RQ/가설)에서 본다.
2. 표현 분석의 근거를 볼 때는 `01`(seen18 latent 분석, 길이/instruction confound·cross-task 실패 구조 포함)을 읽는다.
3. COAST 재현은 실패(원인 미상)로 종료했고 상세 문서는 정리했다 — conceptor 수학 provenance 는 `src/conceptor/README.md`.
4. 다음 steering 방법론 후보를 정할 때는 `07`을 보고, pathway 분리력 결과(VL early / DiT late, steering target)는 `08`에서 본다.
5. instruction confound 판정과 VL/DiT LDA 사분면 방법은 `11_instruction_confound`, fixed-instruction 재수집·conceptor fit·steering eval 계획은 `11_phase4`에서 본다.
6. LeRobot/멀티벤치 SAFE 수집 확장은 `05`를 별도 track으로 본다.
7. 랩미팅 발표 노트(슬라이드 지도·revision gap·그림/수치 출처 매핑)는 `13_lab_meeting_ppt_notes`에서 본다.
8. 코드 흐름과 수식을 한 화면에서 볼 때는
   [GR00T Latent Steering Explorer](../groot/00_groot_steering_explorer.html)의 탭/검색/code map을 연다.

## Reading Order

> **메인스트림 thesis(먼저 읽기)**: [14 Pathway+Phase Online Steering](14_pathway_phase_online_steering.md)
> — 현재 연구 방향·핵심 가설·open problem·ablation 사다리의 단일 출처.

1. [01 seen18 Latent Analysis](01_seen18_latent_analysis.md) — GR00T-N1.6 RoboCasa seen18
   잠재공간에서 succ/fail 이 구분되는지·어떤 조건에서 드러나는지. **길이 confound 통제**가
   모든 해석의 전제. cross-task 실패 공유 구조(COAST Sec 4.4 미재현)·steering 의 표현 측 근거.
2. [05 SAFE lerobot Collection](05_safe_lerobot_collection.md) — SAFE latent 수집을
   lerobot 정책(pi0.5/pi0+FAST/X-VLA/GR00T N1.5) × 멀티 벤치로 확장 (plan + status + handoff).
3. [07 Steering Methods Survey](07_steering_methods_survey.md) — conceptor steering(현 방식,
   COAST 계열) + 이후 적용 후보(CAA, SAE-guided, NOTALL pathway, learned steering)와 권장 순서.
4. [08 Pathway Separation Analysis](08_pathway_separation_analysis.md) — Phase 3 통합: DiT 32-layer
   pre-failure 분리력 + VL(goal) vs DiT(motor) 비교(VL 이른 t≤8 / DiT 늦은 t≥12, goal-vs-motor
   task 분열) + Phase 4 steering target 선택 근거.
5. [11 Instruction Confound](11_instruction_confound.md) — Phase A 판정: 헤드라인 VL AUROC 가
   instruction(in/out) 쏠림 아티팩트일 수 있음. VL/DiT LDA 사분면 분석 방법·근거.
6. [11 Phase 4 N1.5 Instruction-Fixed Plan](11_phase4_n15_instruction_fixed_plan.md) —
   N1.5 instruction-fixed seed selection, paired rollout collection, pathway cache, conceptor fit,
   steering eval 계획과 실행 log.
7. [13 Lab-Meeting PPT Notes](13_lab_meeting_ppt_notes.md) — 랩미팅 발표 덱 작업 노트:
   현재 23-slide 지도·메시지 축·revision gap·편집 규칙 + 원본 아웃라인(슬라이드↔그림↔수치 매핑).
8. [15 Research Structure](15_research_structure.md) — RQ1~4 / 가설 C1~C4 / crossover 검증 설계.

## 결과 위치

| 질문 | 문서 |
|---|---|
| 현재 연구 방향(메인스트림)은? | `14_pathway_phase_online_steering.md` |
| 연구 질문/가설(RQ1~4, C1~C4)·검증 설계는? | `15_research_structure.md` |
| succ/fail latent가 실제로 분리되는가? | `01_seen18_latent_analysis.md` |
| pathway 분리력·최신 steering target은 무엇인가? | `08_pathway_separation_analysis.md` |
| VL AUROC가 instruction 아티팩트인가? | `11_instruction_confound.md` |
| N1.5 instruction-fixed 수집/steering은 어디까지 왔나? | `11_phase4_n15_instruction_fixed_plan.md` |
| LeRobot/멀티벤치 SAFE 수집은 어디까지 왔나? | `05_safe_lerobot_collection.md` |
| 랩미팅 발표 덱은 어떻게 구성/갱신하나? | `13_lab_meeting_ppt_notes.md` |
