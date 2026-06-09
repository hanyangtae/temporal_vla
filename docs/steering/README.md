# Latent Steering (메인 연구 라인)

이 디렉터리는 프로젝트의 **메인 method — latent steering** 문서 기준 위치다. 목표는
GR00T N1.6 RoboCasa rollout에서 성공/실패 latent 신호를 찾고, failure type에 맞는
pathway를 steering하여 Success Rate를 올리는 것이다. 실패하더라도 failure-type 자동 분류와
개입 후보를 남기는 것이 최소 산출물이다.

번호 prefix(`01_`, `02_`, ...)는 읽기 순서다. TTA/VITA progress predictor 방향은 현재 메인이
아니며 [`../ttt/`](../ttt/README.md)에 보존 문서로 둔다.

## 현재 결론

- 초기 SAFE detector와 DiT-only COAST steering은 평균 SR 개선을 만들지 못했다.
- 실패 원인 해석은 NOTALL 가설과 맞는다: GR00T pathway는 goal semantics(VL)와 motor
  program(DiT)이 나뉘며, DiT-only feature는 goal-type failure를 충분히 잡지 못한다.
- Phase 3 결과에서는 VL pathway가 t≤8에서 더 이른 실패 신호를 내고, DiT는 t≥12에서 강해진다.
- 다음 실험 축은 VL pathway conceptor fit, type-matched steering, 그리고 B/C quadrant 비디오
  정성 검증이다.

## 사용 절차

1. 표현 분석의 근거를 볼 때는 `01` → `02`를 먼저 읽는다.
2. COAST 재현과 DiT-only 실패를 이해할 때는 `03` → `04` → `06`을 읽는다.
3. 다음 steering 방법론 후보를 정할 때는 `07`을 보고, 최신 pathway 결과는 `08` → `09`에서 본다.
4. 새 세션을 이어받을 때는 `10_session_handoff.md`를 먼저 보고, 필요한 세부 근거로 역추적한다.
5. LeRobot/멀티벤치 SAFE 수집 확장은 `05`를 별도 track으로 본다.
6. 코드 흐름과 수식을 한 화면에서 볼 때는
   [GR00T Latent Steering Explorer](../groot/00_groot_steering_explorer.html)의 탭/검색/code map을 연다.

## Reading Order

1. [01 seen18 Latent Analysis](01_seen18_latent_analysis.md) — GR00T-N1.6 RoboCasa seen18
   잠재공간에서 succ/fail 이 구분되는지·어떤 조건에서 드러나는지. **길이 confound 통제**가
   모든 해석의 전제. steering 의 표현 측 근거.
2. [02 seen18 Handoff](02_seen18_handoff.md) — cross-task 실패 심화 + 후속 두 세션
   (COAST steering / SAFE detector) 핸드오프(근거·주의·재사용 인프라·실행 스펙).
3. [03 COAST Report](03_coast_report.md) — COAST conceptor algebra 구현 + GR00T N1.6
   rollout feature 적용 진행 보고. 단일 layer 데이터의 한계 정리.
4. [04 COAST Reproduction Map](04_coast_reproduction_map.md) — 우리 코드 ↔ COAST 논문
   식/섹션 매핑 (충실 재현 / 의도적 변경 / 미구현).
5. [05 SAFE lerobot Collection](05_safe_lerobot_collection.md) — SAFE latent 수집을
   lerobot 정책(pi0.5/pi0+FAST/X-VLA/GR00T N1.5) × 멀티 벤치로 확장 (plan + status + handoff).
6. [06 COAST GR00T N1.6 Summary](06_coast_groot_n16_summary.md) — DiT 32-layer COAST 재현,
   layer selection, SR eval 결과. 평균 ΔSR≤0으로 DiT-only steering 실패를 정리.
7. [07 Steering Methods Survey](07_steering_methods_survey.md) — COAST 이후 적용 후보
   (CAA, SAE-guided, NOTALL pathway, learned steering)와 권장 순서.
8. [08 Phase 3 DiT32 Separation](08_phase3_dit32_separation.md) — DiT 32-layer pre-failure
   분리력 분석 결과.
9. [09 Phase 3 VL vs DiT Comparison](09_phase3_vl_dit_comparison.md) — VL(goal)과
   DiT(motor) pathway 비교, Phase 4 steering target 선택 근거.
10. [10 Session Handoff](10_session_handoff.md) — 최신 연구 목적, 완료 실험, 다음 세션
    우선순위, 주요 파일 위치.

## 결과 위치

| 질문 | 문서 |
|---|---|
| succ/fail latent가 실제로 분리되는가? | `01_seen18_latent_analysis.md` |
| DiT-only COAST steering은 왜 실패했나? | `06_coast_groot_n16_summary.md` |
| 최신 steering target은 무엇인가? | `09_phase3_vl_dit_comparison.md` |
| 다음 세션은 무엇부터 해야 하나? | `10_session_handoff.md` |
| LeRobot/멀티벤치 SAFE 수집은 어디까지 왔나? | `05_safe_lerobot_collection.md` |
