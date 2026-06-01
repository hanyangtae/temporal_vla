# Latent Steering (메인 연구 라인)

이 디렉터리는 프로젝트의 **메인 method — latent steering** (succ/fail latent 구분 →
추론 시 활성화를 성공 부분공간으로 steer 하여 SR↑, COAST 계열) 의 표현 분석·재현·진행
기록을 모은다. 번호 prefix(`01_`, `02_`, ...)는 **읽기 순서**다.

> 연구 방향 전반은 루트 `CLAUDE.md` 의 "연구 방향" 절 참조. TTA(VITA progress predictor)는
> 무기한 연기됐고 (`../ttt/`), latent steering 이 현재 메인이다.

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
