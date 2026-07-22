# Gate1 원장 — exp4 계획 반론 (2026-07-21~22)

- 게이트: Gate 1 (계획 토론). thread_id: `019f83f4-18cd-7141-9fcb-3298f80cc3fe`. 왕복 1회로 종료.
- 대상: docs/steering/24_exp4_shared_plan.md, 24a_exp4-1_oracle_rescue_plan.md, 24b_exp4-2_perturb_conceptor_plan.md (초안 v1).
- 결과: Codex 지적 18건 (P1 10 / P2 7 / P3 1). 수용 15건·부분수용 1건·기각 2건. 사용자 최종 결정 4건 별도.

## 라운드 1 요지와 처리

| # | 지적 (요지) | 처리 |
|---|---|---|
| 1 | t0_record=floor(t0/5)는 최대 4 env-step look-ahead | **수용** — ceil로 교체, floor/ceil 민감도 사전등록 (24a §2.2) |
| 2 | 제거형(I−βr̂r̂ᵀ)은 좌표를 0으로 밀 뿐 성공 평균으로 안 감; setpoint/additive를 primary로 | **수용(사용자 결정)** — Ms = setpoint형 h−β[(h·r̂)−s]r̂ primary. 문헌 검증: ACE(2411.09003)·LEACE·WA-LQR 선행 확인, novelty 아님 |
| 3 | 무작위 방향 위약은 dose-matched 아님 | **수용** — label-permutation 방향 + held-out ‖Δh‖ 분포 일치 (24a §4.2) |
| 4 | 수동 t0 주석의 hindsight·분모 선별 편향 | **부분수용** — ITT 분모 고정·주석 동결·claim 등급 "hindsight 포함 oracle 상한" 명기. blind/규칙 기반 주석으로의 교체는 기각(사용자 설계: 영상 보고 직접 지정) |
| 5 | 시간분리 +2는 washout 근거 아님, 주입 흔적이 창 이후에 남을 수 있음 | **수용** — perturbed-succ에서 injected vs sham lag별 AUROC로 cutoff 산출, 잔존 시 "aftermath" 등급 강등 (24b §4.3-2) |
| 6 | TYPE(goal/motor) 라벨이 주입 경로·donor와 완전 혼입 | **수용** — "intervention-source" 라벨로 강등, LODO + 행동 phenotype 대조 요구 (24b §4.3-4) |
| 7 | clean succ 혼합 fit은 물리 서명 학습 | **수용** — primary fit = perturbed-fail vs perturbed-succ (동일 변형·dose), clean succ은 secondary/transfer 전용 (24b §3·4.1) |
| 8 | R-가중 이득 게이트는 자기참조·무검정 | **수용** — held-out + label-permutation null로 임계값, sanity gate 한정 (공유 §3) |
| 9 | 선택·검정 데이터 계약 부재 (winner's curse) | **수용** — calibration/fit/locked-test 3분할 + manifest hash 교집합 0 (24b §3) |
| 10 | 변형당 fail 10ep 과소, per-record CI는 pseudoreplication | **수용** — P1 진입 변형 fail ≥20ep, cluster bootstrap (24b §3·4.3) |
| 11 | A는 감쇠 단독 대조가 아님 | **수용** — A는 legacy 기준선으로만, 감쇠 해석 삭제 (공유 §2) |
| 12 | phase-bin만으로 dwell/길이 confound 미제거 | **수용** — 공통 post-trigger horizon + record 수 균등 + episode 가중 (24b §4.3 공통 통제) |
| 13 | 50–90% 실패율 게이트가 극단 섭동 선호 | **수용** — 40–70%로 하향, 독립 calibration seed CI (24b §1.3) |
| 14 | arm×GPU/serve/순서 confound | **수용(경량)** — GPU id·slot 기록 + arm 순환 배정 (24a §3). 전면 Latin-square는 비채택 |
| 15 | Track P "자연스러운 실패" 전제 미검증 | **수용** — metadata-only baseline 병기 (24b §4.3-1) |
| 16 | 더 싼 exp4-1: 이벤트 anchor + action-blend | **기각(대체로서)** — 수동 주석은 사용자 명시 설계, action-space 개입은 다른 질문. 단 action-replay 12/77을 결과 참조선으로 병기 (24a §6) |
| 17 | P1 전 bridge 게이트 (유도축↔자연축 정렬) | **수용** — P0 후 분석-only 게이트 신설, 미정렬 시 P1 중단 (24b §3) |
| 18 | A0 77판 전량 재실행은 중복 | **수용** — sentinel 12판 우선, 불일치 시만 전량 (24a §8-5) |

## 사용자 최종 결정 (2026-07-22)

1. 연산자는 **setpoint형(Ms) + 기존 conceptor(A)**만. 제거형 arm 제외 (s 값 보고로 갈음).
2. 축은 **within-instruction + cross-scene**만, cross-instruction 유예 (steering fit/eval 축 기준; exp4-2 B1 주입 메커니즘은 유지).
3. **WA-LQR**: 타당성 게이트(24a §5 F1) 통과 시 W arm으로 추가 시도 (참고 24c).
4. **Task 4종: CloseFridge, OpenDrawer, PPCC-bread, PPCC-beer.**

## 후속 결정 (2026-07-22, 같은 날)

- **Task 교체: CloseFridge → OpenStandMixerHead** — CloseFridge는 실행5/예측16에서 SR 0/14 (chunk-길이 함정, docs/steering/25), mixer는 실행5에서도 생존 (docs/steering/26).
- **Scene 실현가능성 필터 채택** (`docs/steering/NOTICE_scene_feasibility_for_exp4.txt`): 기하 불가 seed(실측 mixer 100010)를 정책 무관 관절 스윕으로 fit·eval 양쪽 동일 제외, manifest에 기록. rescue 분모 오염 방지.
- 동시 실행 규칙 명문화 (exp4-1 본 트리 / exp4-2 worktree) — ca7dc40.

raw 응답은 커밋하지 않음 (규약). 스크래치: `$CLAUDE_JOB_DIR/tmp/codex_gate1/`.
