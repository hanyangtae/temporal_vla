# PPCC 추가 수집 및 eval 후보

plan `4fa6496cd684`, kanu, GR00T N1.5 RoboCasa365 ckpt120000. 신규5scene × 5j × n10 = 250판. 2026-09-09 원격 archive metadata와 로컬 완료 payload 합집합250셀, noise inference seed1300000..1300009 검증. 08:02 UTC 확인: shipped_cells 250개 unique, 로컬 잔여 rollout.pkl 0개로 승준 전송도 완료.

| Scene | Layout/style | j0 S/F | j1 S/F | j2 S/F | j3 S/F | j4 S/F | 선택 target | other40 S/F |
|---|---|---|---|---|---|---|---|---|
| bread s3 | 10/10 | 10/0 | 10/0 | 10/0 | 10/0 | 10/0 | 불가: 실패 없음 | — |
| bread s4 | 2/2 | 7/3 | 6/4 | 5/5 | 8/2 | 0/10 | j3 | 18/22 |
| jug s3 | 8/8 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 불가: 성공 없음 | — |
| marshmallow s3 | 1/1 | 10/0 | 5/5 | 10/0 | 0/10 | 6/4 | j4 | 25/15 |
| marshmallow s4 | 6/6 | 10/0 | 0/10 | 5/5 | 1/9 | 10/0 | j2 | 21/19 |

선정은 target S>=1/F>=1 및 나머지40판 양 클래스 조건, 조건 충족 j 중 성공판 많은 후보 우선. Detector target 전체10판 제외; 연산자는 target 성공만 제외. 이 자료는 수집 라벨 기반 후보 진단이며 detector 학습 완료나 intervention 효과를 뜻하지 않는다. 이후 eval은 수집 머신 kanu에서 재현.

| Confound audit | 판정 | 근거 |
|---|---|---|
| Length | N/A | activation 분리/학습 성능 주장 없음 |
| Task identity | pass | instruction·scene·j별 원래 성공 라벨 집계 |
| Instruction balance | pass | j당10판, scene당50판 분리 |
| In-sample rescue | N/A | 구제 성능 측정 전, detector split 계약만 명시 |
| Rollout pooling | N/A | activation pooling 안 함 |
| Phase/dwell | N/A | phase별 성능 주장 없음 |
| Observation vs causation | pass | candidate diagnostic evidence로 제한 |
| Scene-local vs general | pass | 해당5scene 범위만 보고 |

근거: staging shipped_cells.txt, 승준 archive `groot/n15/grid/4fa6496cd684/kanu/**/base/meta.json`, 남은 로컬 `base/{meta.json,rollout.pkl}`, local ep_meta. 로그 DONE만으로 완료 판정하지 않음.
