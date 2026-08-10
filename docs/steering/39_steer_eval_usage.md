# steering eval 용례 — exp5-3 러너에서 증류 (차기 실험 참고)

exp5-3 러너 5개(archive 됨, 2026-08-10 S6 판정)의 **재사용 가치 있는 패턴만** 남긴 문서.
차기 steering eval 은 좌표 규약(collect_grid.sh 패턴 + `--grid-root/--plan-json` +
armsig 자동)으로 새로 짜되, 아래 실험 설계 패턴은 그대로 가져간다.
원본 복원: `git checkout <S6 적용 커밋 직전> -- scripts/safe/groot_n15/robocasa/steer/exp5_3/`

## 1. arm 구성 패턴 (eval_ws_steer.sh)

- **A0 앵커 먼저**: 무개입 base 를 같은 머신에서 같은 그리드로 — cross-machine 비교는
  각주용으로만 (결정론은 머신-로컬). 새 규약에선 base = 수집 rollout 그 자체라 생략 가능.
- **LOO fold 마다 serve 재기동**: fold k 의 NPZ(loo_seed{k})로 serve 를 다시 띄우고
  scene 전체 × seed k 를 돈다 — fit-seed ↔ eval-seed 분리를 프로세스 수준에서 강제.
- **latch**: `--steer-from-record 0` + phase-mode global → POST /steering_phase "steer".

## 2. β sweep 패턴 (beta_sweep.sh)

- 목적 분해가 먼저: 해악이 (a)용량 과다 (b)setpoint 오차 (c)방향 유해 중 무엇인지
  가르도록 β={1.0, 0.5, 0.2} 를 **같은 40셀 라틴 그리드**에 paired 로.
- 라틴 배치: scene i → seed {i%8, (i+5)%8} — scene×seed 전수(160판) 대신 40판으로
  두 축을 모두 커버.
- A0 재실행에 부가 목적을 실어도 됨 (예: action_kinematics jerk base 확보 + 머신
  결정론 재검증).

## 3. scene-matched 수집 패턴 (collect_sm_mixer / mixer_sm_collect.sh)

- 한 scene 안에 succ/fail 혼재를 만드는 그리드: scenario_seed 20종 × inference_seed
  8종 (= 새 규약의 instruction × s × n 그리드가 이걸 일반화한 것).
- scene 은 feasibility JSON 의 feasible 앞 N 개 — 기하 불가 seed 는 fit·eval **양쪽
  동일 제외** (scene-feasibility-filter 규칙).
- serve 는 host conda 3개 (16GB 카드 4개는 OOM — 2026-07-27 실측).
- 판마다 annotate_phase_video 로 상단 배너(instruction+phase+step) 주석 영상 생성.

## 4. 배송 패턴 (mixer_sm_pull.sh)

- home→승준 직결 불가 → **승준 쪽에서 5분 주기 rsync pull** (3분 경과분 = 쓰기 완료).
- 전 종류 pull (아카이브 완전성 — 종류 골라 include 금지), 원본 삭제는 하지 않음.
- 새 규약에선 ship_to_archive.sh(롤링 push + shipped_cells.txt)가 이 역할.

## 5. serve 구성 상수 (당시 실측)

- home(4070Ti 16GB): serve 3개, 포트 8600–8602, host conda. 4개는 OOM.
- srv50(A100): GPU 당 serve 6개.
- fit/NPZ 배치: per-scene setpoint registry 를 `deploy/permanent/loo_seed{k}` 로 —
  새 규약에선 opsig·config.json 이 이 역할.
