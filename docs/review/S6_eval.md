# S6 — eval 실행·큐 스테이지 카드 (2026-08-10)

기계 판독분: [`S6_files.tsv`](S6_files.tsv) · 판정 UI: `python3 scripts/review/ledger_ui.py`

## 규모 — 103파일 12.4k줄. 그러나 파일 정독 스테이지가 아니다

거의 전부 라운드 전용 `.sh` 러너다. **무리 단위 처분**이 맞고, 개별로 읽을 가치가
있는 건 벤치 eval 2개(★현행)와 queue 인프라 정도다.

| 무리 | 파일 | 상태 | 처분 힌트 |
|---|---|---|---|
| exp2 러너·큐 | 24 | ★종결(null) | 최대 무리. S1·S4 archive 로 이미 깨진 것 다수 |
| exp3 러너·큐 | 10 | ★종결(null) | 〃 |
| exp4-1 러너 | 8 | 완료 | 원격 fit 러너 포함 — 재현 기록 성격 |
| exp5-3 러너 | 5 | 현행연계 | per-scene setpoint 확립 라운드 — 다음 steering eval 의 직계 조상 |
| exp5-4·patchceil | 5 | 종결 | probe 수집·donor 이식 실행 |
| perturb(구 induced) 러너 | 4 | keep 축 | S4 에서 perturb 상설 결정 — 러너도 따라감 |
| queue 시스템 | 10 | 인프라 | srv50 병렬 표준(queue_lib·lane_runner) — 차기 eval 도 쓸 후보 |
| steer 루트 잡다 | 17 | 구라운드 | master/heldout/30x30 등 1회용 |
| n16 steer 러너 | 6 | 종결 | fit_conceptor_steering 참조 2개는 이미 깨짐 |
| 벤치 eval (`scripts/eval`) | 4 | ★현행/진단 | robocasa_eval·zmq_eval = per_episode.tsv 표준 구현 |
| 타모델 eval | 6 | 범위제외 | S5 와 같은 처분 |
| TTT(phase1) | 2 | 이동? | 검토 축 밖 — S9 잡파일 때 |

## 판정 축 — 질문 2개

1. **종결 라운드 러너 ~60개를 일괄 archive 하나?** 근거: (a) 전부 구 stem 레이아웃
   전제라 좌표 규약에서 재사용 불가 — 차기 eval 러너는 collect_grid.sh 처럼 새로 쓴다.
   (b) S1·S4·S5 archive 가 깨뜨린 참조의 대부분이 이 무리 안 — 지우면 잔존 참조
   문제가 함께 소멸. (c) 재현 기록은 git 이력 + RESULTS.md 로 충분.
   반론: exp5-3 러너는 현행 fit(keep)의 사용례 문서 역할 — 대표 보존 가치.
2. **queue 시스템(10파일)의 거취.** srv50 병렬 eval 의 실전 검증된 인프라.
   차기 steering eval 을 좌표 규약으로 새로 짤 때 (a) queue_lib 재사용 or
   (b) collect_grid.sh 패턴으로 재작성. (a)면 keep+재배치(RENAME_PLAN),
   (b)면 archive. — 다음 eval 설계 전에 정할 필요는 없고 미정 가능.

## 예약된 정리 (판정 무관)

- S1 archive 잔존 참조: run_one_steer_cell.sh·run_instruction_steering_eval.py·
  master_6p_pipeline.sh·exp3_c0_scan.sh·exp2 p1/p2 .sh — 전부 이 스테이지 무리 안.
- S5 archive 잔존 참조: eval_steer_pathways.sh·eval_steer_vl.sh (n16), 사본 serve
  참조 .sh 6개 (exp4-2·patchceil 무리 안).
