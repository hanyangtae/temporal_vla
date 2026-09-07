# v6 LR 동일 데이터 예산 실험

목적: 같은 N개의 고유 episode로 학습할 때 단독 방향 N판과 LR 각각 N/2판의
detector·setM을 비교한다. 추가 데이터 효과가 아니라 반대 방향 데이터로 대체한
효과를 측정한다. 기존 raw-record 가중 모델은 기준선으로 재사용하지 않는다.

## 고정 설계

- 설정: `configs/experiments/lr_equal_budget_v6_20260907/{protocol,mapping,requests}.json`.
- 작업은 drawer/oven/dish, 주방은 layout/style로 분리한다. LR만 합친다.
- 각 target-side/jitter에 단독·혼합 한 쌍. 서로 다른 target에는 별도 혼합 모델을
  학습한다. 하나의 공통 checkpoint를 모든 LR target에 재사용하는 실험은 아니다.
- 쌍 안에서 고유 N, 성공 S/실패 F, 성패×jitter별 개수, 현재 실패 1판의 ID가 동일.
  혼합은 각 성패 안에서 L:R=1:1. 현재 실패도 N에 포함한다.
- target jitter의 성공은 양쪽 모두 제외. 반대쪽 target jitter 실패도 제외하여
  현재 실패 예산을 동일하게 고정한다. 허용된 현재 실패는 fit과 평가에 겹칠 수 있다.
- 선택 성공 모두를 score-band LOO calibration에 재사용한다. 추가 calibration
  데이터는 없으며, 모델 학습에서 독립된 conformal 보장은 주장하지 않는다.
- cap은 양 모델의 선택 성공 교집합에서 phase별 ceil(mean+std)로 계산한다.
  detector/정규화는 episode당 1/N, episode 내부 timestep 평균. 양쪽 update 수 동일.
- 연산자는 capped records에서 phase를 유지한다. plain은 side/class/jitter의
  균형 평균, lr_jfair는 각 side 내부 mixed jitter 차이를 평균한 후 LR을 균등 평균.
  lr_jfair는 각 side에서 mixed jitter 2개 이상 및 target 실패 phase anchor가 필요하다.
  미등록 phase는 기록하고, 등록 phase가 0인 variant는 평가하지 않는다.
- GT phase, beta 0.8, failure alpha 0.1, reseed offset 900000, 16예측/5실행 유지.
  실험은 GT phase를 사용하는 개입 평가이며 실전 phase 추정 성능을 입증하지 않는다.
- 반복 0/1/2는 **데이터 부분집합 선택 반복**이다. 같은 평가 좌표의 반복을 독립
  환경 표본처럼 합쳐 유의성을 계산하지 않는다.

## 선택 결과와 범위

원본 snapshot은 `outputs/lr_equal_budget_20260907/sources/`에 있다. srv48의
1,800행 인덱스에서 900개 LR row를 사용했다. 원본 replay 키와 canonical 키는
명시 mapping으로 구분한다(oven/washer 좌우 이름 변경).

90 target 후보 중 30개는 원래 실패가 없고, 실패가 있는 60개 중 10개가 엄격한
할당을 충족한다. drawer 8개, dish 2개, oven 0개다. Oven의 전승/전패 및 jitter별
class 지원 부족 때문에 성립하지 않으며 제약을 몰래 완화하지 않는다.

| canonical 대상 | 주방 layout/style | jitter | N | S / F |
|---|---|---|---:|---|
| drawer L | 4/4 | 0, 2 | 32 | 28 / 4 |
| drawer L | 4/4 | 1 | 24 | 20 / 4 |
| drawer R | 4/4 | 4 | 32 | 30 / 2 |
| drawer L | 9/9 | 1, 2, 4 | 32 | 14 / 18 |
| drawer L | 9/9 | 3 | 32 | 10 / 22 |
| dish L | 9/9 | 3 | 24 | 10 / 14 |
| dish L | 9/9 | 4 | 16 | 10 / 6 |

각 행의 각 jitter마다 반복 3회: 30쌍/60모델, manifest 1,728행.
N은 target 사이에서는 달라도 **비교하는 쌍 안에서는 정확히 같다**. F<8은 지원이
얇은 조건으로 표시한다. 전체 LR/task 성능을 대표하는 표본이라고 해석하지 않는다.
예산은 fit/cap/normalization/calibration에 쓰는 고유 episode 수이다. 전체 원장의
성패 metadata를 읽어 가능 조건을 선정한 비용까지 N이라고 주장하지 않는다.

## 실행 단계

코드는 별도 git worktree `exp/lr-equal-budget-20260907`에서 고정 commit으로 배포한다.
진행 중인 srv48/50의 dirty checkout은 checkout/pull하지 않는다. 원격 작업은
`scripts/utils/remote_compute.sh`만 사용한다.

1. manifest를 동결하고 `INPUT_SHA256.json`에 입력 파일 해시를 기록한다.
2. archive-ready: archive 접속, Python, shard 디렉토리, CPU 작업 수를 확인한다.
3. archive-fit: 고정 commit의 별도 worktree 생성 → 작은 입력만 전송 → metadata의
   episode sig로 shard 매칭 → 순차 CPU fit → 30개 pair report 확인 → 모델만 회수.
   archive 파일명만으로 LR을 추정하지 않는다. 실제 loader가 좌표·label·sig 재검증.
4. kanu drawer / srv48 dish 평가를 각각 별도 queue로 실행한다. 두 queue는 FIT_DONE
   전에는 GPU를 점유하지 않는다. 중앙 lease와 원격 로컬 lease를 모두 지킨다.
5. 첫 실패 좌표 ps_base 재현을 통과한 뒤 각 모델의 reseed/plain/lr_jfair를 평가한다.
   각 target의 원래 10개 noise 모두 사용하며 실패 구제와 성공 유지 결과를 분리한다.
6. exact coordinate set, seed, 원래 성패, 중복을 검증한다. 서버가 반환한 exit 0이나
   DONE 문자열만으로 완료 처리하지 않는다. 자기 serve 종료와 GPU 반납을 확인한다.

최대 예약량은 replay gate 10판 + 비교 1,800판이다. phase가 전부 미등록인 연산자는
명시적으로 제외되어 실제 비교 수는 줄 수 있다. `operator_coverage`와 완료 분모를
함께 보고한다. 숫자 해석 전 confound-audit를 적용한다.

## 명령과 상태

새 planner 출력 재생성(기존 동결 출력에 덮어쓰지 말 것):

```bash
python scripts/analysis/grid_phase/lr_equal_budget/planner.py \
  --index-tsv outputs/lr_equal_budget_20260907/sources/index_v6_complete_cells.tsv \
  --mapping-json configs/experiments/lr_equal_budget_v6_20260907/mapping.json \
  --requests-json configs/experiments/lr_equal_budget_v6_20260907/requests.json \
  --out-dir outputs/lr_equal_budget_20260907_replanned
```

대기열 설정은 `outputs/lr_equal_budget_20260907/queue_configs/`에 고정 commit·절대경로로
생성한다. 각각 `queues/{fit,kanu,srv48}/status.json`에서 PENDING/RUNNING/FAILED/COMPLETED를
확인한다. 단순 readiness 실패는 60초 후 재확인하고 실제 stage 실패는 멈춘다.

```bash
python scripts/analysis/grid_phase/lr_equal_budget/run_queue.py \
  --config <queue_configs/fit.json> --state-dir <queues/fit> --poll-seconds 60
```

장기 실행은 반드시 `setsid nohup`으로 분리하고 PPID/프로세스 생존을 확인한다.
GPU 대기는 예약 의사만 기록하며 미리 claim하지 않는다. 평가 직전 빈 GPU 판정은
프로세스 소유자를 확인하며 메모리 여유만 보고 발사하지 않는다. 타인 프로세스나
기존 v6 실행을 종료하지 않는다. 준비된 기본 슬롯은 kanu GPU4와 srv48 GPU2, 각
serve 1개이다. 점유 중이면 대기하며 다른 GPU를 임의로 공유하지 않는다.

현재 최초 점검에서는 archive가 양쪽 경로에서 timeout이었다. 연결 복구 전 상태는
학습 완료가 아니라 대기이다. 재시작 시 RUNNING 잔재/FAILED가 있으면 원격 실제
프로세스와 산출물을 조사하고 새 state-dir로 재등록한다. 부분 fit/eval 덮어쓰기 금지.

검증: stdlib planner/queue/eval 테스트와 Docker robocasa의 합성 shard→fit→실제
online loader 테스트를 사용한다. 실제 activation과 full simulator smoke는 archive
복구 후 실행 단계에 포함되며, 그 전에는 통과했다고 표시하지 않는다.
