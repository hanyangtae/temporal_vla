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
생성한다. 각각 `queues/{fit,kanu_gpu6_parallel,srv48}/status.json`에서 PENDING/RUNNING/FAILED/COMPLETED를
확인한다. 단순 readiness 실패는 60초 후 재확인하고 실제 stage 실패는 멈춘다.

```bash
python scripts/analysis/grid_phase/lr_equal_budget/run_queue.py \
  --config <queue_configs/fit.json> --state-dir <queues/fit> --poll-seconds 60
```

장기 실행은 반드시 `setsid nohup`으로 분리하고 PPID/프로세스 생존을 확인한다.
GPU 대기는 예약 의사만 기록하며 미리 claim하지 않는다. 평가 직전 빈 GPU 판정은
프로세스 소유자를 확인하며 메모리 여유만 보고 발사하지 않는다. 타인 프로세스나
기존 v6 실행을 종료하지 않는다. kanu는 GPU6에서 포트 8866/8867 두 lane을 사용한다. srv48 GPU2 예약은
현재 serve 1개이다. 점유 중이면 대기하며 다른 GPU를 임의로 공유하지 않는다.

재시작 시 RUNNING 잔재/FAILED가 있으면 원격 실제 프로세스와 산출물을 조사하고
새 state-dir로 재등록한다. 부분 fit/eval 덮어쓰기 금지.

검증: stdlib planner/queue/eval 및 Docker robocasa의 합성 shard→fit→실제 online
loader 테스트 32개 통과. 실제 activation fit도 아래 시점에 완료했다. simulator
재현 gate와 정책 효과 검증은 별도이며, 학습 완료를 평가 통과로 표시하지 않는다.

## 실행 기록 (2026-09-07 UTC)

- 코드 revision: `5e5fdf8c9443f9796cc09450580dd2415d8640e5`.
- 연결 복구 후 archive fit이 06:34 시작, 06:42 완료. 30쌍/60모델의 완료 report와
  N 일치를 확인했고, 모델 산출물 약 452 MB를 회수했다.
- plain은 60모델 모두 phase 등록. lr_jfair는 혼합 30모델 모두 각 side의 mixed
  jitter 지원 부족으로 미등록이다. 단독 모델 15개만 등록되어, 현재 산출물로는
  단독 대 혼합 jfair 효과 비교가 성립하지 않는다. plain 비교는 진행한다.
- kanu GPU4에 타인 작업이 시작되어, 실행 전 대기 중이던 우리 queue만 중단했다.
  빈 GPU6을 재검증하고 `queue_configs/kanu_gpu6.json`, `queues/kanu_gpu6/`으로
  새 예약을 만들었다. 06:47 readiness 통과 후 drawer 재현 gate 실행을 시작했다.
  기존 kanu queue 기록은 보존한다.
- srv48 dish queue는 GPU2가 점유 중이므로 대기한다. 타인 작업과 GPU를 공유하지 않는다.
- 이 기록 시점에는 구제율·성공 유지율 결과가 확정되지 않았다.

## kanu GPU6 두 모델 병렬 실행

사용자 요청에 따라 kanu GPU당 serve 2개 규칙을 적용한다. 기존 한 조건짜리
runner에 `SERVES_PER_GPU=2`만 주면 두 번째 worker에 할 일이 없으므로,
`parallel_eval.py`가 서로 다른 조건을 두 개의 runner에 하나씩 배정한다.

- GPU lease 소유자는 하나이며 포트는 8866/8867로 고정한다.
- 각 두 조건 묶음 시작 전 GPU가 비었는지 확인한다. 첫 서버 health 확인 후
  두 번째 서버를 로드해 동시 모델 로딩 피크를 피한다.
- 두 번째 발사 전 GPU PID의 UID·정확한 port·detector 경로·failure-task 또는
  collector endpoint·output 경로가 첫 작업과 일치하는지 검사한다. 검사를 통과한
  child만 GPU busy 검사를 우회하며, 다른 세션과 GPU를 공유하지 않는다.
- 해당 target의 실패 재현 gate가 통과해야 비교 작업을 배정한다. 출력 경로와
  exact-coordinate 검증은 순차 실행과 동일하다. 완료 조건은 재실행하지 않는다.
- 두 조건 모두 끝나고 서버가 정리된 뒤 다음 묶음을 시작한다. 한 조건이 실패하면
  이미 실행 중인 다른 조건은 마무리하고 이후 묶음은 중단한다.
- 전환 때 순차 orchestrator만 SIGSTOP하여 새 조건 시작을 막고, 현재 runner는
  그대로 끝까지 실행한다. runner 종료 뒤 기존 owner를 종료하고 lease를 반납한
  후 새 queue가 시작된다. `drain_sequential.log`에 전환 기록을 남긴다.

```bash
python scripts/analysis/grid_phase/lr_equal_budget/parallel_eval.py \
  --root outputs/lr_equal_budget_20260907 --repo . --gpu 6 --ports 8866 8867
```

새 예약은 `queue_configs/kanu_gpu6_parallel.json`과
`queues/kanu_gpu6_parallel/status.json`을 사용한다. 원래 queue의 중단 상태는
계획된 전환 기록이며, 새 queue의 완료 여부를 대신하지 않는다.
검증: Docker robocasa에서 eval 관련 8개 테스트 통과. 실제 자식 프로세스 두 개의
실행 시간 겹침, 출력 분리, 실패 시 두 번째 발사 차단, target gate 및 소유 판정을 확인했다.
