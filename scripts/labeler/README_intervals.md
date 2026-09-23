# 사건 점·구간 라벨 (2026-09-23)

Linear VLA-12 누리 의견 1(구간)·3(저장/export) 구현. 2(task success onset)는 보류. 기존 성공적 재시도 유형과 최종 succ는 변경하지 않는다.

## 사용자 동작

- M: 기존 순간 사건의 점 마크. 유형·메모는 기존대로 입력.
- I: 구간 시작. 종료 시점으로 이동한 후 O: 종료. 선택한 점 마크에 O를 누르면 해당 점을 시작으로 하는 구간으로 전환.
- 목록에서 구간 선택 후 시작/종료 프레임 입력으로 수정. 시작/점 이동 버튼은 현재 재생 시각으로 시작을 수정.
- 여러 구간, 재발, 겹치는 구간 허용. 종료 미지정/역전/영상 범위 밖 경계는 저장 불가.
- 구간은 [start_frame, end_frame): 종료는 상태가 끝난 첫 프레임. 관측 종료까지 지속돼 실제 끝을 모르면 임의로 종료를 추정하지 않고 메모에 남긴다. 우측 검열 구간 전용 기능은 이번 범위에 없음.
- 기존 S/저장 후 다음 유지. 저장 실패 시 변경 사항을 유지하고 다음 영상으로 이동하지 않는다.

## 데이터 계약

정본 TSV 경로 및 기존 열 유지. marks JSON 안에 구간 필드를 추가한다. 기존 point 라벨을 일괄 재작성하지 않으며 kind 누락은 point로 읽는다.

```json
{"kind":"interval","t":2.0,"frame":40,"start_t":2.0,"start_frame":40,"end_t":4.0,"end_frame":80,"type":"collision_stuck","note":""}
```

- t/frame은 구간 시작의 호환 별칭. 기존 최초사건 소비 코드는 시작 시점을 읽을 수 있다.
- t_fail/frame/type 요약은 successful_retry를 제외한 첫 사건(구간이면 시작). 성공적 재시도는 실패 시각에 포함하지 않는다.
- `succ`는 원본 task outcome이며 사건 유형과 독립.
- `/api/export.csv`: 기존 판 단위 CSV. marks JSON에 전체 시작·종료 포함.
- `/api/events.csv`, `/api/events.tsv`: 사건당 한 행. kind, start_t/start_frame, end_t/end_frame, 식별자, 유형·메모를 명시적 열로 export. 점의 start는 t/frame, end는 빈 값. 무마크 판은 kind=none 행으로 보존.
- 새 UI는 schema_version=2를 요청에 포함. 구간이 이미 저장된 판을 구 UI가 저장하려 하면 HTTP409로 새로고침 요청.

## 배포·검증

운영은 승준 ~/workspace/labeler, 8767 포트. index HTML은 실행 시작 시 읽으므로 교체 후 해당 라벨러만 재시작해야 한다.

`add_interval_ui.py --index <현재 HTML> --out <후보 HTML>`는 목록/필터/기존 8유형을 보존하고 interval_labels.js를 init 앞에 삽입한다. 기존 이벤트 바인딩이 새 함수 선언을 사용하도록 동일 함수명의 구현을 대체한다. 새 서버에는 label_events.py가 함께 필요하다.

`tests/check_intervals.py <후보 HTML> [기존 Chromium 경로]`는 임시 서버/임시 TSV에서 legacy 읽기, 구간 저장·재조회, 두 export, 잘못된 경계, 구 클라이언트 덮어쓰기 방지, 성공적 재시도 제외를 검증한다. Chromium 사용 시 실제 DOM/키 이벤트·경계 수정·실패 저장 보존도 검증한다. 운영 TSV에 테스트 라벨을 쓰지 않는다.
