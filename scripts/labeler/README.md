# RoboCasa 사건 라벨러

승준 `~/workspace/labeler`에서 실행하는 웹 라벨러. 운영 포트는 8767이다.

## 포함 기능

- 여러 plan의 원본 영상을 연결하고 plan/machine/scene/jitter/noise/sig로 출처를 구분한다.
- 성공적 재시도(8번), 기존 점 마크(M), 구간 시작(I)·종료(O), 여러 구간·재발·경계 수정.
- 점·구간 저장 및 사건별 CSV/TSV export. [구간 데이터 계약](README_intervals.md) 참고.
- 전체 라벨 삭제 버튼: 현재 필터와 관계없이 전부 삭제하며 확인창 한 번을 띄운다. 단축키는 없다. 원본 영상과 outcome은 삭제하지 않는다.
- 전체삭제 직전에 기존 TSV를 `deleted_label_backups/labels_<time_ns>.tsv`로 보관한다. 취소하거나 요청 실패 시 UI 라벨을 유지한다.
- 없는 kanu Coffee 영상 항목 제거, 올바른 수집 머신의 추가 marshmallow/bread/dishwasher scene 등록.

## 운영 파일과 배포

런타임 디렉터리에는 `index.html`, `labeler_server_multiplan.py`, `label_events.py`, `start.sh`가 필요하다. HTML에는 현재 등록된 영상 목록이 포함되므로 배포 때 빈 템플릿으로 덮어쓰지 않는다. 라벨 정본은 `v6_failure_onset_labels.tsv`이며 배포로 재작성하지 않는다.

추가 UI 변경은 `add_*_ui.py`로 현재 HTML에서 후보 HTML을 생성한다. 각 추가 스크립트는 이미 적용된 변경의 중복 실행을 거부한다. 기존 최신 HTML을 사용할 때는 다시 적용할 필요가 없다.

배포 시 기존 서버 PID·실행 경로·포트를 확인하고 백업한 뒤, 그 라벨러만 재시작한다. 새 서버는 시작 시 HTML과 TSV를 읽는다. 원본 영상·목록·기존 라벨 보존을 확인하고 HTTP `/`, `/api/labels`, `/api/events.csv`, `/api/events.tsv`를 검증한다.

## 검증

```bash
python3 scripts/labeler/tests/check_intervals.py <현재 또는 후보 HTML> [Chromium 경로]
python3 scripts/labeler/tests/check_delete_all.py <전체삭제 버튼 포함 HTML> <Chromium 경로>
bash -n scripts/labeler/start_multiplan.sh
```

테스트는 임시 HTTP 서버와 임시 TSV만 사용한다. 운영 라벨에 테스트 삭제를 실행하지 않는다.
