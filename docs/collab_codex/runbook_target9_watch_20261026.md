# Target9 n40 원격 수집 자동 대기

사용자 승인: srv48·srv50에서 30분마다 GPU 점유 확인 후 빈 GPU에서 원래 머신의 수집 재개.

- 실행기: `scripts/collect/watch_target9_n40.py --machine srv48` (srv50 동일).
- srv48: 기존 n10~19 누락 108판 → 신규 n20~39 300판. srv50: 신규 400판.
- 원장: common checkout `outputs/collect/target9_n40_watch_20261026/<machine>/status.json`.
- `--dry-run --once`: archive 실사, 계약 검증, GPU 확인. 실제 발사 없음.
- 단일 flock, 프로세스 없는 GPU + 100MiB 미만만 사용. 공통 하네스가 한 서버 1GPU×6serve 및 충돌을 최종 검사.
- 기존 전송기 코드 변경을 피하려고 별도 `collect-n40-watch` worktree로 git 배포한다.
- 48 backfill은 `grid_n20_srv48_resume_20261026`, 신규는 `grid_n40_srv48_20261026` / `grid_n40_srv50_20261026`. plan_id·seed·좌표는 frozen 계약 유지.
- 전송: 30초 간격, 셀 수정 후 60초 정숙, SHA256 일치 후 staging 삭제. staging 8GB 이상이면 수집 대기.
- 시작 전 로컬·승준 각20GiB 여유 필요. 실제 완료 판정은 승준 4파일이 모두 존재하고 비어 있지 않은 경우.
- 감시·전송·launcher 로그 회전. 자기 실행 worker/serve 로그는 10초마다 확인, 20MiB 초과 시 마지막10MiB만 같은 파일에 유지.
- SIGTERM은 watcher만 중지한다. 이미 실행된 수집/전송은 유지하며 PID를 기록한다. 완료 시 자신이 발사한 전송기만 정리.
- 계약 오류, 컨테이너 불일치, 3회 연속 무진척은 자동발사 중지. 상태파일 확인 후 원인 해결 필요.
- 서버 재부팅 후 자동기동 서비스는 등록하지 않음. 살아 있는 프로세스가 30분 주기로 동작.
