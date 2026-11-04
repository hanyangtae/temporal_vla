# deadline_stop 영수증 (2026-11-04)

## 3차 (11-04): `deadline_stop.py` — 공유계약 만료(09:29:06 UTC = 18:29:06 KST) 전 안전 종료 단발 CLI
- 기존 6파일 수정 없이 새 파일만 추가. 새 guardian/watchdog은 기동하지 않음. GuardianV4(lock만 사용) + v1 `act()`를 그대로 재사용.
- pin(전부 정확히 일치해야 함, 하나라도 어긋나면 rc 6이고 신호 0):
  - consumer: pid+start, registration 파일(pid/start/receipt), ledger의 receipt session/machine/[gpu]와 `consumer_pid`
  - follower: 탐지 결과(v3 outer 포함)가 선택적 pin과 같아야 함
  - old guardian: pid+start+cmdline `foreign_gpu_guardian`이 맞을 때만 SIGSTOP 먼저(동시 act 방지) → 정지 후 TERM+CONT. 다른 식별자면 신호하지 않음
- **교체된 consumer는 재타겟하지 않음**(rc 7 또는 6, 신호 없음). 새 receipt를 자동으로 검증해 따라가는 기능은 **지원하지 않음** — 새 pin으로 재실행하는 것은 main/사용자 판단.
- `--dry-run`: pin 확인 + 실제 `act()`를 observe-only로 실행(ownership 검증, follower/wrapper 탐지, ACTION 기록). 신호 없음.
- 실제 실행: v1 순서(follower → consumer → arm 동결, arm SIGINT, serve pg / docker exec / 검증된 root env-container collector TERM→KILL, consumer·follower TERM).
  release는 우리 프로세스 부재 + harness wrapper 종료 확인 뒤 공식 `groot_harness.py release`로만 함(wrapper가 먼저 반납했으면 `released_elsewhere`). 원장을 손으로 고치지 않음.
- 계약 승인 프로세스와 다른 foreign은 owned가 아니므로 신호하지 않음. progress/claims/부분 결과 파일은 손대지 않음(테스트 tearDown에서 byte 비교).
- `<queue_dir>/guardians/gpu<G>.lock`으로 중복 실행 차단(새 파일 1개 생성).
- 증거: `--out` JSON(새 경로만) — pins, follower, owned, 결과, post(consumer/follower/old guardian 생존 여부, ledger의 receipt, GPU pids, 우리 pid, 승인 pid 생존). 이벤트는 `--state-dir`(새 경로)에 기록.
- rc: 0 released/released_elsewhere/dry_run_ok · 5 kept · 6 refused/pin mismatch/lock · 7 consumer 없음 · 2 경로 충돌.
- 시간 예산: v1 grace 90 + kill_wait 30×2 + release 600 = 최악 약 12.5분. 18:00 KST에 실행하면 만료까지 29분 여유.
- 테스트: `tests/test_deadline_stop.py` 10개(전체 142 passed).
