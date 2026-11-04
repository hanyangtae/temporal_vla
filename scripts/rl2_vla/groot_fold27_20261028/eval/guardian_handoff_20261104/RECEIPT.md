# guardian handoff (2026-11-04, 2차 watchdog 포함) — srv50 guardian 재시작 공백 재발 방지

**상태: 코드·테스트만 작성. 현재 실행 중인 run(consumer 2553299, serve 2553407·2555644, guardian 2587681)에는 적용하지 않았음.**
GPU·원격·서비스 변경 없음. 기존 파일(foreign_gpu_guardian.py / _v2 / _v3, eval_gpu_queue_v3.sh, wait_resources_shared.py,
groot_harness.py, 공유 계약 파일)은 수정하지 않음. 배포 여부는 main이 테스트 검토 후 결정.

## 원인 (코드 근거)
1. **v3 outer loop를 follower로 못 잡음** — `eval/runtime/foreign_gpu_guardian.py:46`
   `FOLLOWER_MARK = ('eval_gpu_queue_v2.sh', 'eval_gpu_queue.sh')`. `eval_gpu_queue_v3.sh`는 어느 쪽과도 부분일치하지 않음
   (`eval_gpu_queue.sh` ≠ `eval_gpu_queue_v3.sh`). 그래서 act()의 follower SIGSTOP/TERM이 생략되고 outer loop가 살아남음.
   (peer 관측 일치: ACTION 기록에 follower 없음, consumer freeze만 있음.) 테스트 `CauseTest.test_v1_follower_mark_misses_v3...`로 재현.
2. **rc 241 = SIGTERM된 consumer** — `scripts/utils/groot_harness.py` `run()`은 `rc = child.wait()` 후 `sys.exit(rc)`.
   consumer가 SIGTERM으로 죽으면 rc = −15 → 종료코드 241. v3 loop의 `case` 문에서 `*)` 분기 = "race or refusal"로 보고
   `POLL_S` 후 재시도 → may-start / wait_resources_shared / harness run을 다시 통과하면 **새 consumer가 GPU에서 기동**.
3. **guardian은 1회성** — v1 `run()`: `if confirmed: return self.act(confirmed)`. act → release는 최대 `release_wait_s`(600 s)
   동안 old OWNED 식별자와 old harness wrapper 종료를 기다린 뒤 프로세스가 끝남. old OWNED에만 묶여 있으므로 그 사이·그 이후
   기동한 새 consumer는 아무도 보호하지 않음(공백).

## 수정 (새 파일만, 이 디렉토리)
| 파일 | 역할 |
|---|---|
| `guardian_v4.py` | `GuardianV3` 상속. (a) act 후에도 계속 감시(지속 loop). (b) follower mark에 v3/v4 추가. (c) **handshake로만 새 consumer 인수**. (d) `kept`/`refused`면 `HOLD.json` 래치(운영자가 지울 때까지 인수 금지). (e) `<queue_dir>/guardians/gpu<G>.lock` flock으로 GPU당 guardian 1개. (f) `handoff/guardian.json` heartbeat. 공유 계약·foreign 판정·신호 절차·release는 v1~v3 코드 그대로 |
| `handoff_gate.py` | launcher 쪽: `ready`(heartbeat 신선·guardian pid+start 생존·같은 boot·HOLD 아님), `request`(요청 후 ack 대기; ack가 nonce/receipt/pid/start/boot과 heartbeat guardian을 정확히 되돌려줘야 통과; 실패·timeout이면 rc 75, 요청 철회) |
| `eval_gpu_queue_v4.sh` | v3와 동일 + 두 게이트: outer loop는 `ready`일 때만 예약. `--inner`는 harness verify 직후·**consumer exec 전(=GPU 사용 전)** `request` ack 없으면 exit 75. exec는 pid 유지 → 인수된 pid가 곧 consumer |
| `consumer_watchdog.py` | (2차) consumer별 watchdog. guardian 상실·hang 시 **우리 consumer만** v1 절차로 정지 — 아래 §2차 |
| `tests/test_guardian_handoff.py` | 40 tests (fake OS; 실제 신호·nvidia-smi·harness 없음) |

인수 조건(`GuardianV4.judge`): 같은 boot; 요청자 pid+start 생존(PID 재사용 거부); cmdline `eval_gpu_queue_v4.sh --inner`; 부모가
`groot_harness.py run`; host/gpu/session 일치; harness ledger에 receipt가 이 session/machine/[gpu]이고 `consumer_pid == 요청 pid`;
이 guardian이 이미 본 receipt 아님(old receipt 반환 거부); 이미 인수한 live consumer 없음·등록 consumer 없음(중복 launcher 거부);
직전 poll이 깨끗함(GPU query 성공·미인식 프로세스 없음·low headroom 연속 없음); HOLD 아님.
공유 계약(외부 PID 2795788 / start 1377893575 / reserve 16384 MiB / 2 slot / expiry)은 v2/v3의 `tolerated()`가 그대로 판정 — 계약 불변.

## 2차 (11-04): ACK 후 guardian crash → 무보호 consumer 해결
- launcher(`--inner`)가 request **전에** `consumer_watchdog.py`를 consumer 트리 밖에(subshell+setsid, 재부모화) 띄우고,
  request에 watchdog pid와 nonce를 넣음. guardian은 watchdog이 살아 있고 cmdline이 이 consumer와 nonce를 가리키며 consumer 트리 밖에 있을 때만 인수.
  `GUARDIAN_CONFIG` env 추가 필요.
- watchdog 판정 (heartbeat 나이**만**으로는 절대 정지하지 않음):
  - **ended**: consumer 식별자가 사라짐 → 종료, 아무 조치 없음. 정상 receipt 반납은 harness run wrapper 경로 그대로.
  - **lost**: 알려진 guardian 식별자(pid+start+boot)가 confirm_polls+1회, confirm_s 이상 모두 사라지고 **GPU guardian lock을 잡을 수 있을 때만** 진행.
    lock을 쥔 채로 GuardianV4를 만들어 v1 `act()`를 그대로 실행: ownership 재검증(ledger consumer_pid 포함), 신호 직전 매번 식별자 재확인,
    외부·공유 승인 프로세스는 owned가 아니므로 신호하지 않음, release는 자기 프로세스가 없어진 뒤에만 → 끝나면 HOLD.
    lock이 바쁘면(새 guardian 기동) 보류하고 그 guardian을 따라감.
  - **hung** (peer 지적 반영, 무한 alive 예외 없음): guardian이 살아 있어도 heartbeat가 phase별 **유한 deadline**을 넘으면 hang으로 판정.
    - watch: `hang_s = 3·poll_s + 60(nvidia-smi) + 2·120(harness) + 60`
    - acting(act 진입 전 heartbeat에 기록): `act_deadline = grace_s + 2·kill_wait_s + release_wait_s + 3·120 + 60 + 60` (v1 기본값으로 1230 s)
    - 따라서 정상 act/release(최대 약 750 s)는 hang으로 오판하지 않음.
    - deadline 초과가 confirm_polls+1회 이어지면 **우리 guardian 프로세스만**(식별자·`guardian_v4.py` cmdline 재확인) TERM→KILL.
      lock은 커널이 해제하므로 이후 lost 경로를 탐.
  - **orphan** (peer 지적 반영, lock 없는 정지에 경합 없음): 살아 있고 hang도 아닌 guardian이 이 consumer를 orphan_grace_s 동안 인수하지 않으면
    (재기동 후 HOLD 등) watchdog이 `fence.<pid>.<start>.json`을 씀.
    guardian은 fence된 식별자를 handshake·등록 어느 경로로도 인수하지 않고 heartbeat `fenced`에 표시함.
    watchdog은 **fence 이후 시각의 heartbeat가 fence를 표시하고 여전히 인수하지 않은 경우에만**, 정지 직전 한 번 더 확인한 뒤 lock 없이 같은 절차로 정지.
    guardian이 그 전에 인수하면 fence를 철회하고 조치하지 않음.
- guardian 쪽: act 진입 전 heartbeat phase=`acting` 기록, HOLD.json을 매 step 다시 읽음(watchdog이 쓴 HOLD 반영),
  watchdog이 죽으면 HOLD만 걸고 consumer는 계속 보호(신호 없음).

## 테스트 (필수 항목 대응)
- PID 재사용: 요청 직후 재사용 → nack; 인수 consumer 종료 후 같은 pid를 외부가 재사용 → 외부로 판정, 신호 안 보냄; 계약 pinned pid start 변경 → 미허용
- 새 foreign: 인수 전 → nack; 인수 후 → 우리 consumer·arm·serve·**v4 outer loop** 정지, 외부 프로세스는 신호 없음
- low VRAM: free < reserve(16384) → 우리 것만 정지, 회복 전 새 인수 거부, 회복 후 인수
- old receipt 반환 → nack(`retired`); ledger에 없음 / consumer_pid 불일치 / GPU 불일치 → nack
- 중복 guardian → SystemExit(flock); 중복 launcher → nack
- stale OWNED.json / 과거 ack 파일 → 신뢰 안 함; 죽은 launcher가 남긴 request → nack
- crash: guardian 죽음 / heartbeat stale → `ready` 실패 → 예약 안 함; launcher timeout → 요청 철회, 늦은 인수 불가
- `kept` → HOLD, guardian 재시작 후에도 유지
- guardian 재시작 시 기존 등록 consumer는 ledger `consumer_pid` 일치할 때만 인수
- (2차 watchdog) ACK 후 crash → 우리 consumer만 정지·HOLD, 이후 나타난 foreign 보존; ACK와 heartbeat 사이 crash;
  700 s act(release_wait 600)에도 오판 없음 → deadline 초과 시 우리 guardian만 TERM → 이후 정지; watch 단계 hang 유한;
  guardian PID 재사용 = 상실; 식별자 소실 + lock 보유 중 → 보류; 재기동 guardian이 인수 → 조치 없음; orphan은 fence ack 후에만 정지(foreign·guardian 미신호);
  fence된 consumer는 HOLD 해제 후에도 미인수; consumer 정상 종료 → 조치·release 호출 없음; 중복 watchdog 거부; ownership 불일치 → 신호 0;
  미무장 watchdog은 조치 안 함; watchdog crash → guardian HOLD, kill 없음; handshake는 이 nonce의 트리 밖 watchdog 필수(nonce 1회용)
- shell: `GUARDIAN_STATE_DIR` 없으면 즉시 실패; guardian 미준비면 may-start 전에 rc 75; inner에서 handoff 거부 시 consumer exec 없음; 승인 시 request가 exec보다 먼저

실행: `.venvs/rl2_iid/bin/python -B -m pytest -q -p no:cacheprovider scripts/rl2_vla/groot_fold27_20261028/eval/guardian_handoff_20261104/tests scripts/rl2_vla/groot_fold27_20261028/tests/test_foreign_gpu_guardian.py scripts/rl2_vla/groot_fold27_20261028/tests/test_foreign_gpu_guardian_v2.py scripts/rl2_vla/groot_fold27_20261028/tests/test_foreign_gpu_guardian_v3.py scripts/rl2_vla/groot_fold27_20261028/tests/test_eval_gpu_queue_v2_sh.py` → kanu 로컬 **132 passed** (신규 40 + 기존 92).

## 적용 절차 (main 결정 후, 현재 run 종료 뒤에만 — hot 교체 금지)
1. code-only 배포(git) → srv50 deploy tree에서 위 pytest 재실행.
2. 현재 consumer 2553299와 guardian 2587681이 **정상 종료된 뒤**에 진행. v1~v3 guardian은 lock을 잡지 않으므로 같은 GPU에서 v4와 동시 실행 금지.
3. guardian_v4 기동(기존 v3 config 그대로, 같은 `sharing_authorization`; state dir은 새 경로):
   `setsid nohup .venvs/rl2_iid/bin/python scripts/rl2_vla/groot_fold27_20261028/eval/guardian_handoff_20261104/guardian_v4.py --config <v3 config> --state-dir <new state dir> &`
4. `python3 .../handoff_gate.py ready --state-dir <state> --host srv50 --gpu 3 --session <session>` → `ready: true` 확인.
5. 공식 harness 경로로 launcher 기동: 기존 v3 env + `GUARDIAN_STATE_DIR=<state>` `GUARDIAN_CONFIG=<v4 config>` 로 `bash .../guardian_handoff_20261104/eval_gpu_queue_v4.sh`.
6. `<state>/events.jsonl`에 `handoff_adopted`가 찍히고 `OWNED.json`에 새 consumer가 들어간 것을 확인.

## 남는 위험·충돌 범위
- (1차의 'ACK 후 guardian crash → 무보호'는 2차 watchdog으로 해결.) 남는 한계:
  - **guardian과 watchdog이 둘 다 죽으면** 무보호. watchdog만 죽으면 guardian이 HOLD로 표시하고 계속 보호.
  - 정지까지 지연: crash는 약 (confirm_polls+1)·poll_s 이상 + confirm_s. hang은 phase deadline(watch 약 375 s / acting 약 1230 s, 기본값) + 확인 polls + kill_wait.
  - orphan 정지: orphan_grace_s(기본 120 s)에 fence ack 대기가 더해짐.
  - **watchdog 자신이 hang**(살아 있으나 멈춤)인 경우는 감지하지 않음. guardian이 계속 보호하므로 이중 실패에 해당.
  - watchdog이 쓰는 v1 `act()`의 release 대기(최대 release_wait_s) 동안에는 새 guardian이 lock을 잡지 못해 기동이 거부됨(의도된 단일 actor).
  - ownership 검증이 실패하면(ledger·receipt 불일치) watchdog은 신호 없이 ALERT·HOLD만 남김 → consumer는 운영자 판단 대상.
- 기본 동작은 v1 설계대로 act 시 follower(outer loop)도 정지 → 그 GPU queue는 끝남. 재개는 수동 재기동(handshake 경유).
- v3 launcher로 띄운 consumer는 handshake가 없음: v4가 다른 consumer를 인수 중이면 HOLD, 인수 없으면 ledger 검증 후 인수.
- 새 파일 `<queue_dir>/guardians/gpu<G>.lock` 생성(기존 queue 파일은 건드리지 않음).
- 이 작업 중 kanu 로컬 pytest가 `eval/runtime/__pycache__`, `tests/__pycache__`에 .pyc를 만들었음(코드 파일 변경 없음; srv50 무관).
