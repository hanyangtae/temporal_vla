# persist 모드 + 6 serve 전환 영수증 (srv50 GPU1, 2026-11-04)

사용자 지시(메인 경유): "GPU1 2serve→총 6serve", "일단 돌렸으면 다른애 들어와도 뺐기지말고 계속 해". 외부 PID가 나타난 것만으로 멈추는 조건은 철회됐고, 실제 VRAM/OOM 보호·타인 무신호·하네스 6 slot 한도는 유지한다.

## 기존 기능만으로 가능한가 — 불가
- **serve 수:** consumer의 serve 수와 포트는 launch 때 `SERVES_PER_GPU`와 receipt(serves=2)로 고정된다. 하네스 verify도 receipt 기준으로 검사한다. hot 변경 경로는 없으므로 6 serve는 새 receipt로 재예약(재launch)해야 한다.
- **진행 중 episode 경계에서 멈추기:** 그런 drain 기능은 기존에 없다. 유일한 경로는 TRAIN_REQUEST 기반 yield인데, trainer가 없으므로 쓰면 원장 조작이 된다. 쓰지 않는다.
- **외부 프로세스 판정:** sharing 계약이 없는 v4는 모든 미등록 GPU 프로세스를 2 poll 뒤 정지 조건으로 본다. 메모리 보호(reserve)는 계약 경로(v2)에만 있다. 따라서 새 guardian 클래스가 필요하다.

## 신규 코드 (기존 파일 무변경: guardian_v4 be05bf6c, consumer_watchdog 8f7db869)
| 파일 | sha256 앞 16자 |
|---|---|
| persist_guardian_v4.py | 33224b62a55a0ac0 |
| tests/test_persist_guardian_v4.py | 415a0793e176cb66 (8개) |

handoff 폴더 전체 83개 테스트 OK.

동작:
- **외부 프로세스:** 신원만으로는 절대 정지하지 않는다. 매 poll 기록만 남기고(pid/start/uid/cmd/MiB), owned가 아니므로 signal도 받지 않는다.
- **메모리 보호:** 외부 프로세스가 있고 우리 consumer가 돌 때, free < `reserve_mib`가 confirm_polls 연속이면 기존 v1 act가 우리 job만 멈추고 공식 release한다. 외부 프로세스 없이 free가 낮은 것(우리 자신의 사용량)에는 act하지 않는다. NVML 실패 시에는 아무것도 하지 않는다.
- **새 consumer 채택:** 기본적으로 외부 프로세스가 있으면 거부한다(`adopt_with_foreign` 기본 false). launch 승인은 기존 하네스·wait_resources 규칙을 그대로 따른다.
- HOLD, watchdog, fence, release는 v4 그대로다. 파일명에 `guardian_v4.py`가 들어 있어 watchdog의 hung-guardian 식별 검사도 그대로 맞는다.

## 전환 절차 (메인 실행)
1. **사전조건:** GPU1에 외부 프로세스가 없어야 한다. 있으면 하네스가 6 serve 재예약을 거부하므로, 지금 run을 멈추지 말고 대기한다.
   - 포트 확인: srv50의 GPU1×6 serve 포트는 port0(8890)+100+6..11 = 8996~9001이다. `queue/srv50/consumers/gpu*.json`에 살아 있는 다른 consumer가 있으면 그 포트(base+gpu×serves)와 겹치지 않는지 확인한다.
2. **새 cfg:** exclusive cfg를 새 파일로 복사하고 다음 키만 추가한다. state dir도 새로 만든다.
   - `"foreign_policy": "persist"`
   - `"persist_approval": "<사용자 원문>"`
   - `"reserve_mib": 16384`
   - 확인: `sharing_authorization`이 없는지, gpu=1인지, `env_container`가 gpu1인지.
3. **지금 run 정지:** `failover_stop.py --dry-run`이 rc 0인지 확인한 뒤 실제로 실행한다.
   - 고정값: guardian 3654998, consumer 3742000, receipt 9d0955b67d2243f589be3e67c28deaf7, watchdog pid/start, nonce, `--config`/`--state-dir`(guardian cmdline 문자열 그대로).
   - 진행 중 episode(최대 2개)는 버려지고 새 attempt로 재수행된다. 완료 행은 hash로 보존된다.
   - 정지 후 확인:
     - receipt가 원장에서 사라졌는지, GPU1에 우리 pid가 0인지
     - `rows/n*/a*`에 meta.json은 있는데 progress line이 없는 attempt(orphan)가 없는지. orphan이 있으면 그 job은 resume 때 계약 오류로 실패하므로 재개 전에 보고한다.
4. **persist guardian 기동 (detached):** `setsid nohup python3 scripts/rl2_vla/groot_fold27_20261028/eval/guardian_handoff_20261104/persist_guardian_v4.py --config <새cfg> --state-dir <새state> </dev/null >> <새state>.log 2>&1 &`
5. **v4 outer loop 재개:** 기존 launch 명령과 같게 하되 다음을 바꾼다.
   - `GUARDIAN_CONFIG=<새cfg>`, `GUARDIAN_STATE_DIR=<새state>`, `SERVES_PER_GPU=6`
   - `env -u HARNESS_SHARING_AUTH`, POOL에 1 포함, 같은 eval root와 PIPE_ARGS
   - 확인: handoff ack, watchdog armed, heartbeat adopted, 하네스 receipt serves=6.

## 한계
- episode 경계 정지는 아니다. 진행 중 최대 2판을 잃는다(재수행됨).
- 메모리 보호가 act하면 우리 run이 멈춘다. 사용자가 유지를 요구한 조건이다.
- 랩 규칙(docs/05: 타인 프로세스가 있는 GPU에는 발사 금지)은 launch에 대한 것이고, 이 모드는 launch 승인을 바꾸지 않는다. 바뀌는 것은 이미 실행 중인 run의 정지 조건뿐이다.
