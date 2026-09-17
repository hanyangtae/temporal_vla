# Astra에서 Codex·Claude 세션에 지시하기

## 목적과 현재 설치

메인은 GPT-6 Astra로 유지하고, 독립된 로컬 Codex·Claude Code 세션에 지시를 보낸다.
세션의 기존 대화와 수신 측 권한 설정을 유지한다.

- 구현: `scripts/utils/session_bridge.py` (호스트 Linux, Python 3.11 이상, uv).
- 의존성: `websockets==15.0.1`, uv가 별도 환경에서 관리한다.
- 사용자 명령: `~/.local/bin/session-bridge`, `~/.local/bin/astra-sessions`.
- Codex 사용자 MCP: `session-bridge` 등록 완료. 다음 새 세션부터 도구를 로드한다.
- 공용 Codex App Server: `~/.codex/app-server-control/app-server-control.sock`. CLI·모바일·브리지가 관리형 원격 제어 서버를 공유한다.
- 기존 CLI 설치·기본 모델·프로젝트 권한 설정은 변경하지 않는다.
- 이 문서의 검증 버전: Codex CLI 0.153.4, Claude Code 2.1.263.

## 메인 Astra 실행

```bash
cd /home/dongkyu/pkt_ws/temporal_vla
astra-sessions
```

명령은 필요한 경우 로컬 App Server를 시작하고, `gpt-6-astra`로 접속한다.
메인에 다음처럼 요청한다.

> session-bridge로 세션 목록을 확인하고, Claude의 '전체 파이프라인' 세션에 수집 상황을 확인해 달라고 보내줘. 답변이 오면 읽어줘.

제공하는 MCP 도구:

| 도구 | 기능 |
| --- | --- |
| `sessions_list` | 제공자·이름·ID·작업 폴더·상태·전송 가능 여부 조회 |
| `sessions_send` | 정확한 이름이나 ID로 지시 전달 |
| `sessions_read` | 지정 세션의 최근 메시지 확인 |

현재 열려 있는 메인 대화에서는 새 MCP가 즉시 추가되지 않을 수 있다. 아래 CLI는 지금부터 사용할 수 있다.

## CLI로 직접 사용

```bash
session-bridge list --cwd /home/dongkyu/pkt_ws/temporal_vla
session-bridge send --provider claude '전체 파이프라인' '현재 수집 상황을 요약해줘.'
session-bridge read --provider claude '전체 파이프라인' --limit 6
```

이름이 중복되면 임의로 선택하지 않고 오류를 반환한다. `claude:UUID` 또는 `codex:UUID`로 지정한다.
여러 줄 메시지는 셸에서 보간하지 말고 stdin으로 전달할 수 있다.

```bash
session-bridge send 'claude:대상-UUID' - <<'MESSAGE'
진행 중인 실험의 상태를 확인해줘.
성공 여부와 실행 중인 작업을 구분해서 답해줘.
MESSAGE
```

## Codex 대상 세션 연결

전송 대상 Codex 세션도 같은 App Server에 연결되어 있어야 한다.
새 터미널에서 `astra-sessions`를 실행하면 메인·작업 세션 모두 서로 지정할 수 있다.

기존 Codex 세션이 목록에 `sendable: false`로 보이면 현재 서버에 로드되지 않은 것이다.
기존 세션의 작업을 마친 후 원래 클라이언트에서 빠져나와 다음과 같이 이어서 연다.

```bash
astra-sessions resume 대상-UUID
```

브리지는 다른 서버의 활성 세션을 강제로 재개하거나 잠금을 빼앗지 않는다.
Claude는 기존 실행 중인 세션의 peer socket을 발견하므로 이 재접속 과정이 필요 없다.

## 전달 상태의 의미

- Codex `queued`: `codex queue`가 메시지를 받아들였다. 작업 완료를 뜻하지 않는다.
- Claude `sent_unconfirmed`: 인증된 peer socket에 썼다. 수신 측에서 승인 대기·거절될 수 있다.
- `sendable: true`: 지원하는 전송 경로가 있다. 수신 승인이나 작업 성공을 보장하지 않는다.
- 반환된 `message_id`가 포함된 수신 메시지와 뒤따르는 답변을 `read`로 확인한다.
- 자동 재전송하지 않는다. 응답 지연과 전달 실패를 구분하지 못한 재전송은 중복 작업을 만들 수 있다.
- 이번 구현에는 메인 대화로 자동 완료 알림을 밀어 넣는 기능이 없다. 답변은 `sessions_read`로 가져온다.
- Claude 읽기는 로컬 JSONL에 기록된 텍스트를 대상으로 한다. 다른 저장 형식의 세션은 별도 어댑터가 필요하다.

Claude 전송은 설치된 CLI에서 확인한 **내부 peer protocol v1** 어댑터다. 공개 안정 API라는 보장은 없다.
CLI 업데이트 후 소켓·인증 키·등록 형식이 바뀌면 다시 검증한다.
수신 세션에 원래 적용된 `crossSessionInbound`와 권한 판단을 유지하며, 보류를 우회하지 않는다.
키는 메모리에서만 사용하고 출력하지 않는다. 동일 사용자 소유권, 파일 모드, 실제 peer PID와 시작 시각을 검사한다.
Codex에 보내는 본문에도 peer instruction임을 표시한다.

## 서버·MCP 관리

```bash
session-bridge start-codex-server
codex mcp get session-bridge
```

관리형 서버 로그: `~/.codex/app-server-daemon/app-server.stderr.log`.
서버가 없으면 `astra-sessions`가 `codex remote-control start --json`으로 관리형 서버를 시작한다. 사용자 지정 소켓은 연결만 하며, 별도 서버를 자동 생성하지 않는다.
기존 사용자 Codex 설정과 인증을 사용한다. 대상 세션에 필요한 승인은 연결된 터미널에서 처리한다.

설치를 다른 경로에서 재현하려면 다음 명령의 절대 경로를 조정한다.

```bash
codex mcp add session-bridge -- uv run --script \
  /home/dongkyu/pkt_ws/temporal_vla/scripts/utils/session_bridge.py mcp
```

등록 해제: `codex mcp remove session-bridge`. 사용자 명령의 심볼릭 링크를 지워도 소스 파일은 남는다.
서버를 종료할 때는 연결된 세션이 작업 중인지 확인하고 이 브리지의 서버 PID만 종료한다.
`SESSION_BRIDGE_STATE`, `SESSION_BRIDGE_CODEX_SOCKET`, `CLAUDE_CONFIG_DIR`로 경로를 지정할 수 있다.

## 검증

```bash
python -m unittest discover -s scripts/utils/tests -p test_session_bridge.py -v
python -m py_compile scripts/utils/session_bridge.py
bash -n scripts/utils/astra_sessions.sh
git diff --check
```

2026-09-08 호스트에서 수행:

- 이름 중복·정확한 ID 선택·오래된 PID·키 파일 권한·심볼릭 링크 거절·UTF-8 크기 제한 검증.
- 실제 Unix socket을 사용한 인증 프레임과 여러 줄 한글 메시지 전달 검증.
- Codex queue의 argv 전달과 권한 변경 플래그 부재 검증.
- MCP 초기화·도구 목록·실패 응답 및 기존 서버 보호 검증. 총 10개 테스트 통과.
- 별도 Claude 테스트 세션에 보내고 `CLAUDE_BRIDGE_ACK_20260908` 응답을 읽음.
- 별도 Astra/Codex 테스트 세션에 보내고 `CODEX_BRIDGE_ACK_20260908` 및 completed turn을 읽음.
- 실제 테스트 이후 Claude 테스트 세션을 중지하고 Codex 테스트 세션을 보관 처리함.
- 새 Codex 세션에서 MCP `runtimeStatus: connected`와 3개 도구 로딩 확인.

관련 공개 문서: [Codex App Server](https://learn.chatgpt.com/docs/app-server),
[websockets Unix client](https://websockets.readthedocs.io/en/15.0/reference/sync/client.html).

## 2026-09-11 공용 서버 전환

Claude peer 연결은 그대로 유지하고 Codex 기본 연결만 관리형 서버로 통일했다.
`CODEX_HOME`을 설정하면 해당 경로의 `app-server-control/app-server-control.sock`을 사용한다.

```bash
codex remote-control start
astra-sessions resume 대상-UUID
```

일반 `codex resume` 대신 위 명령을 사용해야 공용 서버에 연결된다.
기존 MCP 프로세스는 시작 당시 주소를 유지하므로 클라이언트를 재접속해 새 MCP를 로드한다.
기존 브리지 서버와 그 서버에 열린 세션은 자동 종료하거나 이동하지 않는다.
현재 응답과 도구 작업이 끝난 후 기존 클라이언트를 닫고 공용 서버로 재접속한다.
잠금이 계속되면 기존 서버의 해당 thread가 아직 로드된 상태인지 확인하고 개별 세션을 해제한다.
다른 작업이 있을 수 있으므로 서버 전체 종료나 잠금 파일 삭제로 전환하지 않는다.
모바일에서 동일 세션을 열고 CLI와 왕복 입력하는 검증은 실제 기기에서 별도로 수행한다.
