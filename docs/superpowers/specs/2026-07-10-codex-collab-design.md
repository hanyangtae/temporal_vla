# Claude×Codex 협업 프로토콜 설계

- 날짜: 2026-07-10 (R3 — Codex 2차 리뷰 반영)
- 상태: **구현 완료 — 이 문서는 비규범 설계 기록** (결정 배경·개정 이력 보존용).
  실행 규약의 단일 출처는 `.claude/skills/codex-collab/SKILL.md`.
- 토론 원장: [`docs/collab_codex/2026-07-10-codex-collab-protocol.md`](../../collab_codex/2026-07-10-codex-collab-protocol.md)
- 대상 도구: Claude Code 2.1.206, codex-cli 0.144.1 (이 레포는 Codex 쪽 `trusted` 등록됨)

## 0. 이 머신 전제조건 (2026-07-10 검증)

codex 기본 sandbox 백엔드(bwrap)가 이 머신의 AppArmor 제한
(`kernel.apparmor_restrict_unprivileged_userns=1`)과 충돌해 셸 실행이 전부 실패한다
(`bwrap: loopback: Failed RTM_NEWADDR`). 해결:

- `codex features enable use_legacy_landlock` 적용됨 (`~/.codex/config.toml`의
  `features.use_legacy_landlock=true`). Landlock 백엔드는 userns가 필요 없어 정상 동작.
- 검증: read-only sandbox에서 레포 파일 읽기 성공, 쓰기 차단 확인.
- **주의**: 이 플래그는 deprecated — codex 업그레이드 후엔
  `codex sandbox -- head -1 README.md`(무료)로 재확인. 플래그가 제거되면 대안은
  AppArmor 프로필 추가(sudo 필요, 사용자 작업).

## 1. 목적

두 모델의 강점을 역할 분담으로 결합한다. 사용자 창구는 Claude 세션 하나다.

| 역할 | 담당 | 근거 (모델 특성) |
|---|---|---|
| 레포 파악·아키텍처·알고리즘 설계·코드 작성 | Claude | 대규모 레포 이해, 설계·코딩 품질 우위 |
| 브레인스토밍 반론·코드 리뷰 | Codex | 독립 관점의 교차 검증, 저비용 |
| 반복 디버깅·장기 실행 위임 | Codex | 에이전트 실행 지구력 우위, 저비용 장기 실행 |

## 2. 확정 결정사항

1. **토폴로지**: Claude 단일 창구. Codex는 Claude가 CLI로 소환하고 그 결과를 요약·중계한다.
2. **Codex 권한 — 두 lane**:
   - **Review lane** (Gate 1·2·수시): `--sandbox read-only`. 커밋 금지, findings/패치 제안만 반환.
   - **Debug lane** (Gate 3): 격리 worktree에서 `workspace-write`. 산출물은 diff로만 회수.
3. **호출 시점**: 표준 게이트(§4) 자동 + 사용자 수시 요청.
4. **메커니즘**: `codex exec` / `codex exec … resume` / `codex review` CLI 직접 호출.
   MCP 등록은 비범위(추후 옵션).
5. **메인 워크트리의 유일한 작성자는 Claude**. Codex 변경은 항상 diff 리뷰 게이트를 거쳐서만 유입.

## 3. 아키텍처

```
사용자 ↔ Claude 세션 (단일 창구, 설계·코드 작성 주체)
              │
              ├─ codex exec ─────────────── 새 토론 시작 (read-only)
              ├─ codex exec … resume ────── 같은 thread 왕복 토론 (read-only)
              ├─ codex review ───────────── diff 기반 코드 리뷰 (read-only)
              └─ codex exec (workspace-write, 격리 worktree) ── Gate 3 위임
              ↓
        docs/collab_codex/*.md (토론 기록, git 추적)
```

- 프로토콜 단일 출처: `.claude/skills/codex-collab/SKILL.md` (구현 산출물, git 추적).
  기존 `.claude/skills` 3개(robocasa-steer-eval 등)와 동일한 운영 패턴.
- `CLAUDE.md`에는 포인터 1–2줄만 추가.

## 4. 게이트 정의

### Gate 1 — 실험 계획 토론 (Review lane)

1. Claude가 계획 초안을 문서로 작성.
2. Codex에게 문서 **경로**를 주고 비판적 반론 요청 (역할 지시: "반례·confound·더 단순한 대안 중심").
3. `resume`으로 최대 3왕복.
4. 합의점과 남은 이견을 정리해 사용자에게 제시. 결정은 사용자.

### Gate 2 — 코드 리뷰 (Review lane)

1. 설계/구현 완료 후 `codex review --uncommitted`(작업 중 변경) 또는
   `codex review --base <sha>`(브랜치 전체). 대상이 명확한 전용 서브커맨드를 쓰고,
   리뷰를 일반 `codex exec` 프롬프트로 대체하지 않는다.
   브랜치 리뷰 기준은 로컬 `dev`가 아니라 **fetch 후 `origin/dev`의 SHA**로 고정하고
   그 SHA를 기록한다 (로컬 dev는 뒤처져 있을 수 있음 — repo 관례도 origin/dev 비교).
2. 지적사항을 Claude가 **코드로 직접 검증한 뒤** 수용/반박 (무검증 채택 금지 — verify-before-relay).
3. 수정 적용은 Claude만.

### Gate 3 — 디버깅·장기 실행 위임 (Debug lane)

**발동 기준 (상향됨)**: 다음의 경우에만 worktree 위임. 그 외 작은 디버깅은
Review lane에서 진단 + 패치 제안으로 처리하고 Claude가 적용한다.

- 수정→실행→관찰을 여러 차례 반복해야 하는 긴 루프
- 여러 파일을 직접 고쳐야 하는 작업
- 빌드/실행 산출물이 작업 트리를 오염시키는 작업

git 흐름:

```
dev
 └─ exp/foo                    ← 작업 브랜치. Claude가 메인 워크트리에서 작업
      └─ exp/foo-codex-<작업명> ← 일회용 브랜치 (git worktree 분리, workspace-write)
      ┌─ ← 완료 후: Claude가 diff 전체 리뷰 → 통과분만 squash 병합
      ▼    (한글 커밋, 메시지에 Codex 출처 명시)
 exp/foo (계속)  → 작업 완료 시 dev로 PR (기존 관례대로)
```

규칙:

- **입력 스냅샷**: 메인 작업 브랜치의 미커밋 변경은 worktree에 나타나지 않으므로,
  위임 대상에 미커밋 변경이 포함되면 **WIP 커밋으로 고정한 뒤 그 SHA에서 분기**한다.
  로그에 이 위임 입력 SHA(delegate_sha)를 base_sha와 함께 기록.
- 회수 시 작업 브랜치 HEAD가 위임 시점에서 움직였으면 codex 브랜치를 최신 HEAD에
  rebase(또는 patch 재적용) 후 **재검증**하고 병합. "codex 브랜치에서 테스트 통과"만으로
  게이트를 통과시키지 않는다.
- **회수 절차**: 전체 채택 = `git merge --squash` 후 `git diff --cached`·`git status --short`로
  staged 내용과 untracked 산출물 검사. 부분 채택 = patch/hunk 단위 적용.
  검증은 codex 브랜치가 아니라 **최종 staged 내용 기준**으로 수행.
- codex 브랜치는 **일회용** — 병합 또는 폐기 즉시 브랜치·워크트리 삭제.
- 병합은 **squash** — 리뷰 통과한 최종 diff를 커밋 1개로.
- **병합/폐기와 동시에 해당 thread도 폐기** — squash로 SHA·경로 전제가 낡은 세션을
  이어 쓰면 Codex가 낡은 상태 기준으로 판단한다. 후속 작업은 새 thread에서
  base/head SHA를 명시해 시작.
- worktree는 **변경 격리 수단이지 보안 경계가 아니다** — worktree들은 refs·lock 등
  repo 메타데이터를 공유한다. Gate 3 실행 중 Claude는 이 저장소의 **git metadata를
  변경하는 명령 전반**(commit, rebase, fetch, branch 생성/삭제, submodule 조작)을
  하지 않는다. 메인 worktree에서의 일반 파일 편집은 허용.
- Codex 프로세스가 비정상 종료(타임아웃·중단)하면 **자동 병합 금지** — worktree의
  `git status`·HEAD·최종 이벤트를 검사해 상태를 보고하고 사용자 판단을 받는다.
- 부분 채택 시 채택/기각 사유를 토론 로그에 기록.

### 수시 호출

사용자 요청 시 게이트 무관하게 위 규약 중 맞는 형태로 호출.

## 5. 호출 규약 (0.144.1에서 검증된 정규형)

새 토론 시작:

```bash
codex exec --sandbox read-only --json \
  -o "$SCRATCH/reply.md" - < "$SCRATCH/prompt.md" > "$SCRATCH/events.jsonl"
```

왕복 (공통 옵션은 반드시 `resume` **앞**에 — resume 서브파서는 `--sandbox`를 받지 않음, 검증됨):

```bash
codex exec --sandbox read-only --json \
  -o "$SCRATCH/reply2.md" resume "$THREAD_ID" - < "$SCRATCH/followup.md"
```

- **thread ID**: `--json` 스트림의 `{"type":"thread.started","thread_id":"…"}` 이벤트에서
  파싱 (검증됨). 토론 로그에 기록.
- **성공 판정은 두 층위** (전송 성공 ≠ 작업 성공):
  - `transport_success` = exit code 0 **그리고** 최종 이벤트 `turn.completed` (검증됨).
    중간 error 이벤트는 기록만 하고 최종 상태까지 기다린다. `turn.failed`·비정상 종료는 실패.
  - `task_success` = 사전 정의한 성공 기준(산출물·검증 명령·기대 결과) 충족. Codex가
    "완료하지 못했다"고 정상 응답해도 transport는 성공이다 — 특히 Gate 3는 위임 프롬프트에
    성공 기준을 명시하고, 로그에 실행한 검증·미검증 항목을 남긴다.
- **resume은 최적화이지 전제가 아님**: 세션 유실·만료에 대비해 매 호출에 목표·이전 결론
  요약을 프롬프트에 재전달하고, resume 실패 시 새 thread로 재시작한다.
  thread 수명 = 토론/위임 1건.
- **동시성**: thread당 동시 호출 1개. 자동화에서 `resume --last` 금지 (병렬 세션 환경에서
  어느 thread가 잡힐지 불명확).
- 프롬프트는 스크래치 파일 + stdin (셸 이스케이프 회피). 파일 **경로** 참조 위주.
  레포 규칙 중복 설명 금지 — Codex는 `AGENTS.md → .agents/agent_spec.md`를 자동 인지.
- 오래 걸리는 호출은 background 실행 후 완료 시 회수.
- SKILL.md에 위 argv를 고정하고, codex 업그레이드 감지 시(`codex --version` 변화)
  §0 sandbox 확인과 argv 재검증을 먼저 수행.

## 6. 토론 기록 포맷

`docs/collab_codex/YYYY-MM-DD-<topic>.md` (한글):

```markdown
# <주제> — Claude×Codex 토론
- 게이트: 계획 토론 | 코드 리뷰 | 디버깅 위임 | 수시
- thread_id: <uuid>
- (Gate 3) base_sha: <sha>
## Round 1
### Claude → Codex (요지)
### Codex 응답 (핵심 발췌)
## 결론
- 합의: ...
- 이견(사용자 결정 필요): ...
- 사용자 결정: ...
```

raw 응답 파일은 스크래치에만 두고 커밋하지 않는다. 로그에는 핵심 발췌만.
세션 대화가 아니라 이 로그(+git SHA)가 협업의 원장(ledger)이다.

**기록 의무 범위** (과설계 방지): 필수 = Gate 1(사용자 결정 포함)과 Gate 3(위임·채택 기록).
단순 Gate 2 리뷰는 커밋/PR 설명으로 대체 가능. 수시 호출은 결정에 영향을 준 경우에만 기록.

## 7. 권한 설정

- **광범위한 prefix allow 금지**: `Bash(codex exec*)` 류는
  `codex exec --sandbox workspace-write …`·`--dangerously-*`까지 매칭되어
  "write는 매번 승인"이라는 의도와 정면 모순 (Codex 2차 리뷰 지적, 치명적).
- 대신 **고정 wrapper 스크립트 1개만 allow**: `scripts/utils/codex_ask.sh`가
  `--sandbox read-only`를 하드코딩하고 인자를 검증(위험 플래그 거부).
  allow 규칙은 이 wrapper 경로에만 건다. (경로 단일 출처 원칙과도 일치.)
- Gate 3(workspace-write)는 wrapper를 우회한 직접 호출 → 항상 사용자 승인 프롬프트.
- **권한 매칭 스모크 테스트**를 구현 검증 항목에 포함: wrapper 호출은 무승인 통과,
  `codex exec --sandbox workspace-write` 직접 호출은 승인 요구를 실제로 확인.
- sandbox와 approval은 별개 축임을 SKILL.md에 명시 (sandbox=실행 격리,
  approval=실행 허용 정책; `workspace-write`가 자동 승인을 뜻하지 않음).

## 8. 에러 처리·안전장치

- codex 인증 만료·CLI 실패·sandbox 오류 → 즉시 중단하고 사용자 보고 (stop-on-problems 원칙).
- **자동 재시도는 Review lane의 명확한 전송 실패에만** (부작용 없음이 보장될 때, 1회).
  Debug lane(Gate 3)의 불명확 종료는 재시도 금지 — Codex가 이미 파일 수정·장기 실행을
  수행한 뒤 이벤트만 유실됐을 수 있으므로, §4의 절차대로 worktree 상태를 조사 후 보고.
- 왕복 상한 3회 — 수렴하지 않으면 양쪽 입장을 나란히 정리해 사용자 에스컬레이션.
- Codex 지적의 무검증 채택 금지 — 코드·실행으로 확인 후 수용.
- 호출 1회 = Codex 쪽 쿼터 소모임을 인지하고, 게이트 외 남발하지 않는다.

## 9. 구현 산출물

1. `.claude/skills/codex-collab/SKILL.md` — 실행 규약의 단일 출처.
   **구현 완료 후 이 설계 문서는 결정 배경·개정 이력만 남는 비규범 기록으로 고정**하고,
   실행 규약 섹션은 SKILL.md 링크로 대체 (이중 규범 방지).
2. `scripts/utils/codex_ask.sh` — read-only 고정 wrapper (인자 검증 포함)
3. `docs/collab_codex/` 디렉토리 + 짧은 README
4. `CLAUDE.md` 포인터 1–2줄
5. `.claude/settings.json` allow 규칙 (wrapper 경로에만)
6. 권한 매칭 스모크 테스트: wrapper 무승인 통과 + workspace-write 직접 호출 승인 요구 확인
7. ~~CLI 스모크 테스트~~ → 완료 (2026-07-10): sandbox 수리, thread_id 파싱, resume 정규형,
   세션 기억 유지, transport 성공 판정 모두 검증됨.

## 10. 비범위

- MCP 등록(`codex mcp-server`) — 운영해보고 필요하면 추후 업그레이드.
- Codex의 메인 워크트리 직접 수정 — 영구 금지.
- Codex→Claude 방향 역호출 — 단일 창구 원칙과 상충.

## 11. 개정 이력

- **R1** (2026-07-10): 초안. Codex 1차 리뷰 실행 — 단, 당시 sandbox 고장으로 Codex가
  문서를 직접 읽지 못한 반쪽 리뷰였음.
- **R2** (2026-07-10): R1 리뷰 지적 반영 — sandbox 수리(§0), 두 lane 권한 모델(§2),
  Gate 3 발동 기준 상향·base_sha·thread 폐기 규칙(§4), 검증된 호출 정규형·성공 판정·
  동시성 규칙(§5), 원장은 세션이 아니라 로그(§6). 기각: "단일 작성자 원칙과 Gate 3
  모순" 지적은 two-lane + diff 게이트로 이미 해소되어 용어 명확화만 반영.
- **R3** (2026-07-10, 현재): Codex 2차 리뷰(문서 직접 읽은 정식 리뷰, 판정 "조건부 가능")
  지적 9건 전부 수용 — allow prefix 권한 우회 → wrapper 방식(§7), Gate 3 입력 스냅샷
  WIP 커밋(§4), 재시도를 Review lane 한정(§8), transport/task 성공 분리(§5),
  squash 회수 절차(§4), origin/dev 기준(§4), git metadata 병렬 금지 범위(§4),
  로그 의무 축소(§6), 구현 후 이 문서 비규범화(§9).
