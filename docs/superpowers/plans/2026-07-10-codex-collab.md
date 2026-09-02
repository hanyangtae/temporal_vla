# Claude×Codex 협업 프로토콜 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스펙 `docs/superpowers/specs/2026-07-10-codex-collab-design.md` §9 산출물 구현 — Codex를 read-only 고정 wrapper로 호출하는 협업 인프라(스킬·wrapper·설정·문서).

**Architecture:** 모든 Review-lane Codex 호출은 `scripts/utils/codex_ask.sh` 하나를 거친다 (`--sandbox read-only` 하드코딩, 옵션 인자 거부 — 이 wrapper가 권한 경계). 실행 규약의 단일 출처는 `.claude/skills/codex-collab/SKILL.md`. allow 규칙은 wrapper 경로에만 건다.

**Tech Stack:** bash, codex-cli 0.144.1, Claude Code settings/skills.

## Global Constraints

- 커밋 메시지 한글 + prefix (`script:`, `docs:`, `config:`) — CLAUDE.md 컨벤션.
- wrapper 호출은 항상 repo root 기준 상대경로 `scripts/utils/codex_ask.sh` (allow 규칙 매칭 전제).
- `.claude/settings.json` allow에는 wrapper 경로만. `Bash(codex*)` 광범위 prefix 절대 금지 (스펙 §7).
- write 계열(`--sandbox workspace-write`, `--dangerously-*`)은 어떤 allow에도 넣지 않는다.
- 유료 Codex 호출은 이 계획 전체에서 Task 4의 스모크 1회뿐. 나머지 테스트는 전부 dry-run(무료).
- 이 머신 전제: `~/.codex/config.toml`에 `features.use_legacy_landlock=true` 적용됨 (스펙 §0).

---

### Task 1: codex_ask.sh wrapper + dry-run 테스트

**Files:**
- Create: `scripts/utils/codex_ask.sh`
- Test: `scripts/utils/tests/test_codex_ask.sh`

**Interfaces:**
- Produces (Task 2·3·4가 의존):
  - `scripts/utils/codex_ask.sh ask <prompt_file> <out_dir>` → 새 thread. stdout 마지막 줄 = thread_id. `<out_dir>/reply.md`(응답), `events.jsonl`, `stderr.log` 생성.
  - `scripts/utils/codex_ask.sh resume <thread_id> <prompt_file> <out_dir>` → 왕복. 출력 동일.
  - `scripts/utils/codex_ask.sh review <out_dir> uncommitted [instructions_file]` / `review <out_dir> base <ref> [instructions_file]` / `review <out_dir> commit <sha> [instructions_file]` → `<out_dir>/review.md`.
  - 종료코드: 0 = transport 성공(exit 0 + 최종 `turn.completed`; review는 exit 0), 2 = 인자 거부/전송 실패.
  - `CODEX_ASK_DRY_RUN=1` → 실행 없이 조립된 argv만 출력 (테스트·업그레이드 재검증용).

- [ ] **Step 1: 실패하는 테스트 작성**

`scripts/utils/tests/test_codex_ask.sh`:

```bash
#!/usr/bin/env bash
# codex_ask.sh dry-run 회귀 테스트 — 모델 호출 없음(무료).
# codex 업그레이드 후 argv 재검증(스펙 §5)에도 이 스크립트를 재사용한다.
set -u
cd "$(dirname "$0")/../../.."   # repo root
W=scripts/utils/codex_ask.sh
pass=0; fail=0

t() { # $1=이름 $2=기대(ok|err) $3=출력에 포함될 문자열; 이후=wrapper 인자
  local name="$1" want="$2" substr="$3"; shift 3
  local out rc=0
  out="$(CODEX_ASK_DRY_RUN=1 bash "$W" "$@" 2>&1)" || rc=$?
  if { [[ "$want" == ok && $rc -eq 0 ]] || [[ "$want" == err && $rc -ne 0 ]]; } \
     && [[ "$out" == *"$substr"* ]]; then
    echo "PASS $name"; pass=$((pass+1))
  else
    echo "FAIL $name (rc=$rc): $out"; fail=$((fail+1))
  fi
}

tmp=$(mktemp); echo "test prompt" > "$tmp"
TID=019f4afa-d9f3-72e1-8bf9-95f666953a09

t ask-argv         ok  "--sandbox read-only" ask "$tmp" /tmp/codex_ask_t
t ask-json         ok  "--json"              ask "$tmp" /tmp/codex_ask_t
t resume-argv      ok  "resume $TID"         resume "$TID" "$tmp" /tmp/codex_ask_t
t resume-sandbox   ok  "--sandbox read-only" resume "$TID" "$tmp" /tmp/codex_ask_t
t flag-reject      err "옵션 인자 금지"      ask --sandbox workspace-write "$tmp" /tmp/o
t danger-reject    err "옵션 인자 금지"      ask --dangerously-bypass-approvals-and-sandbox "$tmp" /tmp/o
t badthread-reject err "thread_id 형식"      resume 'foo;rm -rf' "$tmp" /tmp/o
t noprompt-reject  err "prompt 파일 없음"    ask /nonexistent_prompt_file /tmp/o
t review-unc       ok  "--uncommitted"       review /tmp/o uncommitted
t review-base      ok  "--base origin/dev"   review /tmp/o base origin/dev
t review-badtarget err "review 대상"         review /tmp/o everything
rm -f "$tmp"

echo "== $pass pass, $fail fail"
[[ $fail -eq 0 ]]
```

- [ ] **Step 2: 실패 확인**

Run: `bash scripts/utils/tests/test_codex_ask.sh`
Expected: 전 항목 FAIL (wrapper 파일 없음 → `No such file or directory`), 종료코드 1.

- [ ] **Step 3: wrapper 구현**

`scripts/utils/codex_ask.sh`:

```bash
#!/usr/bin/env bash
# Codex Review-lane 고정 wrapper — read-only sandbox 전용.
# 이 파일이 권한 경계다: .claude/settings.json 은 이 경로만 allow 하므로
# 여기서 옵션 인자를 전면 거부해 workspace-write/--dangerously-* 우회를 차단한다.
# 실행 규약 단일 출처: .claude/skills/codex-collab/SKILL.md
# 설계 배경: docs/superpowers/specs/2026-07-10-codex-collab-design.md §5·§7
#
# 사용법 (repo root에서):
#   scripts/utils/codex_ask.sh ask <prompt_file> <out_dir>
#   scripts/utils/codex_ask.sh resume <thread_id> <prompt_file> <out_dir>
#   scripts/utils/codex_ask.sh review <out_dir> uncommitted [instructions_file]
#   scripts/utils/codex_ask.sh review <out_dir> base <ref> [instructions_file]
#   scripts/utils/codex_ask.sh review <out_dir> commit <sha> [instructions_file]
#
# ask/resume: transport 성공(exit 0 + 최종 turn.completed) 시 0 반환,
#   stdout 마지막 줄 = thread_id. 응답은 <out_dir>/reply.md.
# CODEX_ASK_DRY_RUN=1: 실행 없이 조립된 argv 출력 (테스트·업그레이드 재검증).
set -euo pipefail

die() { echo "codex_ask: $*" >&2; exit 2; }

# 보안 경계: 어떤 인자도 옵션 형태('-'로 시작)를 허용하지 않는다.
for a in "$@"; do
  case "$a" in -*) die "옵션 인자 금지: $a (플래그는 wrapper가 고정)" ;; esac
done

dry_or_run() { # argv... — dry-run이면 출력만, 아니면 그대로 실행하지 않고 0 반환
  if [[ "${CODEX_ASK_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"; echo
    return 0
  fi
  return 1
}

run_exec() { # $1=prompt_file $2=out_dir [$3=thread_id]
  local prompt="$1" out="$2" thread="${3:-}"
  [[ -f "$prompt" ]] || die "prompt 파일 없음: $prompt"
  local argv=(codex exec --sandbox read-only --json -o "$out/reply.md")
  [[ -n "$thread" ]] && argv+=(resume "$thread")
  argv+=(-)
  dry_or_run "${argv[@]}" && return 0
  mkdir -p "$out"
  local rc=0
  "${argv[@]}" < "$prompt" > "$out/events.jsonl" 2>"$out/stderr.log" || rc=$?
  if [[ $rc -ne 0 ]] || ! tail -n1 "$out/events.jsonl" | grep -q '"type":"turn.completed"'; then
    die "transport 실패 (rc=$rc, 최종 이벤트: $(tail -n1 "$out/events.jsonl" 2>/dev/null | cut -c1-160))"
  fi
  grep -m1 -o '"thread_id":"[^"]*"' "$out/events.jsonl" | cut -d'"' -f4
}

mode="${1:-}"; [[ -n "$mode" ]] || die "mode 필요 (ask|resume|review)"
shift

case "$mode" in
  ask)
    [[ $# -eq 2 ]] || die "사용법: ask <prompt_file> <out_dir>"
    run_exec "$1" "$2"
    ;;
  resume)
    [[ $# -eq 3 ]] || die "사용법: resume <thread_id> <prompt_file> <out_dir>"
    [[ "$1" =~ ^[0-9a-f][0-9a-f-]{20,40}$ ]] || die "thread_id 형식 오류: $1"
    run_exec "$2" "$3" "$1"
    ;;
  review)
    [[ $# -ge 2 ]] || die "사용법: review <out_dir> <uncommitted|base|commit> [ref] [instructions_file]"
    out="$1" target="$2"; shift 2
    argv=(codex review)
    case "$target" in
      uncommitted) argv+=(--uncommitted) ;;
      base)   ref="${1:-}"; [[ "$ref" =~ ^[0-9a-zA-Z/._-]+$ ]] || die "base ref 형식 오류: ${ref:-빈값}"; argv+=(--base "$ref"); shift ;;
      commit) sha="${1:-}"; [[ "$sha" =~ ^[0-9a-f]{7,40}$ ]] || die "commit sha 형식 오류: ${sha:-빈값}"; argv+=(--commit "$sha"); shift ;;
      *) die "review 대상 오류: $target (uncommitted|base|commit)" ;;
    esac
    inst="${1:-}"
    if [[ -n "$inst" ]]; then
      [[ -f "$inst" ]] || die "instructions 파일 없음: $inst"
      argv+=(-)
    fi
    dry_or_run "${argv[@]}" && exit 0
    mkdir -p "$out"
    rc=0
    if [[ -n "$inst" ]]; then
      "${argv[@]}" < "$inst" > "$out/review.md" 2>"$out/stderr.log" || rc=$?
    else
      "${argv[@]}" > "$out/review.md" 2>"$out/stderr.log" || rc=$?
    fi
    [[ $rc -eq 0 ]] || die "review 실패 (rc=$rc): $(tail -n3 "$out/stderr.log" 2>/dev/null | tr '\n' ' ')"
    ;;
  *) die "unknown mode: $mode (ask|resume|review)" ;;
esac
```

Step 3 후 `chmod +x scripts/utils/codex_ask.sh scripts/utils/tests/test_codex_ask.sh` 실행.

- [ ] **Step 4: 테스트 통과 확인**

Run: `bash scripts/utils/tests/test_codex_ask.sh`
Expected: `== 11 pass, 0 fail`, 종료코드 0. 추가 정적 검증: `bash -n scripts/utils/codex_ask.sh` 무출력.

- [ ] **Step 5: 커밋**

```bash
git add scripts/utils/codex_ask.sh scripts/utils/tests/test_codex_ask.sh
git commit -m "script: codex read-only wrapper + dry-run 테스트 (권한 경계·argv 검증)"
```

---

### Task 2: codex-collab SKILL.md (실행 규약 단일 출처)

**Files:**
- Create: `.claude/skills/codex-collab/SKILL.md`

**Interfaces:**
- Consumes: Task 1 wrapper CLI (`ask`/`resume`/`review` 시그니처, thread_id stdout, DRY_RUN).
- Produces: 이후 모든 Claude 세션이 따를 실행 규약. Task 3의 CLAUDE.md 포인터가 이 파일을 가리킴.

- [ ] **Step 1: SKILL.md 작성**

frontmatter는 기존 스킬(robocasa-steer-eval) 컨벤션. 본문은 스펙 §4·§5·§6·§8을 실행 절차로 옮긴다:

````markdown
---
name: codex-collab
description: Use when consulting Codex (GPT) as a second brain — experiment-plan debate, code review, or long debugging delegation. Triggers: "codex에게 물어봐", "codex 리뷰", 실험 계획 확정 전 반론 게이트, 구현 완료 후 코드 리뷰 게이트, 반복 디버깅 위임. All review-lane calls MUST go through scripts/utils/codex_ask.sh.
---

# Codex 협업 (Claude×Codex)

설계 배경·결정 이유: `docs/superpowers/specs/2026-07-10-codex-collab-design.md` (비규범).
이 파일이 실행 규약의 단일 출처다.

## 원칙

- 사용자 창구는 Claude 하나. Codex 결과는 요약·중계한다.
- 메인 워크트리의 유일한 작성자는 Claude. Codex 변경은 diff 리뷰 게이트로만 유입.
- Codex 지적은 무검증 채택 금지 — 코드·실행으로 확인 후 수용.
- 호출 1회 = Codex 쿼터 소모. 게이트 외 남발 금지. 왕복 상한 3회 —
  수렴 안 하면 양쪽 입장을 정리해 사용자 에스컬레이션.

## Review lane 호출 (반드시 wrapper 경유)

프롬프트를 스크래치 파일로 작성 후:

```bash
# 새 토론 — stdout 마지막 줄이 thread_id
scripts/utils/codex_ask.sh ask "$SCRATCH/prompt.md" "$SCRATCH/round1"
# 왕복 (공통 옵션은 wrapper가 resume 앞에 고정 배치)
scripts/utils/codex_ask.sh resume "$THREAD_ID" "$SCRATCH/followup.md" "$SCRATCH/round2"
# 코드 리뷰 (Gate 2) — 전용 서브커맨드, 일반 프롬프트로 대체 금지
scripts/utils/codex_ask.sh review "$SCRATCH/review1" uncommitted
scripts/utils/codex_ask.sh review "$SCRATCH/review1" base origin/dev "$SCRATCH/review_focus.md"
```

- 응답: `<out_dir>/reply.md` (review는 `review.md`). wrapper 종료 0 = transport 성공
  (exit 0 + 최종 `turn.completed`). **transport 성공 ≠ task 성공** — 내용을 읽고 판단.
- `codex exec`/`codex review` 직접 호출 금지 (권한 우회 방지 — allow는 wrapper뿐).
- resume은 최적화: 매 호출에 목표·이전 결론 요약을 프롬프트에 재전달. resume 실패
  시 새 thread. thread 수명 = 토론 1건. thread당 동시 호출 1개. `--last` 금지.
- 프롬프트 원칙: 파일 경로 참조 위주(Codex가 직접 읽음), 레포 규칙 중복 설명 금지
  (AGENTS.md 자동 인지), 역할 지시 명시("비판적 리뷰어 — 반례·confound·단순 대안 중심").
- 오류: 인증 만료·CLI·sandbox 오류 → 중단·사용자 보고. 자동 재시도는 Review lane의
  명확한 전송 실패에만 1회.

## 게이트

- **Gate 1 (계획 토론)**: 계획 초안 문서 작성 → `ask`로 반론 요청 → `resume` ≤3왕복 →
  합의/이견 정리해 사용자 제시. 결정은 사용자.
- **Gate 2 (코드 리뷰)**: 구현 완료 후 `review uncommitted` 또는 fetch 후
  `review base origin/dev` (기준 SHA 기록). 지적은 코드로 검증 후 수용/반박.
- **Gate 3 (디버깅 위임, Debug lane)**: 아래 참조. wrapper를 쓰지 않는 유일한 경우.

## Gate 3 — 디버깅·장기 실행 위임 (Debug lane)

발동 기준(전부 아니면 Review lane 패치 제안으로): 긴 수정→실행→관찰 루프 /
다파일 직접 수정 / 작업트리 오염 작업.

1. 미커밋 변경 포함 시 WIP 커밋으로 고정 → delegate_sha 기록.
2. `git worktree add .claude/worktrees/codex-<작업명> -b <branch>-codex-<작업명> <delegate_sha>`
3. `codex exec --sandbox workspace-write --cd <worktree> --json - < 위임프롬프트` 직접 호출
   — **의도적으로 allow 밖** → 사용자 승인 프롬프트가 뜨는 것이 정상. 위임 프롬프트에
   성공 기준(검증 명령·기대 결과) 명시.
4. 실행 중 메인 쪽에서 이 저장소의 git metadata 변경 명령(commit/rebase/fetch/branch/
   submodule) 금지. 파일 편집은 허용.
5. 회수: HEAD가 delegate_sha에서 움직였으면 rebase 후 재검증. 전체 채택 =
   `git merge --squash` 후 `git diff --cached`·`git status --short` 검사. 부분 채택 =
   hunk 단위 적용. 검증은 최종 staged 기준. 한글 커밋에 Codex 출처 명시.
6. 병합/폐기 즉시 worktree·브랜치 삭제 + 해당 thread 폐기.
7. 비정상 종료 시 자동 재시도·자동 병합 금지 — worktree 상태 조사 후 사용자 보고.

## 토론 기록 (원장)

필수: Gate 1(사용자 결정 포함)·Gate 3(위임·채택). Gate 2는 커밋/PR 설명으로 대체 가능.
포맷: `docs/collab_codex/YYYY-MM-DD-<topic>.md` — 게이트, thread_id, (Gate 3) base_sha·
delegate_sha, 라운드별 요지/핵심 발췌, 합의·이견·사용자 결정. raw 응답은 커밋 금지.

## codex 업그레이드 후 재검증 (필수)

`codex --version`이 바뀌었으면 호출 전:
1. `codex sandbox -- head -1 README.md` (무료) — sandbox 동작 확인.
   실패 시 `~/.codex/config.toml`의 `features.use_legacy_landlock=true` 유지 여부 확인
   (deprecated 플래그 — 제거됐으면 사용자에게 보고, AppArmor 프로필 대안은 sudo 필요).
2. `bash scripts/utils/tests/test_codex_ask.sh` (무료) — argv 정규형 회귀 확인.
````

- [ ] **Step 2: 검증**

Run: `head -5 .claude/skills/codex-collab/SKILL.md` → frontmatter 확인.
Run: `git check-ignore .claude/skills/codex-collab/SKILL.md; echo rc=$?` → Expected: `rc=1` (무시 안 됨).

- [ ] **Step 3: 커밋**

```bash
git add .claude/skills/codex-collab/SKILL.md
git commit -m "docs: codex-collab 스킬 추가 — Codex 협업 실행 규약 단일 출처"
```

---

### Task 3: 설정·문서 glue (settings allow, docs/collab README, CLAUDE.md 포인터)

**Files:**
- Create: `.claude/settings.json`
- Create: `docs/collab_codex/README.md`
- Modify: `CLAUDE.md` ("## Agent 운영 규칙" 섹션 끝)

**Interfaces:**
- Consumes: Task 1 wrapper 경로, Task 2 스킬 경로.

- [ ] **Step 1: `.claude/settings.json` 생성** (기존에 없음 — 신규. `settings.local.json`은 건드리지 않는다)

```json
{
  "permissions": {
    "allow": [
      "Bash(scripts/utils/codex_ask.sh:*)",
      "Bash(bash scripts/utils/codex_ask.sh:*)"
    ]
  }
}
```

- [ ] **Step 2: JSON 유효성 + ignore 확인**

Run: `python3 -m json.tool .claude/settings.json > /dev/null && echo OK` → Expected: `OK`
Run: `git check-ignore .claude/settings.json; echo rc=$?` → Expected: `rc=1`

- [ ] **Step 3: `docs/collab_codex/README.md` 작성**

```markdown
# docs/collab — Claude×Codex 토론 원장

Claude와 Codex의 협업 기록(원장). 실행 규약: `.claude/skills/codex-collab/SKILL.md`.

- 기록 필수: Gate 1(계획 토론, 사용자 결정 포함), Gate 3(디버깅 위임·채택).
- 파일명: `YYYY-MM-DD-<topic>.md`. raw 응답 파일은 커밋하지 않는다(핵심 발췌만).
```

- [ ] **Step 4: CLAUDE.md 포인터 추가**

"## Agent 운영 규칙" 섹션 마지막에 append:

```markdown
- Codex(GPT) 협업 — 계획 반론·코드 리뷰·디버깅 위임 — 은 `.claude/skills/codex-collab/SKILL.md`
  규약을 따른다. Review-lane 호출은 반드시 `scripts/utils/codex_ask.sh` 경유.
```

- [ ] **Step 5: 커밋**

```bash
git add .claude/settings.json docs/collab_codex/README.md CLAUDE.md
git commit -m "config: codex wrapper allow 규칙 + collab 원장 README + CLAUDE.md 포인터"
```

---

### Task 4: 엔드투엔드 검증 + 스펙 비규범화 + 마무리

**Files:**
- Modify: `docs/superpowers/specs/2026-07-10-codex-collab-design.md` (헤더에 비규범 배너)

**Interfaces:**
- Consumes: Task 1 wrapper, Task 2 SKILL.md, Task 3 settings.

- [ ] **Step 1: wrapper 실전 스모크 (유료 1회, 소액)**

```bash
SCRATCH=$(mktemp -d)
echo ".claude/skills/codex-collab/SKILL.md 를 읽고 frontmatter의 name 값만 한 단어로 출력해라." > "$SCRATCH/p.md"
scripts/utils/codex_ask.sh ask "$SCRATCH/p.md" "$SCRATCH/out"
echo "rc=$?"; cat "$SCRATCH/out/reply.md"
```

Expected: `rc=0`, stdout에 thread_id(uuid), reply.md 내용 = `codex-collab`.
(이 호출이 wrapper 배선 + sandbox + 성공 판정 로직을 한 번에 검증한다.)

- [ ] **Step 2: 권한 매칭 확인 — 검증 수준 기록**

이 세션(백그라운드)에서는 승인 프롬프트 동작을 관측할 수 없다. 다음을 스모크 결과 보고에
명시한다: "allow 규칙 매칭(wrapper 무승인·workspace-write 직접 호출 승인 요구)은 사용자
interactive 세션에서 1회 확인 필요" (스펙 §9-6, 미검증 항목의 정직한 보고).

- [ ] **Step 3: 스펙 문서 비규범화 배너**

스펙 상태 줄을 다음으로 교체:

```markdown
- 상태: **구현 완료 — 이 문서는 비규범 설계 기록** (결정 배경·개정 이력 보존용).
  실행 규약의 단일 출처는 `.claude/skills/codex-collab/SKILL.md`.
```

- [ ] **Step 4: 최종 정리 커밋**

```bash
git add docs/superpowers/specs/2026-07-10-codex-collab-design.md docs/superpowers/plans/2026-07-10-codex-collab.md
git commit -m "docs: codex 협업 스펙 비규범화 + 구현 계획 문서 추가"
```

- [ ] **Step 5: 완료 보고**

사용자에게: 산출물 목록, 스모크 결과(thread_id·응답), 미검증 항목(권한 프롬프트 —
interactive 세션 확인 방법 포함), 첫 실사용 제안(다음 실험 계획에 Gate 1 적용).
