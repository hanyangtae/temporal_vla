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
