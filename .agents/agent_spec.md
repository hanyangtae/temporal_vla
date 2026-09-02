---
description: Dongkyu-specific agent operating spec for research-code work
---

# Dongkyu Agent Spec

전용 agent의 repo-local 업무 규칙. 이 파일에는 **이 repo에서만 참인 규칙**만 둔다 —
일반적 코딩 규율(작은 diff, 근거 기반 판단, 결론 먼저 등)은 모델 기본 동작에 맡긴다.

## 1. Context Intake

- Entrypoint chain은 이 파일에서 멈춘다. `AGENTS.md`/`CLAUDE.md`는 pointer로 취급하고,
  이후 문서(관련 README, `docs/**`, ADR, runbook)는 task context로 필요한 섹션만 읽는다.
- 시작 시 확인: `git status -sb`, target env(host / container / conda / uv 중 무엇인지).
- Instruction 충돌 시 우선순위: 작업 파일의 local rule > 관련 ADR·runbook > repo-wide
  instruction > 일반 workflow.

## 2. Environment And Validation

Runtime 판단은 **target env 검증**을 기준으로 한다. 이 repo는 env가 쪼개져 있다
(robocasa=Docker py3.11, calvin=3.8, 모델 서버=컨테이너/uv).

- Repo가 지정한 container/conda/uv env를 우선한다. **Host-side Python import 실패는
  syntax/static signal로만 다룬다** (host에 deps가 없는 게 정상).
- 컨테이너 내부 path(`/temporal_vla`, `/cache`)와 host path가 다르면 명령마다 실행 위치를 명시한다.
- 정적 검증: `git diff --check`, `bash -n`, `py_compile`(또는 container syntax check),
  CLI `--help` smoke, stale reference 검색.
- Runtime 변경은 target env에서 최소 smoke run. 생략한 runtime 검증은 final answer에 명시한다.

## 3. Documentation And Research Claims

문서는 목적별 분리: **runbook**(실행 순서·명령어, 그 자리에서 실행 가능하게) /
**report**(결과·평가 범위·해석·다음 검증 축; 명령어는 runbook으로) / **ADR**(결정 배경·유지할
contract·변경 조건).

Research claim 규율:

- Claim 단위는 label 단위와 맞춘다. diagnostic evidence / detector 성능 / policy 성능 /
  intervention 성능을 구분해서 쓴다.
- 시각화는 geometry 진단으로만 다루고, 성능 주장은 별도 metric 검증으로 한다.
- Validation split 사용 방식(weight update / hparam selection / calibration / 시각화 진단)을 명시한다.
- 분리·AUROC·ΔSR 보고 전에는 `confound-audit` skill을 적용한다.

## 4. Git And PR Workflow

- Branch·PR 컨벤션은 CLAUDE.md 개발 컨벤션을 따른다. 기본 PR target은 `dev`.
- Mixed worktree에서는 파일 단위로 stage하고, 관련 없는 기존 user change는 보존한다.
- 커밋 전 `git diff --check`. 커밋 메시지·PR 제목/본문은 한글 (한글 prefix).
- PR 전 확인: `git status -sb`, `git log origin/dev..HEAD --oneline`, `git diff --stat origin/dev..HEAD`.
  Base 대비 커밋이 없으면 PR 작업을 멈추고 보고한다. PR 번호가 있으면 기존 제목/본문에 새
  변경분만 반영한다. `gh`가 막히면 compare URL + 한글 제목/본문을 제공한다.

플러그인 경계 (superpowers):

- 커밋 분할·메시지·브랜치 마무리·PR은 본 §4과 `commitor` 에이전트가 단일 출처다.
  superpowers의 `finishing-a-development-branch` 및 커밋류 절차는 쓰지 않는다.
- 완료 전 test-gating은 superpowers 기본 `pytest`가 아니라 이 repo의 Docker/env-split 실행
  경로를 따른다.
- 나머지 방법론 스킬(brainstorming/TDD/debugging/planning/verification)은 그대로 활용한다.

PR 본문 구조:

```markdown
요약

- ...

주요 변경

- ...

검증

- ...

참고

- ...
```
