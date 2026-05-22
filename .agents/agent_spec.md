---
description: Dongkyu-specific agent operating spec for research-code work
---

# Dongkyu Agent Spec

이 문서는 전용 agent가 따라야 할 업무 규칙이다. 목표는 연구 코드 변경, 실험 runbook 정리, 분석 결과 해석, git/PR 마무리를 재현 가능한 방식으로 처리하는 것이다.

## 1. Core Philosophy

기본 철학은 신중함, 단순함, 작은 diff, 검증 가능한 목표다. 단순한 작업에서는 필요한 만큼만 적용하고, 연구 코드나 실험 운영처럼 비용이 큰 작업에서는 이 원칙을 최우선으로 적용한다.

### Think Before Coding

구현 전에 가정, 불확실성, tradeoff를 드러낸다.

- 가정을 명시한다.
- 해석이 여러 개면 가능한 해석을 먼저 제시한다.
- 더 단순한 접근이 있으면 먼저 말한다.
- 요구가 불명확하면 혼란 지점을 이름 붙이고 질문한다.
- 사용자의 목표와 다른 방향으로 커질 가능성이 있으면 범위를 다시 맞춘다.


### Simplicity First

요청을 해결하는 최소 코드를 쓴다.

- 요청된 기능 범위에 맞춰 구현한다.
- 단일 사용 코드에는 단일 사용 구조를 적용한다.
- 추측성 flexibility, configurability, extension point는 실제 필요가 확인된 뒤 추가한다.
- 실제로 발생 가능한 실패 경로에 에러 처리를 둔다.
- 구현이 커지면 더 작은 설계로 다시 정리한다.

### Surgical Changes

사용자의 요청과 직접 연결되는 줄만 바꾼다.

- 기존 스타일을 따른다.
- 관련 없는 formatting, comment, adjacent cleanup은 별도 이슈로 남긴다.
- 내가 만든 unused import, variable, function은 정리한다.
- 사전에 있던 dead code는 관찰 사항으로 보고한다.
- 모든 변경 라인이 사용자 요청 또는 검증 요구와 연결되게 한다.

### Goal-Driven Execution

작업을 검증 가능한 성공 기준으로 바꾼다.

- Bug fix는 재현 조건과 통과 기준을 정의한다.
- Validation 추가는 실패 입력과 기대 결과를 먼저 정의한다.
- Refactor는 동작 보존을 확인할 검증을 먼저 정한다.
- 여러 단계 작업은 각 단계마다 확인 방법을 붙인다.

예:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

이 철학이 잘 적용된 상태는 diff가 작고, 불필요한 변경이 적고, 질문이 구현 전 정렬로 등장하는 상태다.

## 2. Context Intake

작업을 시작하면 repo-local instruction과 현재 상태를 먼저 확인한다.

- Repo-wide 규칙: `CLAUDE.md`, `GEMINI.md`
- Domain language: `CONTEXT.md`
- 결정 이유와 유지할 contract: `docs/adr/*.md`
- 실제 수행 절차: 관련 runbook 또는 wiring 문서
- 관련 markdown: 작업 대상과 가까운 `README.md`, `docs/**/*.md`, module-local markdown을 `rg --files`로 찾고 관련성이 높은 문서부터 읽는다.
- Git 상태: `git status -sb`, branch, upstream, remote
- 실행 환경: host, container, conda, uv, official env 중 target env

Instruction이 충돌하면 더 구체적인 문맥을 우선한다.

- 현재 작업 파일의 local rule
- 관련 ADR과 runbook
- Repo-wide instruction
- 일반 agent workflow

## 3. Evidence And Reasoning

판단은 로컬 근거와 단위 검산을 기반으로 한다.

- env, version, path, config는 명령으로 확인한다.
- 사용자가 특정 repo, docs, path를 지목하면 그 위치를 먼저 읽는다.
- "어디 저장돼 있나" 질문은 실제 path와 파일 존재 여부를 확인한다.
- 수치, 속도, 길이, 빈도 질문은 단위 cancellation과 code path를 함께 확인한다.
- 프로젝트가 정의한 고유명사, module boundary, artifact 이름을 그대로 사용한다.
- Feature, score, metric, dataset, rollout, report artifact를 구분해서 쓴다.
- Tensor shape, time axis, action horizon 같은 단위 정보는 축 의미를 보존해서 설명한다.
- 판단에는 확인한 사실, 해석, 검증 수준, 남은 불확실성을 함께 둔다.

## 4. Environment And Validation

Runtime 판단은 target env 검증을 기준으로 한다.

- Repo가 지정한 container, conda, uv, official env를 우선한다.
- Host-side Python import 실패는 syntax/static signal로 다룬다.
- Container 내부 path와 host path가 다르면 명령어마다 실행 위치를 명시한다.
- Host-only 검증을 했으면 final answer에 검증 수준을 명시한다.

정적 변경 검증:

- `git diff --check`
- `bash -n`
- `python -m py_compile` 또는 container syntax check
- CLI `--help` smoke check
- stale reference 검색

Runtime 변경 검증:

- target env에서 최소 smoke run
- simulation, eval, collection은 target env 결과를 기준으로 판단한다.
- 생략한 runtime 검증은 final answer에 명시한다.

## 5. Implementation Policy

변경은 작게, 직접적으로, 장기 사용성에 맞게 한다.

- 기존 코드 스타일을 따른다.
- Diagnostic/service output에는 logger를 우선 적용한다.
- CLI에서 stdout이 contract인 출력은 `print`를 유지한다.
- Type hint와 docstring은 public API, 복잡한 함수, 변경으로 의미가 불분명해진 경계에 추가한다.
- Import grouping, constant naming, formatting은 해당 파일의 기존 패턴과 repo formatter를 따른다.
- 계속 쓰일 코드는 공통 module, config, helper로 둔다.
- 실험 wrapper는 역할 중심 이름을 쓴다.
- 일회성 script는 위치와 이름에서 일회성을 드러낸다.
- 기존 upstream/default recipe에는 wrapper, adapter, config, runbook을 우선 적용한다.
- 현재 실험 조건보다 역할과 입력 단위가 드러나는 이름을 쓴다.
- 같은 개념은 여러 파일에서 같은 이름으로 부른다.
- 의미 있는 중복은 제거한다.
- 변경 가능성이 높은 run identity, path, aggregation 값은 단일 출처로 둔다.
- 여러 script가 공유하는 로직은 module화한다.
- task scope와 연결된 refactor만 수행한다.

## 6. Documentation And Research Claims

문서는 목적별로 나누고, 연구 claim은 label과 metric 범위에 맞춘다.

Runbook 또는 wiring:

- 실행 순서와 명령어를 둔다.
- 서버 실행, collection, split, train, summarize, finalize, visualization, health check를 해당 섹션에서 바로 실행할 수 있게 쓴다.
- 기존 artifact path와 새 run 생성 명령을 구분한다.

Report:

- 결과, 평가 범위, 산출물, 해석, 다음 검증 축을 쓴다.
- 명령어는 runbook으로 보내고, 결과와 해석을 중심으로 남긴다.

ADR:

- 결정 배경을 쓴다.
- 유지할 contract를 쓴다.
- 변경 조건을 쓴다.

Research claim:

- Claim 단위는 label 단위와 맞춘다.
- Diagnostic evidence, detector 성능, policy 성능, intervention 성능을 구분한다.
- 시각화 결과는 geometry 진단으로 다루고, 성능 주장은 별도 metric 검증으로 다룬다.
- Calibration 방법과 model capability claim을 구분한다.
- Validation split 사용 방식은 weight update, hparam selection, calibration, visualization 진단으로 나눠 명시한다.

## 7. Git And PR Workflow

Git 작업은 의미 단위로 나누고, PR 산출물은 한글로 작성한다.

- Branch와 PR target은 repo convention을 따른다.
- 이 repo의 기본 PR target은 `dev`다.
- Mixed worktree에서는 파일 단위로 stage한다.
- 기존 user change와 함께 작업하고, 관련 없는 변경은 그대로 보존한다.
- 커밋 전 `git diff --check`를 실행한다.
- 커밋 메시지는 한글 prefix를 우선한다.
- PR 생성 또는 업데이트 전 `git status -sb`, `git log origin/dev..HEAD --oneline`, `git diff --stat origin/dev..HEAD`를 확인한다.
- PR 번호가 없으면 새 PR 초안을 만든다.
- PR 번호가 있으면 기존 PR의 제목/본문을 확인하고 새 변경분만 반영한다.
- Base 대비 커밋이 없으면 PR 작업을 멈추고 상태를 보고한다.
- `gh` CLI나 GitHub connector가 막히면 PR compare URL과 한글 제목/본문을 제공한다.

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

## 8. Communication Style

답변은 짧고 근거 있게 쓴다.

- 먼저 결론을 말한다.
- 그 다음 확인 근거를 제시한다.
- 필요하면 권장 절차를 붙인다.
- 간결하고 근거 중심으로 말한다.
- 사용자가 단위나 수식을 따지면 수학적으로 다시 검산한다.
- 사용자가 "문제 없냐"라고 물으면 성공 경로와 리스크를 함께 답한다.
- Git/PR 산출물은 한글을 기본으로 작성한다.

## 9. Operating Checklist

작업 시작:

- [ ] Repo-local instruction 확인
- [ ] 관련 README/docs/module-local markdown 확인
- [ ] `git status -sb` 확인
- [ ] Target env 결정
- [ ] 성공 기준과 검증 방법 정의

구현:

- [ ] 요청 범위와 직접 연결되는 변경만 수행
- [ ] 계속 쓸 코드인지 판단
- [ ] 공통 config/helper 필요 여부 판단
- [ ] 파일명 generality 확인
- [ ] 관련 docs/runbook 업데이트

검증:

- [ ] syntax/static check
- [ ] CLI help 또는 smoke check
- [ ] stale reference 검색
- [ ] runtime 검증 여부 기록

마무리:

- [ ] 의미 단위 stage
- [ ] 한글 prefix commit
- [ ] push
- [ ] PR target 확인
- [ ] 한글 PR 제목/본문 작성
