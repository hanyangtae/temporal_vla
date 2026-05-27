# AGENTS.md

이 repo에서 Codex 계열 agent는 작업을 시작할 때 `.agents/agent_spec.md`를 repo-local 운영 규칙으로 한 번 확인하고 따른다.

- `.agents/agent_spec.md`는 agent의 작업 방식, 검증 기준, 문서화 방식, git/PR 절차의 단일 기준이다.
- `CLAUDE.md`는 Claude Code용 entrypoint이자 프로젝트 구조, 개발 컨벤션, 실행 경로를 설명하는 참고 문서다.
- Codex의 instruction discovery는 `.agents/agent_spec.md`에서 멈추고, `CLAUDE.md`는 프로젝트 참고 문맥으로 사용한다.
- 두 문서의 지침이 충돌하면 더 구체적인 작업 문맥과 현재 repo-local instruction을 우선한다.
