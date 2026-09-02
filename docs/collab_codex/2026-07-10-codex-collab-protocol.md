# Claude×Codex 협업 프로토콜 설계 — Claude×Codex 토론

- 게이트: 계획 토론 (Gate 1 형식의 첫 실전 — 프로토콜 설계 자체를 리뷰 대상으로)
- 대상 문서: `docs/superpowers/specs/2026-07-10-codex-collab-design.md`
- thread_id: R1 `019f4af1-e4ec-72a0-8cd8-70c0e313de42` / R2 `019f4afd-8ea2-7463-8e09-3090ebe7f0c0`
  (R1 thread는 sandbox 고장 상태의 반쪽 리뷰라 폐기, R2는 새 thread — "thread 수명 = 토론 1건" 규칙 적용)

## Round 1 (R1 초안 리뷰 — 반쪽: Codex가 sandbox 고장으로 문서를 직접 못 읽음)

### Claude → Codex (요지)

협업 프로토콜 초안(단일 창구, read-only 리뷰어, 3게이트, codex exec/resume/review) 비판 리뷰 요청.

### Codex 응답 (핵심)

14건 지적. 검증 후 수용된 핵심: 세션ID는 `thread.started.thread_id`(로컬 확인),
`exec resume` 서브파서에 `--sandbox` 없음(로컬 `--help` 확인 — 초안의 커맨드는 실제로 깨짐),
resume은 최적화일 뿐 전제 금지, thread당 동시 1호출·`--last` 금지,
성공 판정 = exit 0 + `turn.completed`, squash 후 낡은 thread 재사용 금지,
worktree ≠ 보안 경계, Gate 3 발동 기준 상향(작은 디버깅은 패치 제안으로).
기각: "단일 작성자 원칙 vs Gate 3 모순" — two-lane + diff 게이트로 이미 해소.

### 부산물 — 인프라 블로커 발견·수리

Codex 기본 bwrap sandbox가 이 머신의 `kernel.apparmor_restrict_unprivileged_userns=1`과
충돌해 셸 전면 불능(`bwrap: loopback: Failed RTM_NEWADDR`, 무료 재현 확인).
`codex features enable use_legacy_landlock`로 수리 — 읽기 성공·쓰기 차단 검증.
플래그가 deprecated라 codex 업그레이드 시 재확인 필요.

## Round 2 (R2 개정판 리뷰 — 정식: Codex가 문서·CLI help·git 상태를 직접 대조)

### Claude → Codex (요지)

R1 반영분이 정확한지 + 문서를 실제 읽으니 보이는 새 문제 + 구현 착수 가능 판정 요청.

### Codex 응답 (핵심, 9건 전부 수용)

1. [치명적] `Bash(codex exec*)` prefix allow는 `--sandbox workspace-write`·`--dangerously-*`까지
   무승인 통과 — "write는 매번 승인" 의도와 정면 모순 → read-only 고정 wrapper만 allow.
2. [높음] worktree 분기는 미커밋 변경을 못 받음 → 위임 입력을 WIP 커밋으로 고정(delegate_sha).
3. [높음] 무조건 1회 재시도는 write lane에서 중복 수정 위험 → 재시도는 Review lane 전송 실패만.
4. [높음] `turn.completed`는 전송 성공이지 작업 성공이 아님 → transport/task 성공 분리.
5. [중간] "통과분만 squash"의 실제 절차 부재 → merge --squash + `git diff --cached` 검사 /
   부분 채택은 hunk 적용, 검증은 최종 staged 기준.
6. [중간] `--base dev`는 로컬 dev가 낡으면 리뷰 범위 왜곡 → fetch 후 origin/dev SHA 고정.
7. [중간] 병렬 금지가 branch 조작만으론 좁음 → git metadata 변경 명령 전반으로 확대.
8. [낮음] 모든 호출을 원장 기록은 과설계 → 필수는 Gate 1·3만.
9. [낮음] 설계 문서·SKILL.md 이중 규범 → 구현 후 설계 문서는 비규범 기록으로.

**Codex 최종 판정**: 조건부 가능 — 1·2·3 수정 + 권한 매칭 스모크 테스트 후 구현 착수 가능.

## 결론

- 합의: R3에 지적 9건 전부 반영 완료. 남은 조건은 구현 단계의 "권한 매칭 스모크 테스트"(§9-6).
- 이견: 없음.
- 사용자 결정: 접근 A(CLI 직접 호출) 확정, Codex 기본 read-only + Gate 3 worktree 위임 승인,
  구현 착수는 R3 스펙 리뷰 후.
