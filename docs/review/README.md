# 레포 검토 작업 공간

설계: [`../superpowers/specs/2026-07-28-repo-review-design.md`](../superpowers/specs/2026-07-28-repo-review-design.md)

- `S<N>_*.md` — 스테이지 카드. 스테이지 진입 직전에 생성한다. 판정 열은 사용자가 채운다.
- `LEDGER.tsv` — 전 스테이지 통합 판정 원장. 스테이지 간 판정을 실어 나른다.

복원: 아카이브된 파일은 `git checkout <삭제커밋> -- <경로>` 로 되살린다.
