---
name: feedback_no_pr_creation
description: This machine has no gh CLI — commitor's job always ends at push, never attempt PR creation
type: feedback
---

Never run `gh pr create` (or any PR-creation step) in this repo, even when asked
to "정리해서 커밋/푸시" a feature branch. Commit + push to the remote tracking
branch is the full scope of this agent's job here.

**Why:** The user has stated explicitly (twice, across sessions) that this
machine has no `gh` CLI installed. PR creation is handled by the user via a
separate remote-agent session that does have `gh`. This overrides the default
"Creating pull requests" workflow described in this agent's own system prompt
— that section assumes `gh` is available, which is not true on this host.

**How to apply:** Stop right after `git push`. Report the branch/commit(s) and
let the user (or their remote agent) open the PR. Do not suggest running
`gh pr create` as a "next step" either — it will just fail or prompt for auth
that isn't set up here.
