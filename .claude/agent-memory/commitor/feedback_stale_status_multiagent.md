---
name: feedback_stale_status_multiagent
description: In multi-agent sessions, the git status/diff snapshot given at task launch can already be stale by the time this agent runs — re-verify before committing
type: feedback
---

This session's launch prompt listed several files as pending (e.g.
`docs/steering/17_steering_experiment_redesign.md`,
`docs/steering/18_apple_success_rejudge.md`,
`docs/superpowers/specs/2026-07-10-codex-collab-design.md`,
`scripts/safe/groot_n15/robocasa/eval/rejudge_success.py`,
`scripts/safe/groot_n15/robocasa/steer/aggregate_final_scene.py`) as
uncommitted/modified. By the time this agent actually ran `git status` /
`git log`, all of those had already been committed by another concurrently
active agent in the same session (this session had `main`, `48`, `fork`
addressable via SendMessage alongside this one).

**Why:** The `<system-reminder>` git-status block at conversation start is a
one-time snapshot, not live. In a session with multiple parallel agents
touching the same repo, files can be committed between snapshot time and
this agent's execution — re-committing them (or committing based on the
stale file list) would be wrong or redundant.

**How to apply:** Never trust the launch-time git status snapshot as
ground truth for what to commit. Always re-run `git status` / `git log
--oneline -1 -- <path>` for every file the task asks about immediately
before staging. If a named file's log shows a commit newer than the task
context implies, treat it as already handled — skip it, don't re-add or
re-message it. Only act on what the *current* `git status` actually shows.
See also the global memory "Verify before relay" (agent claims must be
checked against primary evidence before being treated as fact) — this is
the same principle applied specifically to git state.
