---
name: feedback_push_auth_failure
description: git push over HTTPS can fail here with "could not read Username for 'https://github.com'" — no credential helper in this agent shell; report, don't retry
type: feedback
---

On 2026-07-13, `git push origin <branch>` failed identically for both the
main repo (`hanyangtae/temporal_vla`) and the `src/benchmarks/robocasa`
submodule fork (`hanyangtae/robocasa`), both origin remotes over HTTPS:

```
fatal: could not read Username for 'https://github.com': No such device or address
```

**Why:** This agent's shell has no interactive terminal and no git
credential helper configured for github.com, so any HTTPS push that needs
auth fails immediately — not a network or remote-permission issue, just
nothing to prompt a username with. Same host-capability gap as
[[feedback_no_pr_creation]] (no `gh` CLI either).

**How to apply:** Do not retry a failed push or try workarounds (`gh auth
login`, credential.helper hacks, switching remotes). This agent's job ends
at attempting `git push`; report the exact failure output verbatim and
stop. The user or a separately-authenticated session handles the actual
push. Do not assume push "always works just because it's git" — verify by
checking the actual command output before reporting success.
