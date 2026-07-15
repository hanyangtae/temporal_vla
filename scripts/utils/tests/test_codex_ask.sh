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
t review-inst      ok  "test prompt"         review /tmp/o base origin/dev "$tmp"
t review-badtarget err "review 대상"         review /tmp/o everything
rm -f "$tmp"

echo "== $pass pass, $fail fail"
[[ $fail -eq 0 ]]
