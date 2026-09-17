#!/usr/bin/env bash
# Codex 호출 고정 wrapper — 레인별로 sandbox 등급이 고정된다.
#   ask/resume/review  : --sandbox read-only       (읽기 전용)
#   write/write_resume : --sandbox workspace-write (2026-09-08 사용자 결정으로 신설)
# 이 파일이 여전히 권한 경계다: settings.json 은 이 경로만 allow 하고,
# 여기서 옵션 인자를 전면 거부해 --dangerously-* 등 임의 플래그 주입을 차단한다.
# sandbox 등급은 레인이 정하며 호출자가 인자로 바꿀 수 없다.
# 실행 규약 단일 출처: .claude/skills/codex-collab/SKILL.md
# 설계 배경: docs/superpowers/specs/2026-07-10-codex-collab-design.md §5·§7
#
# 사용법 (repo root에서):
#   scripts/utils/codex_ask.sh ask <prompt_file> <out_dir>
#   scripts/utils/codex_ask.sh resume <thread_id> <prompt_file> <out_dir>
#   scripts/utils/codex_ask.sh write <prompt_file> <out_dir> [cd_path]
#   scripts/utils/codex_ask.sh write_resume <thread_id> <prompt_file> <out_dir> [cd_path]
#   scripts/utils/codex_ask.sh review <out_dir> uncommitted [instructions_file]
#   scripts/utils/codex_ask.sh review <out_dir> base <ref> [instructions_file]
#   scripts/utils/codex_ask.sh review <out_dir> commit <sha> [instructions_file]
#
# ask/resume: transport 성공(exit 0 + 최종 turn.completed) 시 0 반환,
#   stdout 마지막 줄 = thread_id. 응답은 <out_dir>/reply.md.
# CODEX_ASK_DRY_RUN=1: 실행 없이 조립된 argv 출력 (테스트·업그레이드 재검증).
set -euo pipefail

die() { echo "codex_ask: $*" >&2; exit 2; }

# 보안 경계: 어떤 인자도 옵션 형태('-'로 시작)를 허용하지 않는다.
for a in "$@"; do
  case "$a" in -*) die "옵션 인자 금지: $a (플래그는 wrapper가 고정)" ;; esac
done

dry_or_run() { # argv... — dry-run이면 출력 후 0, 아니면 1 반환(실행은 호출부에서)
  if [[ "${CODEX_ASK_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"; echo
    return 0
  fi
  return 1
}

run_exec() { # $1=prompt_file $2=out_dir [$3=thread_id] — 읽기 전용 레인
  run_lane read-only "" "$@"
}

run_write() { # $1=prompt_file $2=out_dir [$3=thread_id] — 쓰기 레인, cd 는 WRITE_CD
  run_lane workspace-write "${WRITE_CD:-}" "$@"
}

run_lane() { # $1=sandbox $2=cd_path $3=prompt_file $4=out_dir [$5=thread_id]
  local sandbox="$1" cd_path="$2" prompt="$3" out="$4" thread="${5:-}"
  [[ -f "$prompt" ]] || die "prompt 파일 없음: $prompt"
  local argv=(codex exec --sandbox "$sandbox" --json -o "$out/reply.md")
  if [[ -n "$cd_path" ]]; then
    [[ -d "$cd_path" ]] || die "cd 경로가 디렉토리가 아님: $cd_path"
    argv+=(--cd "$cd_path")
  fi
  [[ -n "$thread" ]] && argv+=(resume "$thread")
  argv+=(-)
  dry_or_run "${argv[@]}" && return 0
  mkdir -p "$out"
  local rc=0
  "${argv[@]}" < "$prompt" > "$out/events.jsonl" 2>"$out/stderr.log" || rc=$?
  if [[ $rc -ne 0 ]] || ! tail -n1 "$out/events.jsonl" | grep -q '"type":"turn.completed"'; then
    die "transport 실패 (rc=$rc, 최종 이벤트: $(tail -n1 "$out/events.jsonl" 2>/dev/null | cut -c1-160))"
  fi
  grep -m1 -o '"thread_id":"[^"]*"' "$out/events.jsonl" | cut -d'"' -f4
}

mode="${1:-}"; [[ -n "$mode" ]] || die "mode 필요 (ask|resume|write|write_resume|review)"
shift

case "$mode" in
  ask)
    [[ $# -eq 2 ]] || die "사용법: ask <prompt_file> <out_dir>"
    run_exec "$1" "$2"
    ;;
  resume)
    [[ $# -eq 3 ]] || die "사용법: resume <thread_id> <prompt_file> <out_dir>"
    [[ "$1" =~ ^[0-9a-f][0-9a-f-]{20,40}$ ]] || die "thread_id 형식 오류: $1"
    run_exec "$2" "$3" "$1"
    ;;
  write)
    # 쓰기 레인 — codex 가 파일을 직접 수정한다. cd_path 생략 시 현재 디렉토리.
    [[ $# -eq 2 || $# -eq 3 ]] || die "사용법: write <prompt_file> <out_dir> [cd_path]"
    WRITE_CD="${3:-}" run_write "$1" "$2"
    ;;
  write_resume)
    [[ $# -eq 3 || $# -eq 4 ]] || die "사용법: write_resume <thread_id> <prompt_file> <out_dir> [cd_path]"
    [[ "$1" =~ ^[0-9a-f][0-9a-f-]{20,40}$ ]] || die "thread_id 형식 오류: $1"
    WRITE_CD="${4:-}" run_write "$2" "$3" "$1"
    ;;
  review)
    [[ $# -ge 2 ]] || die "사용법: review <out_dir> <uncommitted|base|commit> [ref] [instructions_file]"
    out="$1" target="$2"; shift 2
    argv=(codex review)
    case "$target" in
      uncommitted) argv+=(--uncommitted) ;;
      base)   ref="${1:-}"; [[ "$ref" =~ ^[0-9a-zA-Z/._-]+$ ]] || die "base ref 형식 오류: ${ref:-빈값}"; argv+=(--base "$ref"); shift ;;
      commit) sha="${1:-}"; [[ "$sha" =~ ^[0-9a-f]{7,40}$ ]] || die "commit sha 형식 오류: ${sha:-빈값}"; argv+=(--commit "$sha"); shift ;;
      *) die "review 대상 오류: $target (uncommitted|base|commit)" ;;
    esac
    inst="${1:-}"
    if [[ -n "$inst" ]]; then
      [[ -f "$inst" ]] || die "instructions 파일 없음: $inst"
      # codex 0.145.0 실증(2026-07-27): --uncommitted/--base/--commit 전부 PROMPT 와
      # clap conflict — scope 플래그 있는 리뷰엔 instructions 를 줄 수 없다.
      # (0.144.x 에선 uncommitted+PROMPT 허용이었음 — 업그레이드 재검증에서 재발 확인.)
      # instructions 가 필요하면 ask 레인에 focus+diff 를 인라인으로 넣어라.
      die "codex 0.145.0 제약: scope 리뷰는 instructions 동시 사용 불가 — 생략하거나 ask 레인 사용"
      [[ "$(head -c1 "$inst")" != "-" ]] || die "instructions 파일은 '-' 로 시작 금지"
      argv+=("$(cat "$inst")")
    fi
    dry_or_run "${argv[@]}" && exit 0
    mkdir -p "$out"
    rc=0
    "${argv[@]}" > "$out/review.md" 2>"$out/stderr.log" || rc=$?
    [[ $rc -eq 0 ]] || die "review 실패 (rc=$rc): $(tail -n3 "$out/stderr.log" 2>/dev/null | tr '\n' ' ')"
    ;;
  *) die "unknown mode: $mode (ask|resume|write|write_resume|review)" ;;
esac
