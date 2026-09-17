#!/usr/bin/env bash
# Source at the beginning of a GR00T runner. Does not replace the runner's traps.
groot_launch_guard() {
  local kind=$1 script=$2; shift 2
  local utils harness machine serves
  utils=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  harness="$utils/groot_harness.py"
  machine=${HARNESS_MACHINE:-}
  if [ -z "$machine" ] && [ "$(hostname -s)" = kanu ]; then machine=kanu; fi
  [ -n "$machine" ] || { echo '[harness] HARNESS_MACHINE=kanu|srv48|srv50 required' >&2; return 2; }
  [ -n "${HARNESS_SESSION:-}" ] || { echo '[harness] HARNESS_SESSION required (actual owning session)' >&2; return 2; }
  case "$machine" in
    kanu) serves=${SERVES_PER_GPU:-2} ;;
    srv48|srv50) serves=${SERVES_PER_GPU:-6} ;;
    *) echo '[harness] unknown machine' >&2; return 2 ;;
  esac
  export HARNESS_MACHINE="$machine" SERVES_PER_GPU="$serves"
  local args=(--machine "$machine" --gpus "${GPUS:?GPUS required}" --serves "$serves" --session "$HARNESS_SESSION" --kind "$kind")
  if [ "$kind" = collect ]; then
    [ -n "${HARNESS_COLLECTION_CONTRACT:-}" ] || { echo '[harness] HARNESS_COLLECTION_CONTRACT required' >&2; return 2; }
    python3 "$harness" check-collection --contract "$HARNESS_COLLECTION_CONTRACT" \
      --plan "${PLAN_JSON:?}" --instructions "${INSTRUCTIONS:?}" --noise-limit "${NOISE_LIMIT:-10}" \
      --done-list "${DONE_LIST:?explicit parent DONE_LIST required}" --grid-root "${GRID_ROOT:?}" --machine "$machine" >&2 || return $?
  fi
  if [ "${DRY_RUN:-0}" = 1 ]; then
    python3 "$harness" check-resources "${args[@]}" >&2
  elif [ -n "${HARNESS_RECEIPT:-}" ]; then
    python3 "$harness" verify "${args[@]}" --receipt "$HARNESS_RECEIPT" --consumer-pid "$$" >&2
  else
    exec python3 "$harness" run "${args[@]}" -- bash "$script" "$@"
  fi
}
