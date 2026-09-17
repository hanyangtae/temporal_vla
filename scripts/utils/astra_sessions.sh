#!/usr/bin/env bash
# Astra terminal attached to the same local App Server used by session-bridge.
set -euo pipefail
bridge_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
bridge_codex_root="${CODEX_HOME:-${HOME}/.codex}"
bridge_socket="${SESSION_BRIDGE_CODEX_SOCKET:-${bridge_codex_root}/app-server-control/app-server-control.sock}"
uv run --script "${bridge_dir}/session_bridge.py" start-codex-server >&2
exec codex --remote "unix://${bridge_socket}" -m gpt-6-astra "$@"
