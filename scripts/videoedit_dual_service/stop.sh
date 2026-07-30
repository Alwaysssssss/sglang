#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${VIDEOEDIT_DUAL_CONFIG:-${SCRIPT_DIR}/config.env}"
if [[ ! -r "$CONFIG_FILE" ]]; then
  echo "Missing readable config: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

stop_one() {
  local name="$1"
  local pid_file="$2"
  local expected="$3"
  [[ -r "$pid_file" ]] || return 0
  local pid cmdline
  pid="$(<"$pid_file")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "Ignoring invalid $name pidfile: $pid_file" >&2
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$pid_file"
    return 0
  fi
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "$cmdline" != *"$expected"* ]]; then
    echo "Refusing to stop PID $pid: it is not the expected $name process." >&2
    return 1
  fi
  echo "Stopping $name (PID $pid)..."
  kill -TERM "$pid"
  for _ in {1..60}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f -- "$pid_file"
      return 0
    fi
    sleep 1
  done
  echo "$name did not exit after 60 seconds; leaving it for manual inspection." >&2
  return 1
}

result=0
stop_one gateway "$PID_DIR/gateway.pid" "dual_service_gateway" || result=1
stop_one dmd "$PID_DIR/dmd.pid" "--port ${DMD_PORT}" || result=1
stop_one normal "$PID_DIR/normal.pid" "--port ${NORMAL_PORT}" || result=1
exit "$result"
