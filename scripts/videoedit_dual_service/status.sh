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

process_status() {
  local name="$1"
  local pid_file="$2"
  if [[ -r "$pid_file" ]]; then
    local pid
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$name: running (PID $pid)"
      return
    fi
  fi
  echo "$name: stopped"
}

health() {
  local name="$1"
  local url="$2"
  local response
  response="$(curl -fsS --max-time 2 "$url" 2>/dev/null || true)"
  if [[ -n "$response" ]]; then
    echo "$name health: $response"
  else
    echo "$name health: unavailable"
  fi
}

process_status gateway "$PID_DIR/gateway.pid"
process_status normal "$PID_DIR/normal.pid"
process_status dmd "$PID_DIR/dmd.pid"
health gateway "http://127.0.0.1:${GATEWAY_PORT}/health"
health normal "http://127.0.0.1:${NORMAL_PORT}/health"
health dmd "http://127.0.0.1:${DMD_PORT}/health"

curl -fsS --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/admin/queue?limit=10" 2>/dev/null || true
echo
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || true
grep -E '^(MemAvailable|SwapFree):' /proc/meminfo || true
for file in memory.current memory.max memory.events; do
  [[ -r "/sys/fs/cgroup/$file" ]] && { echo "$file:"; cat "/sys/fs/cgroup/$file"; }
done
