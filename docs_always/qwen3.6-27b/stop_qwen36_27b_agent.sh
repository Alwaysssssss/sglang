#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/qwen36_27b_agent}"
PID_FILE="${PID_FILE:-${LOG_DIR}/qwen36_27b_agent.pid}"
STOP_TIMEOUT_SECONDS="${STOP_TIMEOUT_SECONDS:-60}"

stop_pid() {
  local pid="$1"
  local cmd

  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  cmd="$(ps -p "$pid" -o cmd= || true)"
  if [[ "$cmd" != *"sglang.launch_server"* ]] || {
    [[ "$cmd" != *"$MODEL_PATH"* ]] && [[ "$cmd" != *"$SERVED_MODEL_NAME"* ]]
  }; then
    echo "Refusing to stop PID $pid because it does not look like this Qwen3.6 agent service:" >&2
    echo "$cmd" >&2
    return 1
  fi

  echo "Stopping PID $pid"
  kill "$pid"

  local deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Stopped PID $pid"
      return 0
    fi
    sleep 1
  done

  echo "PID $pid did not exit after ${STOP_TIMEOUT_SECONDS}s; sending SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
}

stopped=0

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' < "$PID_FILE" || true)"
  if [[ -n "${pid:-}" ]]; then
    stop_pid "$pid"
    stopped=1
  fi
  rm -f "$PID_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
  port_check_file="${TMPDIR:-/tmp}/qwen36_agent_listen.$$"
  if lsof -nP -iTCP:"$SGLANG_PORT" -sTCP:LISTEN >"$port_check_file" 2>/dev/null; then
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      stop_pid "$pid"
      stopped=1
    done < <(lsof -tiTCP:"$SGLANG_PORT" -sTCP:LISTEN)
  fi
  rm -f "$port_check_file"
fi

if [[ "$stopped" == "0" ]]; then
  echo "No matching Qwen3.6 agent SGLang service found"
else
  echo "Stop command completed"
fi
