#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${ROOT_DIR:=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang}"
: "${SGLANG_PORT:=30000}"
: "${MODEL_PATH:=/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B}"
: "${SERVED_MODEL_NAME:=qwen3.6-27b}"
: "${LOG_DIR:=${ROOT_DIR}/logs/qwen36_27b_agent_online}"
: "${PID_FILE:=${LOG_DIR}/qwen36_27b_agent_online.pid}"
: "${STOP_TIMEOUT_SECONDS:=60}"

export \
  ROOT_DIR \
  SGLANG_PORT \
  MODEL_PATH \
  SERVED_MODEL_NAME \
  LOG_DIR \
  PID_FILE \
  STOP_TIMEOUT_SECONDS

: "${QWEN36_AGENT_STOP_SCRIPT:=${SCRIPT_DIR}/stop_qwen36_27b_agent.sh}"

if [[ ! -x "$QWEN36_AGENT_STOP_SCRIPT" ]]; then
  echo "Qwen3.6 agent stop script is not executable: ${QWEN36_AGENT_STOP_SCRIPT}" >&2
  exit 1
fi

exec "$QWEN36_AGENT_STOP_SCRIPT" "$@"
