#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${ROOT_DIR:=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang}"
: "${SGLANG_HOST:=127.0.0.1}"
: "${SGLANG_PORT:=30000}"
: "${SERVED_MODEL_NAME:=qwen3.6-27b}"

: "${CONTEXT_LENGTH:=262144}"
: "${MAX_RUNNING_REQUESTS:=4}"
: "${MAX_RUNNING_REQUESTS_CAP:=4}"
: "${MAX_OUTPUT_TOKENS:=128000}"
: "${MEMORY_TARGET_FRACTION:=0.65}"

: "${LOG_REQUESTS:=0}"
: "${LOG_REQUESTS_LEVEL:=0}"
: "${EXPORT_METRICS_TO_FILE:=0}"
: "${ENABLE_REQUEST_TIME_STATS_LOGGING:=0}"
: "${ENABLE_METRICS:=1}"
: "${ENABLE_MFU_METRICS:=0}"

: "${LOG_DIR:=${ROOT_DIR}/logs/qwen36_27b_agent_online}"
: "${PID_FILE:=${LOG_DIR}/qwen36_27b_agent_online.pid}"

export \
  ROOT_DIR \
  SGLANG_HOST \
  SGLANG_PORT \
  SERVED_MODEL_NAME \
  CONTEXT_LENGTH \
  MAX_RUNNING_REQUESTS \
  MAX_RUNNING_REQUESTS_CAP \
  MAX_OUTPUT_TOKENS \
  MEMORY_TARGET_FRACTION \
  LOG_REQUESTS \
  LOG_REQUESTS_LEVEL \
  EXPORT_METRICS_TO_FILE \
  ENABLE_REQUEST_TIME_STATS_LOGGING \
  ENABLE_METRICS \
  ENABLE_MFU_METRICS \
  LOG_DIR \
  PID_FILE

: "${QWEN36_AGENT_START_SCRIPT:=${SCRIPT_DIR}/start_qwen36_27b_agent.sh}"

if [[ ! -x "$QWEN36_AGENT_START_SCRIPT" ]]; then
  echo "Qwen3.6 agent start script is not executable: ${QWEN36_AGENT_START_SCRIPT}" >&2
  exit 1
fi

exec "$QWEN36_AGENT_START_SCRIPT" "$@"
