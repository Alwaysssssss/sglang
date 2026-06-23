#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang}"
SGLANG_PY="${SGLANG_PY:-/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3}"
MODEL_PATH="${MODEL_PATH:-/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"
API_KEY_FILE="${API_KEY_FILE:-/etc/sglang/qwen36_openai_api_key}"
if [[ -z "${OPENAI_API_KEY:-}" && -f "$API_KEY_FILE" ]]; then
  OPENAI_API_KEY="$(tr -d '[:space:]' < "$API_KEY_FILE")"
else
  OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
fi
ALLOW_EMPTY_API_KEY="${ALLOW_EMPTY_API_KEY:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1048576}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-flashinfer}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/qwen36_27b}"
PID_FILE="${PID_FILE:-${LOG_DIR}/qwen36_27b.pid}"
WAIT_FOR_READY="${WAIT_FOR_READY:-1}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-600}"

cd "$ROOT_DIR"

if [[ ! -x "$SGLANG_PY" ]]; then
  echo "SGLang python is not executable: $SGLANG_PY" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ "$OPENAI_API_KEY" == "EMPTY" ]]; then
  if [[ "$ALLOW_EMPTY_API_KEY" != "1" ]]; then
    echo "OPENAI_API_KEY is EMPTY. Create $API_KEY_FILE or set OPENAI_API_KEY." >&2
    echo "For local-only testing, set ALLOW_EMPTY_API_KEY=1 explicitly." >&2
    exit 1
  fi
  echo "Warning: OPENAI_API_KEY is EMPTY. This is only appropriate for local testing." >&2
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE" || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Service appears to be running already. PID: $old_pid" >&2
    exit 1
  fi
fi

if lsof -nP -iTCP:"$SGLANG_PORT" -sTCP:LISTEN >/tmp/qwen36_port_check.$$ 2>/dev/null; then
  echo "Port $SGLANG_PORT is already in use:" >&2
  cat /tmp/qwen36_port_check.$$ >&2
  rm -f /tmp/qwen36_port_check.$$
  exit 1
fi
rm -f /tmp/qwen36_port_check.$$

mkdir -p "$LOG_DIR"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${LOG_DIR}/qwen36_27b_tp${TP_SIZE}_128k_${stamp}.log"

redact_log_stream() {
  sed -u -E \
    -e "s/api_key='[^']*'/api_key='<redacted>'/g" \
    -e "s/admin_api_key='[^']*'/admin_api_key='<redacted>'/g" \
    -e "s/(--api-key )[[:graph:]]+/\1<redacted>/g"
}

setsid env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  "$SGLANG_PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$SGLANG_HOST" \
  --port "$SGLANG_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --dtype "$DTYPE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --sampling-backend "$SAMPLING_BACKEND" \
  --api-key "$OPENAI_API_KEY" \
  --disable-piecewise-cuda-graph \
  > >(redact_log_stream > "$log_file") 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "Started Qwen3.6-27B SGLang service"
echo "PID: $pid"
echo "Log: $log_file"
echo "Base URL: http://${SGLANG_HOST}:${SGLANG_PORT}/v1"
echo "Model: $SERVED_MODEL_NAME"

if [[ "$WAIT_FOR_READY" != "1" ]]; then
  exit 0
fi

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Process exited before readiness. Last log lines:" >&2
    tail -n 120 "$log_file" >&2 || true
    exit 1
  fi

  if curl --noproxy '*' -fsS \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    "http://${SGLANG_HOST}:${SGLANG_PORT}/health" >/dev/null 2>&1; then
    echo "Service is ready"
    exit 0
  fi

  sleep 2
done

echo "Timed out waiting for readiness after ${READY_TIMEOUT_SECONDS}s. Last log lines:" >&2
tail -n 120 "$log_file" >&2 || true
exit 1
