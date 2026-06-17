#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang}"
SGLANG_PY="${SGLANG_PY:-/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3}"
MODEL_PATH="${MODEL_PATH:-/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"
API_KEY_FILE="${API_KEY_FILE:-/etc/sglang/qwen36_openai_api_key}"
ALLOW_EMPTY_API_KEY="${ALLOW_EMPTY_API_KEY:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-128000}"
MEMORY_TARGET_FRACTION="${MEMORY_TARGET_FRACTION:-0.65}"
RESPECT_CURRENT_GPU_USAGE="${RESPECT_CURRENT_GPU_USAGE:-1}"
MAX_RUNNING_REQUESTS_CAP="${MAX_RUNNING_REQUESTS_CAP:-4}"
KV_BYTES_PER_TOKEN_PER_GPU="${KV_BYTES_PER_TOKEN_PER_GPU:-16384}"
STATIC_OVERHEAD_MIB="${STATIC_OVERHEAD_MIB:-2048}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-flashinfer}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
RADIX_EVICTION_POLICY="${RADIX_EVICTION_POLICY:-lru}"
SAMPLING_DEFAULTS="${SAMPLING_DEFAULTS:-model}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/qwen36_27b_agent}"
PID_FILE="${PID_FILE:-${LOG_DIR}/qwen36_27b_agent.pid}"
WAIT_FOR_READY="${WAIT_FOR_READY:-1}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-900}"
DRY_RUN="${DRY_RUN:-0}"

LOG_LEVEL="${LOG_LEVEL:-info}"
LOG_LEVEL_HTTP="${LOG_LEVEL_HTTP:-warning}"
LOG_REQUESTS="${LOG_REQUESTS:-1}"
LOG_REQUESTS_LEVEL="${LOG_REQUESTS_LEVEL:-2}"
LOG_REQUESTS_FORMAT="${LOG_REQUESTS_FORMAT:-json}"
DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-16}"
ENABLE_REQUEST_TIME_STATS_LOGGING="${ENABLE_REQUEST_TIME_STATS_LOGGING:-1}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
ENABLE_MFU_METRICS="${ENABLE_MFU_METRICS:-0}"
EXPORT_METRICS_TO_FILE="${EXPORT_METRICS_TO_FILE:-1}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"
SHOW_TIME_COST="${SHOW_TIME_COST:-0}"

trim() {
  local value="$*"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

gpu_is_visible() {
  local idx="$1"
  local dev

  if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == "all" ]]; then
    return 0
  fi

  IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  for dev in "${visible_devices[@]}"; do
    dev="$(trim "$dev")"
    if [[ "$dev" == "$idx" ]]; then
      return 0
    fi
  done

  return 1
}

redact_log_stream() {
  sed -u -E \
    -e "s/api_key='[^']*'/api_key='<redacted>'/g" \
    -e "s/admin_api_key='[^']*'/admin_api_key='<redacted>'/g" \
    -e "s/(--api-key )[[:graph:]]+/\1<redacted>/g" \
    -e "s/(Authorization: Bearer )[[:graph:]]+/\1<redacted>/g"
}

quote_redacted_command() {
  local -a redacted=()
  local i=0
  while (( i < ${#server_cmd[@]} )); do
    if [[ "${server_cmd[$i]}" == "--api-key" ]]; then
      redacted+=("--api-key" "<redacted>")
      i=$((i + 2))
      continue
    fi
    redacted+=("${server_cmd[$i]}")
    i=$((i + 1))
  done
  printf '%q ' "${redacted[@]}"
  printf '\n'
}

log() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$START_LOG_FILE"
}

die() {
  log "ERROR: $*"
  exit 1
}

if [[ -z "${OPENAI_API_KEY:-}" && -f "$API_KEY_FILE" ]]; then
  OPENAI_API_KEY="$(tr -d '[:space:]' < "$API_KEY_FILE")"
else
  OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
fi

cd "$ROOT_DIR"

mkdir -p "$LOG_DIR"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
START_LOG_FILE="${START_LOG_FILE:-${LOG_DIR}/qwen36_27b_agent_start_${stamp}.log}"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-${LOG_DIR}/qwen36_27b_agent_tp${TP_SIZE}_256k_${stamp}.log}"
REQUEST_LOG_DIR="${REQUEST_LOG_DIR:-${LOG_DIR}/requests_${stamp}}"
METRICS_FILE_DIR="${METRICS_FILE_DIR:-${LOG_DIR}/metrics_${stamp}}"
CRASH_DUMP_FOLDER="${CRASH_DUMP_FOLDER:-${LOG_DIR}/crash_dumps_${stamp}}"
CLIENT_DEFAULTS_FILE="${CLIENT_DEFAULTS_FILE:-${LOG_DIR}/qwen36_27b_agent_client_defaults_${stamp}.json}"
mkdir -p "$REQUEST_LOG_DIR" "$METRICS_FILE_DIR" "$CRASH_DUMP_FOLDER"
: > "$START_LOG_FILE"

if [[ ! -x "$SGLANG_PY" ]]; then
  die "SGLang python is not executable: $SGLANG_PY"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  die "Model path does not exist: $MODEL_PATH"
fi

if [[ "$OPENAI_API_KEY" == "EMPTY" ]]; then
  if [[ "$ALLOW_EMPTY_API_KEY" != "1" ]]; then
    die "OPENAI_API_KEY is EMPTY. Create $API_KEY_FILE or set OPENAI_API_KEY. For local-only testing, set ALLOW_EMPTY_API_KEY=1 explicitly."
  fi
  log "Warning: OPENAI_API_KEY is EMPTY. This is only appropriate for local testing."
fi

if (( MAX_OUTPUT_TOKENS >= CONTEXT_LENGTH )); then
  die "MAX_OUTPUT_TOKENS (${MAX_OUTPUT_TOKENS}) must be smaller than CONTEXT_LENGTH (${CONTEXT_LENGTH})."
fi

gpu_query_output=""
gpu_count=0
min_gpu_total_mib=0
min_service_budget_mib=0
auto_mem_fraction_static="$MEMORY_TARGET_FRACTION"

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_query_output="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
fi

if [[ -n "$gpu_query_output" ]]; then
  while IFS=',' read -r gpu_idx gpu_name gpu_total_mib gpu_used_mib gpu_free_mib gpu_util; do
    gpu_idx="$(trim "$gpu_idx")"
    gpu_name="$(trim "$gpu_name")"
    gpu_total_mib="$(trim "$gpu_total_mib")"
    gpu_used_mib="$(trim "$gpu_used_mib")"
    gpu_free_mib="$(trim "$gpu_free_mib")"
    gpu_util="$(trim "$gpu_util")"

    [[ "$gpu_total_mib" =~ ^[0-9]+$ ]] || continue
    [[ "$gpu_used_mib" =~ ^[0-9]+$ ]] || continue
    gpu_is_visible "$gpu_idx" || continue

    target_budget_mib="$(awk -v total="$gpu_total_mib" -v used="$gpu_used_mib" -v target="$MEMORY_TARGET_FRACTION" -v respect="$RESPECT_CURRENT_GPU_USAGE" 'BEGIN {
      budget = total * target
      if (respect == "1") {
        budget -= used
      }
      printf "%d", int(budget)
    }')"

    if (( target_budget_mib <= 0 )); then
      die "Visible GPU ${gpu_idx} is already above the target memory fraction ${MEMORY_TARGET_FRACTION}: total=${gpu_total_mib}MiB used=${gpu_used_mib}MiB."
    fi

    gpu_count=$((gpu_count + 1))
    if (( min_gpu_total_mib == 0 || gpu_total_mib < min_gpu_total_mib )); then
      min_gpu_total_mib="$gpu_total_mib"
    fi
    if (( min_service_budget_mib == 0 || target_budget_mib < min_service_budget_mib )); then
      min_service_budget_mib="$target_budget_mib"
    fi
  done <<< "$gpu_query_output"

  if (( gpu_count > 0 )); then
    if (( gpu_count < TP_SIZE )); then
      die "Only ${gpu_count} visible GPUs were detected, but TP_SIZE=${TP_SIZE}."
    fi
    auto_mem_fraction_static="$(awk -v budget="$min_service_budget_mib" -v total="$min_gpu_total_mib" 'BEGIN { printf "%.3f", budget / total }')"
  fi
fi

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-$auto_mem_fraction_static}"

if [[ -z "${MODEL_SIZE_MIB:-}" ]]; then
  MODEL_SIZE_MIB="$(du -sm "$MODEL_PATH" | awk '{print $1}')"
fi

model_shard_mib=$(( (MODEL_SIZE_MIB + TP_SIZE - 1) / TP_SIZE ))
if (( min_gpu_total_mib > 0 )); then
  service_budget_mib="$(awk -v total="$min_gpu_total_mib" -v fraction="$MEM_FRACTION_STATIC" 'BEGIN { printf "%d", int(total * fraction) }')"
else
  service_budget_mib=0
fi

if (( service_budget_mib > 0 && service_budget_mib <= model_shard_mib + 512 )); then
  die "Estimated per-GPU service budget (${service_budget_mib}MiB) is too small for the model shard (${model_shard_mib}MiB). Free GPU memory, raise MEMORY_TARGET_FRACTION, or set RESPECT_CURRENT_GPU_USAGE=0 only if the existing usage is expected to disappear."
fi

if [[ -z "${MAX_RUNNING_REQUESTS:-}" ]]; then
  MAX_RUNNING_REQUESTS_SOURCE="auto_estimate"
  if (( service_budget_mib > 0 )); then
    kv_budget_mib=$(( service_budget_mib - model_shard_mib - STATIC_OVERHEAD_MIB ))
    if (( kv_budget_mib > 0 )); then
      estimated_total_tokens=$(( kv_budget_mib * 1048576 / KV_BYTES_PER_TOKEN_PER_GPU ))
      MAX_RUNNING_REQUESTS=$(( estimated_total_tokens / CONTEXT_LENGTH ))
    else
      estimated_total_tokens=0
      MAX_RUNNING_REQUESTS=1
    fi
    if (( MAX_RUNNING_REQUESTS < 1 )); then
      MAX_RUNNING_REQUESTS=1
    fi
    if (( MAX_RUNNING_REQUESTS_CAP > 0 && MAX_RUNNING_REQUESTS > MAX_RUNNING_REQUESTS_CAP )); then
      MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS_CAP"
      MAX_RUNNING_REQUESTS_SOURCE="auto_estimate_capped"
    fi
  else
    kv_budget_mib=0
    estimated_total_tokens=0
    MAX_RUNNING_REQUESTS=4
    MAX_RUNNING_REQUESTS_SOURCE="fallback_no_gpu_profile"
  fi
else
  kv_budget_mib=0
  estimated_total_tokens=0
  MAX_RUNNING_REQUESTS_SOURCE="explicit"
fi

MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-$((MAX_RUNNING_REQUESTS * 8))}"
PREFILL_MAX_REQUESTS="${PREFILL_MAX_REQUESTS:-$MAX_RUNNING_REQUESTS}"

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(tr -d '[:space:]' < "$PID_FILE" || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "Service appears to be running already. PID: $old_pid"
    fi
  fi

  if command -v lsof >/dev/null 2>&1; then
    port_check_file="${TMPDIR:-/tmp}/qwen36_agent_port_check.$$"
    if lsof -nP -iTCP:"$SGLANG_PORT" -sTCP:LISTEN >"$port_check_file" 2>/dev/null; then
      log "Port $SGLANG_PORT is already in use:"
      cat "$port_check_file" | tee -a "$START_LOG_FILE"
      rm -f "$port_check_file"
      exit 1
    fi
    rm -f "$port_check_file"
  else
    log "Warning: lsof not found; skipping port check for ${SGLANG_PORT}."
  fi
else
  log "DRY_RUN=1; skipping PID and port checks."
fi

cat > "$CLIENT_DEFAULTS_FILE" <<EOF
{
  "base_url": "http://${SGLANG_HOST}:${SGLANG_PORT}/v1",
  "model": "${SERVED_MODEL_NAME}",
  "context_length": ${CONTEXT_LENGTH},
  "max_tokens": ${MAX_OUTPUT_TOKENS},
  "max_completion_tokens": ${MAX_OUTPUT_TOKENS}
}
EOF

server_cmd=(
  "$SGLANG_PY" -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host "$SGLANG_HOST"
  --port "$SGLANG_PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --max-queued-requests "$MAX_QUEUED_REQUESTS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --prefill-max-requests "$PREFILL_MAX_REQUESTS"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --schedule-policy "$SCHEDULE_POLICY"
  --radix-eviction-policy "$RADIX_EVICTION_POLICY"
  --dtype "$DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
  --sampling-backend "$SAMPLING_BACKEND"
  --sampling-defaults "$SAMPLING_DEFAULTS"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --log-level "$LOG_LEVEL"
  --log-level-http "$LOG_LEVEL_HTTP"
  --decode-log-interval "$DECODE_LOG_INTERVAL"
  --uvicorn-access-log-exclude-prefixes /health /metrics
  --crash-dump-folder "$CRASH_DUMP_FOLDER"
  --api-key "$OPENAI_API_KEY"
  --reasoning-parser "$REASONING_PARSER"
)

# if [[ -n "$REASONING_PARSER" ]]; then
#   server_cmd+=(--reasoning-parser "$REASONING_PARSER")
# fi

if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  server_cmd+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

if [[ "$LOG_REQUESTS" == "1" ]]; then
  server_cmd+=(
    --log-requests
    --log-requests-level "$LOG_REQUESTS_LEVEL"
    --log-requests-format "$LOG_REQUESTS_FORMAT"
    --log-requests-target "$REQUEST_LOG_DIR"
  )
fi

if [[ "$ENABLE_REQUEST_TIME_STATS_LOGGING" == "1" ]]; then
  server_cmd+=(--enable-request-time-stats-logging)
fi

if [[ "$ENABLE_METRICS" == "1" ]]; then
  server_cmd+=(--enable-metrics)
fi

if [[ "$ENABLE_MFU_METRICS" == "1" ]]; then
  server_cmd+=(--enable-mfu-metrics)
fi

if [[ "$EXPORT_METRICS_TO_FILE" == "1" ]]; then
  server_cmd+=(--export-metrics-to-file --export-metrics-to-file-dir "$METRICS_FILE_DIR")
fi

if [[ "$DISABLE_PIECEWISE_CUDA_GRAPH" == "1" ]]; then
  server_cmd+=(--disable-piecewise-cuda-graph)
fi

if [[ "$SHOW_TIME_COST" == "1" ]]; then
  server_cmd+=(--show-time-cost)
fi

if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  read -r -a extra_server_args <<< "$EXTRA_SERVER_ARGS"
  server_cmd+=("${extra_server_args[@]}")
fi

log "Qwen3.6-27B agent launch summary"
log "ROOT_DIR=${ROOT_DIR}"
log "MODEL_PATH=${MODEL_PATH}"
log "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "TP_SIZE=${TP_SIZE}"
log "CONTEXT_LENGTH=${CONTEXT_LENGTH}"
log "MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}"
log "MEMORY_TARGET_FRACTION=${MEMORY_TARGET_FRACTION}"
log "RESPECT_CURRENT_GPU_USAGE=${RESPECT_CURRENT_GPU_USAGE}"
log "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
log "MAX_RUNNING_REQUESTS_CAP=${MAX_RUNNING_REQUESTS_CAP}"
log "MAX_RUNNING_REQUESTS_SOURCE=${MAX_RUNNING_REQUESTS_SOURCE}"
log "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
log "MAX_QUEUED_REQUESTS=${MAX_QUEUED_REQUESTS}"
log "PREFILL_MAX_REQUESTS=${PREFILL_MAX_REQUESTS}"
log "CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}"
log "MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS}"
log "TOOL_CALL_PARSER=${TOOL_CALL_PARSER}"
log "REASONING_PARSER=${REASONING_PARSER}"
log "LOG_REQUESTS=${LOG_REQUESTS}"
log "LOG_REQUESTS_LEVEL=${LOG_REQUESTS_LEVEL}"
log "LOG_REQUESTS_FORMAT=${LOG_REQUESTS_FORMAT}"
log "ENABLE_REQUEST_TIME_STATS_LOGGING=${ENABLE_REQUEST_TIME_STATS_LOGGING}"
log "ENABLE_METRICS=${ENABLE_METRICS}"
log "EXPORT_METRICS_TO_FILE=${EXPORT_METRICS_TO_FILE}"
log "REQUEST_LOG_DIR=${REQUEST_LOG_DIR}"
log "METRICS_FILE_DIR=${METRICS_FILE_DIR}"
log "CRASH_DUMP_FOLDER=${CRASH_DUMP_FOLDER}"
log "CLIENT_DEFAULTS_FILE=${CLIENT_DEFAULTS_FILE}"
log "SERVER_LOG_FILE=${SERVER_LOG_FILE}"
log "START_LOG_FILE=${START_LOG_FILE}"
log "MODEL_SIZE_MIB=${MODEL_SIZE_MIB}"
log "MODEL_SHARD_MIB_ESTIMATE=${model_shard_mib}"
if (( service_budget_mib > 0 )); then
  log "SERVICE_BUDGET_MIB_ESTIMATE=${service_budget_mib}"
  log "KV_BUDGET_MIB_ESTIMATE=${kv_budget_mib}"
  log "KV_TOTAL_TOKENS_ESTIMATE=${estimated_total_tokens}"
fi
if [[ -n "$gpu_query_output" ]]; then
  log "GPU snapshot:"
  printf '%s\n' "$gpu_query_output" | tee -a "$START_LOG_FILE"
else
  log "GPU snapshot unavailable; nvidia-smi did not return usable data."
fi
log "Note: SGLang enforces per-request completion <= context length. Use ${CLIENT_DEFAULTS_FILE} or pass max_tokens/max_completion_tokens=${MAX_OUTPUT_TOKENS} from clients to apply the requested agent output cap."
log "Launch command:"
quote_redacted_command | tee -a "$START_LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN=1; not starting SGLang."
  exit 0
fi

setsid env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  "${server_cmd[@]}" \
  > >(redact_log_stream > "$SERVER_LOG_FILE") 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"

log "Started Qwen3.6-27B SGLang agent service"
log "PID=${pid}"
log "PID_FILE=${PID_FILE}"
log "Base URL=http://${SGLANG_HOST}:${SGLANG_PORT}/v1"
log "Health URL=http://${SGLANG_HOST}:${SGLANG_PORT}/health"

if [[ "$WAIT_FOR_READY" != "1" ]]; then
  exit 0
fi

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if ! kill -0 "$pid" 2>/dev/null; then
    log "Process exited before readiness. Last server log lines:"
    tail -n 160 "$SERVER_LOG_FILE" | tee -a "$START_LOG_FILE" || true
    exit 1
  fi

  if curl --noproxy '*' -fsS \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    "http://${SGLANG_HOST}:${SGLANG_PORT}/health" >/dev/null 2>&1; then
    log "Service is ready"
    exit 0
  fi

  sleep 2
done

log "Timed out waiting for readiness after ${READY_TIMEOUT_SECONDS}s. Last server log lines:"
tail -n 160 "$SERVER_LOG_FILE" | tee -a "$START_LOG_FILE" || true
exit 1
