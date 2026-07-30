#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${VIDEOEDIT_DUAL_CONFIG:-${SCRIPT_DIR}/config.env}"
if [[ ! -r "$CONFIG_FILE" ]]; then
  echo "Missing readable config: $CONFIG_FILE" >&2
  echo "Copy ${SCRIPT_DIR}/config.env.example to config.env first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

export PYTHONPATH="${PROJECT_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
PROBE="${SCRIPT_DIR}/resource_probe.py"
LOCK_FILE="${RUNTIME_DIR}/start.lock"
NORMAL_PID_FILE="${PID_DIR}/normal.pid"
DMD_PID_FILE="${PID_DIR}/dmd.pid"
GATEWAY_PID_FILE="${PID_DIR}/gateway.pid"
NORMAL_METRICS="${RUNTIME_DIR}/normal-startup.json"
DMD_METRICS="${RUNTIME_DIR}/dmd-startup.json"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$PID_DIR" \
  "$OUTPUT_DIR/normal" "$OUTPUT_DIR/dmd" \
  "$INPUT_DIR/normal" "$INPUT_DIR/dmd"
chmod 700 "$RUNTIME_DIR" "$PID_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another dual-service start is already running." >&2
  exit 1
fi

is_running() {
  local pid_file="$1"
  [[ -r "$pid_file" ]] || return 1
  local pid
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

stop_pid() {
  local pid_file="$1"
  local expected="$2"
  [[ -r "$pid_file" ]] || return 0
  local pid cmdline
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ -n "$cmdline" && "$cmdline" == *"$expected"* ]]; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  fi
  rm -f -- "$pid_file"
}

validate_transformer() {
  local path="$1"
  if ! "$PYTHON_BIN" "$PROBE" validate-transformer "$path"; then
    echo "Transformer checkpoint is incomplete or incompatible: $path" >&2
    return 1
  fi
}

if is_running "$NORMAL_PID_FILE" || is_running "$DMD_PID_FILE" || is_running "$GATEWAY_PID_FILE"; then
  echo "One or more dual-service processes are already running; use status.sh." >&2
  exit 1
fi

[[ -r "${BASE_MODEL}/model_index.json" ]] || {
  echo "Base model is not readable: $BASE_MODEL" >&2
  exit 1
}
validate_transformer "$NORMAL_TRANSFORMER" || exit 1
dmd_eligible=1
if ! validate_transformer "$DMD_TRANSFORMER"; then
  dmd_eligible=0
  echo "DMD checkpoint is not ready; normal-only mode will be used." >&2
fi

ports=(
  "$GATEWAY_PORT"
  "$NORMAL_PORT" "$NORMAL_BROKER_PORT" "$NORMAL_MASTER_PORT"
  "$NORMAL_SCHEDULER_PORT" "$NORMAL_NCCL_PORT"
)
if (( dmd_eligible )); then
  ports+=(
    "$DMD_PORT" "$DMD_BROKER_PORT" "$DMD_MASTER_PORT"
    "$DMD_SCHEDULER_PORT" "$DMD_NCCL_PORT"
  )
fi
"$PYTHON_BIN" "$PROBE" check-ports "${ports[@]}"

start_backend() {
  local variant="$1"
  local transformer="$2"
  local port="$3"
  local master_port="$4"
  local scheduler_port="$5"
  local nccl_port="$6"
  local pid_file="$7"
  local log_file="$8"

  nohup env \
    PYTHONPATH="$PYTHONPATH" \
    VIDEOEDIT_QUEUE_CAPACITY="$VIDEOEDIT_QUEUE_CAPACITY" \
    SGLANG_USE_RUNAI_MODEL_STREAMER="$SGLANG_USE_RUNAI_MODEL_STREAMER" \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    "$SGLANG_BIN" serve \
      --model-type diffusion \
      --backend sglang \
      --model-path "$BASE_MODEL" \
      --transformer-path "$transformer" \
      --host 127.0.0.1 \
      --port "$port" \
      --master-port "$master_port" \
      --scheduler-port "$scheduler_port" \
      --nccl-port "$nccl_port" \
      --scheduler-response-timeout "$SCHEDULER_RESPONSE_TIMEOUT" \
      --strict-ports true \
      --num-gpus 2 \
      --sp-degree 2 \
      --ulysses-degree 2 \
      --ring-degree 1 \
      --dit-layerwise-offload true \
      --dit-offload-prefetch-size 0 \
      --dit-cpu-offload false \
      --text-encoder-cpu-offload true \
      --image-encoder-cpu-offload true \
      --vae-cpu-offload true \
      --pin-cpu-memory "$PIN_CPU_MEMORY" \
      --warmup false \
      --output-path "$OUTPUT_DIR/$variant" \
      --input-save-path "$INPUT_DIR/$variant" \
      >"$log_file" 2>&1 &
  STARTED_PID=$!
  printf '%s\n' "$STARTED_PID" >"$pid_file"
}

wait_monitor_and_process() {
  local monitor_pid="$1"
  local service_pid="$2"
  while kill -0 "$monitor_pid" 2>/dev/null; do
    if ! kill -0 "$service_pid" 2>/dev/null; then
      kill -TERM "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
      return 1
    fi
    sleep 1
  done
  wait "$monitor_pid"
}

start_gateway() {
  if is_running "$GATEWAY_PID_FILE"; then
    return 0
  fi
  nohup env \
    PYTHONPATH="$PYTHONPATH" \
    VIDEOEDIT_DUAL_QUEUE_DB="$QUEUE_DB" \
    VIDEOEDIT_NORMAL_BACKEND="http://127.0.0.1:${NORMAL_PORT}" \
    VIDEOEDIT_DMD_BACKEND="http://127.0.0.1:${DMD_PORT}" \
    VIDEOEDIT_GATEWAY_POLL_INTERVAL="$VIDEOEDIT_GATEWAY_POLL_INTERVAL" \
    VIDEOEDIT_GATEWAY_HEALTH_TIMEOUT="$VIDEOEDIT_GATEWAY_HEALTH_TIMEOUT" \
    "$PYTHON_BIN" -m \
      sglang.multimodal_gen.runtime.videoedit.dual_service_gateway \
      --host "$GATEWAY_HOST" \
      --port "$GATEWAY_PORT" \
      >"$LOG_DIR/gateway.log" 2>&1 &
  printf '%s\n' "$!" >"$GATEWAY_PID_FILE"
}

echo "Starting normal VideoEdit backend..."
"$PYTHON_BIN" "$PROBE" monitor-startup \
  --health-url "http://127.0.0.1:${NORMAL_PORT}/health" \
  --output "$NORMAL_METRICS" \
  --timeout "$STARTUP_TIMEOUT" \
  --stable-seconds "$STARTUP_STABLE_SECONDS" \
  >"$LOG_DIR/normal-resource.log" 2>&1 &
normal_monitor_pid=$!
start_backend \
  normal "$NORMAL_TRANSFORMER" "$NORMAL_PORT" "$NORMAL_MASTER_PORT" \
  "$NORMAL_SCHEDULER_PORT" "$NORMAL_NCCL_PORT" "$NORMAL_PID_FILE" \
  "$LOG_DIR/normal.log"
normal_pid=$STARTED_PID
if ! wait_monitor_and_process "$normal_monitor_pid" "$normal_pid"; then
  echo "Normal backend failed to become healthy." >&2
  stop_pid "$NORMAL_PID_FILE" "--port ${NORMAL_PORT}"
  exit 1
fi

if (( ! dmd_eligible )); then
  start_gateway
  echo "VideoEdit gateway started in normal-only mode on ${GATEWAY_HOST}:${GATEWAY_PORT}."
  exit 0
fi

if ! "$PYTHON_BIN" "$PROBE" gate-second \
  --metrics "$NORMAL_METRICS" \
  --output "$RUNTIME_DIR/second-service-gate.json" \
  --gpu-headroom-gib "$MIN_STARTUP_GPU_HEADROOM_GIB" \
  --host-headroom-gib "$MIN_HOST_HEADROOM_GIB" \
  --cgroup-headroom-gib "$MIN_CGROUP_HEADROOM_GIB"; then
  echo "Second-service resource gate failed; starting normal-only gateway." >&2
  start_gateway
  exit 0
fi

echo "Starting DMD VideoEdit backend..."
"$PYTHON_BIN" "$PROBE" monitor-startup \
  --health-url "http://127.0.0.1:${DMD_PORT}/health" \
  --output "$DMD_METRICS" \
  --timeout "$STARTUP_TIMEOUT" \
  --stable-seconds "$STARTUP_STABLE_SECONDS" \
  >"$LOG_DIR/dmd-resource.log" 2>&1 &
dmd_monitor_pid=$!
start_backend \
  dmd "$DMD_TRANSFORMER" "$DMD_PORT" "$DMD_MASTER_PORT" \
  "$DMD_SCHEDULER_PORT" "$DMD_NCCL_PORT" "$DMD_PID_FILE" \
  "$LOG_DIR/dmd.log"
dmd_pid=$STARTED_PID
if ! wait_monitor_and_process "$dmd_monitor_pid" "$dmd_pid"; then
  echo "DMD backend failed; keeping normal available through the gateway." >&2
  stop_pid "$DMD_PID_FILE" "--port ${DMD_PORT}"
  start_gateway
  exit 0
fi

if ! "$PYTHON_BIN" "$PROBE" gate-idle \
  --output "$RUNTIME_DIR/dual-idle-gate.json" \
  --gpu-headroom-gib "$MIN_IDLE_GPU_HEADROOM_GIB" \
  --host-headroom-gib "$MIN_HOST_HEADROOM_GIB" \
  --cgroup-headroom-gib "$MIN_CGROUP_HEADROOM_GIB"; then
  echo "Dual idle resource gate failed; stopping DMD and using normal only." >&2
  stop_pid "$DMD_PID_FILE" "--port ${DMD_PORT}"
fi

start_gateway
echo "VideoEdit gateway started on ${GATEWAY_HOST}:${GATEWAY_PORT}."
