#!/usr/bin/env bash
set -euo pipefail

# Start the existing VideoEdit normal+DMD dual-service stack in one container.
# Override any variable at invocation time, for example:
#   RECREATE=1 HOST_GPUS=2,3 bash /root/VideoEdit/sglang/scripts/start_videoedit_container.sh
# Both backends use both GPUs; the gateway serializes requests on port 30000.
# If the container already exists, the default is to remove and recreate it.
# Set RESTART_EXISTING=1 to restart the existing container instead.

IMAGE_NAME="${IMAGE_NAME:-sglang-mgtv:1.0}"
CONTAINER_NAME="${CONTAINER_NAME:-videoedit_reset}"

PROJECT_ROOT="${PROJECT_ROOT:-/root/VideoEdit}"
HOST_REPO_DIR="${HOST_REPO_DIR:-/root/VideoEdit/sglang}"
CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/sgl-workspace/sglang}"
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-/root/VideoEdit/sglang}"
DUAL_SERVICE_DIR_HOST="${DUAL_SERVICE_DIR_HOST:-${HOST_REPO_DIR}/scripts/videoedit_dual_service}"
DUAL_SERVICE_CONFIG_HOST="${DUAL_SERVICE_CONFIG_HOST:-${DUAL_SERVICE_DIR_HOST}/config.env}"
DUAL_SERVICE_CONFIG_CONTAINER="${DUAL_SERVICE_CONFIG_CONTAINER:-${CONTAINER_REPO_DIR}/scripts/videoedit_dual_service/config.env}"

INPUT_SAVE_DIR="${INPUT_SAVE_DIR:-/root/VideoEdit/tmp/sglang-videoedit-cloud-inputs}"
VIDEOEDIT_OUTPUT_DIR="${VIDEOEDIT_OUTPUT_DIR:-/root/VideoEdit/tmp/sglang-videoedit-outputs}"
VIDEOEDIT_REQUEST_LOG_DIR="${VIDEOEDIT_REQUEST_LOG_DIR:-/root/VideoEdit/tmp/sglang-videoedit-request-logs}"
VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES="${VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES:-true}"
CACHE_DIR="${CACHE_DIR:-/root/VideoEdit/tmp/sglang-cache}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_DIR}/flashinfer}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}/xdg}"

HOST_GPUS="${HOST_GPUS:-2,3}"
CONTAINER_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES:-0,1}"
HOST_PORT="${HOST_PORT:-30000}"
CONTAINER_PORT="${CONTAINER_PORT:-30000}"
AWS_REQUEST_CHECKSUM_CALCULATION="${AWS_REQUEST_CHECKSUM_CALCULATION:-WHEN_REQUIRED}"
AWS_RESPONSE_CHECKSUM_VALIDATION="${AWS_RESPONSE_CHECKSUM_VALIDATION:-WHEN_REQUIRED}"

VIDEOEDIT_QUEUE_CAPACITY="${VIDEOEDIT_QUEUE_CAPACITY:-1}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_AS_USER="${RUN_AS_USER:-root}"
RESTART_EXISTING="${RESTART_EXISTING:-0}"
RECREATE="${RECREATE:-0}"
if [[ "$RECREATE" == "1" ]]; then
  RESTART_EXISTING=0
fi

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path on host: $path" >&2
    exit 1
  fi
}

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found" >&2
  exit 1
fi

existing_container_id="$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")"
if [[ -n "$existing_container_id" ]]; then
  existing_status="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")"
  if [[ "$RESTART_EXISTING" == "1" ]]; then
    if [[ "$existing_status" == "running" ]]; then
      echo "Restarting existing container '$CONTAINER_NAME'"
      docker restart "$CONTAINER_NAME" >/dev/null
    else
      echo "Starting existing container '$CONTAINER_NAME' with status: $existing_status"
      docker start "$CONTAINER_NAME" >/dev/null
    fi

    echo "Container is running. Useful commands:"
    echo "  docker logs -f ${CONTAINER_NAME}"
    docker ps --filter "name=^/${CONTAINER_NAME}$"
    echo "  docker exec -it ${CONTAINER_NAME} nvidia-smi"
    exit 0
  fi

  echo "Removing existing container '$CONTAINER_NAME' with status: $existing_status"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Docker image not found locally: $IMAGE_NAME" >&2
  exit 1
fi

require_path "$PROJECT_ROOT"
require_path "$HOST_REPO_DIR"
require_path "$HOST_REPO_DIR/python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py"
require_path "$WORKDIR_IN_CONTAINER"
require_path "$DUAL_SERVICE_DIR_HOST/start.sh"
require_path "$DUAL_SERVICE_DIR_HOST/status.sh"
require_path "$DUAL_SERVICE_DIR_HOST/stop.sh"
require_path "$DUAL_SERVICE_CONFIG_HOST"

mkdir -p "$INPUT_SAVE_DIR" "$VIDEOEDIT_OUTPUT_DIR" "$VIDEOEDIT_REQUEST_LOG_DIR" "$CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"

docker_gpu_arg="\"device=${HOST_GPUS}\""

echo "Starting container '$CONTAINER_NAME' from image '$IMAGE_NAME'"
echo "Host GPUs: ${HOST_GPUS}; container CUDA_VISIBLE_DEVICES: ${CONTAINER_CUDA_VISIBLE_DEVICES}"
echo "Unified gateway URL: http://0.0.0.0:${HOST_PORT}"
echo "Both normal and DMD backends use both GPUs; requests are serialized by the gateway"
echo "Request logs: ${VIDEOEDIT_REQUEST_LOG_DIR}"
echo "Log sensitive request values: ${VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES}"
echo "S3 checksum mode: request=${AWS_REQUEST_CHECKSUM_CALCULATION}, response=${AWS_RESPONSE_CHECKSUM_VALIDATION}"

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus "$docker_gpu_arg" \
  --ipc=host \
  --user "$RUN_AS_USER" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  -v "${HOST_REPO_DIR}:${CONTAINER_REPO_DIR}" \
  -w "$WORKDIR_IN_CONTAINER" \
  -e CONTAINER_REPO_DIR="$CONTAINER_REPO_DIR" \
  -e VIDEOEDIT_DUAL_CONFIG="$DUAL_SERVICE_CONFIG_CONTAINER" \
  -e INPUT_SAVE_DIR="$INPUT_SAVE_DIR" \
  -e VIDEOEDIT_OUTPUT_DIR="$VIDEOEDIT_OUTPUT_DIR" \
  -e VIDEOEDIT_REQUEST_LOG_DIR="$VIDEOEDIT_REQUEST_LOG_DIR" \
  -e VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES="$VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES" \
  -e CACHE_DIR="$CACHE_DIR" \
  -e CUDA_VISIBLE_DEVICES="$CONTAINER_CUDA_VISIBLE_DEVICES" \
  -e VIDEOEDIT_QUEUE_CAPACITY="$VIDEOEDIT_QUEUE_CAPACITY" \
  -e FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE_BASE" \
  -e XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  -e PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e PYTHONPATH="${CONTAINER_REPO_DIR}/python" \
  -e AWS_REQUEST_CHECKSUM_CALCULATION="$AWS_REQUEST_CHECKSUM_CALCULATION" \
  -e AWS_RESPONSE_CHECKSUM_VALIDATION="$AWS_RESPONSE_CHECKSUM_VALIDATION" \
  "$IMAGE_NAME" \
  bash -lc '
    set -euo pipefail

    mkdir -p "$INPUT_SAVE_DIR" "$VIDEOEDIT_OUTPUT_DIR" "$VIDEOEDIT_REQUEST_LOG_DIR" "$CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"
    python3 - <<'"'"'PY'"'"'
import importlib.util
import sglang

print(f"Using sglang from: {sglang.__file__}", flush=True)
print(
    "WanVideoEditPipeline module: "
    f"{importlib.util.find_spec('"'"'sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline'"'"')}",
    flush=True,
)
PY

    dual_service_dir="$CONTAINER_REPO_DIR/scripts/videoedit_dual_service"
    # shellcheck disable=SC1090
    source "$VIDEOEDIT_DUAL_CONFIG"

    tail_pid=""
    cleanup() {
      trap - EXIT INT TERM
      if [[ -n "$tail_pid" ]]; then
        kill -TERM "$tail_pid" 2>/dev/null || true
        wait "$tail_pid" 2>/dev/null || true
      fi
      bash "$dual_service_dir/stop.sh" || true
    }
    trap cleanup EXIT INT TERM

    mkdir -p "$LOG_DIR"
    tail -n +1 -F \
      "$LOG_DIR/normal.log" \
      "$LOG_DIR/dmd.log" \
      "$LOG_DIR/gateway.log" &
    tail_pid=$!

    bash "$dual_service_dir/start.sh"

    normal_pid="$(<"$PID_DIR/normal.pid")"
    gateway_pid="$(<"$PID_DIR/gateway.pid")"

    for pid in "$normal_pid" "$gateway_pid"; do
      if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "VideoEdit startup did not leave normal and gateway running." >&2
        exit 1
      fi
    done

    dmd_pid=""
    if [[ -r "$PID_DIR/dmd.pid" ]]; then
      candidate_dmd_pid="$(<"$PID_DIR/dmd.pid")"
      if [[ "$candidate_dmd_pid" =~ ^[0-9]+$ ]] && kill -0 "$candidate_dmd_pid" 2>/dev/null; then
        dmd_pid="$candidate_dmd_pid"
      fi
    fi

    if [[ -n "$dmd_pid" ]]; then
      echo "VideoEdit dual-service stack is ready on gateway port $GATEWAY_PORT"
      echo "normal=$normal_pid dmd=$dmd_pid gateway=$gateway_pid"
    else
      echo "VideoEdit is running in degraded normal-only mode on gateway port $GATEWAY_PORT" >&2
    fi

    while kill -0 "$normal_pid" 2>/dev/null \
      && kill -0 "$gateway_pid" 2>/dev/null; do
      if [[ -n "$dmd_pid" ]] && ! kill -0 "$dmd_pid" 2>/dev/null; then
        echo "DMD backend exited; keeping normal and gateway running." >&2
        dmd_pid=""
      fi
      sleep 5
    done

    echo "The normal backend or gateway exited; stopping the stack." >&2
    exit 1
  '

echo
echo "Container started. Useful commands:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "  docker ps --filter name=^/${CONTAINER_NAME}$"
echo "  docker exec -it ${CONTAINER_NAME} nvidia-smi"
echo "  docker exec ${CONTAINER_NAME} bash scripts/videoedit_dual_service/status.sh"
echo "  docker stop ${CONTAINER_NAME}"
echo "  curl --noproxy '*' -sS http://127.0.0.1:${HOST_PORT}/health"
echo "Model routing: videoedit-normal -> normal; videoedit-dmd -> DMD"
