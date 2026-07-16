#!/usr/bin/env bash
set -euo pipefail

# Start the local VideoEdit SGLang service from an existing Docker image.
# Override any variable at invocation time, for example:
#   HOST_GPUS=2,3 HOST_PORT=30000 bash docs_tyx/scrips/start_videoedit_container.sh
# If the container already exists, the default is to remove and recreate it.
# Set RESTART_EXISTING=1 to restart the existing container instead.

IMAGE_NAME="${IMAGE_NAME:-sglang-mgtv:1.0}"
CONTAINER_NAME="${CONTAINER_NAME:-videoedit_reset}"

PROJECT_ROOT="${PROJECT_ROOT:-/root/VideoEdit}"
HOST_REPO_DIR="${HOST_REPO_DIR:-/root/VideoEdit/sglang}"
CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/sgl-workspace/sglang}"
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-/root/VideoEdit/sglang}"
MODEL_PATH="${MODEL_PATH:-/root/VideoEdit/model/DifusserEdit/pretrain_models/VideoEdit-diffusers-model}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${MODEL_PATH}/transformer}"

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
BACKEND="${BACKEND:-sglang}"
AWS_REQUEST_CHECKSUM_CALCULATION="${AWS_REQUEST_CHECKSUM_CALCULATION:-WHEN_REQUIRED}"
AWS_RESPONSE_CHECKSUM_VALIDATION="${AWS_RESPONSE_CHECKSUM_VALIDATION:-WHEN_REQUIRED}"

VIDEOEDIT_QUEUE_CAPACITY="${VIDEOEDIT_QUEUE_CAPACITY:-1}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"

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
require_path "$MODEL_PATH"
require_path "$TRANSFORMER_PATH"

mkdir -p "$INPUT_SAVE_DIR" "$VIDEOEDIT_OUTPUT_DIR" "$VIDEOEDIT_REQUEST_LOG_DIR" "$CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"

docker_gpu_arg="\"device=${HOST_GPUS}\""

echo "Starting container '$CONTAINER_NAME' from image '$IMAGE_NAME'"
echo "Host GPUs: ${HOST_GPUS}; container CUDA_VISIBLE_DEVICES: ${CONTAINER_CUDA_VISIBLE_DEVICES}"
echo "Torch compile: ${ENABLE_TORCH_COMPILE}"
echo "Service URL: http://0.0.0.0:${HOST_PORT}"
echo "VideoEdit outputs: ${VIDEOEDIT_OUTPUT_DIR}"
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
  -e MODEL_PATH="$MODEL_PATH" \
  -e TRANSFORMER_PATH="$TRANSFORMER_PATH" \
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
  -e BACKEND="$BACKEND" \
  -e AWS_REQUEST_CHECKSUM_CALCULATION="$AWS_REQUEST_CHECKSUM_CALCULATION" \
  -e AWS_RESPONSE_CHECKSUM_VALIDATION="$AWS_RESPONSE_CHECKSUM_VALIDATION" \
  -e ENABLE_TORCH_COMPILE="$ENABLE_TORCH_COMPILE" \
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

    exec sglang serve \
      --model-type diffusion \
      --backend "$BACKEND" \
      --model-path "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 30000 \
      --num-gpus 2 \
      --sp-degree 2 \
      --ulysses-degree 2 \
      --ring-degree 1 \
      --enable-torch-compile "$ENABLE_TORCH_COMPILE" \
      --dit-cpu-offload true \
      --dit-layerwise-offload true \
      --text-encoder-cpu-offload true \
      --image-encoder-cpu-offload true \
      --vae-cpu-offload true \
      --vae-tiling true \
      --vae-config.use-tiling true \
      --vae-config.use-temporal-tiling false \
      --vae-config.use-feature-cache true \
      --warmup true \
      --warmup-steps 1 \
      --output-path "$VIDEOEDIT_OUTPUT_DIR" \
      --input-save-path "$INPUT_SAVE_DIR" \
      --videoedit-request-log-dir "$VIDEOEDIT_REQUEST_LOG_DIR" \
      --videoedit-request-log-sensitive-values "$VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES" \
      --transformer-path "$TRANSFORMER_PATH"
  '

echo
echo "Container started. Useful commands:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "  docker ps --filter name=^/${CONTAINER_NAME}$"
echo "  docker exec -it ${CONTAINER_NAME} nvidia-smi"
