#!/usr/bin/env bash
set -euo pipefail

# Build the devcontainer image, create a fresh VideoEdit container, and start serve.
# Override any variable at invocation time, for example:
#   HOST_GPUS=2,3 IMAGE_NAME=sglang-videoedit-reset:latest bash docs_tyx/scrips/rebuild_image_create_videoedit_container.sh

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/root/VideoEdit}"
HOST_REPO_DIR="${HOST_REPO_DIR:-/root/VideoEdit/sglang}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${HOST_REPO_DIR}/.devcontainer/Dockerfile}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${HOST_REPO_DIR}}"

IMAGE_NAME="${IMAGE_NAME:-sglang-videoedit-reset:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-videoedit_reset}"

CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/root/VideoEdit}"
CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/sgl-workspace/sglang}"
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-${CONTAINER_REPO_DIR}}"

MODEL_PATH="${MODEL_PATH:-/root/VideoEdit/model/DifusserEdit/pretrain_models/VideoEdit-diffusers-model}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${MODEL_PATH}/transformer}"

INPUT_SAVE_DIR="${INPUT_SAVE_DIR:-/root/VideoEdit/tmp/sglang-videoedit-cloud-inputs}"
CACHE_DIR="${CACHE_DIR:-/root/VideoEdit/tmp/sglang-cache}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_DIR}/flashinfer}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}/xdg}"

HOST_GPUS="${HOST_GPUS:-2,3}"
CONTAINER_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES:-0,1}"
HOST_PORT="${HOST_PORT:-30000}"
CONTAINER_PORT="${CONTAINER_PORT:-30000}"
BACKEND="${BACKEND:-sglang}"

NUM_GPUS="${NUM_GPUS:-2}"
SP_DEGREE="${SP_DEGREE:-2}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-2}"
RING_DEGREE="${RING_DEGREE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"

VIDEOEDIT_QUEUE_CAPACITY="${VIDEOEDIT_QUEUE_CAPACITY:-1}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BUILD_HOST_UID="${BUILD_HOST_UID:-1003}"
BUILD_HOST_GID="${BUILD_HOST_GID:-1003}"
NO_CACHE="${NO_CACHE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
RUN_AS_USER="${RUN_AS_USER:-root}"
ENTER_SHELL="${ENTER_SHELL:-0}"

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

require_path "$HOST_PROJECT_ROOT"
require_path "$HOST_REPO_DIR"
require_path "$HOST_REPO_DIR/python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py"
require_path "$DOCKERFILE_PATH"
require_path "$BUILD_CONTEXT"
require_path "$MODEL_PATH"
require_path "$TRANSFORMER_PATH"

mkdir -p "$INPUT_SAVE_DIR" "$CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"

if [[ "$SKIP_BUILD" != "1" ]]; then
  build_args=(
    docker build
    -f "$DOCKERFILE_PATH"
    -t "$IMAGE_NAME"
    --build-arg "HOST_UID=${BUILD_HOST_UID}"
    --build-arg "HOST_GID=${BUILD_HOST_GID}"
  )

  if [[ "$NO_CACHE" == "1" ]]; then
    build_args+=(--no-cache)
  fi

  build_args+=("$BUILD_CONTEXT")

  echo "Building image '$IMAGE_NAME' from '$DOCKERFILE_PATH'"
  "${build_args[@]}"
else
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "SKIP_BUILD=1 was set, but image does not exist locally: $IMAGE_NAME" >&2
    exit 1
  fi
  echo "Skipping image build and using existing image: $IMAGE_NAME"
fi

existing_container_id="$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")"
if [[ -n "$existing_container_id" ]]; then
  existing_status="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")"
  echo "Removing existing container '$CONTAINER_NAME' with status: $existing_status"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker_gpu_arg="\"device=${HOST_GPUS}\""

echo "Starting rebuilt container '$CONTAINER_NAME' from image '$IMAGE_NAME'"
echo "Host GPUs: ${HOST_GPUS}; container CUDA_VISIBLE_DEVICES: ${CONTAINER_CUDA_VISIBLE_DEVICES}"
echo "Service URL: http://0.0.0.0:${HOST_PORT}"

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus "$docker_gpu_arg" \
  --ipc=host \
  --user "$RUN_AS_USER" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${HOST_PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" \
  -v "${HOST_REPO_DIR}:${CONTAINER_REPO_DIR}" \
  -w "$WORKDIR_IN_CONTAINER" \
  -e MODEL_PATH="$MODEL_PATH" \
  -e TRANSFORMER_PATH="$TRANSFORMER_PATH" \
  -e INPUT_SAVE_DIR="$INPUT_SAVE_DIR" \
  -e CACHE_DIR="$CACHE_DIR" \
  -e CUDA_VISIBLE_DEVICES="$CONTAINER_CUDA_VISIBLE_DEVICES" \
  -e VIDEOEDIT_QUEUE_CAPACITY="$VIDEOEDIT_QUEUE_CAPACITY" \
  -e FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE_BASE" \
  -e XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  -e PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e PYTHONPATH="${CONTAINER_REPO_DIR}/python" \
  -e BACKEND="$BACKEND" \
  -e SERVE_PORT="$CONTAINER_PORT" \
  -e NUM_GPUS="$NUM_GPUS" \
  -e SP_DEGREE="$SP_DEGREE" \
  -e ULYSSES_DEGREE="$ULYSSES_DEGREE" \
  -e RING_DEGREE="$RING_DEGREE" \
  -e WARMUP_STEPS="$WARMUP_STEPS" \
  --entrypoint bash \
  "$IMAGE_NAME" \
  -lc '
    set -euo pipefail

    mkdir -p "$INPUT_SAVE_DIR" "$CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"
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
      --enable-torch-compile \
      --model-path "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$SERVE_PORT" \
      --num-gpus "$NUM_GPUS" \
      --sp-degree "$SP_DEGREE" \
      --ulysses-degree "$ULYSSES_DEGREE" \
      --ring-degree "$RING_DEGREE" \
      --dit-cpu-offload true \
      --dit-layerwise-offload true \
      --text-encoder-cpu-offload true \
      --image-encoder-cpu-offload true \
      --vae-cpu-offload true \
      --warmup true \
      --warmup-steps "$WARMUP_STEPS" \
      --output-path "" \
      --input-save-path "$INPUT_SAVE_DIR" \
      --transformer-path "$TRANSFORMER_PATH"
  '

echo
echo "Container rebuilt and started. Useful commands:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "  docker ps --filter name=^/${CONTAINER_NAME}$"
echo "  docker exec -it ${CONTAINER_NAME} nvidia-smi"
echo "  docker exec -it ${CONTAINER_NAME} bash"

if [[ "$ENTER_SHELL" == "1" ]]; then
  docker exec -it "$CONTAINER_NAME" bash
fi
