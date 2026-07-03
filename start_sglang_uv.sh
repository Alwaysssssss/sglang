#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PROJECT_DIR="$ROOT_DIR/python"
UV_ENV_DIR="${UV_ENV_DIR:-/home/${USER:-$(id -un)}/uv-envs/sglang-llm-diffusion}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$UV_ENV_DIR}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MODEL_TYPE="${MODEL_TYPE:-}"
MODEL_PATH="${MODEL_PATH:-}"

usage() {
  cat <<'EOF'
Usage: ./start_sglang_uv.sh --model-path MODEL [options] [-- extra sglang args]

Start SGLang from the uv environment created by install_uv_env.sh.

Options:
  --model-path MODEL     Hugging Face model id or local model path. Required unless MODEL_PATH is set.
  --model-type TYPE      Optional model type, e.g. diffusion
  --host HOST            Listen host, default: 0.0.0.0
  --port PORT            Listen port, default: 30000
  --env-dir PATH         Virtual environment path, default: /home/$USER/uv-envs/sglang-llm-diffusion
  -h, --help             Show this help

Environment variables:
  MODEL_PATH             Same as --model-path
  MODEL_TYPE             Same as --model-type
  HOST                   Same as --host
  PORT                   Same as --port
  UV_ENV_DIR             Same as --env-dir
  UV_PROJECT_ENVIRONMENT uv project environment path; defaults to UV_ENV_DIR

Examples:
  ./start_sglang_uv.sh --model-path Qwen/Qwen2.5-0.5B-Instruct
  ./start_sglang_uv.sh --model-type diffusion --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  ./start_sglang_uv.sh --model-path Qwen/Qwen2.5-0.5B-Instruct -- --tp 1 --log-level info
EOF
}

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="${2:?missing value for --model-path}"
      shift 2
      ;;
    --model-type)
      MODEL_TYPE="${2:?missing value for --model-type}"
      shift 2
      ;;
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --env-dir)
      UV_ENV_DIR="${2:?missing value for --env-dir}"
      UV_PROJECT_ENVIRONMENT="$UV_ENV_DIR"
      shift 2
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "Missing required --model-path MODEL" >&2
  usage >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  cat >&2 <<EOF
Cannot find uv environment Python:
  $UV_PROJECT_ENVIRONMENT/bin/python

Create it first:
  ./install_uv_env.sh
EOF
  exit 1
fi

export UV_PROJECT_ENVIRONMENT
cd "$PYTHON_PROJECT_DIR"

args=(
  --model-path "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$MODEL_TYPE" ]]; then
  args+=(--model-type "$MODEL_TYPE")
fi

args+=("${extra_args[@]}")

echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
echo "Starting: python -m sglang.launch_server ${args[*]}"

exec uv run python -m sglang.launch_server "${args[@]}"
