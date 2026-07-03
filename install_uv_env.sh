#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PROJECT_DIR="$ROOT_DIR/python"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
UV_ENV_DIR="${UV_ENV_DIR:-/home/${USER:-$(id -un)}/uv-envs/sglang-llm-diffusion}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$UV_ENV_DIR}"
WITH_TRACING="${WITH_TRACING:-0}"
RECREATE_ENV="${RECREATE_ENV:-0}"
CONFIRM_DELETE_ENV="${CONFIRM_DELETE_ENV:-0}"

usage() {
  cat <<'EOF'
Usage: ./install_uv_env.sh [options]

Install a uv development environment for SGLang LLM + Diffusion.
By default, the virtual environment is created under:
  /home/$USER/uv-envs/sglang-llm-diffusion

Options:
  --python VERSION       Python version to install and pin, default: 3.11
  --env-dir PATH         Virtual environment path, default: /home/$USER/uv-envs/sglang-llm-diffusion
  --with-tracing         Also install the tracing extra
  --recreate             Delete and recreate the target environment
  -h, --help             Show this help

Environment variables:
  PYTHON_VERSION         Same as --python
  UV_ENV_DIR             Same as --env-dir
  UV_PROJECT_ENVIRONMENT uv project environment path; defaults to UV_ENV_DIR
  WITH_TRACING=1         Same as --with-tracing
  RECREATE_ENV=1         Same as --recreate
  CONFIRM_DELETE_ENV=1   Required together with --recreate for deletion safety

Examples:
  ./install_uv_env.sh
  ./install_uv_env.sh --python 3.11 --env-dir /home/$USER/uv-envs/sglang-llm-diffusion
  CONFIRM_DELETE_ENV=1 ./install_uv_env.sh --recreate
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="${2:?missing value for --python}"
      shift 2
      ;;
    --env-dir)
      UV_ENV_DIR="${2:?missing value for --env-dir}"
      UV_PROJECT_ENVIRONMENT="$UV_ENV_DIR"
      shift 2
      ;;
    --with-tracing)
      WITH_TRACING=1
      shift
      ;;
    --recreate)
      RECREATE_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is not installed or not on PATH.
Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
  exit 1
fi

if [[ ! -f "$PYTHON_PROJECT_DIR/pyproject.toml" || ! -f "$PYTHON_PROJECT_DIR/uv.lock" ]]; then
  echo "Cannot find python/pyproject.toml and python/uv.lock under $ROOT_DIR" >&2
  exit 1
fi

if [[ "$RECREATE_ENV" == "1" ]]; then
  if [[ "$CONFIRM_DELETE_ENV" != "1" ]]; then
    cat >&2 <<EOF
Refusing to delete environment without confirmation:
  $UV_PROJECT_ENVIRONMENT

Re-run with CONFIRM_DELETE_ENV=1 if you really want to recreate it:
  CONFIRM_DELETE_ENV=1 ./install_uv_env.sh --recreate
EOF
    exit 1
  fi
  rm -rf "$UV_PROJECT_ENVIRONMENT"
fi

mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

export UV_PROJECT_ENVIRONMENT

cd "$PYTHON_PROJECT_DIR"

sync_args=(--locked --extra dev --extra diffusion)
# Equivalent command: uv sync --locked --extra dev --extra diffusion
if [[ "$WITH_TRACING" == "1" ]]; then
  sync_args+=(--extra tracing)
fi

echo "Repository: $ROOT_DIR"
echo "Python project: $PYTHON_PROJECT_DIR"
echo "Python version: $PYTHON_VERSION"
echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
echo "Extras: dev diffusion$([[ "$WITH_TRACING" == "1" ]] && printf ' tracing')"

uv python install "$PYTHON_VERSION"
uv python pin "$PYTHON_VERSION"
uv sync "${sync_args[@]}"

"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import torch
import sglang
try:
    import diffusers
except Exception as exc:
    raise SystemExit(f"failed to import diffusers: {exc}")
print("sglang:", sglang.__file__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
PY

cat <<EOF

Install finished.
Activate with:
  export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
  source "$UV_PROJECT_ENVIRONMENT/bin/activate"

Start LLM example:
  ./start_sglang_uv.sh --model-path Qwen/Qwen2.5-0.5B-Instruct

Start diffusion example:
  ./start_sglang_uv.sh --model-type diffusion --model-path <diffusion-model-or-local-path>
EOF
