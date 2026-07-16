#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PROJECT_DIR="$ROOT_DIR/python"
DEFAULT_UV_ENV_DIR="/home/${USER:-$(id -un)}/uv-envs/sglang-llm-diffusion"
UV_ENV_DIR="${UV_ENV_DIR:-$DEFAULT_UV_ENV_DIR}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$UV_ENV_DIR}"

usage() {
  cat <<'EOF'
Usage: ./start_sglang_uv.sh [options] [-- command args...]

Start a shell in the uv environment created by install_uv_env.sh.

No model is launched by this script. Pass an optional command after -- to run
inside the environment instead of opening an interactive shell.

Options:
  --env-dir PATH         Virtual environment path, default: /home/$USER/uv-envs/sglang-llm-diffusion
  -h, --help             Show this help

Environment variables:
  UV_ENV_DIR             Same as --env-dir
  UV_PROJECT_ENVIRONMENT uv project environment path; defaults to UV_ENV_DIR,
                         which defaults to /home/$USER/uv-envs/sglang-llm-diffusion

Examples:
  ./start_sglang_uv.sh
  ./start_sglang_uv.sh --env-dir ~/uv-envs/sglang-llm-diffusion
  ./start_sglang_uv.sh -- python -c 'import sglang; print(sglang.__version__)'
EOF
}

command_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      UV_ENV_DIR="${2:?missing value for --env-dir}"
      UV_PROJECT_ENVIRONMENT="$UV_ENV_DIR"
      shift 2
      ;;
    --)
      shift
      command_args+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      command_args+=("$1")
      shift
      ;;
  esac
done

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

if [[ ${#command_args[@]} -eq 0 ]]; then
  command_args=(bash)
fi

echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
echo "Starting uv shell: ${command_args[*]}"

exec uv run --directory "$PYTHON_PROJECT_DIR" --active "${command_args[@]}"
