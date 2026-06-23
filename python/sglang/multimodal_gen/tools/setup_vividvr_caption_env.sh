#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhiheng/sglang"
VENV_PATH="${REPO_ROOT}/.venv-vividvr-caption"
REQUIREMENTS_PATH="${REPO_ROOT}/python/requirements-vividvr-caption.txt"

if [[ ! -f "${REQUIREMENTS_PATH}" ]]; then
  echo "requirements file not found: ${REQUIREMENTS_PATH}" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  echo "uv is required but was not found in PATH" >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BOOTSTRAP="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BOOTSTRAP="$(command -v python3)"
else
  echo "python3 is required to create ${VENV_PATH}" >&2
  exit 1
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${UV_BIN}" venv \
    --seed \
    --python "${PYTHON_BOOTSTRAP}" \
    "${VENV_PATH}"
fi

SITE_PACKAGES_PATH="$("${VENV_PATH}/bin/python" - <<'PY'
import sysconfig

print(sysconfig.get_path("purelib"))
PY
)"

if [[ -z "${SITE_PACKAGES_PATH}" || ! -d "${SITE_PACKAGES_PATH}" ]]; then
  echo "site-packages path not found for ${VENV_PATH}" >&2
  exit 1
fi

printf '%s\n' "${REPO_ROOT}/python" > "${SITE_PACKAGES_PATH}/sglang_local_repo.pth"

"${UV_BIN}" pip install \
  --python "${VENV_PATH}/bin/python" \
  --upgrade \
  --index-strategy unsafe-best-match \
  pip setuptools wheel

"${UV_BIN}" pip install \
  --python "${VENV_PATH}/bin/python" \
  --index-strategy unsafe-best-match \
  -r "${REQUIREMENTS_PATH}"

# Match the original Vivid-VR caption runtime even though the latest wheel
# metadata now advertises a stricter numpy floor than the working upstream env.
rm -rf \
  "${SITE_PACKAGES_PATH}/cv2" \
  "${SITE_PACKAGES_PATH}"/opencv_python-*.dist-info \
  "${SITE_PACKAGES_PATH}"/opencv_python*.libs \
  "${SITE_PACKAGES_PATH}"/opencv_python_headless-*.dist-info

ORIG_SITE_PACKAGES="/home/zhiheng/Vivid-VR/.venv/lib/python3.10/site-packages"
OPENCV_SOURCE_DIR=""
OPENCV_DIST_INFO_DIR=""
OPENCV_LIBS_DIR=""
if [[ -d "${ORIG_SITE_PACKAGES}/cv2" && -d "${ORIG_SITE_PACKAGES}/opencv_python-4.13.0.92.dist-info" ]]; then
  OPENCV_SOURCE_DIR="${ORIG_SITE_PACKAGES}/cv2"
  OPENCV_DIST_INFO_DIR="${ORIG_SITE_PACKAGES}/opencv_python-4.13.0.92.dist-info"
  OPENCV_LIBS_DIR="${ORIG_SITE_PACKAGES}/opencv_python.libs"
else
  OPENCV_SOURCE_DIR="$(find "${HOME}/.cache/uv/archive-v0" -path '*/cv2' | head -n 1)"
  OPENCV_DIST_INFO_DIR="$(find "${HOME}/.cache/uv/archive-v0" -path '*/opencv_python-4.13.0.92.dist-info' | head -n 1)"
  OPENCV_LIBS_DIR="$(find "${HOME}/.cache/uv/archive-v0" -path '*/opencv_python.libs' | head -n 1)"
fi

if [[ -z "${OPENCV_SOURCE_DIR}" || -z "${OPENCV_DIST_INFO_DIR}" || -z "${OPENCV_LIBS_DIR}" ]]; then
  "${UV_BIN}" pip install \
    --python "${VENV_PATH}/bin/python" \
    --index-strategy unsafe-best-match \
    --no-deps \
    opencv-python==4.13.0.92
else
  cp -a "${OPENCV_SOURCE_DIR}" "${SITE_PACKAGES_PATH}/"
  cp -a "${OPENCV_DIST_INFO_DIR}" "${SITE_PACKAGES_PATH}/"
  cp -a "${OPENCV_LIBS_DIR}" "${SITE_PACKAGES_PATH}/"
fi

env -u PYTHONPATH "${VENV_PATH}/bin/python" - <<'PY'
import importlib.util
import sys
from pathlib import Path

module_path = Path("/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py")
spec = importlib.util.spec_from_file_location(
    "vividvr_caption_sidecar_selfcheck",
    module_path,
)
if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load sidecar module for self-check: {module_path}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

header_path = module._ensure_python_dev_headers_for_sidecar()
if header_path is None:
    raise SystemExit(
        "Python dev headers not found; sidecar Triton caption kernels cannot be verified"
    )

print(f"Verified sidecar import without PYTHONPATH; python headers: {header_path}")
PY

echo "VividVR caption env is ready: ${VENV_PATH}"
