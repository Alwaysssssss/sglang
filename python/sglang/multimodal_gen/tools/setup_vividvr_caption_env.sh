#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhiheng/sglang"
VENV_PATH="${REPO_ROOT}/.venv-vividvr-caption"
REQUIREMENTS_PATH="${REPO_ROOT}/python/requirements-vividvr-caption.txt"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_OFFICIAL_INDEX_URL="https://download.pytorch.org/whl/cu121"
PYTORCH_EXTRA_INDEX_URL="${PYTORCH_EXTRA_INDEX_URL:-${PYTORCH_OFFICIAL_INDEX_URL}}"
BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION:-0.44.1}"
BITSANDBYTES_INDEX_URL="${BITSANDBYTES_INDEX_URL:-${PYPI_INDEX_URL}}"
BITSANDBYTES_WHEEL_PATH="${BITSANDBYTES_WHEEL_PATH:-}"
PYTHON_DEV_HEADERS_ROOT="${PYTHON_DEV_HEADERS_ROOT:-${HOME}/tmp_py310_headers}"

NO_PROXY_ENV=(
  env
  -u http_proxy
  -u https_proxy
  -u HTTP_PROXY
  -u HTTPS_PROXY
  -u all_proxy
  -u ALL_PROXY
  -u no_proxy
  -u NO_PROXY
)

ensure_python_dev_headers() {
  local version="3.10"
  local package="libpython${version}-dev"
  local extract_root="${PYTHON_DEV_HEADERS_ROOT}/extracted/${package}"
  local include_dir="${extract_root}/usr/include/python${version}"
  local deb_dir="${PYTHON_DEV_HEADERS_ROOT}/debs"

  if [[ -f "${include_dir}/Python.h" ]]; then
    return 0
  fi

  if ! command -v apt >/dev/null 2>&1; then
    echo "apt is not available; cannot fetch ${package} without sudo" >&2
    return 1
  fi

  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb is not available; cannot extract ${package}" >&2
    return 1
  fi

  mkdir -p "${deb_dir}" "${extract_root}"

  (
    cd "${deb_dir}"
    rm -f "${package}"_*.deb
    "${NO_PROXY_ENV[@]}" apt download "${package}"
  )

  local deb_path
  deb_path="$(find "${deb_dir}" -maxdepth 1 -name "${package}_*.deb" | head -n 1)"
  if [[ -z "${deb_path}" ]]; then
    echo "failed to download ${package}" >&2
    return 1
  fi

  rm -rf "${extract_root}"
  mkdir -p "${extract_root}"
  dpkg-deb -x "${deb_path}" "${extract_root}"

  if [[ ! -f "${include_dir}/Python.h" ]]; then
    echo "downloaded ${package}, but Python.h is still missing" >&2
    return 1
  fi

  echo "Prepared Python dev headers at ${extract_root}"
}

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
  "${NO_PROXY_ENV[@]}" "${UV_BIN}" venv \
    --default-index "${PYPI_INDEX_URL}" \
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

TEMP_REQUIREMENTS_PATH="$(mktemp)"
trap 'rm -f "${TEMP_REQUIREMENTS_PATH}"' EXIT
awk \
  -v pytorch_extra_index_url="${PYTORCH_EXTRA_INDEX_URL}" \
  -v pytorch_official_index_url="${PYTORCH_OFFICIAL_INDEX_URL}" '
    $0 == "--extra-index-url https://download.pytorch.org/whl/cu121" {
      print "--extra-index-url " pytorch_extra_index_url
      if (pytorch_extra_index_url != pytorch_official_index_url) {
        print "--extra-index-url " pytorch_official_index_url
      }
      next
    }
    $0 == "bitsandbytes==0.44.1" {
      next
    }
    { print }
  ' "${REQUIREMENTS_PATH}" > "${TEMP_REQUIREMENTS_PATH}"

"${NO_PROXY_ENV[@]}" "${UV_BIN}" pip install \
  --python "${VENV_PATH}/bin/python" \
  --upgrade \
  --index-url "${PYPI_INDEX_URL}" \
  --index-strategy unsafe-best-match \
  pip setuptools wheel

echo "Installing caption dependencies from ${PYPI_INDEX_URL} with PyTorch index ${PYTORCH_EXTRA_INDEX_URL}"
"${NO_PROXY_ENV[@]}" "${UV_BIN}" pip install \
  --python "${VENV_PATH}/bin/python" \
  --index-url "${PYPI_INDEX_URL}" \
  --index-strategy unsafe-best-match \
  -r "${TEMP_REQUIREMENTS_PATH}"

if [[ -n "${BITSANDBYTES_WHEEL_PATH}" ]]; then
  echo "Installing bitsandbytes from local wheel ${BITSANDBYTES_WHEEL_PATH}"
  "${NO_PROXY_ENV[@]}" "${UV_BIN}" pip install \
    --python "${VENV_PATH}/bin/python" \
    --no-deps \
    "${BITSANDBYTES_WHEEL_PATH}"
else
  echo "Installing bitsandbytes ${BITSANDBYTES_VERSION} from ${BITSANDBYTES_INDEX_URL}"
  "${NO_PROXY_ENV[@]}" "${UV_BIN}" pip install \
    --python "${VENV_PATH}/bin/python" \
    --index-url "${BITSANDBYTES_INDEX_URL}" \
    --index-strategy unsafe-best-match \
    "bitsandbytes==${BITSANDBYTES_VERSION}"
fi

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
if [[ -d "${ORIG_SITE_PACKAGES}/cv2" && -d "${ORIG_SITE_PACKAGES}/opencv_python-4.13.0.92.dist-info" && -d "${ORIG_SITE_PACKAGES}/opencv_python.libs" ]]; then
  OPENCV_SOURCE_DIR="${ORIG_SITE_PACKAGES}/cv2"
  OPENCV_DIST_INFO_DIR="${ORIG_SITE_PACKAGES}/opencv_python-4.13.0.92.dist-info"
  OPENCV_LIBS_DIR="${ORIG_SITE_PACKAGES}/opencv_python.libs"
else
  while IFS= read -r opencv_dist_info_candidate; do
    opencv_archive_root="$(dirname "${opencv_dist_info_candidate}")"
    if [[ -d "${opencv_archive_root}/cv2" && -d "${opencv_archive_root}/opencv_python.libs" ]]; then
      OPENCV_SOURCE_DIR="${opencv_archive_root}/cv2"
      OPENCV_DIST_INFO_DIR="${opencv_dist_info_candidate}"
      OPENCV_LIBS_DIR="${opencv_archive_root}/opencv_python.libs"
      break
    fi
  done < <(find "${HOME}/.cache/uv/archive-v0" -path '*/opencv_python-4.13.0.92.dist-info')
fi

if [[ -z "${OPENCV_SOURCE_DIR}" || -z "${OPENCV_DIST_INFO_DIR}" || -z "${OPENCV_LIBS_DIR}" ]]; then
  "${NO_PROXY_ENV[@]}" "${UV_BIN}" pip install \
    --python "${VENV_PATH}/bin/python" \
    --index-url "${PYPI_INDEX_URL}" \
    --index-strategy unsafe-best-match \
    --no-deps \
    opencv-python==4.13.0.92
else
  cp -a "${OPENCV_SOURCE_DIR}" "${SITE_PACKAGES_PATH}/"
  cp -a "${OPENCV_DIST_INFO_DIR}" "${SITE_PACKAGES_PATH}/"
  cp -a "${OPENCV_LIBS_DIR}" "${SITE_PACKAGES_PATH}/"
fi

ensure_python_dev_headers || true

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
