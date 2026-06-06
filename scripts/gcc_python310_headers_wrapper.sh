#!/usr/bin/env bash
set -euo pipefail

# Triton launcher compilation on this machine resolves Python 3.10 headers to
# /usr/include/python3.10, which is absent. Reuse the locally extracted dev
# headers so fused CUDA kernels can JIT-compile during Phase E acceptance.
PY310_DEV_ROOT="/home/zhiheng/tmp_py310dev/extracted/usr/include"

exec /usr/bin/gcc \
  -I"${PY310_DEV_ROOT}" \
  -I"${PY310_DEV_ROOT}/python3.10" \
  -I"${PY310_DEV_ROOT}/x86_64-linux-gnu/python3.10" \
  "$@"
