#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
vsr_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$vsr_repo_root/output_results/vsr/migration_env210/bin/python" "$@"
