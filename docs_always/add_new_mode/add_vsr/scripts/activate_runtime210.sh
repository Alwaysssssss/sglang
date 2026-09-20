#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Source this file to select the VSR torch 2.10 / cu126 runtime.
vsr_runtime_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$vsr_runtime_repo/output_results/vsr/migration_env210/bin/activate"
export VE_SGLANG_PYTHON="$vsr_runtime_repo/output_results/vsr/migration_env210/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset vsr_runtime_repo
