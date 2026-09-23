#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
export OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4
OUT="${1:-output_results/vsr/all_optimizations_20260923}"
SCRIPT=docs_always/add_new_mode/add_vsr/scripts/benchmark_all.py
REF_PY=/mnt/shanhai-ai/envs/conda/envs/swiftvr/bin/python
OPT_PY=output_results/vsr/migration_env210/bin/python
mkdir -p "$OUT"
nvidia-smi > "$OUT/gpu_start.txt"
CUDA_VISIBLE_DEVICES=3 "$OPT_PY" -u "$SCRIPT" --implementation candidate --gpus 1 --output-dir "$OUT/single" > "$OUT/single.log" 2>&1 &
single_pid=$!
CUDA_VISIBLE_DEVICES=6,7 "$OPT_PY" -u "$SCRIPT" --implementation candidate --gpus 2 --output-dir "$OUT/dual" > "$OUT/dual.log" 2>&1 &
dual_pid=$!
printf 'single=%s dual=%s\n' "$single_pid" "$dual_pid"
failed=0
for pid in "$single_pid" "$dual_pid"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo 'A first-stage benchmark failed; inspect logs.'
  exit 1
fi
echo 'First stage complete; starting three GPUs.'
CUDA_VISIBLE_DEVICES=3,6,7 "$OPT_PY" -u "$SCRIPT" --implementation candidate --gpus 3 --output-dir "$OUT/triple" > "$OUT/triple.log" 2>&1
echo 'Starting original reference on GPU 3.'
CUDA_VISIBLE_DEVICES=3 "$REF_PY" -u "$SCRIPT" --implementation reference --output-dir "$OUT/reference_gpu3" > "$OUT/reference_gpu3.log" 2>&1
nvidia-smi > "$OUT/gpu_end.txt"
echo 'All benchmarks complete.'
