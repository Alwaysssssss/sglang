#!/usr/bin/env bash
# Read-only environment check; does not start containers or alter GPU workloads.
set -u
failed=0
for cmd in docker nvidia-smi; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd" >&2
    failed=1
  fi
done
if ! docker ps --format '{{.Names}} {{.Status}}'; then
  echo 'Cannot access Docker. Benchmark must not start.' >&2
  failed=1
fi
if ! nvidia-smi --id=0,1,2,3 --query-gpu=index,uuid,name,memory.used,memory.total --format=csv; then
  echo 'Cannot inspect GPUs 0-3. Benchmark must not start.' >&2
  failed=1
fi
if ! nvidia-smi topo -m; then
  failed=1
fi
exit "$failed"
