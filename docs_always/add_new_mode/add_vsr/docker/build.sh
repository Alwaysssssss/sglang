#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../../.." && pwd)
image=${VSR_IMAGE:-sglang-vsr:torch210-cu126}
wheel=${VSR_KERNEL_WHEEL:-$repo/output_results/vsr/kernel210-wheel/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl}
if [[ -z ${VSR_KERNEL_WHEEL:-} && ! -f "$wheel" ]]; then
    wheel=$here/vendor/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl
fi
expected=48832f11f26134a2b0cc0234bfa1f7872c901ccf7b0835715d100f38c91b3e49
[[ -f "$wheel" ]] || { echo "Missing validated kernel wheel: $wheel" >&2; exit 1; }
actual=$(sha256sum -- "$wheel")
[[ ${actual%% *} == "$expected" ]] || { echo 'Kernel SHA256 differs from the validated torch 2.10 build.' >&2; exit 1; }
cmd=(docker build --platform linux/amd64 -f "$here/Dockerfile" -t "$image")
if [[ -n ${VSR_BASE_IMAGE:-} ]]; then cmd+=(--build-arg "BASE_IMAGE=$VSR_BASE_IMAGE"); fi
cmd+=("$repo")
if [[ ${1:-} == --dry-run ]]; then
    printf 'Verified kernel SHA256: %s\n' "$expected"
    printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: build.sh [--dry-run]' >&2; exit 2; }
command -v docker >/dev/null || { echo 'Run this script on a Docker build host.' >&2; exit 1; }
mkdir -p "$here/vendor"
destination=$here/vendor/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl
if [[ $(realpath "$wheel") != "$(realpath -m "$destination")" ]]; then cp -- "$wheel" "$destination"; fi
printf '%s  %s\n' "$expected" sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl > "$here/vendor/SHA256SUMS"
if [[ -f "$repo/output_results/vsr/kernel210_wheel_provenance.json" ]]; then
    cp -- "$repo/output_results/vsr/kernel210_wheel_provenance.json" "$here/vendor/provenance.json"
fi
exec "${cmd[@]}"
