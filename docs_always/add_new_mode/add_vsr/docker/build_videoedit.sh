#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../../.." && pwd)
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || { echo 'Usage: build_videoedit.sh [--dry-run]' >&2; exit 2; }
base=${VSR_BASE_IMAGE:-sglang-videoedit-src:l40s}
cmd=(docker build --platform linux/amd64 -f "$here/Dockerfile.videoedit"
    --build-arg "BASE_IMAGE=$base" -t "${VSR_IMAGE:-sglang-vsr:l40s}" "$repo")
if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
docker image inspect "$base" >/dev/null
exec "${cmd[@]}"
