#!/usr/bin/env bash
set -euo pipefail
# Execute inside the server container so video URLs and callback localhost work.
container=${VSR_CONTAINER:-vsr-api}
cmd=(docker exec "$container" bash -c '
    set -euo pipefail
    resolution=320x640
    if [[ ${VSR_MODE:-single} == dual ]]; then resolution=320x1216; fi
    python /opt/vsr/warmup.py --input "$1"
    exec python /opt/vsr/test_server_api.py \
        --base-url "http://127.0.0.1:${VSR_PORT:-30176}" \
        --input "$1" --output-dir "/output/acceptance-$(date +%s)" \
        --warmup-resolution "$resolution"
' _ "${VSR_TEST_INPUT:-/input/input.mp4}")
if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ $# == 0 ]] || { echo 'Usage: smoke.sh [--dry-run]' >&2; exit 2; }
exec "${cmd[@]}"
