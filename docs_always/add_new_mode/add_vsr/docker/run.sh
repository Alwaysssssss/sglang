#!/usr/bin/env bash
set -euo pipefail
mode=${VSR_MODE:-single}
case "$mode" in
    single) devices=${VSR_GPUS:-7}; count=1 ;;
    dual) devices=${VSR_GPUS:-6,7}; count=2 ;;
    *) echo 'VSR_MODE must be single or dual' >&2; exit 2 ;;
esac
IFS=, read -r -a gpu_ids <<< "$devices"
[[ ${#gpu_ids[@]} == "$count" ]] || { echo "Mode $mode requires $count GPU IDs" >&2; exit 2; }
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || { echo 'Use numeric host GPU IDs, e.g. 7 or 6,7' >&2; exit 2; }
done
if [[ $count == 2 && ${gpu_ids[0]} == "${gpu_ids[1]}" ]]; then
    echo 'Dual mode requires two different GPUs' >&2; exit 2
fi
: "${VSR_CHECKPOINT_HOST:?Set VSR_CHECKPOINT_HOST to checkpoint-1300}"
: "${VSR_WAN_HOST:?Set VSR_WAN_HOST to the directory containing vae/}"
: "${VSR_INPUT_HOST:?Set VSR_INPUT_HOST to the input directory}"
: "${VSR_OUTPUT_HOST:?Set VSR_OUTPUT_HOST to the output directory}"
: "${VSR_CACHE_HOST:?Set VSR_CACHE_HOST to the compilation cache directory}"
[[ -d "$VSR_CHECKPOINT_HOST/transformer_ema" && -f "$VSR_CHECKPOINT_HOST/vae_decoder_ema.pt" ]] || { echo 'Invalid checkpoint directory' >&2; exit 1; }
[[ -d "$VSR_WAN_HOST/vae" && -d "$VSR_INPUT_HOST" ]] || { echo 'Missing Wan VAE or input directory' >&2; exit 1; }
for path in "$VSR_CHECKPOINT_HOST" "$VSR_WAN_HOST" "$VSR_INPUT_HOST" "$VSR_OUTPUT_HOST" "$VSR_CACHE_HOST"; do
    [[ $path == /* && $path != *,* ]] || { echo 'Mount paths must be absolute and contain no commas' >&2; exit 2; }
done
# Docker --gpus parses CSV: retain quotes around the whole device=6,7 field.
cmd=(docker run -d --init --name "${VSR_CONTAINER:-vsr-api}" --gpus "\"device=$devices\""
    --shm-size "${VSR_SHM_SIZE:-16g}" --stop-timeout 60
    -p "${VSR_BIND_IP:-127.0.0.1}:${VSR_HOST_PORT:-30176}:30176"
    -e "VSR_MODE=$mode" -e VSR_PORT=30176
    --mount "type=bind,src=$VSR_CHECKPOINT_HOST,dst=/models/checkpoint,readonly"
    --mount "type=bind,src=$VSR_WAN_HOST,dst=/models/wan,readonly"
    --mount "type=bind,src=$VSR_INPUT_HOST,dst=/input,readonly"
    --mount "type=bind,src=$VSR_OUTPUT_HOST,dst=/output"
    --mount "type=bind,src=$VSR_CACHE_HOST,dst=/cache"
    "${VSR_IMAGE:-sglang-vsr:torch210-cu126}")
if [[ ${1:-} == --dry-run ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: run.sh [--dry-run]' >&2; exit 2; }
command -v docker >/dev/null || { echo 'Run this script on a Docker GPU host.' >&2; exit 1; }
mkdir -p "$VSR_OUTPUT_HOST" "$VSR_CACHE_HOST"
exec "${cmd[@]}"
