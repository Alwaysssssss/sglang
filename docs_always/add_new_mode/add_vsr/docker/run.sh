#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || { echo 'Usage: run.sh [--dry-run]' >&2; exit 2; }
count=${VSR_NUM_GPUS:-}
if [[ -z $count ]]; then
    case ${VSR_MODE:-} in
        single) count=1 ;;
        dual) count=2 ;;
        "") if [[ -n ${VSR_GPUS:-} ]]; then
                IFS=, read -r -a selected <<< "$VSR_GPUS"
                count=${#selected[@]}
            else count=1; fi ;;
        *) echo 'Use VSR_NUM_GPUS, or VSR_MODE=single/dual' >&2; exit 2 ;;
    esac
fi
[[ $count =~ ^[1-9][0-9]*$ && ${#count} -le 3 ]] || { echo 'VSR_NUM_GPUS must be a positive integer (1..999)' >&2; exit 2; }
devices=${VSR_GPUS:-}
if [[ -z $devices ]]; then
    for ((i=0; i<count; i++)); do devices+="${devices:+,}$i"; done
fi
[[ $devices =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Use numeric host GPU IDs, e.g. 0 or 2,3' >&2; exit 2; }
IFS=, read -r -a gpu_ids <<< "$devices"
[[ ${#gpu_ids[@]} == "$count" ]] || { echo "Expected $count GPU IDs, got $devices" >&2; exit 2; }
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu == 0 || $gpu =~ ^[1-9][0-9]*$ ]] || { echo 'GPU IDs must not have leading zeroes' >&2; exit 2; }
    [[ -z ${seen[$gpu]:-} ]] || { echo 'GPU IDs must be distinct' >&2; exit 2; }
    seen[$gpu]=1
done
mode=multi
if [[ $count == 1 ]]; then mode=single; elif [[ $count == 2 ]]; then mode=dual; fi
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
    -e "VSR_MODE=$mode" -e "VSR_NUM_GPUS=$count" -e VSR_PORT=30176
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
