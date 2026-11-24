#!/usr/bin/env bash
set -euo pipefail
count=${VSR_NUM_GPUS:-}
if [[ -z $count ]]; then
    case ${VSR_MODE:-single} in
        single) count=1 ;;
        dual) count=2 ;;
        *) echo 'Set VSR_NUM_GPUS to a positive integer' >&2; exit 2 ;;
    esac
fi
[[ $count =~ ^[1-9][0-9]*$ && ${#count} -le 3 ]] || { echo 'Invalid VSR_NUM_GPUS (1..999)' >&2; exit 2; }
tile_args=()
if (( count > 1 )); then
    tile_args=(--tile-devices)
    for ((i=0; i<count; i++)); do tile_args+=("cuda:$i"); done
fi
checkpoint=${VSR_CHECKPOINT_DIR:-/models/checkpoint}
wan=${VSR_WAN_ROOT:-/models/wan}
output=${VSR_OUTPUT_DIR:-/output}
for path in "$checkpoint/transformer_ema" "$wan/vae"; do
    [[ -d "$path" ]] || { echo "Missing model directory: $path" >&2; exit 1; }
done
[[ -f "$checkpoint/vae_decoder_ema.pt" ]] || { echo 'Missing vae_decoder_ema.pt' >&2; exit 1; }
mkdir -p "$output" "${TORCHINDUCTOR_CACHE_DIR:-/cache/inductor}" "${TRITON_CACHE_DIR:-/cache/triton}"
python /opt/vsr/check_runtime.py --gpu --devices "$count"
# One SGLang scheduler; multi-GPU mode creates a persistent replica per visible GPU.
exec python /opt/vsr/serve_vsr.py --host 0.0.0.0 --port "${VSR_PORT:-5409}" \
    --checkpoint-dir "$checkpoint" --wan-root "$wan" --output-dir "$output" \
    "${tile_args[@]}" "$@"
