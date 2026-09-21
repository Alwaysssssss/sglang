#!/usr/bin/env bash
set -euo pipefail
mode=${VSR_MODE:-single}
case "$mode" in
    single) count=1; tile_args=() ;;
    dual) count=2; tile_args=(--tile-devices cuda:0 cuda:1) ;;
    *) echo 'VSR_MODE must be single or dual' >&2; exit 2 ;;
esac
checkpoint=${VSR_CHECKPOINT_DIR:-/models/checkpoint}
wan=${VSR_WAN_ROOT:-/models/wan}
output=${VSR_OUTPUT_DIR:-/output}
for path in "$checkpoint/transformer_ema" "$wan/vae"; do
    [[ -d "$path" ]] || { echo "Missing model directory: $path" >&2; exit 1; }
done
[[ -f "$checkpoint/vae_decoder_ema.pt" ]] || { echo 'Missing vae_decoder_ema.pt' >&2; exit 1; }
mkdir -p "$output" "${TORCHINDUCTOR_CACHE_DIR:-/cache/inductor}" "${TRITON_CACHE_DIR:-/cache/triton}"
python /opt/vsr/check_runtime.py --gpu --devices "$count"
# One SGLang scheduler; dual mode creates a persistent replica per visible GPU.
exec python /opt/vsr/serve_vsr.py --host 0.0.0.0 --port "${VSR_PORT:-30176}" \
    --checkpoint-dir "$checkpoint" --wan-root "$wan" --output-dir "$output" \
    "${tile_args[@]}" "$@"
