#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Measure the per-configuration eps_torch floor: run the *reference*
# implementation of every configuration in requirements.md §3.2 once per
# environment, then compare each pair.
#
# Why every configuration: eps_torch turned out to be configuration-dependent.
# A 320x640 run measured mae_max = 2.78 against the 2.5 gate, while the 4K run
# measured 1.13 -- so the floor has to be established per configuration, not
# extrapolated from one. See requirements.md §4.1.2.
#
# Two environments are compared here, not two implementations:
#     swiftvr  -- the reference env (torch 2.10.0+cu126, diffusers 0.36.0)
#     sglang   -- the SGLang env (torch 2.9.1+cu128, diffusers 0.37.0)
# The difference between them is a floor that no later implementation can go
# below; it is not a measure of the SGLang port's correctness.
#
# Usage: sweep_eps_torch.sh [GPU] [CONFIG_FILTER]      CONFIG_FILTER is a
#        comma-separated list, e.g. "A3,C2,C3". SEQUENTIAL=1 runs the two
#        environments one after the other instead of concurrently.
set -u

GPU=${1:-2}
FILTER=${2:-}

export VSR_REPO=${VSR_REPO:-/mnt/shanhai-ai/shanhai-workspace/zhouhao6/vsr}
export SGLANG_REPO=${SGLANG_REPO:-/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang}
VE_SGLANG_PYTHON=${VE_SGLANG_PYTHON:-/home/root/uv-envs/sglang-llm-diffusion/bin/python}
VSR_PY=${VSR_PY:-/mnt/shanhai-ai/envs/conda/envs/swiftvr/bin/python}
WAN=${VSR_WAN_ROOT:-/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers}
CKPT=${VSR_CHECKPOINT:-/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300}

OUT=$SGLANG_REPO/output_results/vsr
DUMPS=$OUT/dumps
VERIFY=$SGLANG_REPO/python/sglang/multimodal_gen/runtime/vsr/verify
MEDIA=$OUT/media
TEST_MEDIA=/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/zy_test

for f in "$WAN/vae/config.json" "$CKPT/transformer_ema/config.json" "$CKPT/vae_decoder_ema.pt"; do
    [ -s "$f" ] || { echo "missing weight: $f" >&2; exit 1; }
done

# NAME | input | target_resolution (HxW) | color_ref
#
# A1 is first and is a *control*: its numbers are already known from M1
# (frames: ssim_min 0.997850 / mse_max 0.4426 / mae_max 0.3453). If the sweep
# does not reproduce them, the sweep itself is wrong and the rest is worthless.
CONFIGS=(
  "A1|$VSR_REPO/input/input.mp4|3840x2160|global"
  "A2|$VSR_REPO/input/input.mp4|3840x2160|chunk"
  "A3|$VSR_REPO/input/input.mp4|3840x2160|none"
  "B1|$TEST_MEDIA/val_input_480x832.mp4|480x832|global"
  "B2|$TEST_MEDIA/val_input_512x512.mp4|512x512|global"
  "B3|$TEST_MEDIA/val_input_768x1280.mp4|768x1280|global"
  # C1/C2 target resolution: near-native, NOT 320x640. They exist to cover the
  # multi-window paths; a heavy downscale pushed their eps_torch above the mp4
  # mae gate and made them uninformative at that layer. See requirements.md §3.2.
  "C1|$MEDIA/loop64.mp4|1920x1080|global"
  "C2|$MEDIA/loop200.mp4|1920x1080|global"
  "C3|$MEDIA/loop64.mp4|2160x3840|global"
)

#: Free VRAM a single run needs, in MiB. One run peaks around 14-16 GiB; the
#: margin absorbs other tenants growing mid-run.
GPU_NEED_MIB=${GPU_NEED_MIB:-20000}
#: How long to wait for that much free VRAM before giving up, in seconds.
GPU_WAIT_MAX=${GPU_WAIT_MAX:-2400}
#: How many times to retry a run that died with CUDA OOM.
RETRIES=${RETRIES:-4}

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 0
}

wait_for_gpu() {
    local waited=0
    while [ "$(gpu_free_mib)" -lt "$GPU_NEED_MIB" ]; do
        if [ "$waited" -ge "$GPU_WAIT_MAX" ]; then
            echo "  ! GPU $GPU still has only $(gpu_free_mib) MiB free after ${waited}s; giving up"
            return 1
        fi
        sleep 30
        waited=$((waited + 30))
        [ $((waited % 300)) -eq 0 ] && \
            echo "  ... waiting for GPU $GPU (need ${GPU_NEED_MIB} MiB, free $(gpu_free_mib) MiB, waited ${waited}s)"
    done
    return 0
}

run_one() {  # <env-name> <python> <config-line>
    local env_name=$1 py=$2 line=$3
    IFS='|' read -r name input target color <<< "$line"
    local root=$DUMPS/EPS_${name}_${env_name}
    local log=$OUT/EPS_${name}_${env_name}.log
    local attempt=1 st=0

    while :; do
        rm -rf "$root"
        wait_for_gpu || { st=1; break; }
        echo "[$(date +%H:%M:%S)] $name/$env_name start (attempt $attempt, free $(gpu_free_mib) MiB)"
        CUDA_VISIBLE_DEVICES=$GPU "$py" -B "$VERIFY/dump_baseline.py" \
            --vsr-repo "$VSR_REPO" --dump-root "$root" --dump-frames \
            -- --input "$input" --output "$OUT/EPS_${name}_${env_name}.mp4" \
               --checkpoint_dir "$CKPT" --wan_root "$WAN" \
               --target_resolution "$target" --tile_h 320 --tile_w 640 --tile_t 33 \
               --color_ref "$color" --read_queue 2 \
            > "$log" 2>&1
        # Capture the status *before* any command substitution: `echo "... $?"`
        # would report the status of the substitution, not of the run.
        st=$?
        echo "[$(date +%H:%M:%S)] $name/$env_name exit=$st"
        [ "$st" -eq 0 ] && break
        # Only OOM is worth retrying; any other failure will repeat itself.
        if [ "$attempt" -lt "$RETRIES" ] && grep -q "OutOfMemoryError" "$log"; then
            attempt=$((attempt + 1))
            echo "  OOM; retrying in 60s"
            sleep 60
            continue
        fi
        break
    done
    return $st
}

for line in "${CONFIGS[@]}"; do
    name=${line%%|*}
    if [ -n "$FILTER" ] && [[ ",$FILTER," != *",$name,"* ]]; then
        continue
    fi
    # Parallel is faster but needs ~2x the VRAM. These GPUs are shared, so when
    # another job grows, both runs die with CUDA OOM -- seen 2026-09-18 on A3,
    # C2 and C3. Use SEQUENTIAL=1 when the card is crowded.
    if [ -n "${SEQUENTIAL:-}" ]; then
        run_one swiftvr "$VSR_PY"           "$line"
        run_one sglang  "$VE_SGLANG_PYTHON" "$line"
    else
        run_one swiftvr "$VSR_PY"          "$line" &
        P1=$!
        run_one sglang  "$VE_SGLANG_PYTHON" "$line" &
        P2=$!
        wait $P1 $P2
    fi

    # Compare the pair: structural gate first, then frames, then the mp4s.
    R=$DUMPS/EPS_${name}_swiftvr
    C=$DUMPS/EPS_${name}_sglang
    "$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.vsr.verify.structural_check \
        --reference "$OUT/EPS_${name}_swiftvr.mp4" --candidate "$OUT/EPS_${name}_sglang.mp4" \
        --report-json "$OUT/reports/EPS_${name}_structural.json" > "$OUT/reports/EPS_${name}_structural.txt" 2>&1
    "$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.vsr.verify.compare_frames \
        --reference-dir "$R" --candidate-dir "$C" \
        --report-json "$OUT/reports/EPS_${name}_frames.json" > "$OUT/reports/EPS_${name}_frames.txt" 2>&1
    "$VE_SGLANG_PYTHON" "$SGLANG_REPO/python/sglang/multimodal_gen/runtime/videoedit/compare.py" \
        --reference "$OUT/EPS_${name}_swiftvr.mp4" --candidate "$OUT/EPS_${name}_sglang.mp4" \
        --report-json "$OUT/reports/EPS_${name}_mp4.json" \
        --min-ssim 0.97 --max-mse 25.0 --max-mae 2.5 \
        --allow-frame-count-delta 0 --max-failed-frame-ratio 0.0 \
        > "$OUT/reports/EPS_${name}_mp4.txt" 2>&1
    echo "[$(date +%H:%M:%S)] $name compared"
done

# Summary table.
"$VE_SGLANG_PYTHON" - "$OUT/reports" "${CONFIGS[@]}" <<'PY'
import json, os, sys
reports, lines = sys.argv[1], sys.argv[2:]
print(f"{'cfg':4s} {'color':7s} {'layer':7s} {'ssim_min':>9s} {'mse_max':>9s} {'mae_max':>9s} {'fail':>5s}")
for line in lines:
    name = line.split("|")[0]
    color = line.split("|")[3]
    for layer, fn in (("frames", f"EPS_{name}_frames.json"), ("mp4", f"EPS_{name}_mp4.json")):
        p = os.path.join(reports, fn)
        if not os.path.exists(p):
            print(f"{name:4s} {color:7s} {layer:7s} {'<missing>':>9s}")
            continue
        s = json.load(open(p))["summary"]
        print(f"{name:4s} {color:7s} {layer:7s} {s['ssim_min']:9.6f} {s['mse_max']:9.4f} "
              f"{s['mae_max']:9.4f} {len(s['failed_frames']):5d}")
PY
