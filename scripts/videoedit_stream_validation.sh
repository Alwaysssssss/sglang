#!/usr/bin/env bash
# Uses existing environments without installing or changing packages.
set -euo pipefail
VE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VE_RUN_DIR="$VE_REPO/outputs/videoedit-stream-validation"
VE_NATIVE=/mnt/shanhai-ai/liuh/VideoEdit-diffusers
VE_CASE="$VE_NATIVE/datas/edit_val_cases/0008"
VE_MODEL=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
VE_BASE=/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/VideoEdit_diffusers/pretrain_models/Wan2.1-I2V-14B-480P-Diffusers
VE_PACKAGES=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/lib/python3.11/site-packages
# PYTHONPATH does not process the existing nvidia_cutlass_dsl.pth file.
VE_CUTLASS_PACKAGES="$VE_PACKAGES/nvidia_cutlass_dsl/python_packages"
VE_ALGO_PACKAGES=/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/lib/python3.11/site-packages
VE_FRAMES="${VE_FRAMES:-48}"
VE_STEPS="${VE_STEPS:-40}"
VE_REF="${VE_REF:-0}"
VE_NAME="case0008_${VE_FRAMES}f_${VE_STEPS}s_ref${VE_REF}_tight"
mkdir -p "$VE_RUN_DIR/tmp" "$VE_RUN_DIR/cache" "$VE_RUN_DIR/reference" "$VE_RUN_DIR/sglang"
export TMPDIR="$VE_RUN_DIR/tmp" XDG_CACHE_HOME="$VE_RUN_DIR/cache"
export HF_HOME="$VE_RUN_DIR/cache/huggingface" TORCH_HOME="$VE_RUN_DIR/cache/torch"
export TRITON_CACHE_DIR="$VE_RUN_DIR/cache/triton" PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$VE_RUN_DIR/cache/extensions"
export FLASHINFER_WORKSPACE_BASE="$VE_RUN_DIR/cache/flashinfer"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${VE_GPU:-6}"
VE_PROMPT="$(tr -d '\r\n' < "$VE_CASE/prompt.txt")"
case "${1:-}" in
reference)
    cd "$VE_NATIVE"
    sha256sum infer.py utils/video_io_ffmpeg.py utils/chunk_plan.py \
        utils/mask_stabilize.py utils/postprocess.py utils/preprocess.py \
        > "$VE_RUN_DIR/${VE_NAME}_reference_sources.sha256"
    /mnt/shanhai-ai/envs/conda/envs/edit_ruidi/bin/python -B infer.py \
        --video_path "$VE_CASE/video.mp4" --mask_path "$VE_CASE/mask.mp4" \
        --img_path "$VE_CASE/reference.png" --model_path "$VE_BASE" \
        --transformer_path "$VE_NATIVE/ckpts/step_47500" \
        --output_dir "$VE_RUN_DIR/reference" --output_name "$VE_NAME" \
        --prompt "$VE_PROMPT" --num_frames "$VE_FRAMES" --ref_frame_idx "$VE_REF" \
        --infer_len 49 --overlap 5 --bridge_overlap 5 --num_inference_steps "$VE_STEPS" \
        --guidance_scale 5 --seed 42 --dtype bf16 --dynamic_cfg --vae_tiling \
        --use_clip --clip_preprocess diffuser --chunk_bbox_mode tight \
        --bbox_expand_scale 1.6 --bbox_padding 0 --dilate_px 8 --mask_scale 1 \
        --feather_px 8 --adain_boundary_dilate 0 --save_paste --save_crop \
        --no_save_color --no_keep_intermediate \
        2>&1 | tee "$VE_RUN_DIR/${VE_NAME}_reference.log"
    exit 0
    ;;
sglang|check-runtime)
    cd "$VE_REPO"
    export PYTHONPATH="$VE_REPO/python:/opt/conda/lib/python3.11:$VE_PACKAGES:$VE_CUTLASS_PACKAGES:$VE_ALGO_PACKAGES"
    export SGLANG_CACHE_DIT_ENABLED=false
    if [[ "$1" == check-runtime ]]; then
        exec /opt/conda/bin/python -B "$VE_REPO/scripts/videoedit_existing_env.py" --check-runtime
    fi
    /opt/conda/bin/python -B "$VE_REPO/scripts/videoedit_existing_env.py" repair \
        --model-path "$VE_MODEL" --transformer-path "$VE_MODEL/transformer" \
        --video-input-path "$VE_CASE/video.mp4" --mask-input-path "$VE_CASE/mask.mp4" \
        --reference-image-path "$VE_CASE/reference.png" --prompt "$VE_PROMPT" \
        --output-path "$VE_RUN_DIR/sglang" --output-file-name "$VE_NAME.mp4" \
        --num-frames "$VE_FRAMES" --ref-frame-idx "$VE_REF" --infer-len 49 \
        --overlap 5 --bridge-overlap 5 --num-inference-steps "$VE_STEPS" \
        --guidance-scale 5 --seed 42 --dtype bf16 --dynamic-cfg \
        --use-clip --clip-preprocess diffuser --chunk-bbox-mode tight \
        --bbox-expand-scale 0.3 --bbox-padding 0 --dilate-px 8 --mask-scale 1 \
        --feather-px 8 --adain-boundary-dilate 0 --enable-paste-back --save-crop-only \
        --decode-mode stream --preserve-audio --no-dit-cpu-offload \
        --dit-layerwise-offload --dit-offload-prefetch-size 0 \
        --text-encoder-cpu-offload --image-encoder-cpu-offload --vae-cpu-offload \
        --pin-cpu-memory --no-enable-teacache --no-enable-frame-interpolation \
        --no-enable-upscaling --num-gpus 1 --tp-size 1 --sp-degree 1 \
        --ulysses-degree 1 --ring-degree 1 --attention-backend torch_sdpa \
        --master-port "${VE_MASTER_PORT:-30005}" --scheduler-port "${VE_SCHEDULER_PORT:-5565}" \
        --perf-dump-path "$VE_RUN_DIR/sglang/${VE_NAME}_perf.json" \
        2>&1 | tee "$VE_RUN_DIR/${VE_NAME}_sglang.log"
    exit 0
    ;;
compare)
    cd "$VE_REPO"
    VE_COMPARE_STATUS=0
    for VE_VARIANT in crop_only full; do
        VE_SUFFIX="_${VE_VARIANT}"
        VE_SSIM=0.97
        if [[ "$VE_VARIANT" == full ]]; then VE_SUFFIX=""; VE_SSIM=0.98; fi
        PYTHONPATH="$VE_REPO/python:$VE_PACKAGES" /opt/conda/bin/python -B \
            python/sglang/multimodal_gen/runtime/videoedit/compare.py \
            --reference "$VE_RUN_DIR/reference/$VE_NAME$VE_SUFFIX.mp4" \
            --candidate "$VE_RUN_DIR/sglang/$VE_NAME$VE_SUFFIX.mp4" \
            --report-json "$VE_RUN_DIR/${VE_NAME}_${VE_VARIANT}_compare.json" \
            --min-ssim "$VE_SSIM" --max-mse 25 --max-mae 2.5 \
            --allow-frame-count-delta 0 --max-failed-frame-ratio 0 || VE_COMPARE_STATUS=1
    done
    exit "$VE_COMPARE_STATUS"
    ;;
*) echo "Usage: bash $0 reference|sglang|compare|check-runtime" >&2; exit 2 ;;
esac
