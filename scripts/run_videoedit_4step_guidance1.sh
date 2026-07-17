#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_PYTHON="/home/tyx/workspace/zhouhao6/sglang/.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
  else
    PYTHON_BIN="python3"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-/home/tyx/workspace/difusser-model/step-55000-diffusers-lh/transformer}"
VIDEO_INPUT_PATH="${VIDEO_INPUT_PATH:-/home/tyx/workspace/1080/1080.mp4}"
MASK_INPUT_PATH="${MASK_INPUT_PATH:-/home/tyx/workspace/1080/mask_1080_merged.mp4}"
REFERENCE_IMAGE_PATH="${REFERENCE_IMAGE_PATH:-/home/tyx/workspace/1080/local.png}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/videoedit_4step_guidance1}"
OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME:-videoedit_4step_guidance1.mp4}"
PERF_DUMP_PATH="${PERF_DUMP_PATH:-${OUTPUT_PATH}/perf.json}"
PROMPT="${PROMPT:-一个男人在舞台演讲，背后有两排文字。}"

mkdir -p "${OUTPUT_PATH}"

"${PYTHON_BIN}" -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "${MODEL_PATH}" \
  --transformer-path "${TRANSFORMER_PATH}" \
  --prompt "${PROMPT}" \
  --video-input-path "${VIDEO_INPUT_PATH}" \
  --mask-input-path "${MASK_INPUT_PATH}" \
  --reference-image-path "${REFERENCE_IMAGE_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --output-file-name "${OUTPUT_FILE_NAME}" \
  --perf-dump-path "${PERF_DUMP_PATH}" \
  --num-frames "${NUM_FRAMES:-80}" \
  --infer-len "${INFER_LEN:-81}" \
  --overlap "${OVERLAP:-5}" \
  --num-inference-steps 4 \
  --guidance-scale 1.0 \
  --seed "${SEED:-42}" \
  --generator-device "${GENERATOR_DEVICE:-cpu}" \
  --dtype "${DTYPE:-bf16}" \
  --bbox-expand-scale "${BBOX_EXPAND_SCALE:-1.0}" \
  --dilate-px "${DILATE_PX:-0}" \
  --mask-scale "${MASK_SCALE:-1.0}" \
  --bbox-padding "${BBOX_PADDING:-0}" \
  --feather-px "${FEATHER_PX:-0}" \
  --adain-boundary-dilate "${ADAIN_BOUNDARY_DILATE:-0}" \
  --enable-paste-back \
  --no-drop-reference-frame \
  --use-clip \
  --use-repaired-context \
  --init-latent-mode noise \
  --mask-downsample-mode nearest \
  --overlap-commit-mode native_skip \
  --tail-padding-mode native_reverse_mirror \
  --decode-mode "${DECODE_MODE:-stream}" \
  --no-enable-teacache \
  --no-enable-torch-compile \
  --warmup \
  --warmup-steps "${WARMUP_STEPS:-1}" \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --num-gpus "${NUM_GPUS:-2}" \
  --tp-size "${TP_SIZE:-1}" \
  --sp-degree "${SP_DEGREE:-2}" \
  "$@"
