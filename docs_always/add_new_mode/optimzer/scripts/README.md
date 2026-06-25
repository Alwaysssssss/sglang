# VideoEdit Optimizer Scripts

These scripts execute the 50-step optimizer stages from `../optimizer.md`.
Run commands with `--dry-run` first to inspect the exact CLI, serve, submit, or compare action.

## Common Setup

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
```

The scripts use the same default paths as `optimizer.md`. Override them with environment variables when needed:

```bash
export MODEL_PATH=/path/to/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/path/to/transformer
export INPUT_VIDEO=/path/to/input.mp4
export INPUT_MASK=/path/to/mask.mp4
export OUT_DIR=/path/to/outputs
export QUANT_TRANSFORMER_PATH=/path/to/quantized/transformer
```

## Examples

List stages:

```bash
python docs_always/add_new_mode/optimzer/scripts/videoedit_optimizer.py list-stages
```

Dry-run a CLI stage:

```bash
docs_always/add_new_mode/optimzer/scripts/run_cli_stage.sh sp2_no_offload_fa --dry-run
```

Run a CLI stage:

```bash
docs_always/add_new_mode/optimzer/scripts/run_cli_stage.sh sp2_no_offload_fa
```

Start serve for a stage:

```bash
docs_always/add_new_mode/optimzer/scripts/start_serve_stage.sh sp2_no_offload_compile_fa --dry-run
```

Submit and poll a serve request:

```bash
docs_always/add_new_mode/optimzer/scripts/submit_stage.sh sp2_no_offload_compile_fa --dry-run
python docs_always/add_new_mode/optimzer/scripts/videoedit_optimizer.py poll sp2_no_offload_compile_fa
```

Compare an output:

```bash
docs_always/add_new_mode/optimzer/scripts/compare_stage.sh sp2_no_offload_fa --dry-run
docs_always/add_new_mode/optimzer/scripts/compare_stage.sh sp2_no_offload_fa
```

Run the script checks:

```bash
python docs_always/add_new_mode/optimzer/scripts/test_videoedit_optimizer.py
```


CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/home/tyx/workspace/zhouhao6/sglang/.venv/bin:$PATH PYTHONPATH=/home/tyx/workspace/zhouhao6/sglang/python /home/tyx/workspace/zhouhao6/sglang/.venv/bin/python


python -m sglang.multimodal_gen.runtime.videoedit.cli repair 
--model-path /home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model 
--transformer-path /home/tyx/workspace/difusser-model/step-55000-diffusers-lh/transformer
 --prompt "一个男人在舞台演讲，背后有两排文字。"
  --video-input-path /home/tyx/workspace/1080/1080.mp4 
  --mask-input-path /home/tyx/workspace/1080/mask_1080_merged.mp4 
  --reference-image-path /home/tyx/workspace/1080/local.png 
  --output-path /home/tyx/workspace/zhouhao6/outputs/1080_sgdit_2gpu_overlap10_repaired_native_skip_teacache_bbox10 
  --output-file-name 1080_sgdit_2gpu_overlap10_repaired_native_skip_teacache_bbox10.mp4 
  --perf-dump-path /home/tyx/workspace/zhouhao6/outputs/1080_sgdit_2gpu_overlap10_repaired_native_skip_teacache_bbox10/perf.json 
  --num-frames -1 
  --infer-len 81 
  --overlap 10 
  --num-inference-steps 40 
  --guidance-scale 5.0 
  --seed 42 
  --generator-device cpu 
  --dtype bf16 
  --bbox-expand-scale 1.0 
  --dilate-px 0 
  --mask-scale 1.0 
  --bbox-padding 0 
  --feather-px 0
  --adain-boundary-dilate 0 
  --enable-paste-back 
  --no-drop-reference-frame 
  --use-clip 
  --use-repaired-context 
  --init-latent-mode noise 
  --mask-downsample-mode nearest 
  --overlap-commit-mode native_skip 
  --tail-padding-mode native_reverse_mirror
   --decode-mode stream 
   --enable-teacache 
   --no-enable-torch-compile 
   --warmup 
   --warmup-steps 1
    --no-enable-frame-interpolation 
    --no-enable-upscaling 
    --num-gpus 2 
    --tp-size 1 
    --sp-degree 2 
    --ulysses-degree 2 
    --ring-degree 1 
    --dist-timeout -1 
    --dit-layerwise-offload --dit-offload-prefetch-size 0.0 --text-encoder-cpu-offload --image-encoder-cpu-offload --vae-cpu-offload --pin-cpu-memory


source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export MODEL_PATH=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer
export INPUT_VIDEO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/1080.mp4
export reference-image-path=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/local.png 
export INPUT_MASK=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/mask_1080_merged.mp4
export OUT_DIR=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs
--output-file-name=1080_sgdit_2gpu_overlap10_repaired_native_skip_teacache_bbox10.mp4
export PROMPT="一个男人在舞台演讲，背后有两排文字。"

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/local.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --output-file-name 1080_sgdit_2gpu_overlap9_0.3_40.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --num-frames 80 \
  --infer-len 81 \
  --overlap 10 \
  --num-inference-steps 10 \
  --guidance-scale 5.0 \
  --seed 42 \
  --generator-device cpu \
  --dtype bf16 \
  --bbox-expand-scale 1.5 \
  --dilate-px 0 \
  --mask-scale 1.0 \
  --bbox-padding 0 \
  --feather-px 0 \
  --adain-boundary-dilate 0 \
  --enable-paste-back \
  --no-drop-reference-frame \
  --use-clip \
  --use-repaired-context \
  --init-latent-mode noise \
  --mask-downsample-mode nearest \
  --overlap-commit-mode native_skip \
  --tail-padding-mode native_reverse_mirror \
  --decode-mode stream \
  --no-enable-torch-compile \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/1080p_videoedit_perf_1gpu_default.json
```

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/local.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --output-file-name 1080_sgdit_2gpu_overlap9_1.0_40.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --num-frames 80 \
  --infer-len 81 \
  --overlap 9 \
  --num-inference-steps 40 \
  --guidance-scale 5.0 \
  --seed 42 \
  --generator-device cpu \
  --dtype bf16 \
  --bbox-expand-scale 1.0 \
  --dilate-px 0 \
  --mask-scale 1.0 \
  --bbox-padding 0 \
  --feather-px 0 \
  --adain-boundary-dilate 0 \
  --enable-paste-back \
  --no-drop-reference-frame \
  --use-clip \
  --use-repaired-context \
  --init-latent-mode noise \
  --mask-downsample-mode nearest \
  --overlap-commit-mode native_skip \
  --tail-padding-mode native_reverse_mirror \
  --decode-mode stream \
  --no-enable-torch-compile \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/1080p_videoedit_perf_1gpu_default1.0.json

deactivate 

CUDA_VISIBLE_DEVICES=1
python infer.py --video_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/1080.mp4 \
  --mask_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/mask_1080_merged.mp4 \
  --img_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/local.png \
  --use_clip \
  --bbox_expand_scale 1.0 \
  --num_inference_steps 40 \
  --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers/output_1_0


CUDA_VISIBLE_DEVICES=0
python infer.py --video_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/1080.mp4 \
  --mask_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/mask_1080_merged.mp4 \
  --img_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/local.png \
  --use_clip \
  --bbox_expand_scale 0.3 \
  --num_inference_steps 40 \
  --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers/output_0_3


python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/1080/local.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --output-file-name 1080_sgdit_2gpu_overlap9_1.5_40.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --num-frames 80 \
  --infer-len 81 \
  --overlap 9 \
  --num-inference-steps 40 \
  --guidance-scale 5.0 \
  --seed 42 \
  --generator-device cpu \
  --dtype bf16 \
  --bbox-expand-scale 1.5 \
  --dilate-px 0 \
  --mask-scale 1.0 \
  --bbox-padding 0 \
  --feather-px 0 \
  --adain-boundary-dilate 0 \
  --enable-paste-back \
  --no-drop-reference-frame \
  --use-clip \
  --use-repaired-context \
  --init-latent-mode noise \
  --mask-downsample-mode nearest \
  --overlap-commit-mode native_skip \
  --tail-padding-mode native_reverse_mirror \
  --decode-mode stream \
  --no-enable-torch-compile \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/1080p_videoedit_perf_2gpu_default1.5.json
  