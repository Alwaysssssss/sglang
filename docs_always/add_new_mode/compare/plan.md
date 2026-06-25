
# Wan VideoEdit推理流程对比

- SGLang pipeline：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- SGLang stages：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- SGLang VideoEdit runtime：`python/sglang/multimodal_gen/runtime/videoedit/`
- SGLang DiT/VAE：`python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`、`wan_videoedit.py`、`runtime/models/vaes/wanvae.py`
- VideoEdit-diffusers：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py`
- VideoEdit-diffusers pipeline/model：`pipelines/pipeline_wan_edit.py`、`models/transformer_wan.py`、`models/autoencoder_kl_wan.py`、`models/flow_match.py`
- VideoEdit-diffusers utils：`utils/preprocess.py`、`utils/postprocess.py`

## sglang的运行环境

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/reg.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare \
  --output-file-name sglang_wan_videoedit_mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --num-frames 80 \
  --infer-len 81 \
  --overlap 10 \
  --num-inference-steps 40 \
  --guidance-scale 5.0 \
  --seed 42 \
  --generator-device cpu \
  --dtype bf16 \
  --bbox-expand-scale 0.3 \
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