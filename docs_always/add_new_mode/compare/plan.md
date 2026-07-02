
# Wan VideoEdit推理流程对比


## VideoEdit原始算法仓库的实现

- VideoEdit-diffusers：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py`
- VideoEdit-diffusers pipeline/model：`pipelines/pipeline_wan_edit.py`、`models/transformer_wan.py`、`models/autoencoder_kl_wan.py`、`models/flow_match.py`
- VideoEdit-diffusers utils：`utils/preprocess.py`、`utils/postprocess.py`

执行脚本

```bash
python3 infer.py --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers --output_name result --chunks 1 --clip_preprocess diffsynth
```

最终结果视频结果在

/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers 目录下

视频名称 

result.mp4

该算法已经在其他机器完成运行，在这个环境中之需要使用即可，不需要在运行



## sglang VideoEdit 

- SGLang pipeline：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- SGLang stages：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- SGLang VideoEdit runtime：`python/sglang/multimodal_gen/runtime/videoedit/`
- SGLang DiT/VAE：`python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`、`wan_videoedit.py`、`runtime/models/vaes/wanvae.py`

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

需完善参数，保证和

python3 infer.py --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers --output_name result --chunks 1 --clip_preprocess diffsynth

这个参数一致

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/reg.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare \
  --output-file-name result \
  --num-gpus 2 \
  --sp-degree 2 \
  --num-frames 80 \
  --infer-len 81 \
  --overlap 9 \
  --num-inference-steps 10 \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/1080p_videoedit_perf_1gpu_default.json
```

## 对比方案

- 原始算法已经完成运行，只对比最后视频结果
  - 注意首帧问题，可能原始算法首帧没有剔除，需要通过源码判断
  