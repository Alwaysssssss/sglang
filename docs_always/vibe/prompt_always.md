

请依照 @python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model skill 的方法，将 @../VideoEdit-diffusers 的视频编辑模型作为新模型接入，并在 @docs_always/add_new_mode/add_videoedit_diffusers/README.md 中详细完善方案。完善时需重点解决以下逻辑：

1. 脱离原 VideoEdit-diffusers 仓库的耦合，确保集成到 sglang 后，不依赖原 repo 的路径、数据结构和私有调用；
2. 采用 SGLang 推荐的扩展方式，优先复用已有 VAE、DiT 主体和 pipeline 设计，仅补充 VideoEdit 专属的数据组装、pipeline stage 或接口适配代码，避免冗余复制已有通用逻辑；
3. 设计清晰的模型/预处理/后处理解耦机制，以便 sglang 未来升级或 VideoEdit-diffusers 仓库变更后，可便捷同步新特性并无缝合并新版，无需大改现有集成代码；
4. 方案里明确接口分层、数据流转与模块边界，便于后续维护和自动化对齐 upstream 变更。

方案目标：实现模块边界清晰、可配置、松耦合的集成方式，使 sglang 升级与新模型并存无障碍合并。

完善方案，加强输出结果的校验机制：集成后生成的视频结果需与原始 VideoEdit-diffusers 仓库的 infer.py 输出结果逐帧对齐。对齐方式为针对每一帧采用结构相似度（如 SSIM）、均方误差（MSE）等图像处理算法进行自动对比，可容忍微小容差，但不得出现大范围失配或整帧错误。请实现自动化对比和统计功能，便于回归验证与迭代优化，确保改造集成后模型输出在视觉效果和数值尺度上高度一致。建议将对齐逻辑封装为独立测试脚本或模块，便于 CICD 持续回归和 future update 跟进。

https://github.com/shanhai-mgtv/VideoEdit-diffusers
/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410
/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/prompt.txt prompt直接在这个文件内拿匹配的就行

    g.add_argument(
        "--model_path",
        default="/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/"
                "VideoEdit-diffusers/pretrain_models/Wan2.1-I2V-14B-480P-Diffusers/",
        help="Base Wan2.1 diffusers model directory",
    )
    g.add_argument(
        "--transformer_path",
        default="/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/"
                "VideoEdit-diffusers/utils/wan_converted_step_9500/",
        help="Fine-tuned transformer checkpoint directory",
    )

python3 infer.py --video_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/videos/1144932-hd_1920_1080_30fps_short.mp4 --mask_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/masks/1144932-hd_1920_1080_30fps_No_bbox_mask.mp4 --output_name 1144932-hd_1920_1080_30fps_No_bbox_mask.mp4 --prompt "A vibrant pink flower with a yellow center remains the focal point against green foliage throughout the video." --num_frames 81


python3 infer.py --video_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/videos/20729655-hd_1920_1080_25fps_short.mp4 --mask_path /mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/DiffSynth-Studio/test_videos/pexel_test_data_0410/masks/20729655-hd_1920_1080_25fps_No_bbox_mask.mp4 --output_name 20729655-hd_1920_1080_25fps_No_bbox_mask.mp4 --prompt "A person stands on a bridge by a river, facing city buildings, spreading arms wide against a cloudy sky." --num_frames 81

export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/activate.sh && codex --dangerously-bypass-approvals-and-sandbox --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410 --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/wan_eraser


claude --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410 --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/wan_eraser

UV_HTTP_TIMEOUT=1800 uv pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple \
  -e "python[diffusion]" --prerelease=allow

  UV_HTTP_TIMEOUT=1800 uv pip install   --index-url https://pypi.tuna.tsinghua.edu.cn/simple   -e "python[diffusion]" --prerelease=allow



  FlowMatchScheduler是否是一样的？

  根据方案 @sglang/docs_always/add_new_mode/add_videoedit_diffusers/README.md，以及原始算法仓库 @/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers，完善/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/
  pretrain_models/Wan2.1-I2V-14B-480P-Diffusers，要求不新增文件，不要删除文件

python3 infer.py  --video_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 --mask_path  /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 --output_name 15108907_3840_2160_50fps_No_bbox_mask.mp4 --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." --num_frames 81


依照 @/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers 及 @/mnt/shanhai-ai/shanhai-workspace/zhouhao6/wan_eraser 源码，同时参考 @sglang/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md 所述最佳实践，并以当前 @/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/docs_always/add_new_mode/add_videoedit_diffusers/README.md 集成方案为基础，提出优化和可落地的集成重构方案，具体完善如下：

- 最好是新增文件，尽量不要修改原来的文件
- 所有stage的中间变量全部放到WanVideoEditSamplingParams中，先给出stage，在给出WanVideoEditSamplingParams
- 原始仓库基于i2v而来，不采用t2v，要新增任务VIDEO_EDIT任务，不复用t2v了
- 重写WanVideoEditPipeline的forward函数，在forward函数中加一个for循环处理多个阶段，把多帧处理放在stage的外部，所有stage处理每81帧
- 不要冒烟测试，直接端到端测试
- 移除方案中冗余的部分
- 完善文档中的目录


继续完善方案：
1- 移除关于dry-run / 冒烟测试、T2V 的内容；
2- 与原始仓库@VideoEdit-diffusers，详细比较WanTransformer3DModel的模型结构，


根据方案 @docs_always/add_new_mode/add_videoedit_diffusers/README.md，继续完成剩下的开发与测试，尽最大可能不修改原始已有的文件。

1. 代码重构与集成
   - 基于原始算法仓库 @/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers，以及 @/mnt/shanhai-ai/shanhai-workspace/zhouhao6/wan_eraser 源码，参考最佳实践 @sglang/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md、当前集成方案文档，规划并实施 VIDEO_EDIT 任务的新增集成，避免复用 t2v 路径。
   - 明确各流程阶段（stage），所有中间变量统一封装进 WanVideoEditSamplingParams，理清各阶段的作用及数据流，最后汇总并给出 WanVideoEditSamplingParams 的字段结构设计。
   - 重写 WanVideoEditPipeline 的 forward 方法。其内部采用 for 循环实现多阶段处理，保证stage级处理覆盖81帧，不在stage内部做多帧for循环。
   - 秉持“新增文件为主，尽量少改动原有代码”，如需修改原始文件，务必注释缘由与兼容性保障。

2. 代码实现
   - 按上述设计逐步实现各核心功能模块，包括但不限于VIDEO_EDIT相关新任务类型的注册、中间参数容器WanVideoEditSamplingParams、WanVideoEditPipeline的新forward逻辑等。
   - 新增单独文件（如 video_edit_pipeline.py、video_edit_sampling_params.py 等）承载上述新功能模块代码，保持与原仓库结构和扩展性兼容。

3. 测试验证
   - 制定并运行至少一个端到端测试用例，对典型输入（例如81帧的视频及mask、明确的prompt和参数）做完整链路回归，验证集成结果在功能与性能上的正确性和合理性。
   - 检查中间结果与最终输出文件，确保多阶段调度和中间变量保存均符合设计预期。

4. 文档完善
   - 更新集成目录结构描述；
   - 补充 WanVideoEditSamplingParams 字段的详细文档以及各阶段设计（可用表格/项目列表形式说明）。

5. 结构示例（可参考）：
   - video_edit_sampling_params.py: 定义 WanVideoEditSamplingParams 类和各阶段临时变量。
   - video_edit_pipeline.py: 定义 WanVideoEditPipeline，重写 forward，并注册 VIDEO_EDIT 任务类型。
   - tests/video_edit_integration_test.py: 完整测试 VIDEO_EDIT 端到端流程。
   - README/集成方案文档：更新集成目录结构、参数说明和用法示例。

如需扩展或细化某个具体细节，可以分阶段逐步展开并逐步完善各个环节的实现与文档说明。

当前结果已经对齐，但是对于下面两个文件，侵入式的修改了函数内部实现，要求

不修改 /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py 的函数内部实现

不修改 /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py 的函数内部实现


优化：
- 已经在/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python/sglang/multimodal_gen/registry.py完成了注册，，不用修改 /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py 的内部实现，对应或许要修改cli 和 async def create_video_repair(req: VideoRepairRequest):
- 扩展cli和serve参数，支持可调的并行、内存卸载、cache、torch.compiler等等多种优化选项
- 存在默认的negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


## 10. 服务与 CLI 方案

### 10.1 本地 CLI

新增：

```text
python/sglang/multimodal_gen/runtime/videoedit/cli.py
```

命令：

```bash
conda deactivate
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer \
  --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." \
  --video-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 \
  --mask-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --output-file-name 15108907_3840_2160_50fps.mp4 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame
```

### 10.2 Serve API

新增专用 endpoint：

```text
POST   /v1/videos/repairs
GET    /v1/videos/{video_id}
GET    /v1/videos/{video_id}/progress
GET    /v1/videos/{video_id}/content
DELETE /v1/videos/{video_id}
GET    /health
```

`VideoRepairRequest` 新增在 `protocol.py`：

```python
class VideoRepairRequest(BaseModel):
    task_id: str | None = None
    prompt: str
    negative_prompt: str | None = None

    video_input_path: str | None = None
    mask_input_path: str | None = None
    video_url: str | None = None
    mask_url: str | None = None
    video_bucket: str | None = None
    video_object_key: str | None = None
    mask_bucket: str | None = None
    mask_object_key: str | None = None

    callback_url: str | None = None
    output_storage: str = "local"
    output_path: str | None = None
    output_bucket: str | None = None
    output_object_key: str | None = None

    num_frames: int = 81
    infer_len: int = 81
    overlap: int = 0
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 42
    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0
    enable_paste_back: bool = True
    drop_reference_frame: bool = True
```

admission 规则参考 wan_eraser 的 `BoundedSemaphore(1)`：

```python
if active_videoedit_jobs + queued_videoedit_jobs >= queue_capacity:
    raise HTTPException(status_code=429, detail="videoedit_queue_full")
```

服务启动：

```bash
conda deactivate
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

VIDEOEDIT_QUEUE_CAPACITY=1 \
sglang serve \
  --model-type diffusion \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer
```

提交任务：

```bash
curl -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "pexel_15108907_first_81",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps.mp4",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "enable_paste_back": true,
    "drop_reference_frame": true
  }'
```


conda deactivate
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer \
  --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." \
  --video-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 \
  --mask-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --output-file-name 15108907_3840_2160_50fps_sp2_no_offload.mp4 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_perf_sp2_no_offload.json


# no torch.compile
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_1gpu_default.mp4 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_1gpu_default.json"


# yes torch.compile
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_1gpu_default.mp4 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --enable-torch-compile \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_1gpu_default.json"


VIDEOEDIT_QUEUE_CAPACITY=1 \
sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --enable-torch-compile true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"


  curl -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "pexel_15108907_sp2_no_offload",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp2_no_offload.mp4",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": true,
    "perf_dump_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_no_offload_compile.json"
  }'

[05-07 08:16:03] server_args: {"model_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model", "model_id": null, "backend": "auto", "attention_backend": null, "attention_backend_config": {}, "cache_dit_config": null, "nccl_port": null, "trust_remote_code": false, "revision": null, "num_gpus": 1, "tp_size": 1, "sp_degree": 1, "ulysses_degree": 1, "ring_degree": 1, "dp_size": 1, "dp_degree": 1, "enable_cfg_parallel": false, "hsdp_replicate_dim": 1, "hsdp_shard_dim": 1, "dist_timeout": 3600, "pipeline_class_name": null, "lora_path": null, "lora_nickname": "default", "lora_scale": 1.0, "component_paths": {"transformer": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer"}, "transformer_weights_path": null, "lora_target_modules": null, "dit_cpu_offload": false, "dit_layerwise_offload": false, "dit_offload_prefetch_size": 0.0, "text_encoder_cpu_offload": false, "image_encoder_cpu_offload": false, "vae_cpu_offload": false, "use_fsdp_inference": false, "pin_cpu_memory": true, "comfyui_mode": false, "enable_torch_compile": true, "warmup": true, "warmup_resolutions": null, "warmup_steps": 1, "disable_autocast": false, "master_port": 30005, "host": "0.0.0.0", "port": 30000, "webui": false, "webui_port": 12312, "scheduler_port": 5567, "strict_ports": false, "output_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs", "input_save_path": "/tmp/sglang-videoedit-inputs", "prompt_file_path": null, "model_paths": {}, "model_loaded": {"transformer": true, "vae": true, "video_vae": true, "audio_vae": true, "video_dit": true, "audio_dit": true, "dual_tower_bridge": true}, "boundary_ratio": null, "log_level": "info", "uvicorn_access_log_exclude_prefixes": []}


  python -m sglang.multimodal_gen.runtime.videoedit.cli repair   --model-path "$MODEL_PATH"   --transformer-path "$TRANSFORMER_PATH"   --prompt "$PROMPT"   --video-input-path "$INPUT_VIDEO"   --mask-input-path "$INPUT_MASK"   --output-path "$OUT_DIR"   --output-file-name 15108907_3840_2160_50fps_sp2_no_offload.mp4   --num-gpus 2   --sp-degree 2   --ulysses-degree 2   --ring-degree 1   --num-frames 81   --infer-len 81   --overlap 0   --num-inference-steps 20   --guidance-scale 5.0   --seed 42   --dtype bf16   --enable-paste-back   --drop-reference-frame   --perf-dump-path "$OUT_DIR/videoedit_perf_sp2r_no_offload.json"


export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/activate.sh && codex --dangerously-bypass-approvals-and-sandbox --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/activate.sh && codex --dangerously-bypass-approvals-and-sandbox --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers


export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
codex --dangerously-bypass-approvals-and-sandbox --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers

export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
codex --dangerously-bypass-approvals-and-sandbox --add-dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers

export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
codex --dangerously-bypass-approvals-and-sandbox --add-dir /mnt/shanhai-ai/liuh/VideoEdit-diffusers

export HTTP_PROXY="http://localhost:10909"
export HTTPS_PROXY="http://localhost:10909"
export ALL_PROXY="http://localhost:10909"
curl -fsSL https://github.com/SaladDay/cc-switch-cli/releases/latest/download/install.sh | bash




