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


根据方案 @docs_always/add_new_mode/add_videoedit_diffusers/README.md，完成端到端实现和测试。

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