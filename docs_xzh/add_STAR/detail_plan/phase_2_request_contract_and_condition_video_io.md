# Phase 2：请求契约与条件视频输入链路

## 1. 阶段目标

本阶段的目标是把“条件视频”变成 SGLang 内部的一等输入，而不是借用现有 `image_path` 的旁路行为。

阶段完成后，应满足：

1. 用户可以通过独立字段传入 `condition_video_path`
2. `SamplingParams` 和 `Req` 已能表达条件视频语义
3. 运行时已存在独立的“条件视频加载 stage”
4. 该 stage 负责整段视频读帧、采样、裁剪、resize、tensor 化
5. 条件视频数据流已与后续 VAE 编码 stage 对齐

---

## 2. 本阶段范围

### 本阶段处理

1. `SamplingParams` 扩展
2. `Req` 字段扩展
3. 条件视频加载 stage
4. 条件视频元信息记录
5. 输入默认值与显式字段覆盖规则

### 本阶段不处理

1. 条件视频 VAE 编码
2. pipeline registry wiring
3. STAR DiT / VAE / scheduler 具体实现
4. decoding parity

---

## 3. 设计结论

## 3.1 不复用 `image_path`

必须使用独立字段：

1. `condition_video_path`

不能继续使用：

1. `image_path=.mp4`

原因：

1. 当前 `InputValidationStage` 对 mp4 只取首帧
2. 语义层面会混淆“图像条件”和“视频条件”
3. 后续维护时会持续制造隐式行为

## 3.2 不把条件视频塞进 `condition_image`

本阶段结束后，不应要求 `condition_video` 通过 `condition_image` 这类图像字段传递。

建议新增明确字段：

1. `Req.condition_video`
2. `Req.original_condition_video_size`
3. `Req.original_condition_video_fps`
4. `Req.condition_video_indices`

可以继续复用的字段只有：

1. `Req.image_latent`

因为它在 `DenoisingStage` 中表达的是“额外条件 latent”，而不是“图像对象本身”。

---

## 4. 计划涉及的代码文件

### 4.1 新增文件

建议新增：

1. `python/sglang/multimodal_gen/configs/sample/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_loading.py`

### 4.2 修改文件

建议修改：

1. `python/sglang/multimodal_gen/configs/sample/__init__.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py`
3. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/__init__.py`
4. `python/sglang/multimodal_gen/test/unit/test_sampling_params.py`
5. 新增或修改 `python/sglang/multimodal_gen/test/unit/test_input_validation.py`

---

## 5. SamplingParams 设计

## 5.1 新增专用 SamplingParams 子类

建议新增：

1. `StarCogVideoXSRSamplingParams(SamplingParams)`

原因：

1. 本模型有独立的条件视频输入语义
2. 不适合把所有字段直接上提到基础 `SamplingParams`
3. 后续如果出现第二个 video-conditioned 模型，再评估公共抽象是否需要上提

## 5.2 建议字段

建议至少包含：

1. `condition_video_path: str | None = None`
2. `condition_video_start_frame: int | None = None`
3. `condition_video_num_frames: int | None = None`
4. `condition_video_sample_fps: int | None = None`
5. `condition_video_frame_stride: int | None = None`
6. `enable_color_fix: bool = False`
7. `color_fix_mode: str | None = None`

说明：

1. `condition_video_num_frames` 和 `num_frames` 含义可以不同
   `num_frames` 表示目标生成帧数，`condition_video_num_frames` 表示条件视频实际抽取帧数。
2. MVP 阶段允许二者相同，但字段语义要先拆开。

## 5.3 默认值策略

建议：

1. `condition_video_path` 默认 `None`
2. `condition_video_num_frames` 默认 `None`
3. `num_frames` 默认由 pipeline config 决定
4. `height/width` 默认由 pipeline config 或条件视频 loading stage 决定

不要在 `SamplingParams` 中硬编码 STAR 当前某个临时实验用的帧数。

---

## 6. Req 运行时字段设计

建议在 `Req` 中新增以下字段：

1. `condition_video: torch.Tensor | list | None = None`
2. `original_condition_video_size: tuple[int, int] | None = None`
3. `original_condition_video_fps: float | None = None`
4. `condition_video_indices: list[int] | None = None`
5. `condition_video_num_frames: int | None = None`

### 推荐张量格式

建议统一使用：

1. `condition_video` 在 loading stage 输出为 `[B, T, C, H, W]`

理由：

1. 更接近用户视角和视频加载逻辑
2. 后续 VAE 编码 stage 可以自行 permute 成 `[B, C, T, H, W]`
3. 便于中间调试和保存可视化

---

## 7. 条件视频加载 stage 设计

## 7.1 建议新增 stage

建议新增：

1. `STARConditionVideoLoadingStage`

文件建议放在：

1. `runtime/pipelines_core/stages/video_condition_loading.py`

理由：

1. 它本质是通用“条件视频读取 + 预处理”能力
2. 后续其他 video-conditioned pipeline 也可能复用
3. 不应一开始就放到 `model_specific_stages/`

## 7.2 stage 责任

该 stage 应负责：

1. 校验 `condition_video_path` 是否存在
2. 使用 `load_video()` 读取整段帧列表
3. 获取原始 fps、帧数、分辨率
4. 根据请求或 pipeline config 决定：
   - 抽哪些帧
   - 最终输出多少帧
   - 输出尺寸是多少
5. 对每帧执行统一 resize / crop
6. 将帧列表转为 `[B, T, C, H, W]` 张量
7. 归一化到 `[-1, 1]`
8. 写入 `batch.condition_video`
9. 写入元信息字段

## 7.3 不建议该 stage 负责的内容

不建议在本 stage 中做：

1. VAE encode
2. image_latent 写入
3. scheduler 相关逻辑
4. 文本相关逻辑

---

## 8. 尺寸与抽帧规则

## 8.1 显式字段优先

如果用户在请求中显式指定了：

1. `width`
2. `height`
3. `num_frames`

则 loading stage 必须优先尊重这些字段。

推荐使用：

1. `batch.extra["explicit_fields"]`

来判断哪些字段是用户显式传入的。

## 8.2 若用户未显式指定尺寸

建议规则：

1. 优先使用 pipeline config 的默认分辨率
2. 若 pipeline config 未指定，则由模型转换产物中的原生推荐分辨率决定
3. 条件视频要被裁剪/缩放到该目标尺寸

不要让尺寸决策隐式依赖 `InputValidationStage` 的通用 720p 默认值。

## 8.3 若用户未显式指定帧数

建议规则：

1. 优先用 pipeline config 的默认 `num_frames`
2. 如果模型权重只对某些帧数做过验证，应在 config 中限制合法帧数范围
3. loading stage 按目标帧数从条件视频做均匀抽帧或固定 stride 抽帧

---

## 9. 与 InputValidationStage 的关系

## 9.1 保留 InputValidationStage

保留原因：

1. 它已经负责 seed/generator 生成
2. 它已经负责 prompt / negative prompt / guidance 基础校验
3. 这是标准 pipeline 前置阶段

## 9.2 不修改 InputValidationStage 去支持整段视频

本阶段不建议为了 STAR 去大改 `InputValidationStage`。

原因：

1. `InputValidationStage` 是通用基础 stage
2. 条件视频读取是模型族特定需求
3. 改动过大容易影响既有 I2V/TI2V 行为

### 推荐策略

1. `InputValidationStage` 只做通用校验
2. `STARConditionVideoLoadingStage` 负责：
   - 校验 `condition_video_path`
   - 覆盖 `batch.height/width/num_frames` 的模型特定决策

---

## 10. 实现步骤

建议按以下顺序推进：

1. 新增 `StarCogVideoXSRSamplingParams`
2. 把它导出到 `configs/sample/__init__.py`
3. 在 `Req` 中新增 `condition_video` 等字段
4. 新建 `video_condition_loading.py`
5. 先实现最小加载逻辑：
   - 本地 mp4
   - 全量读帧
   - 统一 resize/crop
   - 输出 tensor
6. 再补抽帧规则和元信息
7. 再补单元测试

### 推荐核心伪代码

```python
class STARConditionVideoLoadingStage(PipelineStage):
    def forward(self, batch, server_args):
        path = batch.condition_video_path
        if not path:
            raise ValueError("condition_video_path is required")

        frames = load_video(path)
        meta = inspect_video_meta(path, frames)

        target_width, target_height = resolve_target_size(
            batch=batch,
            pipeline_config=server_args.pipeline_config,
        )
        target_frames, frame_indices = resolve_target_frames(
            batch=batch,
            meta=meta,
            pipeline_config=server_args.pipeline_config,
        )

        frames = select_frames(frames, frame_indices)
        frames = [resize_and_crop(f, target_width, target_height) for f in frames]
        tensor = frames_to_tensor(frames)   # [1, T, C, H, W], range [-1, 1]

        batch.condition_video = tensor
        batch.original_condition_video_size = (meta.width, meta.height)
        batch.original_condition_video_fps = meta.fps
        batch.condition_video_indices = frame_indices
        batch.condition_video_num_frames = tensor.shape[1]
        batch.width = target_width
        batch.height = target_height
        return batch
```

---

## 11. 建议补充的测试

建议新增：

1. `python/sglang/multimodal_gen/test/unit/test_star_sampling_params.py`
2. `python/sglang/multimodal_gen/test/unit/test_star_condition_video_loading.py`

## 11.1 `test_star_sampling_params.py`

至少覆盖：

1. `condition_video_path` 字段能正常构建
2. `explicit_fields` 能正确记录用户显式传入的尺寸
3. 子类默认值不会污染基类模型

## 11.2 `test_star_condition_video_loading.py`

至少覆盖：

1. 本地 mp4 能读取整段帧
2. 输出张量 shape 为 `[B, T, C, H, W]`
3. 帧数选择逻辑正确
4. 用户显式传入 `width/height` 时优先级正确
5. 不设置 `condition_video_path` 会报清晰错误

---

## 12. 阶段验收标准

本阶段结束的标准是：

1. 已有 STAR 专用 `SamplingParams`
2. `Req` 已支持条件视频字段
3. 已有独立的条件视频 loading stage
4. 该 stage 会读整段视频而非首帧
5. shape、尺寸和帧数策略可通过单元测试验证

---

## 13. 失败信号与止损点

如果出现以下情况，不要进入阶段 3：

1. 仍然需要复用 `image_path`
2. 仍然需要把整段视频塞到 `condition_image`
3. `batch.condition_video` 的 shape 还不稳定
4. 尺寸和帧数决策没有明确规则

这些问题如果不先收敛，后面 pipeline 和 VAE stage 会反复返工。
