# Phase 3：Pipeline 组装与 Stage 接线

## 1. 阶段目标

本阶段的目标是把 STAR-CogVideoX-SR 接入到 SGLang 的 pipeline 框架里，但此时还不要求最终结果完全对齐。

阶段完成后，应满足：

1. `StarCogVideoXSRPipeline` 可以被 registry 发现
2. pipeline 的 stage 顺序已经固定
3. 条件视频 loading / VAE encoding / decoding 的位置已固定
4. `PipelineConfig`、`SamplingParams`、`registry.py` 已接线完成
5. 可以完成不依赖真实权重结果的 dry-run 或 smoke-run

---

## 2. 本阶段范围

### 本阶段处理

1. pipeline 类
2. pipeline config 类
3. stage 组装顺序
4. registry wiring
5. 组件需求声明

### 本阶段不处理

1. DiT 最终结构细节
2. VAE 最终结构细节
3. scheduler 结果精确对齐
4. decode parity

---

## 3. 推荐 pipeline 形态

## 3.1 采用单独 pipeline 类

建议新增：

1. `runtime/pipelines/star_cogvideox_sr_pipeline.py`

类名建议：

1. `StarCogVideoXSRPipeline`

原因：

1. 这不是现有 Wan / LTX / MOVA 的简单变体
2. 条件视频和 decode 路径都带有 STAR 特有逻辑
3. 用独立 pipeline 更容易保持边界清晰

## 3.2 task_type 选择

建议使用：

1. `ModelTaskType.T2V`

而不是：

1. `ModelTaskType.TI2V`

理由：

1. 当前 `TI2V` 在框架中带有 Wan 特化语义
2. STAR 的条件机制是完整视频 latent 通道拼接
3. 走 `T2V + condition_video` 更干净

---

## 4. 计划涉及的代码文件

### 4.1 新增文件

建议新增：

1. `python/sglang/multimodal_gen/runtime/pipelines/star_cogvideox_sr_pipeline.py`
2. `python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py`
3. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py`
4. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py`

### 4.2 修改文件

建议修改：

1. `python/sglang/multimodal_gen/runtime/pipelines/__init__.py`
2. `python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py`
3. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/__init__.py`
4. `python/sglang/multimodal_gen/registry.py`

---

## 5. PipelineConfig 设计

## 5.1 新增配置类

建议新增：

1. `StarCogVideoXSRPipelineConfig(PipelineConfig)`

建议包含：

1. `task_type = ModelTaskType.T2V`
2. STAR 专用 `dit_config`
3. STAR 专用 `vae_config`
4. 文本编码器配置
5. scheduler 配置
6. 默认分辨率 / 默认帧数
7. 条件视频抽帧和尺寸策略
8. 可选 color fix 配置

## 5.2 应实现的 hook

建议在该 config 里优先实现：

1. `prepare_latent_shape`
2. `prepare_pos_cond_kwargs`
3. `prepare_neg_cond_kwargs`
4. `postprocess_vae_encode`
5. `normalize_vae_encode`
6. `get_decode_scale_and_shift`
7. 如有需要再补 `post_denoising_loop`

### `prepare_pos_cond_kwargs` / `prepare_neg_cond_kwargs`

这里负责把文本条件转换成 transformer forward 需要的 kwargs。

推荐返回字段至少包括：

1. `encoder_hidden_states`
2. 如果模型需要，再加 `encoder_attention_mask`

如果 STAR transformer forward 需要更多字段，也在这里统一准备，不要在 `DenoisingStage` 内部再加 STAR 分支。

---

## 6. Stage 顺序设计

建议 `create_pipeline_stages()` 显式组装，不调用 `add_standard_ti2v_stages()`。

推荐顺序：

1. `InputValidationStage()`
2. `STARConditionVideoLoadingStage()`
3. `TextEncodingStage(...)`
4. `STARConditionVideoVAEEncodingStage(...)`
5. `LatentPreparationStage(...)`
6. `TimestepPreparationStage(...)`
7. `DenoisingStage(...)`
8. `STARCogVideoXSRDecodingStage(...)`

### 为什么不用 `add_standard_ti2v_stages()`

原因：

1. 它的条件图像路径假设并不适合整段视频
2. 我们需要插入自定义 loading 和 video VAE encoding stage
3. decoding 也不是通用 decode

### 为什么仍然保留标准 `DenoisingStage`

因为它已经提供：

1. scheduler 主循环
2. CFG 串行双前向
3. `batch.image_latent` channel concat
4. SP gather / shard 支持框架

---

## 7. 条件视频 VAE 编码 stage 设计

## 7.1 建议新增 stage

建议新增：

1. `STARConditionVideoVAEEncodingStage`

文件建议：

1. `runtime/pipelines_core/stages/video_condition_vae_encoding.py`

## 7.2 责任

该 stage 应负责：

1. 接收 `batch.condition_video`，shape `[B, T, C, H, W]`
2. permute 成 `[B, C, T, H, W]`
3. 调用 STAR VAE `encode`
4. 做 scale / shift / normalize
5. 把结果写到 `batch.image_latent`

注意：

1. `batch.image_latent` 的最终 shape 必须与 `batch.latents` 匹配
2. 它是给标准 `DenoisingStage` 拼接使用的

## 7.3 不建议复用 `ImageVAEEncodingStage`

不建议直接继承并小修，除非实现时发现：

1. 它的输入张量流程和我们 80% 以上一致

当前预判不建议直接复用的原因：

1. 它偏单图条件，内部有“首帧图像 + 后续补零帧”的假设
2. STAR 需要编码的是完整视频序列

---

## 8. Decoding stage 设计

## 8.1 建议新增专用 decoding stage

建议新增：

1. `STARCogVideoXSRDecodingStage`

文件建议：

1. `runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py`

原因：

1. STAR 的 decode 不是标准整段 decode
2. 它带有明确的时间窗口分块逻辑
3. 不应污染通用 `DecodingStage`

## 8.2 本阶段只搭骨架

本阶段先实现：

1. 继承 `PipelineStage`
2. 接受 `vae`
3. 预留 `decode_windows()` / `postprocess_output()` 方法
4. 能跑通 shape 级别的 smoke test

结果对齐留到阶段 5。

---

## 9. Pipeline 类实现步骤

建议按以下顺序实现：

1. 新增 pipeline 类并声明 `pipeline_name`
2. 声明 `_required_config_modules`
3. 在 `initialize_pipeline()` 中构造 scheduler
4. 在 `create_pipeline_stages()` 中手工组装 stage
5. 导出 `EntryClass = StarCogVideoXSRPipeline`
6. 确保 registry 自动发现或手工注册可用

### 推荐伪代码

```python
class StarCogVideoXSRPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "StarCogVideoXSRPipeline"

    _required_config_modules = [
        "tokenizer",
        "text_encoder",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args):
        self.modules["scheduler"] = build_star_scheduler(server_args.pipeline_config)

    def create_pipeline_stages(self, server_args):
        self.add_stage(InputValidationStage())
        self.add_stage(STARConditionVideoLoadingStage())
        self.add_stage(TextEncodingStage(...))
        self.add_stage(STARConditionVideoVAEEncodingStage(...))
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_stage(STARCogVideoXSRDecodingStage(...))
```

---

## 10. Registry 接线设计

## 10.1 必须新增的注册项

需要在 `registry.py` 中接入：

1. `StarCogVideoXSRSamplingParams`
2. `StarCogVideoXSRPipelineConfig`
3. `StarCogVideoXSRPipeline`
4. 模型 detector 或明确的 model id

## 10.2 推荐注册方式

优先采用：

1. 手工 `register_configs(...)`
2. pipeline 通过 `EntryClass` 自动发现

建议在模型 detector 中使用：

1. 转换后模型目录的稳定命名模式
2. 或显式的 `--model-id`

不建议依赖 STAR 原始 checkpoint 文件名猜测。

---

## 11. 测试计划

建议新增：

1. `python/sglang/multimodal_gen/test/unit/test_star_pipeline_registry.py`
2. `python/sglang/multimodal_gen/test/unit/test_star_stage_wiring.py`

### `test_star_pipeline_registry.py`

至少覆盖：

1. pipeline 能被发现
2. `register_configs()` 后能解析 model info
3. `sampling_param_cls` 和 `pipeline_config_cls` 对应正确

### `test_star_stage_wiring.py`

至少覆盖：

1. stage 顺序正确
2. `InputValidationStage` 后会进入 `STARConditionVideoLoadingStage`
3. `DenoisingStage` 仍然在主链路中
4. decoding stage 为 STAR 专用 stage

---

## 12. 阶段验收标准

本阶段结束时，至少应满足：

1. pipeline / config / registry 接线完成
2. `create_pipeline_stages()` 顺序固定
3. 条件视频 loading 与 VAE encoding 的位置固定
4. 标准 `DenoisingStage` 仍在主去噪路径上
5. smoke test 能打印并执行完整 stage 列表

---

## 13. 失败信号与止损点

出现以下情况时，不要进入阶段 4：

1. pipeline 仍依赖 STAR repo import 才能初始化
2. stage 顺序还在摇摆
3. `image_latent` 的写入位置还不明确
4. registry 还不能稳定解析模型

这些问题如果不先收敛，后面接 DiT/VAE/scheduler 时会反复改 wiring。
