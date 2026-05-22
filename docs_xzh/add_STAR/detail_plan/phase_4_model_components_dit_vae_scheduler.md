# Phase 4：模型组件适配（DiT / VAE / Scheduler）

## 1. 阶段目标

本阶段的目标是完成 STAR-CogVideoX-SR 的核心模型组件适配，让 pipeline 真正具备推理能力。

阶段完成后，应满足：

1. STAR DiT 能在 SGLang `DenoisingStage` 下被调用
2. STAR 3D VAE 能完成 encode / decode
3. scheduler adapter 能与 `TimestepPreparationStage` + `DenoisingStage` 配合
4. 条件视频 latent 与噪声 latent 能在 channel 维正确拼接
5. 基本 forward shape 与 dtype 流程稳定

---

## 2. 本阶段范围

### 本阶段处理

1. STAR DiT 适配
2. STAR 3D VAE 适配
3. STAR scheduler adapter
4. 条件 latent 与文本条件的 forward 契约

### 本阶段不处理

1. 最终 decode parity
2. color fix
3. 多 GPU 优化收尾
4. 完整性能 profiling

---

## 3. 计划涉及的代码文件

### 3.1 新增文件

建议新增：

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py`
3. `python/sglang/multimodal_gen/runtime/models/schedulers/star_vpsde_dpmpp2m.py`
4. `python/sglang/multimodal_gen/configs/models/dits/star_cogvideox_sr.py`
5. `python/sglang/multimodal_gen/configs/models/vaes/star_cogvideox_vae.py`

### 3.2 可能修改文件

可能需要修改：

1. `python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py`

---

## 4. DiT 适配方案

## 4.1 目标

目标不是把 SAT `BaseModel` 原样搬进来，而是提供一个 **SGLang 运行时可直接调用的推理版 DiT**。

建议目标 forward 契约：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    guidance: torch.Tensor | None = None,
    encoder_hidden_states: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    ...
```

这要与 `DenoisingStage._predict_noise()` 的调用方式对齐。

## 4.2 输入语义

进入 transformer 之前，`DenoisingStage` 会完成：

1. `latent_model_input = latents`
2. 如果 `batch.image_latent` 存在，则沿 channel 维拼接：
   `torch.cat([latents, image_latent], dim=1)`

因此 STAR DiT 应假设：

1. `hidden_states` 已经是拼接后的张量
2. 输入 shape 为 `[B, C_total, T, H, W]`

### 关于 `in_channels`

原 STAR patch embed 内部使用 `proj_sr(in_channels * 2 -> hidden_size)`。

实施建议：

1. 配置层仍保留“基础 latent channel 数”，例如 `base_in_channels=16`
2. 模型内部显式计算 `total_in_channels = base_in_channels * 2`
3. patch embed 直接吃 `hidden_states` 当前真实 channel 数

这样可以避免外部再重复拼接逻辑。

## 4.3 结构适配优先级

建议分三层适配：

### 第一层：必须保真

1. patch embedding 的 `proj_sr`
2. time embedding
3. 文本条件投影
4. AdaLN modulation
5. 局部空间增强 `spa_local`
6. 局部时间增强 `temp_local`
7. 最终输出头

### 第二层：可复用则复用

1. attention backend 选择
2. layernorm / RMSNorm 实现
3. 通用 linear / MLP 结构

### 第三层：后续优化

1. TP / SP
2. 融合 kernel
3. layerwise offload

第一版不要因为追求 TP/SP 直接打乱结构保真。

## 4.4 推荐实现路径

建议优先实现一个 **单卡保真版**，再考虑抽取和复用现有底层组件。

推荐路径：

1. 先按 STAR 原结构写推理版模块树
2. 保持 state dict key 与转换后权重尽量一致
3. 在 block 内部尽量替换为 SGLang 的通用 attention / norm 实现
4. 单卡 parity 稳定后，再看 TP/SP

## 4.5 `PipelineConfig` 如何喂文本条件

建议在 `StarCogVideoXSRPipelineConfig` 中实现：

1. `prepare_pos_cond_kwargs`
2. `prepare_neg_cond_kwargs`

目标输出：

1. `encoder_hidden_states`
2. 如模型需要则输出 `encoder_attention_mask`
3. 如模型需要则输出 `text_length`

不要在 `DenoisingStage` 内部为 STAR 增加模型判断分支。

---

## 5. VAE 适配方案

## 5.1 目标

VAE 要满足两类使用场景：

1. `STARConditionVideoVAEEncodingStage` 里的 encode
2. `STARCogVideoXSRDecodingStage` 里的 decode

## 5.2 设计原则

建议将：

1. **VAE 算子本体**
2. **分块 decode 策略**

拆开实现。

即：

1. `star_cogvideox_vae.py` 只负责 `encode()` / `decode()`
2. 时间窗口调度放在自定义 decoding stage

## 5.3 encode 契约

建议 VAE `encode()` 输入：

1. `[B, C, T, H, W]`

输出：

1. `DiagonalGaussianDistribution` 或等价结构
2. 由 encoding stage 再决定 `sample()` / `mode()`

## 5.4 decode 契约

建议 VAE `decode()` 输入：

1. `[B, C, T, H_lat, W_lat]`

输出：

1. `[B, C, T, H, W]`

如原实现需要特殊参数，例如：

1. `clear_fake_cp_cache`
2. `timesteps`

则应在 VAE 类中通过明确签名暴露，而不是依赖 `**kwargs` 暗传。

## 5.5 与现有 VAE 的复用策略

建议先做结构比对：

1. 若现有 `ltx_2_vae.py` 中的 3D block 与 STAR VAE 高度接近，可抽公共子模块
2. 若差异大，则单独建 `star_cogvideox_vae.py`

判断标准建议：

1. 如果复用需要在现有 VAE 上加入太多 STAR 特判，则不要复用
2. 如果只是 block 级公共层复用，则可以抽 helper

---

## 6. Scheduler 适配方案

## 6.1 目标

需要一个能被：

1. `TimestepPreparationStage`
2. `DenoisingStage`

直接使用的 scheduler adapter。

## 6.2 推荐新增类

建议新增：

1. `StarVPSDEDPMPP2MScheduler`

文件：

1. `runtime/models/schedulers/star_vpsde_dpmpp2m.py`

## 6.3 必须支持的接口

至少需要兼容：

1. `set_timesteps(...)`
2. `scale_model_input(...)`
3. `step(...)`

并暴露：

1. `timesteps`
2. 如需要则暴露 `sigmas`

## 6.4 实现策略

初版建议：

1. 严格对齐 STAR 原采样语义
2. 先实现一个薄 adapter
3. 不要一开始就尝试“映射到某个看起来类似的现有 scheduler”

原因：

1. 采样路径偏差会直接影响 parity
2. 结果一旦不对，很难判断是模型问题还是 scheduler 问题

---

## 7. 条件视频 VAE 编码 stage 的细节

该 stage 在本阶段需要真正落地，而不是只留骨架。

## 7.1 输入输出

输入：

1. `batch.condition_video`: `[B, T, C, H, W]`

输出：

1. `batch.image_latent`: `[B, C_lat, T_lat, H_lat, W_lat]`

## 7.2 关键步骤

1. `permute(0, 2, 1, 3, 4)` 转为 `[B, C, T, H, W]`
2. 调用 VAE `encode`
3. 取 `sample()` 或 `mode()`
4. 调用 `pipeline_config.postprocess_vae_encode`
5. 调用 `pipeline_config.normalize_vae_encode`
6. 写入 `batch.image_latent`

## 7.3 shape 校验

必须加显式检查：

1. `batch.image_latent.ndim == 5`
2. `batch.image_latent.shape[0] == batch.latents.shape[0]` 或与 batch size 一致
3. channel 数与 STAR transformer 预期匹配

---

## 8. 推荐实施顺序

建议按下面顺序写：

1. 先写 config dataclass
2. 再写 scheduler adapter
3. 再写 VAE encode/decode 主体
4. 再写 transformer skeleton
5. 再打通 `video_condition_vae_encoding`
6. 最后用随机输入把 `DenoisingStage` 跑通

### 为什么先写 scheduler

因为 `TimestepPreparationStage` 和 `DenoisingStage` 都依赖它，先固定接口能减少联调噪音。

---

## 9. 测试计划

建议新增：

1. `python/sglang/multimodal_gen/test/unit/test_star_scheduler_adapter.py`
2. `python/sglang/multimodal_gen/test/unit/test_star_transformer_shapes.py`
3. `python/sglang/multimodal_gen/test/unit/test_star_vae_shapes.py`
4. `python/sglang/multimodal_gen/test/unit/test_star_condition_video_vae_encoding.py`

### `test_star_scheduler_adapter.py`

至少覆盖：

1. `set_timesteps()` 可运行
2. `timesteps` 长度正确
3. `scale_model_input()` 和 `step()` 输出 shape 稳定

### `test_star_transformer_shapes.py`

至少覆盖：

1. 拼接后的 `hidden_states` 能前向
2. `encoder_hidden_states` 输入 shape 正确
3. 输出 noise shape 与目标 latent shape 匹配

### `test_star_vae_shapes.py`

至少覆盖：

1. encode 输入 `[B, C, T, H, W]`
2. decode 输出时序与空间维正确

### `test_star_condition_video_vae_encoding.py`

至少覆盖：

1. 从 `condition_video` 到 `image_latent` 的整条编码链
2. scale / shift / normalize 不会改变目标 shape

---

## 10. 阶段验收标准

本阶段结束时，至少应满足：

1. transformer / vae / scheduler 三类组件都可实例化
2. `video_condition_vae_encoding` 能写出稳定的 `batch.image_latent`
3. `DenoisingStage` 能把 `latents` 与 `image_latent` 拼接后成功前向
4. 所有 shape、dtype 和基础 forward 测试通过

---

## 11. 失败信号与止损点

如果出现以下问题，不要进入阶段 5：

1. transformer forward 契约还不稳定
2. `image_latent` 与 `latents` shape 还对不上
3. scheduler 行为与 `TimestepPreparationStage` 不兼容
4. VAE encode/decode 仍依赖 STAR 训练框架上下文

这些问题必须在本阶段收敛，否则后面的 parity 验证没有意义。
