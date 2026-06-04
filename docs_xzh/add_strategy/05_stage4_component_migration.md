# Stage 4: 模型组件迁移方案

## 1. 迁移总策略

原则是：

- 基座组件尽量按 `CogVideoX1.5-5B` 原始结构移植
- VividVR 差异只保留最小必要增量
- 不把整棵 `src/diffusers` vendoring 到 SGLang

## 2. 组件迁移表

| 组件 | 来源文件 | 输入输出 | 迁移难度 | 建议方案 |
| --- | --- | --- | --- | --- |
| Transformer 基座 | `src/diffusers/models/transformers/cogvideox_transformer_3d.py` 及 vividvr 变体引用部分 | 输入 `[B,T,C,H,W]` latent、text embeds、timestep，输出 noise pred | 高 | 先移植 base CogVideoX transformer，再在 vividvr 子类叠加增量 |
| VividVR Transformer 增量 | `src/diffusers/models/transformers/cogvideox_vividvr_transformer_3d.py` | 增加 connector/control 分支 | 高 | 独立子类文件，避免污染 base |
| VividVR ControlNet | `src/diffusers/models/controlnet_cogvideox_vividvr.py` | 输入 latent + control state + text，输出 control hidden states | 高 | 第一版局部私有模块，不抽公共 controlnet |
| VAE | `AutoencoderKLCogVideoX` | 视频 tensor <-> latent | 高 | 新增原生 runtime VAE |
| Text Encoder | `T5EncoderModel` | text -> prompt embeds | 低 | 复用现有 T5 loader / TextEncodingStage |
| Scheduler | `scheduling_dpm_cogvideox.py` | step 输入多出 restoration 语义 | 中高 | 新增 `cogvideox_dpm_vividvr.py` |
| Prompt tiling processor | `VRDiT/utils.py` | video -> tile prompt list | 中 | 第一版延后，第二版 helper 化 |
| Postprocess | `colorfix.py` `textfix.py` `enhancer.py` | video -> enhanced video | 中 | 与主 pipeline 解耦，后置 |

## 3. Transformer 迁移建议

## 3.1 最小移植边界

必须保留：

- `patch_embed`
- rotary embedding 相关参数
- `CogVideoXBlock`
- `connectors`
- `control_feat_proj`
- `control_patch_embed`
- `control_hidden_states` 注入逻辑

不需要第一版就移植的：

- 训练相关逻辑
- PEFT 训练辅助逻辑中与推理无关的部分

## 3.2 推荐文件拆分

建议拆成两层：

- `runtime/models/dits/cogvideox.py`
  - 基座 `CogVideoXTransformer3DModel`
- `runtime/models/dits/cogvideox_vividvr.py`
  - `CogVideoXVividVRTransformer3DModel`

这样后续若再接别的 CogVideoX 变体，可复用 base。

## 3.3 输入输出合同

建议在 SGLang 内部统一合同：

- stage 侧使用 `[B, T, C, H, W]`
- 只在 VAE 边界 permute 成 `[B, C, T, H, W]`

理由：

- 这更贴合 VividVR pipeline 的实际实现
- tiling / temporal merge 都是按 `[B, T, C, H, W]` 思考的

## 4. VAE 迁移建议

## 4.1 为什么 VAE 必须移植

当前 `sglang` 没有 `AutoencoderKLCogVideoX` 原生 runtime 类。

而 VividVR 强依赖：

- VAE encode
- VAE decode
- temporal compression ratio
- scaling factor
- tiling / slicing

因此不能只靠现有 Wan/Hunyuan VAE 代替。

## 4.2 建议

新增：

- `configs/models/vaes/cogvideox.py`
- `runtime/models/vaes/cogvideox.py`

最小要求：

- `encode()`
- `decode()`
- `config.scaling_factor`
- `config.temporal_compression_ratio`
- `enable_tiling()`
- `enable_slicing()`

## 5. Scheduler 迁移建议

## 5.1 为什么不能直接复用 generic scheduler

VividVR 的 DPM scheduler 额外依赖：

- `old_pred_original_sample`
- restoration-guided sampling

这会改变每一步的状态传播。

## 5.2 建议实现

新增：

- `runtime/models/schedulers/cogvideox_dpm_vividvr.py`

要求：

- 兼容 `CogVideoXDPMScheduler` 原配置
- 保留 `restoration_guidance_scale`
- 明确 `step()` 返回 `(prev_sample, pred_original_sample)`

## 6. Text Encoder 迁移建议

直接复用：

- `T5EncoderModel`
- `T5Tokenizer`
- `TextEncodingStage.encode_text()`

不建议新建自定义 text encoder runtime 类。

## 7. Processor / Helper 迁移建议

## 7.1 空间 tiling

来源：

- `VRDiT/utils.py:prepare_tiling_infos_generator`
- `pipeline_cogvideox_vividvr.py:prepare_tiling_infos_generator`

建议：

- SGLang 版本独立放到 `runtime/vividvr/tiling.py`
- 不混进公共 `runtime/utils`

## 7.2 长视频时间聚合

来源：

- `VRDiT/inference.py:infer_split_clips`

建议：

- 独立放到 `runtime/vividvr/windowing.py`
- 对外暴露：
  - clip 切分
  - latent 映射构造
  - overlap merge

## 7.3 auto caption

来源：

- `VRDiT/utils.py:prepare_validation_prompts`

建议：

- 第一版延后
- 后续单独放 `runtime/vividvr/captioning.py`
- 当前集成阶段只允许：
  - 读取 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
  - 为未来 `CogVLM2` 接入保留占位
- 不允许在 `sglang` 环境中实时跑 `CogVLM2`

## 7.4 后处理

来源：

- `VRDiT/colorfix.py`
- `VRDiT/textfix.py`
- `VRDiT/enhancer.py`

建议：

- 不进入核心 `DenoisingStage`
- 后续单独放 `runtime/vividvr/postprocess.py`

## 8. 迁移顺序建议

1. `CogVideoX` base transformer
2. `CogVideoX` VAE
3. `CogVideoX` DPM scheduler
4. `VividVR transformer` 增量
5. `VividVR controlnet`
6. `VividVR pipeline + stages`
7. `tiling/windowing helpers`
8. `caption/postprocess`

## 9. 开放决策

- `embeddings_vividvr.py` 中的 positional embedding 是否单独成文件，还是合并进 `cogvideox.py`

建议：

- 若后续还会接其他 CogVideoX 模型，则拆成公共 `cogvideox` embeddings helper
- 若只服务于 VividVR，可先局部保留
