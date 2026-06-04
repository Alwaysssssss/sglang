# Stage 0: 项目摸底与真实推理链路

## 1. 结论先行

### 1.1 Vivid-VR 的架构归类

`Vivid-VR` 应归类为：

- 基座：`CogVideoX1.5-5B`
- 任务形态：视频恢复 / 视频修复，不是通用 T2V，也不是 Wan VideoEdit 那种 mask-edit
- 运行框架：基于定制 diffusers 分支
- 模型结构：自定义 `CogVideoX` transformer + 自定义 controlnet + 自定义 scheduler
- 运行策略：空间 tiling + 长视频 clip 级时间聚合 + 可选 caption + 可选后处理

它不是：

- Wan 系列
- Flux 风格
- 现成 diffusers generic pipeline 直接生产可用的情况
- 只换 checkpoint 的原生 CogVideoX pipeline

### 1.2 与 SGLang 当前现状的关键差距

`sglang.multimodal_gen` 当前已经有：

- Wan
- Hunyuan
- MOVA
- LTX2
- Flux
- 其他扩散模型

但当前没有：

- `CogVideoXTransformer3DModel` 的原生 runtime 实现
- `AutoencoderKLCogVideoX` 的原生 runtime 实现
- `CogVideoXDDIMScheduler` / `CogVideoXDPMScheduler` 的原生 runtime 实现
- `CogVideoX` 风格的 controlnet 组件类型

所以该项目的第一性问题不是“怎么编排 pipeline”，而是“先补足 `CogVideoX` 组件底座，再谈 pipeline 接入”。

## 2. 源码结构

### 2.1 业务入口

核心入口：

- `VRDiT/inference.py`

辅助模块：

- `VRDiT/utils.py`
- `VRDiT/cogvlm2.py`
- `VRDiT/colorfix.py`
- `VRDiT/textfix.py`
- `VRDiT/enhancer.py`

### 2.2 模型与 pipeline 定制源码

关键文件：

- `src/diffusers/pipelines/cogvideo/pipeline_cogvideox_vividvr.py`
- `src/diffusers/models/transformers/cogvideox_vividvr_transformer_3d.py`
- `src/diffusers/models/controlnet_cogvideox_vividvr.py`
- `src/diffusers/models/embeddings_vividvr.py`
- `src/diffusers/schedulers/scheduling_dpm_cogvideox.py`
- `src/diffusers/schedulers/scheduling_ddim_cogvideox.py`

### 2.3 checkpoint 组成

基座组件：

- `ckpts/CogVideoX1.5-5B/text_encoder`
- `ckpts/CogVideoX1.5-5B/tokenizer`
- `ckpts/CogVideoX1.5-5B/transformer`
- `ckpts/CogVideoX1.5-5B/vae`
- `ckpts/CogVideoX1.5-5B/scheduler`

VividVR 增量组件：

- `ckpts/Vivid-VR/connectors.pt`
- `ckpts/Vivid-VR/control_feat_proj.pt`
- `ckpts/Vivid-VR/control_patch_embed.pt`
- `ckpts/Vivid-VR/controlnet/`

可选增强组件：

- `ckpts/cogvlm2-llama3-caption`
- `ckpts/easyocr`
- `ckpts/RealESRGAN/RealESRGAN_x2plus.pth`

## 3. 真实推理链路

## 3.1 模型构造链

`VRDiT/inference.py` 的构造顺序是：

1. 加载 `CogVLM2_Captioner`
2. 加载 `T5EncoderModel`
3. 加载 `CogVideoXVividVRTransformer3DModel.from_pretrained(..., subfolder="transformer")`
4. 手动修改 transformer positional embedding 配置
5. 加载 `AutoencoderKLCogVideoX`
6. 用 `from_transformer()` 构造 `CogVideoXVividVRControlNetModel`
7. 加载 `CogVideoXDPMScheduler`
8. 给 transformer 注入 `connectors.pt`、`control_feat_proj.pt`、`control_patch_embed.pt`
9. 给 controlnet 注入 `ckpts/Vivid-VR/controlnet/`
10. 构造 `CogVideoXVividVRControlNetPipeline`
11. 开启 `pipe.enable_model_cpu_offload()`

这里的工程含义是：

- transformer 不是纯 checkpoint 加载完成，还要额外打补丁式加载三组附加权重
- controlnet 不是标准 diffusers model_index 自动解析出来的公共组件，而是 VividVR 私有逻辑
- pipeline 是“基座 checkpoint + 私有结构 + 私有增量权重”的组合

补充约束：

- 上面描述的是原始 `Vivid-VR` 仓库的真实链路，不等于当前 `sglang` 集成阶段必须完整照搬
- 当前 `sglang` 环境中，`CogVLM2` caption 会输出乱码
- 因此在集成阶段，caption 输入必须固定读取：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 实时 `CogVLM2_Captioner` 链路只作为后续待恢复能力，不纳入当前正确性集成范围

## 3.2 单短视频路径

当 `control_video.size(0) <= num_temporal_process_frames` 时，执行 `infer_whole_video()`：

1. 对帧数做 `8k+1` padding
2. 先按生成分辨率 resize 出 `video_for_caption`
3. 调 `prepare_validation_prompts()`
4. 对空间 tiles 分别 caption，得到 `prompt_list` / `negative_prompt_list`
5. 调用 `pipe(...)`
6. 在 pipeline 内部做：
   - `pre_denoise_process()`
   - timestep loop
   - `post_denoise_process()`
7. 输出 numpy video
8. clip 级收尾：
   - 去掉 padding frame
   - AdaIN color fix
   - 可选 `TextFixer`
   - export video

## 3.3 长视频路径

当 `control_video.size(0) > num_temporal_process_frames` 时，执行 `infer_split_clips()`：

1. 以 `num_temporal_process_frames` 和 overlap 规则切 clip
2. 每个 clip 单独执行：
   - padding
   - spatial caption
   - `pipe.pre_denoise_process()`
3. 统一生成全局 `timesteps`
4. 对每个 timestep：
   - 对每个 clip 执行 `pipe.denoise_process()`
   - 再在 latent 维度做跨 clip 覆盖式 merge
5. 所有 timestep 结束后，对每个 clip 单独 `pipe.post_denoise_process()`
6. 再根据 overlap 规则裁掉首尾重复帧并拼接
7. 然后再做 AdaIN / TextFixer / 输出

关键结论：

- 长视频逻辑并不在 scheduler，也不在 controlnet，而是在外层 orchestration。
- 这部分必须映射到 SGLang 的 pipeline 层，而不是塞进某个公共 stage。

## 4. Pipeline 内部的三段式合同

`pipeline_cogvideox_vividvr.py` 的结构很清晰，基本可以抽象成三段：

### 4.1 `pre_denoise_process()`

负责：

- 输入校验
- prompt encode
- `control_video -> control_latents`
- 初始 noise latents 构造
- latent padding

输出：

- `latents`
- `control_latents`
- `prompt_embeds`
- `negative_prompt_embeds`
- `num_latent_padding_frames`
- `ori_height`
- `ori_width`

### 4.2 `denoise_process()`

负责：

- 生成空间 tiling 计划
- 每个 tile 调 `denoise_step()`
- tile 级 merge
- 返回新的 `latents` 和 `old_pred_original_sample`

### 4.3 `post_denoise_process()`

负责：

- 去除 latent padding frame
- VAE decode
- resize 回原视频尺寸
- 输出 `pil` / `np` / `latent`

## 5. 组件拆解

### 5.1 Transformer 差异

VividVR transformer 相比标准 `CogVideoXTransformer3DModel` 新增：

- `connectors`
- `control_feat_proj`
- `control_patch_embed`
- `control_hidden_states` 注入路径

这不是“配置小改”，而是结构级增量。

### 5.2 ControlNet 差异

VividVR controlnet：

- 直接复用 `CogVideoXBlock`
- 生成 block 级 `control_hidden_states`
- 用于驱动主 transformer 的 connector 注入

这不是 SGLang 当前已有的公共 controlnet 形态。

### 5.3 Scheduler 差异

`CogVideoXDPMScheduler.step()` 被扩展为接受：

- `old_pred_original_sample`
- `restoration_guidance_scale`
- `restoration_ori_latent`

这说明：

- `DenoisingStage` 的标准 scheduler 合同不够
- VividVR 需要自己的 denoising stage 或 scheduler adapter

### 5.4 Caption / Postprocess 差异

caption：

- `prepare_validation_prompts()` 先按空间 tiles 切视频
- 每个 tile 调 `CogVLM2_Captioner`
- 输出 prompt list，而不是单条 prompt

但在当前 `sglang` 集成计划中：

- 不实时调用这条 caption 链
- 统一以现成 `prompt.txt` 作为 caption sidecar 输入
- `CogVLM2` 只保留能力占位

postprocess：

- `adaptive_instance_normalization()`
- `TextFixer` = `Enhancer + EasyOCR detector + text region replace`

这些不属于 diffusion 核心生成合同，应当后置。

## 6. 与 Wan / Diffusers / 原生推理框架的关系

### 6.1 与 Wan 的关系

几乎没有底层结构复用关系。

只有工程上可复用的 SGLang 思路：

- pipeline 编排
- sampling params runtime 状态
- model-specific stages
- registry / loader / executor

### 6.2 与 diffusers 的关系

VividVR 是明显的“基于 diffusers 改出来的模型族私有分支”：

- 组件目录与 model_index 遵循 diffusers 风格
- 运行接口保留了 diffusers pipeline 习惯
- 但其真实可用逻辑已经超出 generic wrapper 的能力

### 6.3 与 SGLang 原生 runtime 的关系

VividVR 最适合复用的不是 diffusers generic wrapper，而是：

- `ComposedPipelineBase`
- `Req`
- `PipelineExecutor`
- registry
- loader
- 现有 T5 text encode 能力

不适合直接复用的是：

- Wan 的 latent packing
- Wan 的 VAE 统计量
- Wan 的 scheduler
- Wan 的 cross-attention 语义

## 7. 对后续实现最重要的判断

### 7.1 最终风格选择

推荐采用：

- hybrid 倾向的 model-specific pipeline

不推荐采用：

- 完全细粒度 modular stage 拆分

原因：

- `pre_denoise_process()` 内部耦合过强
- prompt 是按 tile 组织，而不是单 batch 单 prompt
- long-video merge 明显属于外层 pipeline orchestration
- scheduler step 合同已偏离标准

### 7.2 第一版正确方向

第一版应先实现：

- 单卡
- 单 clip
- 显式 prompt
- 自定义 transformer / controlnet / scheduler 可运行
- 输出视频闭环

而不是先实现：

- auto caption
- TextFixer
- 全长视频时间聚合
- 多卡 TP/SP

## 8. 待确认问题

- `sglang` 第一版是否接受“caption 完全外置，只吃用户 prompt”的使用方式。
- `CogVideoX` VAE 是先做原生 runtime 移植，还是第一阶段先接受一个局部 diffusers VAE 兼容层。
- 长视频 merge 是否要完全复现原始“覆盖式替换”策略，还是改为加权 merge；文档当前按原实现建议先保持一致。
