# VividVR 对齐 `wan_videoedit` Modular 风格实施方案

更新时间：2026-06-04

## 1. 目标

本文档用于指导把当前的 `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
从“3 段式大 stage + 隐式 runtime 状态”的实现，重构为更接近
`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
的 modular 风格。

本次重构的核心目标不是“改模型行为”，而是“改架构表达方式，同时尽量不改数值行为”。

必须同时满足两条：

1. Pipeline 结构上向 `wan_videoedit` 的 modular stage composition 对齐。
2. 推理结果上保持现有 Phase C 验收结果，且与基线指标文件
   `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T070647Z.json`
   尽可能接近。

## 2. 当前实现的准确诊断

当前 VividVR 不是 `wan_videoedit` 风格的 modular pipeline，而是更接近一个定制化 hybrid pipeline。

### 2.1 当前 stage 结构

`vividvr_pipeline.py` 当前只注册了 3 个 stage：

- `VividVRBeforeDenoisingStage`
- `VividVRDenoisingStage`
- `VividVRDecodingStage`

这和 `wan_videoedit_pipeline.py` 的多 stage 组合方式不同。后者把职责拆成：

- validation
- text encoding
- condition encoding
- latent preparation
- timestep preparation
- latent init
- denoising
- decoding
- postprocess

### 2.2 当前 VividVR 的主要架构问题

`VividVRBeforeDenoisingStage` 目前承担了过多职责，至少包含：

- 解析 prompt file 路径
- 读取 prompt 文件
- 组合 positive prompt suffix
- 决定 negative prompt
- T5 文本编码
- CFG 负 prompt 编码
- control video 读取与 padding
- `VideoProcessor.preprocess_video`
- VAE encode control video 得到 control latents
- noise latent 初始化
- tiling info 生成
- runtime/debug 状态写入

这导致几个问题：

1. stage 边界不清晰，后续维护时很难判断某个逻辑应该属于 prompt、condition、latent、还是 postprocess。
2. 数值关键状态主要通过 `batch.extra["vividvr_runtime"]` 和 `batch.extra["vividvr_debug"]` 传递，缺少像 `WanVideoEditSamplingParams.runtime_*` 这样的显式 contract。
3. 如果后续要做 Phase D 的 clip split / merge 或更多 video-edit orchestration，当前单大 stage 的扩展成本会持续升高。

### 2.3 需要避免的误判

这次目标是“对齐 `wan_videoedit` 的 modular 风格”，不是强行把 VividVR 改成最纯的
`add_standard_ti2v_stages()` helper 风格。

原因很简单：

- `wan_videoedit` 自己也不是完全依赖标准 helper 组合。
- `wan_videoedit` 的 modular 本质，是“显式的小 stage 组合 + typed runtime contract”。
- 为了 Phase C 稳定性，VividVR 第一轮重构应优先对齐这一层，而不是一步把 denoising/decoding 全塞进框架标准 hook。

结论：

- 第一目标是 `wan_videoedit` 式 modularization。
- 不是一开始就追求“100% 标准 stage helper 化”。

## 3. 重构边界

### 3.1 本轮架构重构应做的事

- 把 VividVR 的 pre-denoising 大 stage 拆开。
- 建立显式的 `VividVRSamplingParams.runtime_*` 运行态契约。
- 把 decode 后的 crop / frame drop / AdaIN 从 decode 主体里分离成独立 postprocess stage。
- 保持现有单 clip `forward()` 路径和模型行为不变。

### 3.2 本轮架构重构不应顺手做的事

- 不要顺手引入 live `CogVLM2` caption。
- 不要改变 prompt 来源策略。
- 不要改变 scheduler 行为。
- 不要改变 tiling 算法。
- 不要同时开始做 long-video multi-window orchestration。
- 不要为了“更像框架”就重写 `initialize_pipeline()` 的全部模块加载逻辑。

最后这一点很重要。`initialize_pipeline()` 当前手动加载 `text_encoder` / `transformer` / `controlnet`
虽然不够“标准”，但它本身不是造成当前 modular 缺失的主因。第一轮应先把 stage graph 和 runtime contract 整理清楚。

## 4. 不可回退的 Phase C 正确性约束

后续所有实现都必须守住以下语义，不允许因为“重构”而变化：

### 4.1 Prompt / text 相关

- prompt 必须来自 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`，或者请求明确覆盖后的 prompt file。
- `caption_source` 只能是 `prompt_file`。
- 不允许 live `CogVLM2`。
- 文本编码输出长度必须保持 `226`，不能漂到 `512` 或其他长度。
- positive prompt suffix 必须保持当前 `VividVRPipelineConfig.default_positive_prompt_suffix` 的拼接逻辑。
- `vividvr_t5_postprocess_text()` 的“输出长度必须等于 tokenizer 输入长度”的语义不能改。

### 4.2 Video preprocess / latent 相关

- `load_control_video()` 必须保留未 padding 的 `reference_video`。
- padding 逻辑必须保持 `(original_num_frames - 1) % 8 == 0` 的尾帧补齐规则。
- control latents 的 VAE encode、缩放和 `permute` 顺序不能变。
- latent padding 到 `patch_size_t` 的逻辑不能变。
- VAE tiling 默认值必须保持 `tile_sample_min_height=240`、`tile_sample_min_width=360`。

### 4.3 Denoising 相关

- timesteps 必须继续走当前 `retrieve_timesteps()` 路径。
- rotary embedding 的构造方式不能变。
- tiling 的 slice 遍历顺序、权重融合方式、meshgrid accumulate / divide 逻辑不能变。
- CFG 的拼接顺序和计算方式不能变。
- `restoration_guidance_scale`、`restoration_ori_latent` 的 scheduler step 参数不能变。

### 4.4 Decode / postprocess 相关

- latent padding frame 的移除时机不能变。
- decode 前 `latents / self.vae.config.scaling_factor` 的缩放不能变。
- resize 回原始分辨率的顺序不能变。
- `drop first 3 frames` 必须保留。
- crop 尾部 padding frames 必须保留。
- AdaIN 必须继续使用未 padding 的 `reference_video`。
- 最终输出 shape、frame 数和 fps 必须与现有 Phase C 行为保持一致。

## 5. 建议采用的目标架构

建议把 VividVR 改造成“单 clip 的 modular pipeline”，风格参考 `wan_videoedit`，但不强行复用其 long-video orchestration。

推荐的目标 stage 顺序如下：

1. `VividVRInputValidationStage`
2. `VividVRPromptPreparationStage`
3. `VividVRTextEncodingStage`
4. `VividVRConditionEncodingStage`
5. `VividVRLatentPreparationStage`
6. `VividVRTilingPreparationStage`
7. `VividVRTimestepPreparationStage`
8. `VividVRDenoisingStage`
9. `VividVRDecodingStage`
10. `VividVROutputPostprocessStage`

注意：

- 这里没有强制设置 `LatentInitStage`，因为 VividVR 当前不是 `img2video strength<1` 那条路径。
- 这里把 tiling preparation 单独拿成一个 stage，因为它现在是 VividVR 数值路径的重要组成部分。
- `VividVRDenoisingStage` 和 `VividVRDecodingStage` 第一轮仍然建议保留为 VividVR 自定义 stage，而不是立即改造成框架标准 stage。

## 6. 每个 stage 应该负责什么

### 6.1 `VividVRInputValidationStage`

职责：

- 校验 `video_input_path` 存在。
- 校验 `prompt_file_path` / `prompt_path` 可解析。
- 校验 `caption_source == "prompt_file"`。
- 校验 `use_live_cogvlm2_caption == False`。
- 校验 `num_outputs_per_prompt == 1`。
- 校验 `dtype`、`tile_size`、`tile_stride`、`num_temporal_process_frames`、`restoration_guidance_scale` 等 request 约束。

说明：

- 这些校验大多已经在 `VividVRSamplingParams` 里做了，但 modular 风格下仍建议有一个轻量 validation stage，和 `wan_videoedit` 保持一致的阅读体验。
- 这个 stage 不负责读文件，不负责生成 tensor。

### 6.2 `VividVRPromptPreparationStage`

职责：

- 解析最终 prompt file path。
- 读取 prompt 文件文本。
- 组合 `model_prompt_text`。
- 解析 `negative_prompt`。

输出建议写入：

- `params.runtime_prompt_file_path`
- `params.runtime_raw_prompt_text`
- `params.runtime_model_prompt_text`
- `params.runtime_negative_prompt_text`

同时更新最少量 batch 字段：

- `batch.prompt`
- `batch.negative_prompt`

设计理由：

- 这一步不依赖 GPU，不应和 text encoding 混在一个 stage。
- 后续验收或 debug 时，prompt 路径和原文应成为明确的 runtime contract。

### 6.3 `VividVRTextEncodingStage`

职责：

- 使用现有 `_VividVRT5EncoderWrapper + TextEncodingStage` 编码正 prompt。
- 当 `guidance_scale > 1.0` 时编码负 prompt。
- 仅负责文本，不读视频。

输出建议写入：

- `params.runtime_prompt_embeds`
- `params.runtime_negative_prompt_embeds`
- `params.runtime_do_cfg`

必须保持：

- text len 为 `226`
- postprocess 函数仍是 `vividvr_t5_postprocess_text`
- dtype 仍按 `dit_precision` 对齐到 transformer

### 6.4 `VividVRConditionEncodingStage`

职责：

- 调用 `load_control_video()`
- 保留 `reference_video`
- 用 `VideoProcessor.preprocess_video()` 得到模型输入 control video
- 调用当前 `_prepare_latents()` 里的 VAE encode 部分得到 `control_latents`
- 记录原始高度、宽度、原始帧数、padding 帧数、fps

建议拆分方式：

- `load_control_video()` 仍保持在 `runtime/vividvr/preprocess.py`
- VAE encode control video 的逻辑可以抽为 helper，例如 `_encode_control_latents(...)`
- 不建议在这一步生成 noise latents；那属于下一 stage

输出建议写入：

- `params.runtime_control_video`
- `params.runtime_reference_video`
- `params.runtime_control_latents`
- `params.runtime_original_height`
- `params.runtime_original_width`
- `params.runtime_original_num_frames`
- `params.runtime_num_padding_frames`
- `params.runtime_fps`
- `params.runtime_padded_input_frames`

同时更新 batch：

- `batch.height`
- `batch.width`
- `batch.num_frames`
- `batch.fps`

### 6.5 `VividVRLatentPreparationStage`

职责：

- 构造 `torch.Generator(seed)`
- 根据 control video 形状、VAE temporal compression、transformer `patch_size_t`
 生成初始 noise latents
- 记录 `num_latent_padding_frames`

建议保留的 helper 行为：

- 当前 `_prepare_latents()` 中除 VAE encode 之外的 noise latent shape 计算、patch padding、
  `randn_tensor`、`init_noise_sigma` 逻辑

输出建议写入：

- `params.runtime_generator`
- `params.runtime_latents`
- `params.runtime_num_latent_padding_frames`

同时更新 batch：

- `batch.generator`
- `batch.latents`
- `batch.raw_latent_shape`

### 6.6 `VividVRTilingPreparationStage`

职责：

- 生成 `tiling_infos`
- 计算 `tile_count`
- 复制 prompt embeds / negative prompt embeds 到 tile 维度

可选预计算：

- 如果希望降低 denoising stage 内的杂项逻辑，也可以把 `ofs_emb` 和 rotary embedding
 预先准备到 runtime 中；但为了降低行为漂移，第一轮也可以让 rotary 仍在 denoising 内部构造。

输出建议写入：

- `params.runtime_tiling_infos`
- `params.runtime_tile_count`
- `params.runtime_tiled_prompt_embeds`
- `params.runtime_tiled_negative_prompt_embeds`

必须保持：

- `prepare_tiling_infos_generator()` 的遍历顺序不变
- prompt repeat 的时机和次数不变

### 6.7 `VividVRTimestepPreparationStage`

职责：

- 调用当前 `retrieve_timesteps()` 路径
- 记录 timesteps
- 记录 timestep 数量

输出建议写入：

- `params.runtime_timesteps`
- `params.runtime_timestep_count`

同时更新 batch：

- `batch.timesteps`

说明：

- 这一步从当前 denoising stage 中提前拆出，和 `wan_videoedit` 的节奏保持一致。

### 6.8 `VividVRDenoisingStage`

职责：

- 只做 denoising 主循环
- 从 runtime contract 读取：
  `latents`、`control_latents`、`prompt_embeds`、`negative_prompt_embeds`、
  `do_cfg`、`timesteps`、`tiling_infos`
- 执行 controlnet + transformer + scheduler step
- 维护 `runtime_progress`

第一轮建议：

- 保持当前自定义 denoising 逻辑几乎逐行不变，只改变输入来源。
- 不建议第一轮改成继承框架标准 `DenoisingStage` 后大量 override。

原因：

- 当前 VividVR 的 tiled controlnet + transformer + meshgrid accumulation 是数值敏感路径。
- 先把“状态来源”和“职责边界” modular 化，比先改底层 denoising 基类更稳。

输出建议写入：

- `params.runtime_latents`
- `params.runtime_progress`

同时更新 batch：

- `batch.latents`

### 6.9 `VividVRDecodingStage`

职责：

- 只负责 latent 到 decoded tensor 的 VAE decode 主体
- 处理 `num_latent_padding_frames`
- 按当前顺序做 `permute`、`scaling_factor` 还原、VAE decode

不建议在此 stage 内继续做：

- resize 回原图尺寸
- drop first 3 frames
- crop padding frames
- AdaIN

这些应移到下一 stage。

输出建议写入：

- `params.runtime_decoded_video`

### 6.10 `VividVROutputPostprocessStage`

职责：

- resize 到原始尺寸
- `video_processor.postprocess_video`
- `drop first 3 frames`
- crop 掉 `num_padding_frames`
- 用 `reference_video` 做 AdaIN
- 产出最终 `batch.output`

输出建议写入：

- `params.runtime_output_video`

同时保留 debug：

- `output_shape`
- `output_num_frames`

设计理由：

- 这一步和 `wan_videoedit` 的 `WindowPostprocessStage` 对位。
- 从 decode 中分离出来以后，后续无论做 clip merge、caption postprocess 还是输出 sidecar，
 语义都会更清楚。

## 7. 运行态 contract 的推荐设计

当前 VividVR 最大的架构缺口之一，是没有 `WanVideoEditSamplingParams` 这种 typed runtime contract。

建议在 `python/sglang/multimodal_gen/configs/sample/vividvr.py` 中增加显式运行态字段。

### 7.1 推荐字段分组

#### A. 文本与请求派生字段

- `runtime_prompt_file_path`
- `runtime_raw_prompt_text`
- `runtime_model_prompt_text`
- `runtime_negative_prompt_text`
- `runtime_do_cfg`

#### B. 输入视频与元信息

- `runtime_control_video`
- `runtime_reference_video`
- `runtime_original_height`
- `runtime_original_width`
- `runtime_original_num_frames`
- `runtime_num_padding_frames`
- `runtime_padded_input_frames`
- `runtime_fps`

#### C. 文本编码与 latent 条件

- `runtime_prompt_embeds`
- `runtime_negative_prompt_embeds`
- `runtime_control_latents`

#### D. 采样过程字段

- `runtime_generator`
- `runtime_latents`
- `runtime_num_latent_padding_frames`
- `runtime_tiling_infos`
- `runtime_tile_count`
- `runtime_timesteps`
- `runtime_timestep_count`
- `runtime_progress`

#### E. 解码与输出字段

- `runtime_decoded_video`
- `runtime_output_video`

### 7.2 `batch` 与 `params.runtime_*` 的分工

建议明确以下原则：

- `params.runtime_*` 是 stage 间状态传递的主通道。
- `batch` 只保留框架需要直接消费的公共字段，例如
  `prompt`、`negative_prompt`、`latents`、`timesteps`、`height`、`width`、`num_frames`、`fps`。
- `batch.extra["vividvr_debug"]` 可以保留，但只作为 debug 输出，不再承载主业务状态。

### 7.3 为什么不建议继续依赖 `batch.extra["vividvr_runtime"]`

因为它会带来三个问题：

1. 类型和字段集合不显式，维护时很难发现字段遗漏。
2. stage 拆开之后，状态读写关系会继续变得不可追踪。
3. 单元测试很难对 runtime contract 做精确断言。

## 8. 推荐的迁移顺序

为了尽量减小数值漂移，建议按下面顺序实施，而不是一次性大改。

### 阶段 1：先引入 runtime contract，不改变 stage 数量

做法：

- 给 `VividVRSamplingParams` 增加 `runtime_*` 字段。
- 在现有 3-stage 实现内部，先改成“写新 contract，同时保留旧 debug 输出”。
- 尽量不改计算顺序。

目标：

- 先把“状态从哪里来、流向哪里”稳定下来。
- 这一阶段应该是最小风险、几乎 no-op 的重构。

### 阶段 2：拆出 prompt 和 text encoding

做法：

- 从 `VividVRBeforeDenoisingStage` 中先抽出：
  `VividVRPromptPreparationStage`、`VividVRTextEncodingStage`
- 下游继续沿用旧的 control/latent/tiling 逻辑

目标：

- 先把 CPU 侧文本逻辑拆开，几乎不碰数值密集区域。

### 阶段 3：拆出 condition encoding 和 latent preparation

做法：

- 把视频读取、reference 保留、video preprocess、control latent encode
  拆到 `VividVRConditionEncodingStage`
- 把 generator/noise latent 逻辑拆到 `VividVRLatentPreparationStage`

目标：

- 把当前最大的 `BeforeDenoisingStage` 解耦成和 `wan_videoedit` 相似的结构。

### 阶段 4：拆出 tiling 和 timestep

做法：

- 新增 `VividVRTilingPreparationStage`
- 新增 `VividVRTimestepPreparationStage`
- `VividVRDenoisingStage` 改为只读取 runtime contract，不再负责这些预处理

目标：

- 让 denoising 主循环真正只剩 denoising。

### 阶段 5：拆出 output postprocess

做法：

- 把 resize / frame drop / crop padding / AdaIN 从 decode 中拿出来
- decode 只负责 VAE decode

目标：

- 让 output 后处理语义清晰，后续也更方便接 clip merge 或其他后处理模块。

### 阶段 6：清理旧状态通道

做法：

- 移除 `batch.extra["vividvr_runtime"]` 的主逻辑依赖
- 保留 `batch.extra["vividvr_debug"]`
- 如有必要，再讨论是否把更多模块变成 `_required_config_modules`

目标：

- 完成真正的 modular 风格落地。

## 9. 不建议第一轮做的“激进重构”

以下方向理论上可行，但第一轮不建议做：

### 9.1 直接把 VividVR 改成标准 `DenoisingStage` + `DecodingStage` hook 风格

不建议原因：

- 当前 VividVR 的 denoising 比 `wan_videoedit` 更数值敏感。
- controlnet、tiling、meshgrid accumulate、restoration guidance 都是漂移高风险点。
- 第一轮应先保持 denoising 数学路径不变，只重构前后 stage 边界。

### 9.2 同时重写 `initialize_pipeline()`

不建议原因：

- 模块加载方式和 modular stage 风格不是同一个问题。
- 同时改加载路径和执行路径，定位回归会非常困难。

### 9.3 同时开始做 long-video Phase D

不建议原因：

- 架构整理和能力扩展要分开做。
- 先把单 clip modular 化并守住 Phase C，才适合继续向 clip split / merge 扩展。

## 10. 验收要求

后续按本文档改代码时，验收必须包含两层。

### 10.1 硬门槛：Phase C 必须继续通过

必须继续通过：

- `python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py`

其中硬门槛包括：

- `summary["ssim_min"] >= 0.90`
- `summary["mse_max"] <= 150.0`
- `summary["mae_max"] <= 8.0`
- `failed_frame_ratio <= 0.05`
- `summary["pass_compare"] == true`

并且 debug 字段仍应存在：

- `prompt_embed_shape`
- `control_latent_shape`
- `latents_shape`
- `timestep_count`
- `tile_count`

### 10.2 软目标：尽量贴近 2026-06-04 Phase C 基线

基线文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T070647Z.json`

关键基线值：

- `ssim_mean = 0.967716215299506`
- `ssim_min = 0.9473462237832677`
- `mse_mean = 39.878108160836355`
- `mse_max = 81.55096435546875`
- `mae_mean = 3.3365604979651313`
- `mae_max = 3.9912755489349365`
- `failed_frame_ratio = 0.0`
- `prompt_embed_shape = [1, 226, 4096]`
- `control_latent_shape = [1, 20, 16, 90, 120]`
- `latents_shape = [1, 20, 16, 90, 120]`
- `tile_count = 1`
- `padded_input_frames = 73`
- `timestep_count = 50`
- `output_shape = [3, 70, 720, 960]`
- `output_num_frames = 70`

建议把以下内容作为重构期间的内部目标，而不是仅满足宽松硬门槛：

- 输出帧数保持完全一致
- debug shape 保持完全一致
- `tile_count`、`timestep_count`、`padded_input_frames` 保持完全一致
- `ssim_mean`、`ssim_min` 尽量只出现非常小的波动
- `mse/mae` 不应出现明显抬升

如果重构后数值发生明显漂移，优先排查：

1. text postprocess 是否仍保持 `226`
2. prompt suffix 拼接是否改变
3. `reference_video` 是否误用了 padding 后视频
4. latent padding 去除时机是否改变
5. `drop first 3 frames` / crop padding / AdaIN 顺序是否改变
6. tiling 遍历与权重融合顺序是否改变

## 11. 验证执行方式

按照仓库根目录 `AGENTS.md` 的约束，后续所有长时间推理验证必须在 `tmux` 中运行，方便用户实时查看进度。

推荐方式：

```bash
tmux new-session -d -s vividvr_phasec_refactor \
  'cd /home/zhiheng/sglang && \
   SGLANG_RUN_VIVIDVR_ACCEPTANCE=1 PYTHONPATH=python uv run \
   --with pytest --with diffusers==0.37.0 --with imageio==2.36.0 \
   --with imageio-ffmpeg==0.5.1 --with addict==2.4.0 --with PyYAML==6.0.1 \
   --with av==16.1.0 --with scikit-image==0.25.2 --with cache-dit==1.3.0 \
   --with opencv-python-headless==4.10.0.84 --with trimesh \
   python -m pytest \
   python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py -q'
```

查看方式：

```bash
tmux attach -t vividvr_phasec_refactor
```

如果需要更快拿一次单 run 指标，也可运行：

- `python/sglang/multimodal_gen/tools/run_vividvr_phase_c_single.py`

同样必须放在 `tmux` 中执行。

## 12. 推荐的完成标准

当且仅当满足以下条件时，认为这轮 modular 重构完成：

1. `vividvr_pipeline.py` 的 stage graph 已经接近 `wan_videoedit` 风格，不再是单个大 `BeforeDenoisingStage`。
2. `VividVRSamplingParams` 已具备显式 `runtime_*` contract。
3. `batch.extra["vividvr_runtime"]` 不再承担主业务状态传递职责。
4. decode 后处理已拆成独立 stage。
5. Phase C 单测和重推理验收都通过。
6. 结果与 `phase_c_metrics_seed42_20260604T070647Z.json` 保持尽可能小的差异。

## 13. 最后建议

最稳妥的实施策略不是“把 VividVR 改得更像框架”，而是“先把它改得更像 `wan_videoedit`”。

也就是：

- 先做 stage 职责拆分
- 先做 typed runtime contract
- 先守住 Phase C 数值路径

在这个基础上，如果后面还想进一步标准化到框架 helper / hook 风格，再做第二阶段重构会更安全。
