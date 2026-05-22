# STAR CogVideoX-SR 接入 SGLang 总方案

## 1. 文档定位

本文件只做总方案规划，不做任何代码实现。

目标是将 `STAR_mg` 的 **CogVideoX 视频超分分支** 以 **SGLang native pipeline** 的方式接入到 `sglang.multimodal_gen` 中，从而复用 SGLang 现有的推理加速、调度、并行和 pipeline 组织能力。

本轮规划需要同时满足以下约束：

1. 仅接入 `STAR_mg/cogvideox-based` 分支，不接 `I2VGen/VEnhancer` 分支。
2. 接入后运行时不能依赖原 `STAR_mg` 仓库路径、环境变量、数据集类和私有调用链。
3. 方案优先采用 **Style B: Modular 组合式 Pipeline**，尽量复用 SGLang 现有 stage 和 config hook。
4. 需要明确接口分层、数据流、模块边界、权重组织方式和后续 upstream 同步策略。

---

## 2. 结论先行

### 2.1 选型结论

`STAR_mg` 的 CogVideoX-SR 分支适合接入 SGLang，但应该按 **“T2V 主干 + LQ 视频条件分支”** 来建模，而不是复用 SGLang 当前的 `TI2V` 语义。

原因如下：

1. SGLang 当前 `TI2V` 路径带有明显的 **Wan 特化语义**，尤其是首帧保留、特殊 timestep 展开和后处理逻辑，不适合 STAR 的“整段 LQ latent 通道拼接”范式。
2. STAR CogVideoX-SR 的核心条件方式，是把 `LQ video latent` 与 `noisy latent` 按 **channel 维拼接** 后送入 DiT，这与 SGLang 标准 `DenoisingStage` 的 `batch.image_latent` 拼接路径高度一致。
3. 因此最合理的方案是：
   - 保持主任务类型接近 `T2V`
   - 把低清视频作为额外条件输入
   - 让自定义 stage 负责“视频加载与编码”
   - 让标准 `DenoisingStage` 继续负责 scheduler 循环和加速路径

### 2.2 方案总方向

整体采用：

`InputValidationStage`
-> `STARConditionVideoLoadingStage`
-> `TextEncodingStage`
-> `STARConditionVideoVAEEncodingStage`
-> `LatentPreparationStage`
-> `TimestepPreparationStage`
-> `DenoisingStage`
-> `STARCogVideoXSRDecodingStage`

这里的核心思想是：

1. **通用逻辑复用 SGLang**
   - 输入基础校验
   - 文本编码
   - latent 初始化
   - timestep/scheduler 驱动
   - denoising 主循环
   - registry / loader / server 接口
2. **STAR 专属逻辑最小化隔离**
   - LQ 视频加载与预处理
   - LQ 视频 VAE 编码
   - STAR DiT 结构
   - STAR 采样器适配
   - STAR 时序分块解码策略

---

## 3. 上游原理摘要

## 3.1 STAR CogVideoX-SR 的原始推理链路

从原实现看，CogVideoX-SR 的推理逻辑可以概括为：

1. 输入：
   - 文本 prompt
   - LQ 视频 `lq`
2. 预处理：
   - 按固定目标分辨率处理视频
   - 构造 cond / uncond 文本条件
3. 条件视频编码：
   - `lq` 送入 3D VAE 编码成视频 latent
4. 采样：
   - 生成与目标 latent 同形状的噪声
   - 将 `noisy latent` 与 `lq latent` 沿 channel 维拼接
   - 输入 STAR 改造过的 DiT
5. CFG：
   - 原 SAT 版本通过 batch doubling 的方式并行做 cond/uncond
6. 解码：
   - 不是一次性整段 decode
   - 而是按时间窗口分块 decode，再拼接结果
7. 后处理：
   - 可选颜色修正
   - 输出 mp4

### 3.1.1 STAR 的关键模型特征

CogVideoX-SR 分支的关键特征不是“一个普通的 CogVideoX 配置”，而是以下几个结构性改动：

1. **Patch embedding 改造**
   - 输入通道翻倍
   - 用 `proj_sr` 直接处理 `(noisy_latent, lq_latent)` 拼接结果
2. **AdaLN + 局部增强**
   - 每层在 attention 前增加局部空间/时间增强逻辑
   - 这是 STAR 的时空增强核心之一
3. **采样器与 guider**
   - 原始实现使用 SAT/SGM 风格的采样器与 Dynamic CFG
4. **3D VAE 时序分块 decode**
   - 解码策略带有模型实现依赖，不能简单替换成“整段 decode”

### 3.1.2 一个重要适配判断

原 STAR 在 CFG 时会把 `lq latent` 复制两份后与 cond/uncond 一起送入 sampler。

但 SGLang 标准 `DenoisingStage` 的 CFG 是 **串行双前向**，不是 batch doubling，因此接入到 SGLang 后：

1. `batch.image_latent` 不需要手工复制成 cond/uncond 两份
2. 同一份 `lq latent` 可分别参与正负条件两次 forward
3. 这能减少不必要的中间张量放大，也更贴合 SGLang 现有框架

## 3.2 SGLang 当前可复用能力

从 `multimodal_gen` 现有实现看，以下能力可以直接复用：

1. `TextEncodingStage`
   - 已有成熟文本编码链路
2. `LatentPreparationStage`
   - 已有初始噪声 latent 生成逻辑
3. `TimestepPreparationStage`
   - 已有 scheduler `set_timesteps` 与 timestep 分发逻辑
4. `DenoisingStage`
   - 已支持标准 scheduler 循环
   - 已支持 CFG
   - 已支持 `batch.image_latent` 与 `latents` 的 channel 拼接
5. `PipelineConfig` hook 机制
   - 可用来准备 cond kwargs、VAE encode/decode 缩放、SP gather/shard 等

## 3.3 当前框架缺口

当前 SGLang 直接缺的不是“视频扩散主循环”，而是以下几块：

1. **缺少整段条件视频输入通路**
   - 当前 `image_path` 读 `.mp4` 只取首帧
2. **缺少整段条件视频的 VAE 编码 stage**
   - 现有 `ImageVAEEncodingStage` 语义更偏“单张图像条件”
3. **缺少 STAR 的 DiT / VAE / scheduler 适配**
4. **缺少 STAR 专用时序分块 decode 策略**

---

## 4. 总体架构设计

## 4.1 总体原则

### 原则 A：运行时彻底脱离 STAR 仓库

运行时不得依赖：

1. `STAR_mg` 的 Python 包路径
2. `STAR_COG_*` 环境变量
3. `PairedCaptionDataset` 等数据集结构
4. SAT 的训练框架入口
5. 原 YAML 配置作为运行时唯一配置源

运行时应只依赖：

1. SGLang 自己的 pipeline config
2. SGLang 自己的 model config
3. 本地或转换后的模型权重目录

### 原则 B：只迁移推理所需最小闭包

只迁移以下内容：

1. 推理所需 DiT 结构
2. 推理所需 3D VAE 结构
3. 采样器/调度器适配
4. 输入预处理和输出后处理所需最小逻辑

不迁移以下内容：

1. 训练 loss
2. 数据集类
3. 训练脚本
4. checkpoint 管理逻辑
5. 与训练相关的 SAT 调度封装

### 原则 C：优先复用通用 stage，必要时新增薄适配层

不推荐把 STAR 所有前处理都塞进一个超大的 `BeforeDenoisingStage`。

更推荐：

1. 继续使用标准 `TextEncodingStage`
2. 继续使用标准 `LatentPreparationStage`
3. 继续使用标准 `TimestepPreparationStage`
4. 继续使用标准 `DenoisingStage`
5. 仅新增：
   - 条件视频加载 stage
   - 条件视频 VAE 编码 stage
   - 必要时的 STAR 专用 decoding stage

## 4.2 模块分层

建议将接入分为五层：

### 第 1 层：外部接口层

负责接收用户请求参数。

建议暴露：

1. 文本 prompt / negative prompt
2. `condition_video_path`
3. `num_frames`
4. `height` / `width`
5. `fps`
6. `num_inference_steps`
7. `guidance_scale`
8. 可选后处理开关，如 `enable_color_fix`

说明：

1. 不建议继续复用 `image_path` 承载 mp4
2. 应为 STAR-SR 增加独立、清晰的“条件视频输入”字段

### 第 2 层：pipeline 编排层

负责决定 stage 的顺序和数据契约。

建议新增：

1. `StarCogVideoXSRPipeline`

职责：

1. 注册所需模块
2. 组装 stage
3. 明确这是一个 **带条件视频分支** 的 video pipeline

### 第 3 层：stage 适配层

负责把用户输入变成标准 `Req` 字段，供后续标准 stage 使用。

建议新增：

1. `STARConditionVideoLoadingStage`
2. `STARConditionVideoVAEEncodingStage`
3. `STARCogVideoXSRDecodingStage`

### 第 4 层：模型组件层

负责 STAR 专属模型结构。

建议新增：

1. STAR DiT
2. STAR 3D VAE
3. STAR scheduler adapter

### 第 5 层：模型资产与权重转换层

负责把 STAR 原始 checkpoint 变成 SGLang 可直接加载的结构。

建议新增：

1. 独立权重转换脚本
2. 独立模型目录约定

---

## 5. 数据接口设计

## 5.1 SamplingParams 设计

建议新增专用 sampling params，而不是一上来改动所有通用请求结构。

建议新增：

1. `StarCogVideoXSRSamplingParams`

建议字段：

1. `condition_video_path: str | None`
2. `condition_video_start: int | None`
3. `condition_video_num_frames: int | None`
4. `condition_video_sample_fps: int | None`
5. `enable_color_fix: bool`
6. `color_fix_mode: str | None`

设计理由：

1. 对外接口清晰
2. 不污染现有所有模型的基础采样参数
3. 如果后续出现第二个 video-conditioned 模型，再决定是否把 `condition_video_path` 上提到 base `SamplingParams`

## 5.2 Req 运行时字段设计

建议在 `Req` 中增加少量明确字段，而不是全部塞进 `extra`。

建议新增：

1. `condition_video`
   - 含义：条件视频张量或帧序列
2. `original_condition_video_size`
   - 含义：原始条件视频尺寸
3. `condition_video_num_frames`
   - 含义：预处理后实际参与编码的帧数

其余细节可以放入：

1. `batch.extra["condition_video_meta"]`

例如：

1. 原始 fps
2. 抽帧策略
3. resize / crop 方案
4. temporal chunk 信息

## 5.3 条件 latent 复用策略

建议复用现有字段：

1. `batch.image_latent`

虽然命名上偏“image”，但它在标准 `DenoisingStage` 中的实际作用是“额外条件 latent 输入”，对 STAR 来说完全合适。

不建议新增：

1. `batch.lq_latent`
2. `batch.condition_video_latent`

原因：

1. 会导致 denoising 主循环增加分支
2. 会降低复用度
3. 不利于后续更多“条件 latent 拼接型”模型复用

---

## 6. Pipeline 设计

## 6.1 任务类型选择

### 结论

不建议把 STAR-SR 标成 `TI2V`。

更合理的做法是：

1. 视为 `T2V` 主干
2. 再额外挂接 `condition_video_path` / `batch.image_latent`

### 原因

1. 当前 `TI2V` 在 SGLang 内部已经携带 Wan 特有逻辑
2. STAR 的条件机制不是“首帧图像注入”，而是“整段 latent 通道拼接”
3. 如果复用 `TI2V`，会在 `DenoisingStage` 中触发不必要的特化路径和语义冲突

## 6.2 推荐 stage 顺序

建议 pipeline 按如下顺序组装：

1. `InputValidationStage`
2. `STARConditionVideoLoadingStage`
3. `TextEncodingStage`
4. `STARConditionVideoVAEEncodingStage`
5. `LatentPreparationStage`
6. `TimestepPreparationStage`
7. `DenoisingStage`
8. `STARCogVideoXSRDecodingStage`

### 各 stage 责任

#### 1. InputValidationStage

保留其现有职责：

1. prompt / negative prompt 校验
2. seed / generator 生成
3. 通用推理参数合法性检查

本阶段不再负责读取 `.mp4` 条件视频。

#### 2. STARConditionVideoLoadingStage

新增，职责如下：

1. 读取 `condition_video_path`
2. 解析原始帧、fps、分辨率
3. 根据 STAR 兼容策略做抽帧、裁剪、resize
4. 形成统一的 `batch.condition_video`
5. 写入 `original_condition_video_size` 与相关 meta
6. 若用户未显式指定 `height/width/num_frames`，根据配置补齐

#### 3. TextEncodingStage

尽量复用标准实现，仅通过 pipeline config 调整：

1. tokenizer 参数
2. T5 max_length
3. prompt postprocess 逻辑

#### 4. STARConditionVideoVAEEncodingStage

新增，职责如下：

1. 将整段 LQ 视频转为 `[B, C, T, H, W]`
2. 送入 STAR 3D VAE 编码
3. 做 scale / shift / normalize
4. 输出到 `batch.image_latent`

注意：

1. 这里不是“单图变首帧条件”
2. 这里是“整段视频条件 latent”
3. 不应沿用现有 `ImageVAEEncodingStage` 的“单图 + 后续补零帧”语义

#### 5. LatentPreparationStage

直接复用，用来生成待去噪的初始噪声 latent。

#### 6. TimestepPreparationStage

直接复用，前提是 STAR 的 scheduler adapter 实现了标准接口。

#### 7. DenoisingStage

尽量直接复用。

复用条件：

1. STAR DiT forward 签名适配到 SGLang 标准调用方式
2. `batch.image_latent` 维度与 `latents` 匹配
3. scheduler adapter 符合标准 `scale_model_input` / `step` 接口

#### 8. STARCogVideoXSRDecodingStage

建议新增，而不是强塞给通用 `DecodingStage`。

原因：

1. STAR 原始 decode 带时序窗口策略
2. 这不是通用 VAE decode 语义
3. 放入独立 decoding stage 能把模型特有逻辑隔离开

---

## 7. 组件实现规划

## 7.1 PipelineConfig

建议新增：

1. `StarCogVideoXSRPipelineConfig`

职责：

1. 定义模型架构配置
2. 定义精度、VAE 参数、调度器参数
3. 定义 cond kwargs 适配方式
4. 定义 preprocess / decode hook
5. 定义 SP shard / gather 策略

建议重点实现以下 hook：

1. `prepare_pos_cond_kwargs`
2. `prepare_neg_cond_kwargs`
3. `prepare_latent_shape`
4. `postprocess_vae_encode`
5. `normalize_vae_encode`
6. `preprocess_decoding` 或交给自定义 decoding stage
7. 如有需要，`slice_noise_pred`

## 7.2 DiT 适配

建议新增：

1. `runtime/models/dits/star_cogvideox_sr.py`

适配目标：

1. 不依赖 SAT `BaseModel` 运行时
2. forward 签名对齐 SGLang `DenoisingStage`
3. 支持输入 `hidden_states=[B,C,T,H,W]`
4. 内部完成：
   - 通道拼接后的 patch embedding
   - 文本条件注入
   - 位置编码
   - STAR 局部空间/时间增强
   - 最终噪声预测

### DiT 适配原则

1. 保留 STAR 推理等价结构
2. 不把 SAT 的训练/并行框架整包搬过来
3. 如果参数命名无法直接兼容，使用转换脚本做 key remap，而不是在运行时写大量兼容分支

## 7.3 文本编码器

优先级建议如下：

1. 优先复用 SGLang 现有 T5 编码链路
2. 若原 STAR 文本编码权重布局与现有 `T5Config` 可兼容，则不新增 encoder 实现
3. 若存在明显布局差异，再补一个“薄适配 wrapper”，但仍复用 `TextEncodingStage`

目标是：

1. 避免把 STAR 的 conditioner 系统整体引入 SGLang

## 7.4 3D VAE 适配

建议新增：

1. `runtime/models/vaes/star_cogvideox_vae.py`

职责：

1. 提供标准 `encode` / `decode`
2. 保持与 STAR 的 latent scale/shift 行为一致
3. 只保留推理所需路径

### VAE 解耦策略

建议将“VAE 本体”和“时序窗口 decode 策略”拆开：

1. VAE 类只负责 encode/decode 算子本体
2. `STARCogVideoXSRDecodingStage` 负责：
   - 时间分块
   - 窗口调度
   - 拼接输出

这样做的好处：

1. VAE 权重同步更简单
2. decode 策略变动不会污染 VAE 类
3. 未来若同一 VAE 被别的 pipeline 复用，也不受 STAR-SR 的窗口逻辑影响

## 7.5 Scheduler 适配

建议新增：

1. `runtime/models/schedulers/star_vpsde_dpmpp2m.py`

初版原则：

1. 尽量保持 STAR 原采样语义
2. 不在第一版里偷偷替换成“看起来差不多”的现有 scheduler

原因：

1. STAR 的采样器与 guider 组合可能直接影响收敛轨迹和视觉结果
2. 初版目标是“功能和结果先对齐”
3. 后续再评估是否可替换为现成 `UniPC` / `DPM-Solver` 以进一步减少维护成本

## 7.6 条件视频加载与编码 stage

建议新增两个 stage，而不是一个超大 stage：

1. `STARConditionVideoLoadingStage`
2. `STARConditionVideoVAEEncodingStage`

### 为什么拆成两个 stage

1. 输入 I/O / 预处理 与 GPU VAE 编码职责不同
2. 未来更容易单独 profile
3. 后续其他 video-conditioned 模型也能复用前一半逻辑
4. 更符合 modular style

## 7.7 Decoding stage

建议新增：

1. `STARCogVideoXSRDecodingStage`

其职责包括：

1. 应用 latent 反 scale/shift
2. 按 STAR 原始时间窗口策略 decode
3. 拼接完整视频
4. 调用可选后处理
5. 输出标准 `OutputBatch`

### 可选后处理策略

建议分两层：

1. **MVP 默认关闭**
   - color fix 不是接入主干必需能力
2. **保留接口**
   - 若后续需要对齐 STAR 默认输出，可在 decode 后增加可选 color fix hook

---

## 8. 目录与文件级改造计划

以下是建议的文件布局，目的是把“通用扩展”和“STAR 专属适配”分开。

## 8.1 新增的 STAR 专属文件

建议新增：

1. `python/sglang/multimodal_gen/runtime/pipelines/star_cogvideox_sr_pipeline.py`
2. `python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py`
3. `python/sglang/multimodal_gen/configs/sample/star_cogvideox_sr.py`
4. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
5. `python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py`
6. `python/sglang/multimodal_gen/runtime/models/schedulers/star_vpsde_dpmpp2m.py`
7. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py`

## 8.2 新增的可复用通用文件

建议新增：

1. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_loading.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py`

如果实现时发现与现有 `ImageVAEEncodingStage` 高度重合，可改为：

1. 抽公共 helper / mixin
2. 再让图像版和视频版共用

## 8.3 需要变更的通用文件

预计需要小范围修改：

1. `runtime/pipelines_core/schedule_batch.py`
   - 增加 `condition_video` 相关字段
2. `runtime/pipelines_core/stages/__init__.py`
   - 导出新 stage
3. `runtime/pipelines/__init__.py`
   - 确保 pipeline 可被自动发现
4. `configs/pipeline_configs/__init__.py`
5. `configs/sample/__init__.py`
6. `registry.py`
   - 注册 sampling params / pipeline config / model detector

## 8.4 权重与工具文件

建议新增：

1. `python/sglang/multimodal_gen/tools/convert_star_cogvideox_sr.py`
2. `docs_xzh/add_STAR/code_plan/weight_layout.md`（后续可选）

---

## 9. 权重与模型资产策略

## 9.1 不建议直接在运行时读取 STAR 原 YAML + checkpoint 组合

不建议让 SGLang 运行时直接依赖：

1. STAR 的 YAML 配置
2. STAR 的 SAT checkpoint 目录结构
3. STAR 的环境变量约定

原因：

1. 运行时耦合过高
2. 维护困难
3. upstream 改动会直接传导到线上加载路径

## 9.2 推荐使用“离线转换 + 运行时原生加载”

建议采用：

1. 离线转换脚本读取 STAR checkpoint
2. 输出 SGLang 原生模型目录

建议输出目录结构包含：

1. `transformer/`
2. `vae/`
3. `text_encoder/`
4. `scheduler/`
5. `model_index.json` 或等价元信息

### 好处

1. 运行时不依赖 STAR 仓库
2. 模型加载路径稳定
3. 后续升级时可以只更新转换脚本
4. 更适合 CI、部署和自动化测试

## 9.3 upstream 同步策略

建议把同步分为两类：

1. **结构同步**
   - 如果 STAR 改了 DiT / VAE 结构，则更新对应 `runtime/models/*`
2. **权重同步**
   - 如果 STAR 只更新权重，不改结构，则只跑转换脚本

这样可以把“代码同步”和“权重同步”解耦。

---

## 10. 关键设计抉择

## 10.1 不复用 `image_path` 作为 mp4 输入

理由：

1. 现有逻辑对 mp4 只取首帧
2. 语义错误
3. 后续维护会持续混淆“图像条件”和“视频条件”

## 10.2 不复用 `TI2V` 语义

理由：

1. 当前实现带 Wan 特化逻辑
2. 与 STAR 的整段 latent 通道拼接不一致
3. 会增加调试和维护成本

## 10.3 尽量复用标准 `DenoisingStage`

这是整个方案最重要的“复用中心”。

只要以下三点成立，就不应该重写 denoising 主循环：

1. STAR transformer forward 已适配
2. scheduler 接口已适配
3. `batch.image_latent` 能正确提供条件 latent

## 10.4 将 STAR 特殊 decode 逻辑放到专用 decoding stage

理由：

1. STAR 的时间窗口 decode 不是通用行为
2. 把它塞进通用 `DecodingStage` 会污染框架
3. 独立 stage 最便于后续替换和比对

---

## 11. 开发阶段拆分

## 阶段 0：方案确认

目标：

1. 确认本文件作为总设计基线
2. 确认本期只做 CogVideoX-SR
3. 确认走 native SGLang pipeline，不走 diffusers fallback

## 阶段 1：模型资产解耦

目标：

1. 定义 SGLang 原生模型目录结构
2. 设计 checkpoint 转换脚本
3. 明确参数映射规则

交付：

1. 权重转换脚本设计
2. 目录结构规范
3. key remap 规则

## 阶段 2：请求接口与 stage 骨架

目标：

1. 新增 sampling params
2. 新增 `Req.condition_video` 契约
3. 新增条件视频加载与编码 stage 骨架

交付：

1. pipeline skeleton
2. stage skeleton
3. registry 接线方案

## 阶段 3：模型组件适配

目标：

1. 完成 DiT 适配
2. 完成 VAE 适配
3. 完成 scheduler adapter

交付：

1. 单卡 smoke test
2. 基本 shape 对齐

## 阶段 4：解码与结果对齐

目标：

1. 完成时序分块 decode
2. 跑通端到端推理
3. 与原 STAR 做结果对齐

交付：

1. 结果视频
2. 中间 latent / decode parity 记录

## 阶段 5：性能与维护性收尾

目标：

1. 验证是否走到 SGLang 标准加速路径
2. 评估 CPU offload / VAE tiling / SP 兼容性
3. 清理多余耦合

交付：

1. 性能对比记录
2. 风险清单收敛

---

## 12. 测试与验收计划

## 12.1 功能验收

至少验证：

1. 能接收 `prompt + condition_video_path`
2. 能正确读取整段视频而不是首帧
3. 能产生与目标 shape 一致的 `batch.image_latent`
4. 能进入标准 `DenoisingStage`
5. 能完成时序分块 decode
6. 能输出 mp4

## 12.2 结果对齐

建议做三类对齐：

1. **shape 对齐**
   - 文本 embedding shape
   - 条件 latent shape
   - noise latent shape
2. **数值对齐**
   - 固定 seed 下若干中间层统计量
   - 若干 timestep 的 latent 均值/方差
3. **结果对齐**
   - 最终视频主观观感
   - 简单客观指标，如帧级差异统计

## 12.3 性能验收

至少比较：

1. 原 STAR 推理时延
2. SGLang native pipeline 推理时延
3. 峰值显存
4. VAE decode 阶段显存曲线

---

## 13. 主要风险与规避策略

## 风险 1：SAT 模型结构与 SGLang 现有 DiT 抽象差异较大

规避：

1. 直接新增 STAR 专属 DiT 文件
2. 不强行塞进现有 CogVideoX 或其他 DiT 结构
3. 用转换脚本解决权重命名差异

## 风险 2：VAE decode 窗口策略如果处理不当，会出现时序拼接异常

规避：

1. 保留原 STAR 的窗口 decode 语义
2. 将窗口策略封装到专用 decoding stage
3. 单独做 decode parity 验证

## 风险 3：若误用 `TI2V` 路径，会与 Wan 专用逻辑冲突

规避：

1. 明确不走 `TI2V` 语义
2. 采用 `T2V + condition_video` 模型化方式

## 风险 4：若直接复用 `image_path`，后续维护会持续混乱

规避：

1. 独立引入 `condition_video_path`
2. 明确区分图像条件与视频条件

## 风险 5：若运行时继续依赖 STAR repo，将严重影响后续合并升级

规避：

1. 使用离线权重转换
2. 运行时只认 SGLang 原生模型目录

---

## 14. 本方案的最终推荐实现形态

综合可维护性、复用度和后续扩展性，最终推荐实现形态如下：

1. 新建 `StarCogVideoXSRPipeline`
2. 采用 **Modular Pipeline**
3. 继续复用：
   - `InputValidationStage`
   - `TextEncodingStage`
   - `LatentPreparationStage`
   - `TimestepPreparationStage`
   - `DenoisingStage`
4. 新增薄适配 stage：
   - `STARConditionVideoLoadingStage`
   - `STARConditionVideoVAEEncodingStage`
   - `STARCogVideoXSRDecodingStage`
5. 新增 STAR 专属模型组件：
   - DiT
   - VAE
   - scheduler adapter
6. 使用离线权重转换实现与 STAR 仓库解耦
7. 不复用 `TI2V`
8. 不复用 `image_path` 承载视频

这是当前最符合目标文档要求的方案：

1. 模块边界清晰
2. 与原 STAR 仓库松耦合
3. 最大化复用 SGLang 既有加速路径
4. 后续升级 SGLang 或同步 STAR upstream 时，改动面最可控

---

## 15. 下一轮实施建议

下一轮正式开始代码改造时，建议按以下顺序推进：

1. 先落“模型资产解耦与目录规范”
2. 再落“SamplingParams + Req + 条件视频 stage 骨架”
3. 然后接 DiT / VAE / scheduler
4. 最后接 decoding parity 与结果验收

这样可以确保我们先把接口和边界钉住，再接模型主体，避免后面返工。
