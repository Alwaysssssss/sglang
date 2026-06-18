# VividVR 对齐编辑模型服务方式计划

## 1. 背景

- 当前 `VividVR` 已完成 `Phase C / D / E` 语义与默认质量基线收口，后续服务化必须复用现有已验收推理链，不能新造一条脱离基线的简化执行路径。
- `sglang` 现有视频编辑服务已经具备可复用框架，包括：
  - `/v1/videos/repairs` 异步提交入口
  - `VIDEO_STORE` 任务状态存储
  - `prepare_request(...) -> Req -> scheduler` 调度链路
  - `GET /v1/videos/{id}`、`/progress`、`/content` 结果查询方式
- 现有样例服务是 `WanVideoEdit` 风格，请求契约围绕 `prompt + video + mask` 设计；这与 `VividVR` 当前语义不一致，不能直接照搬字段集合。

## 2. 本轮服务化目标

- 目标不是重写一套新服务，而是复用现有视频编辑服务框架，为 `VividVR` 增加一个对外可调用的编辑服务入口。
- 服务启动后，调用方按照约定格式提交视频编辑请求，服务端完成：
  - 输入视频接收
  - 读取现有 demo prompt
  - `VividVR` 推理调度
  - 任务进度记录
  - 输出结果回传或落盘

## 3. 关键约束

### 3.1 基线保护

- 必须复用当前已验收的 `VividVRPipeline` 执行链。
- 不允许为了服务化绕开当前长视频 stage executor 语义。
- 不允许引入 live `CogVLM2` 作为 `VividVR` runtime 内部 caption 路径。

### 3.2 用户请求约束

- 用户不需要输入 `prompt_file_path`。
- 第一版也不要求用户直接输入 `prompt` 文本。
- 用户只提交视频和必要的推理控制参数；当前阶段提示词由服务端直接读取现有 demo 对应 prompt 文件。

### 3.3 与当前 `VividVR` 采样契约的桥接要求

- 当前 `VividVRSamplingParams` 仍以 `prompt_file_path` / `caption_file_path` 为主输入语义。
- 当前阶段为了不破坏已验收主链，服务端应优先采用“内部 prompt 文件桥接”：
  - 服务端固定读取现有 demo prompt 文件
  - 调 `VividVR` 时继续走现有 `prompt_file_path` 语义
  - `prompt_file_path` 仅作为服务端内部配置，不作为外部请求字段
- 后续若接入 caption 模型，再单独设计 `caption_file` 桥接方案，不在当前阶段混入。

## 4. 对齐 Wan 编辑服务方式的总体策略

## 4.1 复用的部分

- 继续复用当前仓库 `/home/zhiheng/sglang` 内部已经存在的视频服务框架。
- 具体复用对象是：
  - [video_api.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:344) 中 `/v1/videos/repairs` 一类异步提交模式
  - [http_server.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:281) 挂载的 `video_api.router`
  - 当前仓库内的任务存储、后台调度、进度查询、结果下载能力
  - 当前仓库内的输入文件下载/落盘、输出路径拆分、后台 `asyncio` dispatch 方式
- 这里的“复用”只指当前项目 `sglang` 仓库内已有实现，不涉及项目外 `/home/zhiheng/sglang_serve` 的代码复用。

## 4.2 不直接复用的部分

- 不复用 `WanVideoEdit` 的 mask 编辑语义。
- 不复用 `WanVideoEditSamplingParams`。
- 不复用 `mask_input_path / mask_url / bbox / feather / paste_back` 一整套局部修补字段。
- 不复用 `infer_len / overlap / dynamic_cfg / init_latent_mode / decode_mode` 这类 Wan 专属窗口控制字段作为 `VividVR` 外部契约。

## 5. VividVR 服务第一版建议请求契约

### 5.1 建议保留的通用服务字段

- `task_id`
- `model`
- `callback_url`
- `output_storage`
- `output_path`
- `output_bucket`
- `output_object_key`

### 5.2 建议保留的输入字段

- `video_input_path`
- `video_url`

约束：

- 二选一，至少提供一个。
- 第一版不支持 `mask_input_path` / `mask_url`。
- 第一版不支持 `reference_image_url`。

### 5.3 建议保留的通用推理字段

- `num_frames`
- `num_inference_steps`
- `guidance_scale`
- `seed`
- `dtype`

### 5.4 建议新增或内部固定的 VividVR 相关字段

对外是否开放应从简，第一版建议尽量少暴露：

- `num_temporal_process_frames`
- `restoration_guidance_scale`

其中：

- 如果当前默认长视频口径已经固定，第一版可以先不暴露，完全走服务端默认值。
- 若后续确有必要，再把它们作为高级可选字段开放。

### 5.5 第一版明确不开放给用户的字段

- `prompt`
- `prompt_file_path`
- `prompt_path`
- `caption_file_path`
- `caption_source`
- `use_live_cogvlm2_caption`
- `cogvlm2_model_path`
- 全部 `mask_*` 字段
- 全部 `bbox_* / feather / paste_back / crop_only` 字段
- `infer_len`
- `overlap`
- `strength`
- `dynamic_cfg*`
- `use_clip`
- `use_repaired_context`
- `vary_seed_by_window`
- `init_latent_mode`
- `mask_downsample_mode`
- `overlap_commit_mode`
- `tail_padding_mode`
- `decode_mode`
- `enable_frame_interpolation*`
- `enable_upscaling*`
- `perf_dump_path`

这些字段要么是 Wan repair 专属，要么会把当前 `VividVR` 服务契约变得过早复杂化。

补充说明：

- `prompt_file_path` 当前并不是被删除语义，而是转为服务端内部固定配置。
- 当前默认 prompt 来源仍应保持与既有基线一致，即现有 demo prompt 文件。

## 6. 第一版服务端内部执行流

建议按下面顺序组织：

1. 请求接入
   - 复用 `/v1/videos/repairs` 或等价异步入口。
   - 完成请求体校验、`task_id` 生成、输入路径归一化。

2. 输入准备
   - 下载或复制 `video_url / video_input_path` 到服务端工作目录。
   - 解析 `num_frames`，建立输出目录和进度文件路径。

3. prompt 准备
   - 服务端直接读取现有 demo 对应 prompt 文件。
   - 当前默认来源保持与既有 `Phase C` 基线一致，即 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`。
   - 该路径作为服务端内部配置使用，不暴露给外部请求。

4. VividVR 请求桥接
   - 构造 `VividVRSamplingParams.from_user_kwargs(...)`
   - 显式传入：
     - `video_input_path`
     - `prompt_file_path=<internal demo prompt path>`
   - 其余默认值继续复用当前 `VividVR` pipeline config 和 sampling params 默认配置。

5. 调度执行
   - 复用 `prepare_request(...) -> Req`
   - 投递给现有 scheduler
   - 输出任务状态、进度和结果文件

6. 结果回传
   - 复用 `VIDEO_STORE`
   - 复用 `/status`、`/progress`、`/content` 结果获取方式
   - 如有需要，继续支持 `callback_url`

## 7. 分阶段实施计划

### Phase S1: 契约冻结

- 明确 `VividVR` 服务是否沿用 `/v1/videos/repairs`，还是新增按模型分发逻辑。
- 冻结第一版对外请求字段。
- 冻结服务端内部 prompt 来源与配置方式。

交付物：

- 最小请求 schema
- 字段白名单 / 黑名单
- 内部 prompt 配置说明

### Phase S2: 路由与请求分发

- 在现有 `video_api` 框架内加入 `VividVR` 模型分支。
- 根据 `model` 或 server 配置，将请求分流到 `VividVR` sampling params 构造逻辑。
- 保持现有 `WanVideoEdit` 路由能力不被破坏。

交付物：

- 请求归一化与模型分发方案
- `VividVR` 服务路由入口

### Phase S3: prompt 内部桥接层

- 把服务请求桥接到当前内部 prompt 文件语义。
- 明确 prompt 文件解析位置、默认值和覆盖策略。
- 保证该桥接层不改变当前 `VividVR` 已验收 prompt 行为。

交付物：

- prompt 内部配置规范
- prompt 注入链路

### Phase S4: 推理链复用接入

- 把内部 prompt 路径注入 `VividVRSamplingParams`
- 复用现有 `VividVRPipeline`
- 打通任务提交、执行、查询、下载全链路

交付物：

- 端到端服务链路跑通

### Phase S5: 回归与验收

- 验证服务化后不会破坏 `Phase C / D / E` 既有基线
- 验证长视频默认口径仍符合当前验收结论
- 验证请求失败、prompt 读取失败、输出失败时的错误状态回传

交付物：

- 服务化回归记录
- 最小验收用例集合

## 8. 需要优先确认的设计点

- 是否保留 `model` 作为外部必填字段，还是在独立服务实例里固定只跑 `VividVR`。
- 当前内部 prompt 是否固定为 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`，还是允许服务端配置覆盖。
- 第一版是否需要保留 `callback_url` 与对象存储上传能力。
- 当前仓库 `video_api` 是在 Wan repair 契约上扩展，还是单独加一条 `VividVR` 分支入口更稳妥。

## 9. 当前建议结论

- 服务框架应复用当前仓库 `sglang` 内已有视频异步服务方式，不重新设计任务系统，也不依赖项目外 `sglang_serve` 代码。
- 对 `VividVR` 来说，请求契约应从 Wan 的 `prompt + mask` 模式切换成“视频输入 + 服务端内部 prompt”模式。
- 当前阶段最稳妥的接法不是引入 caption 模型，而是继续读取现有 demo prompt 文件，并通过 `prompt_file_path` 内部桥接进入现有已验收 pipeline。
- `prompt_file_path` 不应暴露给用户；它应保留为服务端内部默认配置，而不是外部 API 字段。
