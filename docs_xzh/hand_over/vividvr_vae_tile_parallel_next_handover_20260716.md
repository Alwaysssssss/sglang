# Vivid-VR VAE Tile 并行下一阶段交接

## 1. 文档状态

- 日期：2026-07-16
- 当前状态：方案已确认，尚未实现
- 暂定方案：方案一——在 CogVideoX VAE 内实现等价的空间 tile 并行解码
- 目标：复用现有 SP 进程组并行解码 VAE 空间 tiles，降低 `Decode/Trim` 阶段和端到端推理耗时
- 当前代码基线：分支 `sglang_Vivid`，方案确认时 HEAD 为 `66c5c2dd7`

本阶段不是重做 Vivid-VR 解码链，而是在严格保持现有 CogVideoX VAE tiled decode 语义的前提下，仅改变空间 tiles 的执行位置。

## 2. 必须保护的稳定基线

实现和验收期间必须保护已经完成的 Phase C / D / E 语义：

- 单 clip 路径继续保持 `prompt_embed_shape=226`、VAE tile 尺寸 `240 / 360`、未 padding 的 `reference_video`。
- 长视频继续保持 clip split、timestep 级多 clip 编排、latent merge、clip trim 和 stitch。
- VAE decode 后继续保持 drop first 3 frames、crop padding、AdaIN/reference color fix。
- 单卡、双卡和四卡现有正式配置不能因本任务被隐式修改。
- 双卡或四卡下的 attention backend、SP、CFG 和 connector context 语义不变。
- 单卡路径必须继续作为串行等价基线。

当前正式性能结果位于：

- `docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`
- `Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716`

当前 `Decode/Trim` 同步计时基线如下，后续只能用相同机器、相同输入、相同配置且仅切换 VAE 并行开关的 A/B 实验判断收益：

| 方案 | 并行拓扑 | Decode/Trim（秒） |
| --- | --- | ---: |
| R0 | 单卡 SDPA eager | 111.985 |
| R1 | 单卡 FA eager | 99.267 |
| R2 | 单卡 FA compile | 100.704 |
| R3 | SP=2 | 101.167 |
| R4 | SP=4 | 107.193 |
| R5 | CFG=2 × SP=2 | 110.560 |
| R6 | 单卡综合方案 | 98.781 |
| R99 | 双卡综合方案 | 100.274 |
| R100 | 四卡综合方案 | 101.786 |

不同 R 方案之间还包含 attention、compile、SP/CFG 等变量，不能直接用上表推导 VAE 并行的因果收益。

## 3. 当前实现事实与能力缺口

### 3.1 当前 VAE 路径

- Vivid-VR 使用 `CogVideoXVAEConfig` 和 `AutoencoderKLCogVideoX`。
- 默认开启 VAE tiling，空间 tile 最小尺寸为高 `240`、宽 `360`。
- `VividVRDecodingStage` 将 latent 转成 `[B, C, T, H, W]` 后调用 `self.vae.decode(...)`。
- 长视频由 `VividVRMultiClipDecodeTrimStage` 逐 clip 解码，再执行 trim；之后由 stitch/postprocess stage 完成拼接和颜色修复。
- 当前 `vividvr.py` 中 `vae_sp=False`，正式默认配置没有启用 VAE 并行。

### 3.2 CogVideoX tiled decode 的关键语义

当前 diffusers CogVideoX VAE 的空间 tiled decode 对每个 tile 执行完整时序解码：

1. latent 按空间区域切成重叠 tiles；
2. 每个空间 tile 内以固定 temporal batch 解码；
3. tile 内的 temporal convolution cache 在时间批次之间持续传递；
4. 所有 tile 完成后，按原顺序执行 vertical/horizontal blend；
5. 按原边界裁剪并恢复完整输出。

时序 cache、tile 重叠、blend 和 crop 共同决定输出语义，不能为了并行而替换成近似实现。

### 3.3 现有通用并行 VAE 不能直接复用

`runtime/models/vaes/common.py` 中已有 `ParallelTiledVAE.parallel_tiled_decode`，但第一阶段不能直接让 CogVideoX 继承或调用该逻辑：

- 它同时拆分时间和空间 tiles，而本方案只并行空间 tiles。
- CogVideoX 依赖专用的 causal temporal conv cache 和 temporal batch 处理。
- 其中部分 collective 未显式限定到当前 SP group；在 CFG × SP 拓扑下直接使用 WORLD group 有通信错误或死锁风险。
- 当前 CogVideoX runtime 只是复制了部分 tiling 配置字段，并没有真正消费通用并行 tiled decode 路径。

因此，现有通用实现只作为任务划分和通信方式的参考，不作为方案一的直接替换实现。

## 4. 已确认的方案一

### 4.1 范围

第一阶段仅实现以下能力：

- 对 CogVideoX VAE 的空间 tiles 做数据并行式分工。
- 复用当前 SP ranks，不引入专用 VAE GPU 或额外服务进程。
- SP=2 时由两个 SP rank 分担 tiles；SP=4 时由四个 SP rank 分担 tiles。
- CFG=2 × SP=2 时，每个 SP subgroup 内执行二路 VAE tile 并行，不跨 CFG group 通信。
- 每个 rank 最终获得完整 decoded clip，后续 trim、stitch 和 postprocess 保持不变。
- `vae_sp` 未开启或 SP world size 为 1 时，调用原串行路径。

第一阶段明确不做：

- temporal tile 并行；
- temporal clip 并行；
- 跨 CFG group 的四路 VAE 并行；
- 专用 VAE ranks；
- 修改默认正式配置；
- 顺带重构整个通用 VAE 并行框架。

### 4.2 数据流

```text
完整 clip latent
    ↓
所有 SP rank 生成完全一致的空间 tile plan
    ↓
按确定性规则把 tile index 分配给当前 SP group 的 ranks
    ↓
每个 rank 仅解码自己的 tiles
（每个 tile 内仍执行原 temporal loop + conv cache）
    ↓
在当前 SP group 内收集 tile tensor 和必要元数据
    ↓
按全局 tile index 恢复二维 tile 网格
    ↓
复用原 vertical/horizontal blend 和 crop 顺序
    ↓
每个 rank 得到完整 decoded clip
    ↓
沿用现有 Trim → Stitch/Postprocess
```

tile 分配规则应当确定、可测试，并优先做到负载均衡。第一版可以使用按全局 tile index 的 round-robin 分配；如果边界 tile 尺寸或耗时差异明显，再在不改变输出顺序的前提下改为基于 tile 形状的静态均衡。

### 4.3 通信边界

- 所有 collective 必须显式使用 `get_sp_group()` 对应的 device group，禁止默认使用 WORLD group。
- CFG=2 × SP=2 下，两组 SP collective 必须完全隔离。
- 正式实现前必须确认进入 VAE decode 时每个 SP rank 是否已经持有完整 latent；若不是，必须先在同一 SP group 内显式恢复完整 latent，不能假设输入已复制。
- tile tensor 尺寸不同时，应使用确定性的 shape metadata 和张量 collective，优先避免在性能主路径中使用 `all_gather_object`。
- gather 后必须按全局 tile index 重排，不能依赖不同 rank 的返回顺序隐式等于图像空间顺序。

## 5. 方案比较记录

| 方案 | 说明 | 结论 |
| --- | --- | --- |
| 方案一 | 在 CogVideoX VAE 内镜像现有 tiled decode，只并行空间 tile 的执行 | 已选；语义边界清楚，最容易与现有串行结果做严格对齐 |
| 方案二 | 先扩展通用 `ParallelTiledVAE`，再让 CogVideoX 接入 | 暂缓；复用性更好，但第一阶段重构面过大，且容易改变 CogVideoX 时序 cache 语义 |
| 方案三 | 在 VividVR decode stage 外部切 latent tile，再分别调用 VAE | 不采用；stage 难以正确复刻 VAE 内部 overlap、cache、blend 和 crop，语义泄漏严重 |

## 6. 建议代码落点

### 6.1 核心实现

主要修改位置：

- `python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`

建议将以下职责保持在 CogVideoX VAE 内：

- 构造与串行实现一致的 tile plan；
- 根据 SP rank 选择本地 tile；
- 使用原逻辑解码单个 tile，并保持 temporal conv cache；
- 在 SP group 内 gather；
- 按全局 index 恢复 tile grid；
- 复用或等价实现原 blend/crop。

如需抽取共用的数据结构或纯函数，可以对 `runtime/models/vaes/common.py` 做小范围补充，但不要在第一阶段改变其他 VAE 的运行时行为。

### 6.2 开关传递

可能涉及：

- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`

要求：

- `--vae-sp` 必须能明确传递到 CogVideoX VAE runtime。
- `vae_sp=False` 的默认值暂时不变。
- stage 只负责开关、计时和 debug 信息，不承载 tile 切分或 merge 细节。
- 正式测试不得静默 fallback；请求值和实际生效值必须同时记录。

### 6.3 指标与可观测性

每次推理的指标 JSON 建议至少增加或确认以下字段：

- `vae_sp_requested`
- `vae_sp_effective`
- `vae_parallel_group`，第一阶段固定为 `sp`
- `vae_parallel_world_size`
- `vae_total_tiles`
- `vae_local_tiles_per_rank`
- `vae_tile_decode_seconds`
- `vae_gather_seconds`
- `vae_merge_seconds`
- `vae_parallel_fallback_reason`

如字段集合最终发生变化，必须同步更新指标格式说明和相关文档。

## 7. 实现顺序

1. 固化当前串行 tile plan、tile decode、blend/crop 的行为测试。
2. 将 tile 规划、单 tile decode 和 merge 拆成可单测的内部函数，但保持串行输出不变。
3. 增加纯逻辑 tile 分配测试，覆盖 tile 数不能整除 SP world size 的情况。
4. 增加两 rank 分布式小测试，验证 SP group、索引恢复、不同尺寸 tensor 和异常处理。
5. 接入真实 CogVideoX VAE 的 SP=2 路径，并与串行 decoded tensor 对齐。
6. 扩展到 SP=4 和 CFG=2 × SP=2，重点验证 subgroup 隔离和 collective 次序。
7. 增加指标字段和同步计时。
8. 依次完成单 clip、长视频和正式性能 A/B 验收。
9. 只有端到端质量与性能均通过后，才讨论是否修改默认正式配置。

## 8. 验收矩阵

| 编号 | 拓扑 | VAE tile 并行 | 用途 |
| --- | --- | --- | --- |
| C0 | 单卡 | 关闭 | 串行语义控制组 |
| P2-0 | SP=2 | 关闭 | 双卡同拓扑控制组 |
| P2-1 | SP=2 | 开启 | 双卡 VAE 并行实验组 |
| P4-0 | SP=4 | 关闭 | 四卡纯 SP 控制组 |
| P4-1 | SP=4 | 开启 | 四卡纯 SP VAE 并行实验组 |
| C4-0 | CFG=2 × SP=2 | 关闭 | 四卡正式拓扑控制组 |
| C4-1 | CFG=2 × SP=2 | 开启，组内并行度 2 | 四卡正式拓扑实验组 |

每一组都必须只切换 `vae_sp`，其余模型权重、输入、seed、推理步数、attention backend、compile、connector context 和服务生命周期完全一致。

为节省开发期时间，可以先使用 1 step 做通信和流程 smoke test；质量与正式性能结论必须使用当前 Phase E 的 `130f / 20 step` 长视频口径。所有长时间推理验收必须在命名清楚的 tmux session 中运行，并保存日志、指标 JSON 和结果视频。

## 9. 验收标准

### 9.1 正确性

- 单卡关闭 VAE 并行时输出和当前串行基线一致。
- SP=2、SP=4、CFG=2 × SP=2 均无 deadlock、collective mismatch 或 rank 间形状错误。
- decoded tensor 的 shape、帧数和裁剪范围严格一致。
- 先对同一 latent 的串行与并行 decoded tensor 做数值误差比较，再做端到端视频指标和人工检查。
- 不出现 tile 接缝、闪烁、颜色漂移或 trim/stitch 边界变化。
- Phase C / D / E 现有 debug contract 与服务接口不回归。

如果并行 collective 引入非 bitwise 差异，必须记录最大误差、平均误差和端到端质量指标，不能仅凭肉眼宣称一致。

### 9.2 性能

- 计时前后执行必要的 CUDA synchronization，避免把异步工作记到后续 stage。
- 以同日、同机、同拓扑的关闭组为基线，报告至少三轮的中位数。
- 必须同时报告 `Decode/Trim`、纯模型推理用时、端到端总用时、峰值显存和 GPU·秒。
- 单独报告 tile decode、gather 和 merge，使通信开销可定位。
- `Decode/Trim` 应获得稳定正收益，且模型推理和端到端总用时不能回归。
- 不能因为局部 tile kernel 更快就直接晋升默认配置；正式判断以端到端验收为准。

## 10. 主要风险与防护

| 风险 | 防护措施 |
| --- | --- |
| temporal conv cache 被重置或跨 tile 误共享 | 单 tile 内完整复用串行 temporal loop；为 cache 生命周期增加单测 |
| CFG × SP 使用错误通信组导致死锁 | 所有 collective 显式绑定当前 SP subgroup；增加 CFG=2 × SP=2 专项测试 |
| 边界 tile 尺寸不同导致 gather 失败 | 先交换确定性 shape metadata，再执行 tensor gather 和索引恢复 |
| tile 数不能整除 rank 数导致负载不均 | 使用确定性分配并记录每 rank tile 数和耗时 |
| gather 完整 decoded tiles 造成显存峰值上升 | 记录 gather 前后峰值显存；必要时分批 gather，但不改变 merge 顺序 |
| diffusers 版本变化使复制逻辑漂移 | 对照当前依赖版本并用串行行为测试锁住 tile plan 与 blend/crop |
| stage 计时被 CUDA 异步执行污染 | 正式 profiler 使用同步计时，控制组和实验组采用相同计时方式 |
| 开关请求成功但运行时未生效 | JSON 同时记录 requested/effective、并行度、tile 分配和 fallback reason |

## 11. 完成定义

下一阶段只有同时满足以下条件，才可认为 VAE tile 并行完成：

- 方案一的实现和单元测试已落地；
- 单卡串行路径无行为变化；
- SP=2、SP=4、CFG=2 × SP=2 的通信与质量验证通过；
- Phase C 单 clip 和 Phase D/E 长视频回归通过；
- 正式 A/B 结果证明 `Decode/Trim` 与端到端用时获得稳定收益；
- 指标 JSON、结果视频和日志保存到统一验收目录；
- 如默认参数或服务语义发生变化，相关运行文档和 `AGENTS.md` 已同步更新；
- 阶段改动已经独立提交并推送。

在上述验收完成前，`vae_sp` 保持实验开关，不能替换当前正式默认配置。

## 12. 下一轮开始前的快速检查

- 阅读本交接文档及 `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`。
- 检查最新 `git status`，保护用户未提交改动。
- 确认当前 diffusers `AutoencoderKLCogVideoX.tiled_decode` 的实际源码和版本。
- 确认 decode 边界上 latent 在各 SP rank 的分布状态。
- 先写串行行为锁定测试，再开始并行实现。
- 所有真实推理和长时间验收均在 tmux 中执行。
