# VideoEdit UMT5 文本编码优化方案

状态：设计方案，尚未实现。

## 目标与范围

整条视频只进行一次文本条件准备：正提示词编码一次；需要 CFG 时，负提示词也编码一次。结果由同一请求中的所有窗口复用，包括 long / short 两个 pass。

多卡执行同一个请求时，由其中两张卡分别编码正、负提示词，再将结果同步到参与该请求的所有 rank。该优化独立于 DiT 的 TP / SP，不启用 DiT CFG 并行，也不改变窗口生成顺序。

“一次”指每个实际使用的提示词分支编码一次，不是将两个提示词合成一次模型调用。缓存限定于单个视频请求，不跨请求共享。

## 当前实现

- `VideoEditTextEncodingStage.forward()` 在每个窗口内依次编码正、负提示词，各 rank 重复执行完整计算。
- `WanVideoEditSamplingParams.reset_window_runtime()` 会清除 `runtime_prompt_embeds`、`runtime_negative_prompt_embeds` 和 `runtime_do_cfg`。
- `_forward_streaming()` 的窗口回调会重置窗口状态，再执行包含文本编码的全部 stages。
- 文本模型是原生 Hugging Face `UMT5EncoderModel`，各 rank 使用完整模型。
- 开启 `text_encoder_cpu_offload` 时，每次进入文本编码阶段都会将模型搬入 GPU，结束后搬回 CPU。

因此，设一条视频有 W 个窗口、R 个 rank，且启用 CFG，当前整体约执行 `2 × W × R` 次文本模型前向。优化后整个请求共执行 2 次；无 CFG 时执行 1 次。

相关代码：

- [文本编码阶段](../../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py)
- [窗口状态重置](../../../python/sglang/multimodal_gen/configs/sample/videoedit_wan.py)
- [视频请求与窗口执行入口](../../../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py)

## 执行策略

这里的 rank 0、rank 1 均指“同一请求执行组内”的编号，不一定是全局进程编号。分组必须覆盖后续共同执行 DiT 的所有参与者，不能把处理其他视频的进程加入通信。

| 条件 | 编码分工 | 结果同步 |
|---|---|---|
| 单卡，启用 CFG | rank 0 顺序编码正、负提示词 | 无需通信 |
| 两卡及以上，启用 CFG | rank 0 编码正提示词，rank 1 同时编码负提示词 | 正、负 embedding 分别广播给整个请求组 |
| 单卡，不启用 CFG | rank 0 只编码正提示词 | 无需通信 |
| 两卡及以上，不启用 CFG | rank 0 只编码正提示词，其他 rank 不编码 | 正 embedding 广播给整个请求组 |

多于两张卡时，其他 rank 只接收 embedding，不重复编码。每个 rank 最终持有完整的正 embedding，以及需要时的完整负 embedding，继续使用现有 DiT TP / SP 路径。

是否准备负 embedding 沿用当前语义：`guidance_scale > 1.0`。动态 CFG 即使只在部分步骤使用负分支，也在请求首次准备时编码并缓存。DMD 的 `guidance_scale=1.0` 路径只编码正提示词。

```text
请求开始：建立空的请求级文本条件缓存
                       ↓
首窗口文本阶段：缓存未就绪
        ┌──────────────┴──────────────┐
rank 0：UMT5(正提示词)       rank 1：UMT5(负提示词)
        └──────────────┬──────────────┘
             同步状态、广播两份 embedding
                       ↓
             各 rank 保存请求级缓存
                       ↓
        首窗口及后续窗口直接绑定缓存供 DiT 使用
                       ↓
          请求结束、失败或取消：释放缓存引用
```

## 缓存与生命周期

建议新增请求级文本条件对象，包含 `ready`、`prompt_embeds`、`negative_prompt_embeds`、`do_cfg`。对象挂在请求运行时状态上，不放到 pipeline 或 stage 的共享成员中，避免请求间串用。

1. 每次顶层视频请求开始时初始化空缓存。即使复用参数对象，也不能沿用上一次请求的结果。
2. 首个有效窗口进入文本阶段时准备缓存。这样可以保持现有输入验证和 stage 顺序。
3. 正、负分支编码及同步全部成功后才标记 `ready=True`，不允许使用部分完成的缓存。
4. `reset_window_runtime()` 可以继续清空窗口字段，但不得清除请求级缓存；文本 stage 命中缓存时重新绑定三个现有窗口字段。
5. long / short pass 切换也必须复用同一个缓存，不能在 pass 边界重建。
6. 请求结束、失败和取消统一清理请求级及窗口级 embedding 引用，避免占用延续到下一请求。

请求内固定 prompt、negative prompt、模型权重、tokenizer 配置、输出 dtype 和 CFG 配置；若未来支持请求中途修改这些条件，必须显式失效并重建缓存。embedding 应当作为只读输入，不允许后续 stage 原地修改。

## 通信与数值要求

- 保留现有 `TextEncodingStage.encode_text()` 调用路径，包括正提示词 `prompt or " "`、负提示词 `negative_prompt or ""`、tokenizer、截断、padding、attention mask、autocast 和 dtype 行为。
- 确保各编码 rank 使用同一模型权重并处于推理模式；不在本次优化中替换 UMT5 实现、量化或调整精度。
- 先分支并行编码，再按所有 rank 一致的顺序执行通信：编码状态同步 → 正 embedding 广播 → 必要时负 embedding 广播。
- 接收方先获得 shape / dtype 元数据，再在本地设备分配接收 tensor。广播使用对应请求组和正确的源 rank 映射，不广播整个请求对象。
- 所有 rank 必须采用一致的 CFG 分支判断和缓存状态，避免某些 rank 跳过 collective。
- 可捕获的编码异常应先同步失败状态，使整个请求失败；进程退出或通信失效交给分布式超时与 worker 故障处理。不能让部分 rank 回退重算、其他 rank 继续等待广播。

不能仅把 stage 改为 `MAIN_RANK_ONLY`：当前 executor 的该分支只执行主 rank 并做 barrier，不会自动同步 embedding，而且不能表达双 rank 分工。

## 模型驻留与 offload

第一版保留现有模型加载方式，先完成计算分工和请求缓存，降低接入风险。编码阶段仅负责本分支的 rank 将模型搬入 GPU；缓存命中时直接返回，不再执行模型搬运。

启用文本 CPU offload 时，编码 rank 按现有策略在首次编码结束后卸载模型；两份 embedding 留在各卡供后续窗口使用。异常路径仍需执行清理。

未启用 offload 时，保留用户配置的模型驻留行为。本方案第一版不承诺减少模型权重显存：双卡编码仍需要两份完整 UMT5，已有加载流程也可能让非编码 rank 持有副本。后续可单独优化非编码 rank 的模型加载和驻留。

## 实施步骤

1. 新增请求级缓存，接入顶层请求初始化、窗口绑定和最终清理；先验证单卡多窗口仅编码一次。
2. 在文本 stage 中增加请求组内分工、状态同步和 tensor 广播，保留单卡及无 CFG 路径。
3. 检查 offload，使缓存命中不触发模型迁移；覆盖 long / short pass 的缓存复用。
4. 增加编码次数、首次文本准备耗时、缓存命中次数和通信耗时观测，进行正确性及性能验收。

## 验证与验收

### 正确性

- 单卡和双卡分别测试 CFG 开／关、多窗口、单窗口，以及同时包含 long / short pass 的视频。
- 全请求合计 UMT5 前向次数：启用 CFG 为 2，不启用 CFG 为 1；第二个窗口起不再执行文本模型前向。
- 多卡启用 CFG 时：rank 0 只编码正提示词，rank 1 只编码负提示词，其他 rank 不编码；各 rank 接收到的同一分支 embedding 一致。
- 连续执行不同提示词的请求，确认缓存不会串用；覆盖失败、取消和随后新请求。
- 验证空提示词、空负提示词、动态 CFG 和 CPU offload。
- 对照优化前的 embedding、DiT 输出及最终视频。固定模型、输入和随机种子，使用项目既有数值容差；跨硬件或内核存在差异时记录误差，不预先承诺逐位相同。

### 性能

分别记录冷启动／预热后的首次文本准备耗时、后续窗口文本阶段耗时、整条视频耗时、峰值 GPU 显存，以及 CPU↔GPU 模型搬运次数。比较时固定视频、窗口数量、步数、模型、offload 和 DiT 并行配置。

设正负编码耗时为 `Tpos`、`Tneg`，结果通信与状态同步耗时为 `Tcomm`，忽略其他开销：

- 当前文本关键路径约为 `W × (Tpos + Tneg)`，不能再乘 rank 数，因为各 rank 同时重复计算。
- 双卡优化后的文本关键路径约为 `max(Tpos, Tneg) + Tcomm`，再加后续窗口极小的缓存绑定开销。
- 无 CFG 时约为一次 `Tpos + Tcomm`，收益主要来自跨窗口复用。

多窗口越多，请求级缓存的收益越明显；单窗口收益取决于双分支耗时及通信成本。最终视频加速比例必须实测，不能把文本编码加速直接等同于整条视频加速。
