# Vivid-VR Caption Bridge 验收完成与 Caption 双卡服务加速下一步交接

日期：`2026-06-23 UTC`

## 1. 本轮结论

当前 `Vivid-VR` 在 `sglang` 中的主推理集成已经具备可交付的服务链路，且本轮已经把此前未完全跑通的 `caption bridge` 真正补齐到端到端验收通过。

本轮可以确认的状态是：

- `Phase C` 单 clip 基线未回退。
- `Phase D` 长视频 `clip split / latent merge / stitch` 语义基线未回退。
- `Phase E` 默认主推理配置仍维持既有结论：
  - 单卡：`single_gpu_fa_compile`
  - 双卡：`dual_gpu_fa_eager_compile`
- `serve` 路径现在已经支持“请求只给视频，服务端自动补 `caption sidecar`，再继续走 `sglang` 原生 Vivid-VR 推理链”。
- `FlowCut` 兼容入口已经在不显式传 `caption_file_path` 的前提下完成一次真实长视频端到端验收。

这次验收通过并不等于“整条链路严格单 GPU 占用”。本轮实际运行拓扑是：

- `GPU0`：`sglang serve` 主推理
- `GPU1`：原版环境中的 `caption sidecar`

也就是说，当前已经证明“服务自动补 caption 并完成推理”这条链路是可用的，但 `caption` 本身还没有进入下一阶段的双卡加速收口。

## 2. 当前已完成状态

### 2.1 Caption Bridge 契约已经收口

当前 `caption bridge` 的关键契约已经明确：

- 主服务负责生成 `manifest.json`
- sidecar 负责按 `temporal clip` 顺序生成 `caption.txt`
- sidecar 输出固定为“一行一个 `temporal clip caption`”
- `expected_caption_count` 表示 `temporal clip` 数，不再表示 `spatial tile` 总数
- `sglang` 消费端按“每个 `temporal clip` 消费一条 caption”运行，再把该 caption 扩展到当前 clip 的所有 tile prompt

这解决了此前最核心的契约错位问题：旧失败样本里 sidecar 产物按 tile 维度输出，导致消费端出现 `consumed 2, available 308`。当前这条问题已经消失。

### 2.2 端到端验收已经真实通过

本轮通过的是真实 `tmux` 重型验收，而不是仅靠单元测试推断。

已确认的结果：

- `caption bridge` 在线生成的 `manifest` 为 `expected_caption_count = 2`
- 对应 sidecar 文本为 `2` 行
- `FlowCut` 请求最终状态为 `completed`
- callback 最终状态为 `succeeded`
- 输出视频已成功落盘并通过 `ffprobe`

当前通过的输出视频：

- `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/flowcut-caption-bridge-fix-20260623T013528Z_0.mp4`

当前对应的服务侧指标文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/flowcut-caption-bridge-fix-20260623T013528Z.json`

当前对应的质量对比文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/flowcut-caption-bridge-fix-20260623T013528Z_framewise_ssim.json`
- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/flowcut-caption-bridge-fix-20260623T013528Z_vs_single_gpu_fa_compile.json`

### 2.3 质量已经确认与既有单卡基线一致

本轮输出与正式 benchmark 参考口径对比结果为：

- `ssim_mean = 0.9872759400`
- `ssim_min = 0.9831915677`
- `pass_compare = true`

历史单卡正式基线 `single_gpu_fa_compile` 对应结果为：

- `ssim_mean = 0.9870814328`
- `ssim_min = 0.9813359022`
- `pass_compare = true`

同时，本轮输出与历史单卡输出视频直接逐帧对比结果为：

- `ssim_mean = 0.9880320431`
- `pass_compare = true`

结论是：当前 `caption bridge` 接线没有带来可观察到的质量回退，可以视为与既有单卡主推理质量保持一致。

## 3. 当前运行形态

现在的请求主链已经明确如下：

1. 客户端调用 `/v1/videos/repairs` 或 `/v1/videos/repairs/flowcut`
2. 服务端解析 `video_input_path` 或下载 `video_url`
3. 如果请求未显式传 `caption_file_path`，主服务先生成 `manifest.json`
4. 主服务请求本机 `caption sidecar`
5. sidecar 在原版 `/home/zhiheng/Vivid-VR/.venv` 环境中逐个 `temporal clip` 生成 caption，并写成 `caption.txt`
6. 主服务把 `caption_file_path` 回填到 `VividVRSamplingParams`
7. `sglang` 主推理链读取 `caption.txt`
8. 长视频链路做 `temporal window planning -> clip preparation -> denoising -> latent merge -> decode -> trim -> stitch`
9. 服务返回结果路径并完成 callback

这里有两个当前必须记住的边界：

- `caption` 仍然只在原版环境里生成
- 主推理仍然只跑 `sglang` 原生 Vivid-VR 运行时，不依赖原版仓库推理代码

## 4. 关键数据与当前瓶颈

### 4.1 当前主推理耗时

这次成功样本的 `server perf dump` 结果为：

- `total_duration_ms = 1018875.95`

主要阶段耗时：

- `VividVRLongClipPreparationStage = 70872.26 ms`
- `VividVRMultiClipDenoisingStage = 844836.13 ms`
- `VividVRMultiClipDecodeTrimStage = 99304.42 ms`

这说明当前主耗时仍然在长视频 denoising，本轮 `caption bridge` 并没有改变主推理主瓶颈。

### 4.2 当前 Caption Bridge 的真实阻塞成本

虽然主推理主瓶颈仍在 denoising，但 `caption bridge` 已经成为服务链路中明确可见的前置阻塞项。

本轮成功样本里：

- `manifest.json` 生成时间与最终 `caption.txt` 写完时间相差约 `63.44 s`
- 当前 `130f` 样本对应 `2` 个 `temporal clip`
- 粗略折算约为 `31.7 s / clip`

这部分时间发生在主推理真正进入 `pipeline.forward(...)` 之前，因此它会直接拉长客户端首个有效提交到进入主推理的等待时间。

### 4.3 当前 Caption 模型规模

当前 sidecar 默认加载的还是原版 `CogVLM2` caption 模型，checkpoint 为：

- `/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption`

按命名口径，它的语言底座是 `Meta-Llama-3.1-8B-Instruct`。按本地 checkpoint 实际参数量统计，整模型规模约为 `12.5B`。

这意味着后续要做双卡服务加速时，不能把它当成一个轻量模块来处理；无论是双 worker、模型并行，还是远端独立服务，都需要认真看 GPU 常驻成本和资源抢占问题。

## 5. 当前仍需注意的约束

### 5.1 当前 sidecar 仍然是串行逐 clip 生成

sidecar 现在的处理方式仍是：

- 先整段读取视频
- 再按 `manifest.clips` 顺序逐个 clip 生成 caption
- 最后一次性写出 `caption.txt`

这意味着：

- 当前没有 clip 级并行
- 当前没有多 worker 聚合
- 当前没有双卡模型并行
- 当前没有 streaming 输出中间 caption

### 5.2 不能破坏现有主推理基线

后续任何 `caption` 加速都必须遵守：

- 不修改 `/home/zhiheng/sglang/.venv` 的主推理依赖来迁就原版 caption
- 不把原版 Vivid-VR 推理运行时代码重新带回主服务
- 不破坏 `Phase C / D / E` 已验收质量基线
- 不改变“每个 `temporal clip` 一条 caption”的 sidecar 契约

### 5.3 Caption 双卡部署与主推理双卡部署存在资源耦合

下一步如果考虑“caption 模型双卡服务加速”，必须先明确部署前提。

当前有两类完全不同的场景：

1. `caption sidecar` 与主推理不共享同一组 GPU  
   这是最简单的部署形态，适合独立 caption 机器或额外 GPU 资源。

2. `caption sidecar` 与 `dual_gpu_fa_eager_compile` 主推理共享同一台双卡机器  
   这时必须处理 GPU 常驻冲突问题。因为如果 caption 服务长期常驻占满两张卡，主推理双卡就没有可用资源。

因此，“caption 双卡加速”不是一个纯实现问题，它同时是部署拓扑问题。

## 6. 下一步任务：Caption 模型双卡服务加速

### 6.1 推荐目标

下一轮不建议先改主推理，而是把目标收敛为：

- 在不改变现有 `caption bridge` 输入输出契约的前提下，降低 `caption sidecar` 的服务耗时
- 尽量把 `63.44 s` 的前置阻塞显著压缩
- 保持最终 `caption` 行数、顺序和消费语义不变

### 6.2 推荐优先路线

当前最值得优先评估的路线，不是立刻做复杂模型并行，而是先看“按 clip 并行”的双 worker 方案。

推荐顺序：

1. 先把当前 sidecar 的耗时拆清楚  
   至少拆出：
   - 视频读取耗时
   - 模型前处理耗时
   - 单 clip caption 推理耗时
   - 输出写文件耗时

2. 优先评估“每张 GPU 一个 caption worker”的双 worker 服务  
   由调度层把不同 `temporal clip` 分发给两个 worker，最后按 `clip_index` 顺序聚合写回 `caption.txt`。

3. 只有在双 worker 方案不能满足收益预期时，再评估单模型双卡并行  
   例如 `TP` 或更重的模型并行。

当前推荐双 worker 优先于双卡模型并行，原因是：

- 现在的 sidecar 契约天然是“每个 clip 独立生成一条 caption”
- clip 之间没有推理依赖，最适合做任务级并行
- 当前 `130f` 样本恰好是 `2` 个 clip，和双卡 worker 拓扑天然匹配
- 这种方案对现有主服务契约侵入最小

### 6.3 需要先做出的部署决策

下一轮开始前，建议先明确下面这个问题：

`caption` 双卡服务到底是：

- A. 独立于主推理 GPU 的长期常驻服务
- B. 与主推理共享同一台双卡机器，但只在前置 caption 阶段短时间占用 GPU
- C. 单独远端部署的 caption 服务，由主服务通过 HTTP 调用

如果目标是 B，那么要特别注意：

- 长期常驻双卡 sidecar 与双卡主推理会直接冲突
- 需要额外设计 GPU 释放、进程编排或服务拆分策略

如果目标是 C，那么要额外处理：

- 视频访问路径共享
- sidecar 产物回传
- 网络重试
- 超时与错误语义

### 6.4 下一轮验收建议

下一轮如果进入 caption 双卡加速，建议把验收拆成两层：

1. `caption sidecar` 独立 benchmark  
   目标是明确：
   - 单卡耗时
   - 双 worker 耗时
   - 稳态吞吐
   - GPU 占用

2. 重型端到端回归  
   继续要求在 `tmux` 中运行，并至少确认：
   - `caption.txt` 行数与 `temporal clip` 数一致
   - 顺序与 `clip_index` 一致
   - `FlowCut` callback 成功
   - 最终视频质量不低于当前 `caption bridge` 已验收基线

## 7. 重要产物位置

本轮最重要的交接产物包括：

- 交接文档：
  - `/home/zhiheng/sglang/docs_xzh/hand_over/vividvr_caption_sidecar_service_handover_20260622.md`
  - `/home/zhiheng/sglang/docs_xzh/hand_over/vividvr_service_external_access_and_caption_next_handover_20260622.md`
  - `/home/zhiheng/sglang/docs_xzh/hand_over/phase_e_default_configs_and_serve_followups_handover_20260622.md`

- 本轮成功验收日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_caption_sidecar_fix_gpu1_20260623T013321Z.log`
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_serve_single_bridge_fix_20260623T013352Z.log`
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/flowcut-caption-bridge-fix-20260623T013528Z.log`
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/flowcut-caption-bridge-fix-20260623T013528Z_callback.jsonl`

- 本轮成功验收 sidecar 产物：
  - `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/flowcut-caption-bridge-fix-20260623T013528Z.manifest.json`
  - `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/flowcut-caption-bridge-fix-20260623T013528Z.txt`

- 本轮成功验收结果：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/flowcut-caption-bridge-fix-20260623T013528Z_0.mp4`
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/flowcut-caption-bridge-fix-20260623T013528Z.json`
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/flowcut-caption-bridge-fix-20260623T013528Z_framewise_ssim.json`
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/flowcut-caption-bridge-fix-20260623T013528Z_vs_single_gpu_fa_compile.json`

## 8. 一句话交接

当前 `Vivid-VR` 服务已经完成“请求只给视频，服务端自动补 `caption sidecar` 并完成长视频推理”的端到端验收，质量与既有单卡基线一致；下一步不该再纠结桥接能不能跑，而应该把重点转到 `caption` 服务本身的双卡加速和部署拓扑收口上。
