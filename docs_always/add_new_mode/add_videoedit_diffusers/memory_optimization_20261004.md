# VideoEdit 内存与通信优化记录（2026-10-04）

本文记录本仓库截至 2026-10-04 未提交工作区的内存与通信优化、实测结果和后续工作。已实现、默认开启、已部署和已证明加速是不同状态。

本轮采用 L40S、544×480×49、40 步和 TeaCache 0.3；与 [原优化方案](./optimizer.md)、[历史 benchmark](./benchmark_results.md) 的素材、81 帧/20 步及统计口径不同，数据单独记录，不直接混算加速比。

## 已实现的优化

| 优化 | 实现内容 | 开关与边界 |
| --- | --- | --- |
| **DiT 阶段缓存与零回写释放** | 支持部分或全部层驻留，避免每次前向重复上传；阶段结束释放 GPU 副本，保留 CPU 原件 | `SGLANG_VIDEOEDIT_RESIDENT_LAYERS`、`RELEASE_AFTER_DENOISE`；40 层全缓存只验收小尺寸，历史全帧 1920×1088×49 曾 OOM |
| **逐层预取与 GPU 缓冲区复用** | 跳过已驻留层，滚动预取后续层；复用已消费缓冲区，用事件保护跨流生命周期，不为下一次前向无条件绕回预取 | `OVERLAP_PREFETCH`；减少分配和等待不等于必然加速 |
| **编码器与 VAE 生命周期** | 文本/图像编码器和 VAE 保留 CPU 原件，按阶段上传，结束后恢复 CPU 引用；清理正常及异常路径、VAE 缓存 | 编码器由 `ENCODER_CPU_MASTER` 控制；VAE 跟随 `vae_cpu_offload`；首次接管 GPU VAE 可能先回迁 CPU |
| **减少中间张量** | 不生成未使用的完整视频 GPU 副本和重复条件拼接；及时解除 masked video、mask、噪声、嵌入、latent 和解码视频的多余引用 | 默认及时清理；`KEEP_STAGE_TENSORS=1` 保留诊断字段 |
| **有界 pinned staging 与分批 D2H** | 权重使用双缓冲 staging，限制锁页内存；解码帧按原 dtype 分批回传，在 CPU 转 FP32/uint8 | `OFFLOAD_PINNED_MB`、`STAGE_PINNED_MB`、`DECODE_TRANSFER_FRAMES`；增加 CPU 中转或传输批次，可能降速；不是逐帧流式 VAE 解码 |
| **自动驻留与异步阶段加载** | 根据可用显存、可复用 allocator 缓存及预留预算选择驻留层数/预取深度；独立流上传，文本编码时可预取图像编码器 | `AUTO_RESIDENCY`、`ACTIVATION_RESERVE_MB`、`WEIGHT_BUDGET_MB`、`ASYNC_STAGE_LOAD`、`PREFETCH_IMAGE_ENCODER`；尚无按输入尺寸/历史激活峰值学习预算 |
| **注意力通信优化** | 等布局 Q/K/V 合并交换；按注意力头分块，尝试重叠 All-to-All 与 attention | `FUSED_QKV`、`USP_CHUNKS`；分块有后端、形状等限制，不兼容时走原路径；单项低延迟收益未确立 |
| **跨 rank 共享 CPU 权重** | 按内容和 dtype 去重连续 CPU 权重，使用 tmpfs 共享映射与 CUDA 注册，直接 H2D；更新前按层分离私有副本 | `SHARED_CPU_DIR`，默认关闭；与有界 DiT staging 互斥；非连续权重保留私有存储，每卡仍需上传完整权重 |

表中省略前缀的环境变量均带 `SGLANG_VIDEOEDIT_`。缓存、额外滚动预取、自动驻留、异步加载、编码器预取、通信融合/分块及共享 CPU 权重均提供显式开关，不能把代码存在理解为默认全部开启。另增加独立双卡启动配置、normal-only 模式、配置校验、容器防误替换选项，以及隔离基准、分阶段显存/PSS 采集和输出一致性检查。

## 双卡小尺寸实测

统一条件：GPU 4/5（L40S），镜像 `sglang-videoedit-dev-v2:l40s`，挂载 SGLang 当前工作区；实际模型输入 **544×480、49 帧**，40 步、seed 42、CFG 5.0、动态 CFG、TeaCache 0.3。每组预热 1 次、正式 3 次，取中位数。完整请求时间取较慢 rank 的 pipeline 时间，包含处理与本地输出；DiT 包含适用的权重加载、计算、通信和释放。模型加载单列，不计入稳态请求。

**第一轮：新增优化组合筛选。** 基线已有四层预取、40 层阶段缓存、结束释放、编码器 CPU 原件、阶段张量清理、VAE 生命周期及分批 D2H；不是相对所有优化前代码的整体 A/B。

| 配置 | 完整请求（秒） | DiT（秒） | DiT 锁页内存 / rank | DiT 活跃显存峰值 / 卡 |
| --- | ---: | ---: | --- | ---: |
| 基线 | 49.453 | 31.342 | 30.09 GiB CPU 原件 | 31.789 GiB |
| 滚动预取＋通信分块＋异步阶段加载＋图像编码器预取＋64 MiB staging＋自动驻留 | 53.881 | 35.663 | 64 MiB staging | 31.789 GiB |

组合使完整请求耗时增加 **8.95%**，没有体现加速。64 MiB 仅为 DiT staging，CPU 权重仍存在；不能理解为整个进程内存仅 64 MiB。尚未拆分各开关贡献。8 次请求的解码帧、latent 哈希及 TeaCache 决策一致，CUDA/NCCL 检查通过。

**第二轮：仅切换共享 CPU 权重。** 两组均关闭上述额外组合，保留较快基线配置；共享组使用独立 tmpfs，直接注册共享页上传。

| 指标 | 私有 pinned 权重 | 共享 pinned 权重 |
| --- | ---: | ---: |
| 完整请求中位数 | 51.037 秒 | 52.056 秒 |
| DiT 中位数 | 32.546 秒 | 32.033 秒 |
| 两 rank 去噪后 PSS 合计中位数 | 112.982 GiB | 63.082 GiB |
| 容器正式请求内存采样峰值 | 114.412 GiB | 64.358 GiB |
| DiT 活跃显存峰值 / 卡 | 31.789 GiB | 31.789 GiB |
| 启动到健康检查 | 138.090 秒 | 190.093 秒 |

两 rank PSS 合计减少 **49.900 GiB（44.17%）**。共享映射共 80 个文件、30.089 GiB，两个 rank 的映射 PSS 合计仍为 30.089 GiB，私有页为 0，证明确实只保存一份物理权重。明确去重约 30.09 GiB；整体 PSS 更大的下降还包含分配路径/分配器保留差异，未进一步拆分。PSS 是按共享比例计数的进程快照，容器峰值则包含页缓存等，不能混用或用 RSS 简单相加替代。

共享组完整请求约慢 **2.00%**、DiT 约快 **1.58%**，三次范围重叠，不能认定稳定加速或完全没有代价。共享组额外采集 smaps 物理共享证据，开销计入完整请求时间，但不计入 DiT 时间。启动增加约 **52 秒**。34 项回归测试通过；8 次完整请求全部成功，无 OOM，16 个 rank latent 记录及全部输出帧哈希一致，TeaCache 决策一致。

共享模式需私有 tmpfs 目录：本机普通磁盘 `MAP_SHARED` 的 CUDA 注册失败，不能改用可能复制页面的 `MAP_PRIVATE` 冒充共享。共享引用与 CUDA 注册随 tensor/view 生命周期保留；写入通过更新接口先分离副本。进程强杀可能留下文件，必须在所有 rank 退出后清理；本次专用 tmpfs 随测试容器销毁。当前默认关闭，未启用到既有业务服务。

历史小尺寸对照中，全层阶段缓存曾使 DiT 完整阶段从约 50.95 降至 32.18 秒；该结果与本轮测试分开记录。历史单卡与当前双卡代码、预热和计时条件不完全一致，不能据此宣称严格的双卡加速比；本轮也不能外推为全帧或长期负载可用。

## 后续优先级

1. **P0：单项与交错复测。** 分别测试 staging 容量、滚动预取、异步加载、编码器预取和通信分块，补齐当前代码单卡对照，明确内存—延迟取舍。
2. **已实现：按输入与历史峰值估算激活余量。** 后续开发增加了尺寸/帧数感知、跨 rank 最小预算、按形状独立历史及全层/部分驻留切换。配置与验证范围见[自动驻留开发与测试](./adaptive_residency_20261004.md)。仍不将固定预留或单一素材测试视为所有输入无 OOM 的保证。
3. **P1：共享方案启动与更新成本。** 分析内容哈希、发布与 CUDA 注册耗时；量化 LoRA/refit 分离后内存回升，扩展多素材与连续请求验证。
4. **P1：生命周期压力测试。** 覆盖多窗口、尺寸切换、取消、超时、异常后下一请求，跟踪 PSS、锁页内存及 GPU allocated/reserved 是否持续增长。
5. **P2：进一步降低 VAE 解码峰值。** 评估分块/流式解码及其时间、接缝和时序质量；已有分批 D2H 不等于已实现该能力。

## 实现与复核入口

实现和测试入口使用仓库相对链接；`.local` 下的原始报告是本机测试产物，通常不随 Git 分发，关键结果已汇总在本文。

- [优化开关与测试说明](../../../scripts/videoedit_benchmark/README.md)
- [组合筛选报告](../../../.local/crop-dual-20261004/report.md)
- [共享 CPU 权重报告](../../../.local/shared-cpu-benchmark-20261004/report.md)
- [共享权重实现](../../../python/sglang/multimodal_gen/runtime/utils/shared_cpu_weights.py)、[逐层卸载与缓存](../../../python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py)

复现共享方案对照需进入 SGLang 仓库，确认所选 GPU 空闲，并使用新的结果目录：

```bash
cd /home/zhouhao6/VideoEdit/sglang
python3 scripts/videoedit_benchmark/run_shared_cpu.py \
  --out .local/shared-cpu-new --gpus 4,5 \
  --image sglang-videoedit-dev-v2:l40s
```

[返回接入文档](./README.md)
