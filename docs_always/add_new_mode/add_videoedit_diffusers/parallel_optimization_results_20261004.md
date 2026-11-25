# VideoEdit 双卡阶段并行开发与实测（2026-10-04）

本轮实现了[并行方案](./parallel_optimization_requirements_20261004.md)中的请求级文本复用、文本/CLIP owner 分工和 Wan VAE 空间 tile 并行，并用双 L40S 完成小裁剪与全帧真实模型验证。线上 GPU 7 对比采用完全相同的请求模板，未重启或修改现有线上容器。

## 实现

- 服务在最终模型放置和 DiT 分层卸载完成后、报告 ready 前，由文本 owner 编码一次默认负向提示词。最终嵌入保存在各 rank 的 CPU 缓存中。
- 正向提示词每请求只编码一次，请求内所有窗口复用；负向文本只在 CFG 实际消费且有效文本变化时更新。窗口清理仅释放设备副本，请求结束再释放请求级 CPU 引用。
- 多卡请求中 rank 0 运行文本编码，rank 1 运行 CLIP。两个 owner 各自在本进程执行，随后用现有 SP 通信组显式广播条件 tensor；任一 owner 失败时先汇总错误，其他 rank 不会停在后续 tensor collective。
- VAE encode/decode 按 Diffusers Wan VAE 的既有空间 tile、重叠和融合顺序执行。每个 tile 保留完整时间序列，按 tile 序号轮转分配给 rank；结果经 NCCL 广播后，各 rank 以相同顺序融合，因此后续 DiT、carry 和 bridge 仍获得完整结果。
- 新增服务准备生命周期钩子，使启动预计算发生在 layerwise offload 建立之后，避免 30.54 GiB DiT 与 UMT5 同时完整驻留导致 OOM。
- GPU 4/5 专用配置已经启用双 owner 条件准备和 VAE encode/decode 二卡空间并行。通用启动脚本会校验 owner、VAE 模式和并行度范围。

核心入口：

- [请求状态与清理](../../../python/sglang/multimodal_gen/configs/sample/videoedit_wan.py)
- [启动生命周期](../../../python/sglang/multimodal_gen/runtime/managers/gpu_worker.py)
- [VideoEdit pipeline 编排](../../../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py)
- [条件与 VAE stages](../../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py)
- [显式 tensor 广播](../../../python/sglang/multimodal_gen/runtime/videoedit/distributed.py)
- [VAE 空间并行](../../../python/sglang/multimodal_gen/runtime/videoedit/vae_spatial.py)

## 配置

本轮双卡采用以下新增配置，其余 DiT 自适应驻留参数与[此前记录](./adaptive_residency_20261004.md)一致：

```bash
NUM_GPUS=2
SP_DEGREE=2
ULYSSES_DEGREE=2
RING_DEGREE=1

SGLANG_VIDEOEDIT_PARALLEL_CONDITIONING=1
SGLANG_VIDEOEDIT_TEXT_OWNER=0
SGLANG_VIDEOEDIT_CLIP_OWNER=1

VIDEOEDIT_VAE_SP=true
SGLANG_VIDEOEDIT_VAE_SP_MODE=both
SGLANG_VIDEOEDIT_VAE_ENCODE_DEGREE=2
SGLANG_VIDEOEDIT_VAE_DECODE_DEGREE=2
```

该配置已写入 [GPU 4/5 专用配置](../../../scripts/videoedit_dual_service/config.l40s.45.u2.env)。下一次用专用启动脚本创建服务时生效；本轮没有部署或重启线上 GPU 7 服务。

## 正确性与调用次数

固定参数为正常模型、BF16、FA、40 步、动态 CFG、TeaCache 0.3、seed 42、49 帧模型窗口。小裁剪实际模型输入为 544×480×49，全帧为 1920×1088×49；源视频和输出均为 1920×1080、41 帧。

- 小裁剪全部候选及正式/预热输出的解码帧 SHA-256 均为 `2ebdf825e3109140b0e0fe0802e4b1ee29f5e1d24ab862b4e163a9b70030f1ca`。
- 全帧新双卡、旧双卡参考及线上 GPU 7 的有效输出均为 `1ac50c51c261d806a12e155d74cee580da9b95b671533c5b80dbbac98385ce7d`。
- 每个请求两个 rank 的 latent SHA-256 相同，TeaCache 决策序列相同。
- 双卡正式进程连续处理 4 个请求后，rank 0 的正向编码累计次数为 4，负向编码累计次数为 1；rank 1 的文本编码次数为 0。CLIP 在 rank 1 每请求执行一次。
- 默认负向启动预计算耗时 2.925 秒，发生在服务 ready 前。

专项单元测试 5 项通过；VideoEdit pipeline、参数和双服务相关回归共 75 项及 39 个子测试通过。测试覆盖跨窗口请求级引用、负向变化、CFG 关闭、启动默认负向、请求/窗口清理和 tile 顺序。真实模型还验证了 owner 通信、双卡 VAE、启动内存顺序和完整媒体输出。

## VAE 候选筛选

先对同一小裁剪请求各跑一次，所有输出逐帧一致：

| 方案 | VAE encode | VAE decode | 完整推理 |
| --- | ---: | ---: | ---: |
| 单 owner encode / decode | 3.261 秒 | 4.378 秒 | 56.096 秒 |
| 双卡 encode，单 owner decode | 2.109 秒 | 4.388 秒 | 54.272 秒 |
| 双卡 encode / decode | 2.245 秒 | 2.714 秒 | 54.851 秒 |

单样本完整时间存在约 1 秒波动，但双卡 decode 的阶段收益为 1.67 秒，明显大于该阶段重复测量波动，因此正式测试选择 encode/decode 都使用两卡。

## 小裁剪与线上 GPU 7 对比

双卡和线上服务各执行一次预热、三次正式请求。两侧请求模板除任务 ID、输出路径、性能落盘路径和超时外完全一致。正式推理时间如下：

| 服务 | 三次推理时间 | 中位数 | 三次客户端时间 | 客户端中位数 |
| --- | --- | ---: | --- | ---: |
| 新双卡 GPU 4/5 | 46.579 / 47.122 / 47.384 秒 | **47.122 秒** | 47.506 / 48.688 / 48.694 秒 | **48.688 秒** |
| 当前线上 GPU 7 | 79.072 / 79.522 / 78.960 秒 | **79.072 秒** | 80.454 / 80.933 / 80.330 秒 | **80.454 秒** |

按推理时间中位数，新双卡是线上单卡的 **1.678×**，单请求推理延迟减少 **31.950 秒 / 40.41%**。按客户端时间中位数为 **1.652×**，延迟减少 **31.766 秒 / 39.48%**。

阶段中位数取同一请求两个 rank 中较慢者；线上为现有单 rank stage 日志：

| 阶段 | 新双卡 | 线上 GPU 7 |
| --- | ---: | ---: |
| 文本与 CLIP 条件准备 | 1.972 秒 | 文本 12.867 秒 + CLIP 0.992 秒 |
| VAE 条件编码 | 1.826 秒 | 6.012 秒 |
| DiT | 31.592 秒 | 45.905 秒 |
| VAE 解码 | 2.559 秒 | 4.626 秒 |

双卡条件阶段内部的文本中位数为 1.958 秒，CLIP 为 0.535 秒，两者在不同 rank 执行，完整阶段取汇合关键路径 1.972 秒。线上仍顺序运行文本和 CLIP，并且每请求重新装载和编码负向文本。

正式请求的一秒采样显存峰值为双卡每卡最高 44,973 MiB，线上单卡 25,960 MiB。按推理中位数估算，新双卡消耗约 94.24 GPU·秒，线上约 79.07 GPU·秒；新方案以约 19.2% 更多总 GPU 时间换取 40.4% 的单请求延迟下降。

本轮新双卡相对此前同素材双卡中位数 49.456 秒缩短到 47.122 秒，约减少 4.72%。这个历史对照不是交错运行，归因以本轮阶段数据为准。

## 全帧结果

全帧同样执行一次预热、三次正式请求，无 OOM。正式推理时间为 **577.965 / 579.536 / 590.930 秒**，中位数 **579.536 秒**；客户端中位数为 **582.825 秒**。最后一轮 DiT 为 541.964 秒，形成高侧样本，未删除。

| 阶段 | 新双卡三次中位数 | 旧双卡第二次全帧 |
| --- | ---: | ---: |
| 文本与 CLIP 条件准备 | 2.021 秒 | 文本 2.556 秒 + CLIP 0.580 秒 |
| VAE 条件编码 | **14.675 秒** | 24.694 秒 |
| DiT | 529.976 秒 | 532.295 秒 |
| VAE 解码 | **20.443 秒** | 35.836 秒 |
| 完整推理 | **579.536 秒** | 606.838 秒 |

VAE encode 中位数减少 10.019 秒，decode 减少 15.393 秒。完整推理相对旧参考减少 27.301 秒 / 4.50%；两次实验不交错，因此完整时间只作为同模型同输入的历史对照，VAE 阶段差值是本轮主要结论。全帧正式请求的一秒采样峰值为每卡 37,579 MiB。

随后在未重启、未修改配置的当前线上 GPU 7 单卡服务执行相同全帧请求。按用户要求，线上完成一次预热和一次正式请求后停止重复测量；正式输出解码为 1920×1080、41 帧，逐帧哈希与新双卡一致。

| 服务 | 样本口径 | 推理时间 | 客户端时间 |
| --- | --- | ---: | ---: |
| 新双卡 GPU 4/5 | 3 次正式请求中位数 | **579.536 秒** | **582.825 秒** |
| 当前线上 GPU 7 | 1 次正式请求 | **1060.392 秒** | **1062.694 秒** |

按推理时间，新双卡是当前线上单卡的 **1.830×**，单请求延迟减少 **480.856 秒 / 45.35%**；按客户端时间为 **1.823×**，延迟减少 **479.869 秒 / 45.16%**。线上单卡正式请求的主要阶段为文本 13.348 秒、CLIP 1.087 秒、VAE encode 25.664 秒、DiT 971.390 秒和 VAE decode 36.395 秒；一秒采样显存峰值为 35,536 MiB。由于线上全帧只保留一个正式样本，这组加速比是单样本对双卡三次中位数，置信度低于小裁剪的三对三结果。

## 详细阶段耗时

2026-10-05 补充：按“逐请求相减后取中位数”重新统计的前后处理和接口额外开销，见[双卡服务报告补充](../../../docs_handoff/videoedit_service_gpu45_u2_5403.l40s.md#2026-10-05-补充前后处理及接口额外开销)。补充记录同时区分本次双模型小裁剪和此前 normal-only 全帧实验，并解释与本节内层钩子计时及中位数之差口径的区别。

下表单位均为秒。小尺寸双方和大尺寸双卡取三次正式请求的逐项中位数；大尺寸线上单卡为一次正式请求。双卡模块耗时取同一请求两个 rank 的较慢值，再跨请求取中位数。各阶段独立取中位数，表内数值相加不保证严格等于完整请求中位数。

| 阶段 | 小尺寸双卡 | 小尺寸线上单卡 | 小尺寸加速 | 大尺寸双卡 | 大尺寸线上单卡 | 大尺寸加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 客户端请求端到端 | **48.687921** | 80.453688 | **1.652×** | **582.824711** | 1062.693932 | **1.823×** |
| 服务端完整推理 | **47.122396** | 79.072012 | **1.678×** | **579.536459** | 1060.391897 | **1.830×** |
| 客户端/服务边界开销 | 1.565525 | 1.381676 | — | 3.288252 | 2.302035 | — |
| 窗口参数校验 | 0.000109 | 0.0000 | — | 0.000126 | 0.0000 | — |
| 条件准备关键路径 | **1.972437** | 13.858400 | **7.026×** | **2.020908** | 14.435000 | **7.143×** |
| 文本编码 | **1.958456** | 12.866600 | **6.570×** | **2.005956** | 13.348000 | **6.654×** |
| CLIP 图像编码 | **0.534894** | 0.991800 | **1.854×** | **0.560334** | 1.087000 | **1.940×** |
| VAE 条件编码 | **1.826358** | 6.011900 | **3.292×** | **14.675092** | 25.664400 | **1.749×** |
| latent 准备 | 0.004605 | 0.004700 | 1.021× | **0.039584** | 0.052100 | **1.316×** |
| timestep 准备 | 0.000363 | 0.000300 | — | 0.000827 | 0.000800 | — |
| latent 初始化 | 0.000049 | 0.0000 | — | 0.000050 | 0.000100 | — |
| 已打点模型前处理合计 | **3.803921** | 19.875300 | **5.225×** | **16.736587** | 40.152400 | **2.399×** |
| DiT 去噪 | **31.591746** | 45.904900 | **1.453×** | **529.975955** | 971.390200 | **1.833×** |
| VAE 解码 | **2.559003** | 4.626300 | **1.808×** | **20.443303** | 36.394700 | **1.780×** |
| 窗口后处理 | 0.000022 | 0.0000 | — | 0.000024 | 0.0000 | — |
| 已打点模型后处理合计 | **2.559025** | 4.626300 | **1.808×** | **20.443327** | 36.394700 | **1.780×** |
| 其余请求内开销 | 8.636803 | 8.554212 | 0.990× | **12.362661** | 12.454597 | 1.007× |

双卡文本和 CLIP 分别在 rank 0、rank 1 并行执行，因此“文本编码”和“CLIP 图像编码”是“条件准备关键路径”的子项，不能再次相加。线上单卡顺序执行两者，所以其条件准备关键路径等于文本与 CLIP 之和。“已打点模型前处理合计”只计算窗口校验、条件准备关键路径、VAE 条件编码、latent/timestep 准备和 latent 初始化，不重复计算文本、CLIP 子项。

“其余请求内开销”按每次服务端完整推理减去已打点关键路径后再取中位数，包括当前没有独立计时点的输入视频、mask 和参考图读取与转换、bbox/crop 编排、跨 rank 请求协调、贴回、音频处理、媒体编码写盘及运行时空隙。“客户端/服务边界开销”使用表中客户端中位数减服务端中位数，包含请求准入、状态轮询和响应观测误差，不作为模型模块加速比。线上日志只保留到 0.1 毫秒，因此其中的 `0.0000` 表示低于日志显示精度，并非证明绝对耗时为零。

按不重叠的关键路径归因，双卡节省如下：

| 延迟来源 | 小尺寸减少 | 大尺寸减少 |
| --- | ---: | ---: |
| 条件准备关键路径 | 11.885963 秒 | 12.414092 秒 |
| VAE 条件编码 | 4.185542 秒 | 10.989308 秒 |
| DiT 去噪 | 14.313154 秒 | **441.414245 秒** |
| VAE 解码 | 2.067297 秒 | 15.951397 秒 |
| 其余请求内开销变化 | -0.082591 秒 | 0.091936 秒 |
| 服务端完整推理总计 | **31.949616 秒** | **480.855438 秒** |

小尺寸的收益分布在条件准备、DiT 和 VAE；大尺寸总收益中约 91.8% 来自 DiT 去噪减少的 441.414 秒。表中模块差值使用模块中位数，总计行使用服务端完整推理中位数，因此模块差值之和与总计可能有独立取中位数造成的小差异。

## 复现与记录

```bash
# 候选筛选
python3 scripts/videoedit_benchmark/run_parallel_optimization.py \
  --out .local/parallel-optimization-screen-new \
  --gpus 4,5 --variants owner1,encode2,both2 \
  --geometry crop --requests 1

# 双卡正式测试：每个目录一次预热加三次正式请求
python3 scripts/videoedit_benchmark/run_parallel_optimization.py \
  --out .local/parallel-optimization-crop-new \
  --gpus 4,5 --variants both2 --geometry crop --requests 4
python3 scripts/videoedit_benchmark/run_parallel_optimization.py \
  --out .local/parallel-optimization-full-new \
  --gpus 4,5 --variants both2 --geometry full --requests 4 \
  --request-timeout 1500 --container-timeout 7000

# 不改线上配置，对现有服务执行相同请求
python3 scripts/videoedit_benchmark/run_online_comparison.py \
  --out .local/parallel-optimization-online-new \
  --request-template .local/parallel-optimization-crop-new/both2/request-template.json \
  --requests 4

# 线上全帧：在相同模板上强制完整画面模型输入
python3 scripts/videoedit_benchmark/run_online_comparison.py \
  --out .local/parallel-optimization-online-full-new \
  --request-template .local/parallel-optimization-full-new/both2/request-template.json \
  --geometry full --requests 2 --request-timeout 3600
```

本机最终记录：

- `.local/parallel-optimization-crop-formal-20261004`
- `.local/parallel-optimization-full-formal-20261004`
- `.local/parallel-optimization-online-gpu7-20261004`
- `.local/parallel-optimization-online-gpu7-full-20261004`
- `.local/parallel-optimization-screen-20261004`
- `.local/parallel-optimization-smoke-20261004-v5`

每个双卡目录包含请求结果、逐帧 MD5、rank 事件、stage perf、显存采样、启动日志、镜像 ID 和源文件哈希。线上目录另保存容器 inspect、请求前后健康状态和本轮服务日志片段。

## 结论边界

- 已验证双 L40S、正常模型、BF16、FA、40 步、TeaCache 0.3、batch 1、49 帧窗口以及 544×480 和 1920×1088 两种实际输入。
- 当前空间实现依赖 Wan VAE 的既有 tiling 语义；未触发空间 tiling 的更小输入继续走原路径。patchified VAE 会明确拒绝该路径。
- 配置允许在更大的 SP world 中指定 encode/decode degree，但本轮没有进行四卡真实模型验收，因此四卡不作为已验证配置发布。
- 小裁剪对比采用双方各三次正式请求的中位数；全帧线上单卡按用户要求只保留一次正式请求，因此全帧加速比需要后续多样本测试才能给出同等置信度。
- 现有线上服务没有变更。GPU 4/5 正式服务配置会在下一次创建专用服务时启用本轮能力。

[返回优化汇总](./memory_optimization_20261004.md)
