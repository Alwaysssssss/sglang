# L40S 双卡并行、卸载与通信实测（2026-09-29）

后续补充：[DiT 完整常驻的受控单/双卡对照](videoedit_dit_resident_benchmark.l40s.md)测得约 1.555 倍加速。本报告的“没有加速”仅适用于下述卸载/FSDP 配置及历史基线比较，不能推广到完整常驻 SP。

## 范围与方法

使用物理 GPU 1、2，独立容器 `videoedit_bench_12`、端口 5404；没有在 GPU 7 上运行测试，没有重启 GPU 0 的现有服务。镜像 `sglang-videoedit-dev-v2:l40s`。容器内逻辑 GPU 0/1 对应物理 GPU 1/2。容器内存上限 300 GiB，memory-swap 同值（不允许额外 swap）。

复用历史请求 `4d187b56-62b7-4121-8d7b-7528150e35d9` 的输入和生成参数，固定种子，40 步、TeaCache 0.3、动态 CFG；输出 41 帧。移除回调和远端输出配置，只写本地文件。每个成功配置顺序运行一次预热、一次正式计时、一次 Nsight Systems 采集。正式计时包含 HTTP 提交到轮询完成，约 1 秒轮询误差；采集轮次单独列出，不混入正式计时。单次正式计时不能证明小幅差异具有统计显著性。

## 请求耗时与资源

| 配置 | 预热 s | 正式 s | 采集轮 s | 正式峰值显存 GiB/卡（两卡最大） | 正式容器内存峰值 GiB |
|---|---:|---:|---:|---:|---:|
| u_layer1 | 109.03 | 101.15 | 102.27 | 17.11 | 122.11 |
| r_layer1 | 111.44 | 103.55 | 107.17 | 17.10 | 122.57 |
| u_fsdp | 104.17 | 92.00 | 96.35 | 32.84 | 42.94 |
| r_fsdp | 100.96 | 93.44 | 96.83 | 32.86 | 43.54 |
| u_fsdp_simple | 109.74 | 95.74 | 97.17 | 32.84 | 44.00 |
| u_fsdp_simple_textresident | 88.08 | 76.30 | 78.41 | 36.00 | 23.36 |

容器内存来自 cgroup `memory.current`，包括可计费的文件缓存等，不等于纯 Python RSS；峰值来自每秒采样，不能代表瞬时最高点，也不包含启动阶段峰值。

## 正式请求的阶段耗时

| 配置 | 文本编码 s | 图像编码 s | 条件编码 s | 去噪 s | 解码 s |
|---|---:|---:|---:|---:|---:|
| u_layer1 | 17.31 | 3.21 | 8.40 | 57.10 | 5.22 |
| r_layer1 | 16.34 | 2.50 | 8.53 | 62.00 | 5.24 |
| u_fsdp | 15.93 | 3.40 | 6.65 | 51.93 | 5.02 |
| r_fsdp | 16.87 | 2.79 | 9.88 | 48.71 | 5.18 |
| u_fsdp_simple | 19.02 | 3.00 | 10.12 | 49.34 | 5.29 |
| u_fsdp_simple_textresident | 0.12 | 2.86 | 9.88 | 48.31 | 5.15 |

阶段来自服务原生 perf dump（CPU 侧墙钟计时，本轮未设置 SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1，因此异步 GPU 工作可能在后续阶段才完成；尤其不能把文本常驻后的 0.12 s 当作完整 GPU 文本计算时间）；视频读取、输出编码等开销未全部包含在上表，因此阶段之和不等于端到端耗时。

## Nsight GPU 时间线

以下仅列物理 GPU 1（逻辑 0）的采集轮；另一张卡数据见各目录 `analysis.json`。每类耗时是该卡活动区间的并集，跨类别可能重叠，不能相加得到请求耗时。“非 NCCL kernel”包括矩阵运算、注意力、打包/拷贝等 GPU kernel，并非纯数学计算。NCCL kernel 时间包括等待，不等于纯链路传输时间。H2D/D2H 分别是 CPU→GPU/GPU→CPU。

| 配置 | 非 NCCL kernel s | NCCL s | H2D s | D2H s | H2D GiB | D2H GiB | kernel/NCCL 重叠 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| u_layer1 | 27.77 | 14.66 | 56.78 | 16.03 | 524.64 | 13.07 | 0.00 |
| r_layer1 | 30.02 | 17.75 | 61.27 | 14.50 | 524.64 | 13.07 | 2.16 |
| u_fsdp | 44.42 | 26.53 | 2.04 | 17.47 | 13.15 | 13.07 | 17.42 |
| r_fsdp | 48.61 | 27.99 | 1.85 | 15.73 | 13.15 | 13.07 | 22.81 |
| u_fsdp_simple | 44.98 | 26.73 | 2.74 | 18.26 | 13.15 | 13.07 | 18.00 |
| u_fsdp_simple_textresident | 44.53 | 26.53 | 0.42 | 2.59 | 2.56 | 2.49 | 17.52 |

逐层卸载 Ulysses 的互斥分解：H2D 单独执行 22.07 s、H2D 与非 NCCL kernel 重叠 22.04 s、H2D 与 NCCL 重叠 12.67 s、D2H 单独执行 16.03 s、非 NCCL kernel 单独执行 5.73 s、NCCL 单独执行 2.00 s、无已追踪 GPU 活动 10.83 s，其余约 0.03 s；合计 GPU 活动观察窗口约 91.39 s。窗口外的 CPU/HTTP/编码开销另计。

FSDP Ulysses 的互斥分解：H2D 2.04 s、D2H 17.47 s、非 NCCL kernel 单独执行 27.00 s、NCCL 单独执行 9.11 s、kernel 与 NCCL 重叠 17.42 s、无已追踪 GPU 活动 12.01 s，其余约 0.03 s。没有 GPU 活动的空隙不能直接标成 CPU 计算时间。本次关闭 CPU sampling/context-switch 采样，未精确拆解 CPU 编码、调度和等待。

## 配置含义与失败案例

- `u_layer1`：SP=2、Ulysses=2、Ring=1，DiT 逐层卸载、prefetch=1。`r_layer1` 改为 Ulysses=1、Ring=2。两卡各自搬运相同量级权重，序列分片不会自动让权重搬运减半。
- `u_fsdp` / `r_fsdp`：关闭 DiT 逐层卸载，启用 `--use-fsdp-inference true --hsdp-shard-dim 2 --hsdp-replicate-dim 1`；权重分片常驻 GPU，通过 AllGather 临时聚合。二者仅 Ulysses/Ring 不同。
- `u_fsdp_simple`：在 u_fsdp 基础上设 `NCCL_PROTO=Simple`。端到端没有测出收益，去噪阶段略快但其他阶段波动，不能据单次请求认定协议性能有确定排序。
- `u_fsdp_simple_textresident`：在 Simple 配置基础上关闭文本编码器 CPU 卸载，图像编码器和 VAE 仍卸载。应与 u_fsdp_simple 成对比较，避免混淆协议和卸载影响。
- 以上均 `--dit-cpu-offload false --pin-cpu-memory true`。FSDP 模式下逐层卸载关闭，prefetch 参数不参与逐层预取。
- `u_resident`：关闭逐层卸载且不使用 FSDP，完整 DiT 在两卡各保留一份。首个请求约 25.37 s 时在文本编码器搬入显存阶段 CUDA OOM；采样显存约 44.39 GiB/卡。说明仅使用序列并行不足以容纳完整常驻模型组合。没有继续测试同样内存布局的 Ring 常驻模式。

## 已确定与尚未确定的原因

逐层卸载的主要限制是大量 CPU/GPU 权重搬运：每卡 H2D 约 524.64 GiB，显著超过 FSDP 的 13.15 GiB。FSDP 以更多 GPU 权重通信换取更少主机搬运，端到端有所改善，但通信与计算重叠会共享 GPU/内存资源。默认 FSDP trace 的 AllGather 使用 `RING_LL`；同一种 GEMM 的 5550 次调用累计时间从约 6.95 s 增至 21.14 s，提示重叠期间可能存在资源竞争，尚不能单凭 trace 证明具体硬件瓶颈。

Ulysses 与 Ring 的单次端到端差异较小，且 FSDP 下 Ring 去噪阶段反而略快；因此只能报告本输入下的结果，不能推广为所有 PCIe 卡或所有分辨率都应选同一种并行方式。此前单卡同输入约 79 s 是历史基线，不是本轮同时重测；已有数据不支持为降低单请求延迟而把生产服务切到双卡。

## 原始证据与复现

所有本轮脚本、日志、输出和 trace 位于 `.local/dual-gpu-benchmark/`（本地实验产物，未纳入 Git）。每个变体目录保存 `command.json`、`server.log`、`nccl.*.log`、三轮 `*-result.json` / `*-samples.jsonl` / `*-perf.json`、`trace.nsys-rep`、`trace.sqlite` 和 `analysis.json`。用 Nsight Systems 打开 `.nsys-rep` 可逐流查看 CUDA kernel、memcpy 和 NCCL 的重叠。自定义 BENCH NVTX 包装未传播至 worker，阶段归因使用原生 perf dump，GPU 分类使用 CUDA/NCCL 原生采集。

测试容器的 GPU 只能设为 1、2；内存限制 300g、memory-swap 300g；挂载仓库至 `/sgl-workspace/sglang` 并保留模型与输入所用原路径。设置 `PYTHONPATH=/sgl-workspace/sglang/python`、`CUDA_VISIBLE_DEVICES=0,1`、`SGLANG_USE_RUNAI_MODEL_STREAMER=false`、`PYTORCH_ALLOC_CONF=expandable_segments:True`。容器存活且 30000 端口空闲时，可执行：

```sh
docker exec -u root videoedit_bench_12 python3 /sgl-workspace/sglang/.local/dual-gpu-benchmark/matrix.py u_fsdp
```

必须等脚本完成采集导出和后端关闭后再启动下一组；`invalid_simple_early_start` 是重试前的无效运行，不纳入结果。复跑请先归档既有同名目录，避免覆盖证据。

## P2P 实际验证

本机这对 GPU 在容器内支持 P2P，模型运行日志实际选择 `via P2P/CUMEM`。启动日志中的 NET/Socket 探测不代表模型张量经 Socket 传输，应看最终通道选择。

同一个静态 CUDA 程序，启用 peer access、64 MiB 拷贝、5 次预热、30 次计时，两个方向依次测试。每轮先将源缓冲区填 7、目标清零，拷贝后校验全部字节，均通过：

| 方向（容器逻辑 GPU） | 宿主机 GB/s | 容器 GB/s |
|---|---:|---:|
| 1 → 0 | 26.672 | 26.660 |
| 0 → 1 | 26.266 | 26.264 |

这是单向 cudaMemcpyPeerAsync 带宽，不是双向聚合带宽，也不是模型 AllGather 的有效带宽。数据见 `p2p-host.txt`、`p2p-container.txt`，源码 `p2p.cu`，校验二进制 `p2p_checked`。

另用两 rank 的 `torch.distributed.all_to_all_single` 测试：每尺寸 5 次预热、30 次计时，CUDA event 计时取较慢 rank，校验接收的两段数据。这里只调整 `NCCL_P2P_DISABLE`，未强制 Simple：

| 每 rank 总输入大小 MiB | P2P 开启 ms | P2P 禁用 ms |
|---|---:|---:|
| 0.0625 | 0.0645 | 0.1132 |
| 4 | 0.1463 | 3.5350 |
| 32 | 0.9725 | 24.8694 |
| 128 | 3.8296 | 98.4005 |

128 MiB 输入中一半发往另一张卡；按远端有效数据计，每 rank 带宽约 17.52 GB/s 与 0.682 GB/s。日志确认禁用后走 `SHM/direct/direct`。该差异只适用于本机、本微基准，不能直接换算成完整视频请求加速倍数。证据为 `probe-p2p-result.txt`、`probe-nop2p-result.txt` 和对应日志。无需给容器增加 privileged 或关闭 P2P。

## 文本常驻的收益与建议配置

文本常驻相对同为 Simple 的文本卸载配置：正式请求 95.74 → 76.30 s；采集轮 97.17 → 78.41 s；每卡 H2D 13.15 → 2.56 GiB，D2H 13.07 → 2.49 GiB。逻辑 GPU 0 的 D2H 活动 18.26 → 2.59 s，非 NCCL kernel 44.98 → 44.53 s，NCCL 26.73 → 26.53 s。两轮独立观察都支持主要收益来自减少文本编码器来回搬运。

当前最有希望的双卡候选是 **Ulysses=2 + FSDP shard=2 + 文本编码器常驻 + VAE/图像编码器卸载**。本轮成功测过的完整参数组合如下（模型路径、端口与启动器见该变体 `command.json`）：

```sh
# 只映射物理 GPU 1、2，容器内部使用逻辑 0、1
NCCL_PROTO=Simple
# sglang serve 的相关参数：
--num-gpus 2 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
--use-fsdp-inference true --hsdp-shard-dim 2 --hsdp-replicate-dim 1 \
--dit-layerwise-offload false --dit-cpu-offload false \
--text-encoder-cpu-offload false --image-encoder-cpu-offload true \
--vae-cpu-offload true --pin-cpu-memory true
```

上面的 Simple 是本轮该成功组合的实验条件，不代表已证明 Simple 更优。尚未测试“文本常驻 + 默认协议”；因此下一步应成对复测这两个协议，并用多个不同输入、至少三次正式请求检验收益与显存余量。当前峰值约 36 GiB/卡仅适用于这个样本，不保证更大输入也能运行。无需再增大 `dit-offload-prefetch-size`：该候选已经关闭 DiT 逐层卸载。

此前单卡约 79 s 的结果来自网关计时，本轮直接请求后端，硬件使用状态也不同。不能据 76.30 vs 79 s 宣称双卡稳定更快。生产服务保持现有单卡配置；如果目标是多任务吞吐，双卡各自跑单卡副本是待测的另一方案，不能用本轮单请求数据证明其收益。

## 输出与收尾检查

六组成功配置共 18 次输出均可解码为 41 帧。所有 Ulysses 配置的解码帧哈希相同；所有 Ring 配置相同；Ulysses 与 Ring 哈希不同，但代表输出间 SSIM All=0.999996。SSIM 不是人工质量验收，也不保证其他输入的结果一致。证据在 `output-validation.json`、`ring-vs-ulysses-ssim.log`。

测试完成后停止独立测试容器，保留其定义和仓库内证据供复查。GPU 0 的 5403 服务和 GPU 7 的 5402 服务健康接口均返回 HTTP 200、normal=true、dmd=false；符合各自 normal-only 状态。本轮没有修改其配置或重启它们。
