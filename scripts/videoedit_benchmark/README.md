# TeaCache / SP 实验入口

只运行正常模型，物理 GPU 限定 0–3；每项创建独立容器，容器无外网、不映射服务端口，不使用业务队列。所有缓存、临时文件和输出放在仓库内。模型目录只读挂载。启动前要求所用 GPU 空闲；退出时销毁所属容器，确认显存释放后才进入下一项。CPU/RAM 不设硬限制。

```bash
# 无 GPU 单元测试
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/videoedit_benchmark -p 'test_*.py' -v
# 查看 30 项配置的执行顺序，不启动容器
python3 scripts/videoedit_benchmark/run.py --out .local/teacache-sp/example --dry-run
# 首项，输出目录必须尚不存在
python3 scripts/videoedit_benchmark/run.py \
  --out .local/teacache-sp/full-single --variants single_prefetch1
# 30 项串行矩阵；可耗时数小时以上
python3 scripts/videoedit_benchmark/run.py --out .local/teacache-sp/full-matrix
# 从已结束且已释放 GPU 的配置之后续跑，保留全部既有结果
python3 scripts/videoedit_benchmark/run.py --out .local/teacache-sp/full-matrix --resume
# 查看实时配置/请求/去噪进度
python3 scripts/videoedit_benchmark/status.py .local/teacache-sp/full-matrix --write
# 按已测结果指定剩余配置的优先顺序
python3 scripts/videoedit_benchmark/run.py \
  --out .local/teacache-sp/selected --variants u2_prefetch2,u2_prefetch4,u2_prefetch8
# 裁剪复测必须另开输出目录
python3 scripts/videoedit_benchmark/run.py \
  --out .local/teacache-sp/crop-check --geometry crop --variants single_prefetch1,u2_prefetch1
```

每项加载一次模型，然后提交 1 次完整请求预热、3 次完整请求计时。固定 FlashAttention (`fa`)、BF16、pin memory 和进程级 THP 禁用；不启用 FSDP/量化/编译。`--request-timeout` 默认 3600 秒，不通过缩小输入绕过失败。结果目录已存在时拒绝运行，避免覆盖失败证据。

本机历史输入来自 `.local/videoedit-l40s.0123/dual/` 的请求 `4d187b56-62b7-4121-8d7b-7528150e35d9`。脚本读取其原始文本、视频、mask、参考图以及窗口策略，输出输入 SHA-256 和净化后的本地请求。`full` 几何只替换 chunk bbox 和对齐尺寸，保留原始编辑 mask；源尺寸不是 1920×1080 时直接报错。模型实际输入应为 49 帧、1920×1088，不能从输出帧数推断。

`stage_resident` 在启动时将 DiT 放 CPU；条件编码后卸载 VAE，在去噪前载入完整 DiT，完成去噪即卸载 DiT，再载入 VAE 解码并卸载。每个窗口和请求重复这一过程。权重传输纳入阶段时间。通过测试专用 `PYTHONPATH` 钩子启用，业务服务不受影响。

结果包含：

- `manifest.json`、`container.json`、`command.json`：镜像 ID、环境和实际启动配置。
- `load.json`：后端启动到健康检查通过，含模型加载和初始化；不算入请求时间。
- `rank*.jsonl`：同步后的各阶段耗时、allocated/reserved 峰值、形状、每 pass TeaCache 决策、首 block 实际执行次数、最终 latent 哈希、完整 pipeline 请求耗时。
- `*-samples.jsonl`：每秒 GPU 显存/利用率/功耗、容器内存与累计 CPU 使用量；采样可能漏掉瞬时峰值。
- `*-result.json`、`*.framemd5`：完整请求状态、视频尺寸/帧数和解码帧哈希。
- `summary.json` / `summary.md`：仅三次正式请求均完整且所有 rank 记录齐全时计算中位数、最小/最大值、去噪耗时和 GPU 秒；同目录单卡基线存在时计算加速比。

端到端主指标取各 rank pipeline 请求时间最大值，包含流式输入、模型计算/搬运、解码及本地视频编码输出，也包含同步和采集开销；`client_seconds` 另含 HTTP 提交及轮询。latent 哈希复制开销单独记录。去噪取各 rank 去噪阶段总和的最大值。输出哈希不同不代表质量等价，需再用仓库 `runtime/videoedit/compare.py` 做逐帧误差/PSNR/SSIM。未经质量比较、裁剪复测及交错复测，不据此修改业务配置。

## 显存预取与通信重叠（2026-10-02）

新增路径保持显式启用，业务配置默认不变：

- `SGLANG_VIDEOEDIT_OVERLAP_PREFETCH=1`：逐层滚动预取，复用已消费的平坦权重缓冲区。copy stream 仅在复用槽位前等待该槽位的消费完成事件；compute stream 仅等待当前层权重。均匀平坦权重的缓冲区槽位数不超过“驻留层数 + 预取层数 + 1”，不包含激活、通信或分配器缓存。非连续权重继续保留原 stride，异构层和非连续权重不保证同样的字节上界。
- 预取不再绕回下一次 forward，避免最后一次去噪或 TeaCache 跳过时仍搬运无用权重。只读 CPU 权重保留，逐层释放无需 D2H；`release_all()` 同时释放复用池。更新 CPU 权重前等待未完成的 H2D，防止修改 DMA 源。
- `SGLANG_VIDEOEDIT_USP_CHUNKS=2`：按 head 分块，每块仍对完整全局序列计算 attention。下一块 QKV 交换与当前块 attention、上一块输出交换与后续 attention 可重叠。分块增加 collective 和 kernel 启动次数，不能假定一定提速。
- 通信流水线仅用于无梯度 CUDA、FlashAttention、纯 Ulysses、等形状/类型/设备 QKV、无 dropout，且 head 数可被 `Ulysses × chunks` 整除的路径。Ring、GQA、复制 token 路径等仍使用既有实现。所有 rank 必须使用相同配置。`USP_CHUNKS=1` 关闭流水线；`SGLANG_VIDEOEDIT_FUSED_QKV` 控制既有的整块合并交换。

不需要服务依赖的验证入口（需要 PyTorch）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  scripts.videoedit_benchmark.test_overlap_pipeline -v

# 先确认设备空闲；这里的 0、2 仅是显式选择示例。
CUDA_VISIBLE_DEVICES=0,2 PYTHONDONTWRITEBYTECODE=1 \
  torchrun --standalone --nproc_per_node=2 \
  scripts/videoedit_benchmark/test_overlap_pipeline.py --backend nccl

CUDA_VISIBLE_DEVICES=0,2 PYTHONDONTWRITEBYTECODE=1 \
  torchrun --standalone --nproc_per_node=2 \
  scripts/videoedit_benchmark/benchmark_usp_overlap.py \
  --sequence-length 1024 --heads 40 --head-dim 128 --chunks 2 4 \
  --trace-dir .local/overlap-validation/traces
```

微基准使用 BF16 PyTorch SDPA，输出最慢 rank 的延迟中位数/范围、各 rank 中最大 allocated/reserved 峰值及最大绝对输出误差。峰值包括固定输入和参考输出。不整除的分块数明确跳过。trace 独立于计时采集，可检查 NCCL、attention 和 memcpy 的真实时间重叠；微基准不等于生产 FlashAttention 后端或完整视频请求性能。

本轮已验证：PyTorch 2.6 CPU 下，24 组预取/驻留/stride 组合的重复推理、缓冲区复用上界、释放后重新加载；16 组模拟 2/4 rank 通信的分块 attention 与全局 attention 数值对齐；原有实验隔离/聚合的 6 个测试通过。CPU 测试模拟事件与传输，不验证 CUDA 生命周期或 NCCL 行为。NCCL 测试入口另外覆盖非默认流连续排队 20 次 forward、同步前释放、stride 权重与驻留缓存。

当前会话权限变化后，Docker socket、GPU 驱动及进程间 socket 均不可访问，因此 GPU/NCCL 实测未完成，尚无加速比、GPU 峰值改善或完整视频质量结论。后续先运行 NCCL 正确性测试，再用相同输入、TeaCache 轨迹与配置交错测完整请求；权重 H2D 和 NCCL 会竞争 PCIe 带宽，需要在两项同时开启时复测。

## 阶段内存与主机 staging 优化

VideoEdit 默认及时释放仅供前一阶段使用的张量：不再额外生成未使用的原始视频 GPU 副本；条件编码后释放 masked video 和完整 mask；去噪后释放条件与文本/图像嵌入；解码完成后释放 latent 和 GPU 视频引用。原始噪声在初始化 latent 后解除额外引用，随机数生成和精度保持不变。`SGLANG_VIDEOEDIT_KEEP_STAGE_TENSORS=1` 可保留原诊断字段。

`vae_cpu_offload=true` 时，VAE 在编码和解码阶段分别上传，结束后恢复 CPU 权重引用，不回拷只读权重；正常退出和异常退出都清理 VAE 的 `clear_cache()`。首次接管已在 GPU 的 VAE 时会先迁回 CPU。关闭该服务参数时保留原驻留行为。

解码 D2H 使用两个 pinned 帧批次缓冲区，按解码 dtype 传输，随后在 CPU 转 FP32 和 uint8，与原像素转换顺序一致。不会在 GPU 生成整段 FP32 输出副本。当前优化重叠的是 D2H 与 CPU 帧转换；VAE 仍按原逻辑解码整窗，不声称已实现逐帧 VAE 解码与回传重叠。

| 配置 | 默认 | 行为 |
| --- | --- | --- |
| `SGLANG_VIDEOEDIT_ASYNC_STAGE_LOAD` | `0` | 为只读阶段权重使用独立 CUDA copy stream；同 dtype 连续权重合并 GPU 存储，并通过有界 staging 合并小 H2D 拷贝。消费前等待事件，消费后通过 `record_stream` 保护 GPU 内存。编码器需同时开启既有 `ENCODER_CPU_MASTER=1`。 |
| `SGLANG_VIDEOEDIT_PREFETCH_IMAGE_ENCODER` | `0` | 后台线程在文本编码时加载下一阶段 CLIP。要求图像编码器 CPU offload 和 `ENCODER_CPU_MASTER=1`，且显存检查通过。窗口、阶段异常退出或 reset 会等待并清理预取句柄。 |
| `SGLANG_VIDEOEDIT_STAGE_PREFETCH_RESERVE_MB` | `4096` | 提前加载 CLIP 后必须留下的显存余量，用于文本激活等；需按实际负载调整。 |
| `SGLANG_VIDEOEDIT_STAGE_PINNED_MB` | `64` | 每个异步阶段加载器的双缓冲区总大小；`0` 不创建 staging 池。并发加载时，各加载器预算相加。 |
| `SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB` | `0` | 正数时，DiT CPU 权重保留在普通内存，每个 offload manager 只创建指定大小的双 staging 缓冲区；`0` 保留既有全量 pin 策略。还需启用服务的 pin memory。 |
| `SGLANG_VIDEOEDIT_AUTO_RESIDENCY` | 服务启动脚本为 `1`；直接后端为 `0` | 每个去噪阶段根据可用显存和可复用 allocator 缓存重新选择驻留前缀和预取深度，自动启用滚动预取；阶段退出释放权重。 |
| `SGLANG_VIDEOEDIT_ACTIVATION_RESERVE_MB` | `4096` | 自适应预算中排除的未来激活/工作区显存；不是自动学习的峰值预测器，首次使用前需根据负载校准。 |
| `SGLANG_VIDEOEDIT_WEIGHT_BUDGET_MB` | `0` | 正数时进一步限制所有自适应 offload manager 的权重预算总和，按模型权重大小分配。预算不足以容纳最大层时报错；仅够一层时使用零 lookahead 的按需加载。 |
| `SGLANG_VIDEOEDIT_DECODE_TRANSFER_FRAMES` | `4` | 每个 D2H 缓冲区的帧数，必须为正数。 |

预算是权重调度容量，不是进程显存硬限制：激活、NCCL、allocator 碎片及异构层暂时存活的旧分配仍需额外余量。有界 pinned staging 会增加 CPU 内存拷贝，并在重用 CPU 槽位前等待其 DMA 事件；非连续权重和 pin 失败会显式回退为同步 pageable 拷贝，不能把“开启 staging”直接等同于提速。

阶段报告新增 `effective_resident_layers`、`effective_prefetch_layers`、`pinned_master_bytes`、`pinned_staging_bytes`、`pageable_fallback_copies`，用于核实实际策略与回退。建议分别测逐项开启与组合开启，避免把减少 pin 内存带来的额外 CPU 开销归因到通信。

测试入口：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  scripts.videoedit_benchmark.test_videoedit_memory \
  scripts.videoedit_benchmark.test_overlap_pipeline \
  scripts.videoedit_benchmark.test_benchmark -v
```

CPU 测试使用真实预处理、阶段代码和调度器以及小型替代模型，覆盖保留中间张量/释放中间张量两种模式的完整阶段链路、latent 和像素逐值一致、重复请求、VAE 缓存清理、部分上传失败、预取句柄回收、驻留预算和 pin 失败回退。它不是生产模型端到端质量测试。CUDA 可用时另运行非默认流权重加载、分批 D2H、有限 pinned 池、多轮逐层卸载以及移除钩子后完整加载的测试。

本轮结果：上述命令共 22 项，20 项通过，2 项 CUDA 测试因设备不可访问而跳过；另通过 Python AST、shell 语法和 `git diff --check`。日志位于仓库 `.local/overlap-validation/memory-tests.log`。尚未执行生产模型、真实多卡和完整服务请求测试，不能据此给出实际显存降幅或延迟收益。自适应预算目前使用实时可用量与配置的激活余量，未实现按历史峰值自动学习余量。

## 跨 rank 共享只读 CPU 权重（2026-10-04）

`SGLANG_VIDEOEDIT_SHARED_CPU_DIR` 指向所有本机 rank 可见的同一个私有目录时，DiT 逐层卸载将连续 CPU 权重按内容和 dtype 去重，映射到相同文件页。默认空值关闭。非连续权重仍采用原有保留 stride 的私有存储。共享的是 CPU 原件，每张 GPU 仍需分别上传，不能据此宣称减少 H2D 总字节数或推理必然加速。

开启 `pin_cpu_memory` 时，**该目录需位于 tmpfs**。本机 CUDA 13/L40S 对普通磁盘文件的 `MAP_SHARED` 注册返回 invalid argument；不能改用可能触发页面复制的 `MAP_PRIVATE` 来冒充共享。代码通过 `cudaHostRegister` 注册共享页，直接 H2D，不经过 pageable→pinned staging；与 `SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB>0` 互斥。注册失败明确报错，不静默创建完整私有 pinned 副本。tmpfs、锁页内存均消耗主机 RAM，需给目录足够容量；本模型建议该测试挂载 40 GiB。

生命周期：原子发布完整文件；各映射持有文件锁；tensor 及其派生 view 持有映射和 CUDA 注册。最后一个本进程引用释放前等待 CUDA，注销后才释放映射；最后一个进程引用正常退出时删除数据文件。进程被强杀可能留下文件，因此每个服务/实验使用独立目录，并在全部 rank 退出后清理。以下测试容器退出时销毁整个专用 tmpfs，不在磁盘长期保存权重缓存。

共享原件只读。现有 `update_cpu_weights` 和 `sync_layer_to_cpu` 路径在写入前等待 H2D，按层创建私有副本，保证 LoRA/refit 不污染其他 rank。发生更新的层不再共享，内存可能回升；`iter_cpu_weights` 返回的 tensor 仅供读取，不支持调用者绕过更新接口原地写入。该实现不是可变权重的跨进程同步协议。

可复现的小尺寸单变量对照（两张卡必须空闲）：

```bash
python3 scripts/videoedit_benchmark/run_shared_cpu.py \
  --out .local/shared-cpu-benchmark-new --gpus 4,5 \
  --image sglang-videoedit-dev-v2:l40s
```

该入口限定本轮获准的 GPU 4/5/6 中两张，模型只读挂载，无外网或业务回调。两组均为四层预取、40 层阶段缓存、结束释放、编码器 CPU 原件模式；仅共享 CPU 策略不同。每组 1 次预热 + 3 次正式请求，自动销毁测试容器。日志记录 `shared_cpu_weight_bytes`、`host_memory_bytes`（含 PSS）、`shared_master_mapping_bytes`、共享文件总字节数及原有 GPU 指标。PSS 按共享比例计费，跨 rank 比较应累加 PSS；不能把两个共享映射的 RSS 或 `pinned_master_bytes` 直接相加作为物理占用。

新增测试 `test_shared_cpu_weights` 覆盖内容隔离、原子发布失败、跨进程物理共享、rank 退出、派生 view 生命周期、非默认流异步复制，以及 refit/LoRA 写入前分离。真实 CUDA 测试必须在 tmpfs 下设置 `TMPDIR`；没有 CUDA 时相应测试会跳过。开发验证日志位于 `.local/shared-cpu-validation/`。

实现依据：[PyTorch frombuffer 所有权说明](https://docs.pytorch.org/docs/stable/generated/torch.frombuffer)、[CUDA 主机内存注册](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)。

本机小尺寸验收已完成：`.local/shared-cpu-benchmark-20261004/report.md`。两组各 4 次请求全部成功，跨组 latent 和解码帧哈希一致。两 rank 去噪后 PSS 合计中位数 112.982→63.082 GiB；共享映射本身合计 PSS 30.089 GiB、私有页为 0。完整请求中位数 51.037→52.056 秒，DiT 32.546→32.033 秒，启动到健康检查 138.090→190.093 秒。样本范围重叠，且共享组包含额外 smaps 证据采集；不宣称稳定加速。默认配置保持关闭，未部署到既有业务服务。

## 尺寸感知自动驻留（2026-10-04）

专用入口 `run_adaptive_residency.py` 使用镜像 `sglang-videoedit-dev-v2:l40s`，支持从 GPU 4/5/6 中选两张空闲卡。先跑固定全层缓存小尺寸基线，再在同一自适应服务中交替跑小尺寸与真实 1080p。`report_adaptive_residency.py` 检查各尺寸输出、双 rank 预算一致和缓存释放，单独报告两次全帧耗时。

新增 `SGLANG_VIDEOEDIT_FULL_FRAME_RESERVE_MB=16384` 和 `SGLANG_VIDEOEDIT_RESIDENCY_MARGIN_MB=2048`。VideoEdit 服务启动脚本默认开启自动驻留；直接启动后端时设置 `SGLANG_VIDEOEDIT_AUTO_RESIDENCY=1`，显式设为 `0` 可关闭。范围、完整配置与复现命令见[自动驻留文档](../../docs_always/add_new_mode/add_videoedit_diffusers/adaptive_residency_20261004.md)。
