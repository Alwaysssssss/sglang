# L40S：其他模型卸载、DiT 常驻的单卡与双卡对照

日期：2026-09-29。测试物理 GPU 1，以及 GPU 1+2；未使用 GPU 7，未重启生产服务。

## 结论

在其他模型权重卸载、DiT 完整常驻、不开 FSDP、关闭 TeaCache 的本轮条件下，双卡 Ulysses **获得 1.555 倍加速**：单卡中位数 147.393 秒，双卡 94.808 秒，耗时降低 35.68%。三次正式计时稳定，单/双卡所有初始 latent、文本 embedding 以及最终 latent 哈希一致。

这修正了“L40S 上 SP 没有加速”的泛化说法：此前测试只能说明特定逐层卸载/FSDP 配置没有比历史单卡快，不能用于评价完整常驻的 SP。本轮证明常驻时有明显加速，但没有达到 1.8 倍，也不代表端到端服务同样加速 1.555 倍。

本轮同时改变了权重布局并关闭 TeaCache，因此不能仅通过新旧测试差值，量化卸载和缓存各自造成的损失；确定的结果是本轮受控单/双卡对照的加速比。尚未测量 TeaCache 开启时完整常驻 SP 的加速比。也没有采集本轮的完整 Nsight trace，不能把距理想两倍的差额全部归为通信。

## 测量条件

本轮专门回答 DiT 常驻时 SP 是否加速，不使用上一轮的卸载/FSDP 结果充当单卡基线。

- 复用同一历史输入、seed 42、40 个去噪步骤和动态 CFG；两边均关闭 TeaCache，避免缓存跳步影响计算量。与之前 TeaCache 开启时的 45 秒去噪结果不可直接比较。
- 文本编码器、图像编码器、VAE 完成输入准备后全部转到 CPU。每卡完整 DiT 参数约 30.5386 GiB，随后一次性搬入 GPU。
- 计时期间 DiT 不卸载、不做逐层预取、不启用 FSDP。双卡只使用 SP/Ulysses（SP=2，Ulysses=2，Ring=1），使用默认 NCCL 协议。
- 先预热一次，再测量三次完整去噪循环。每次恢复初始 latent 和调度器状态；计时前后显式 CUDA synchronize，双卡开始前 barrier，按每轮较慢 rank 的耗时统计。
- 计时包括真实去噪循环的 DiT forward、调度器、进度更新及 SP 通信，不包括模型加载、输入编码、最后的视频解码与输出。
- 这是一组请求内部的重复去噪测量，没有重做每次请求的前置编码；不是端到端服务压测。
- 初始 latent 形状为 [1,16,13,60,68]。保存各 rank 的输入/输出哈希以检查重复运行一致性。
- 容器上限 300 GiB，额外 swap 为 0；逐秒采样 GPU 和容器内存。PyTorch 峰值分配与 nvidia-smi 总显存不是同一个指标。

## 加载顺序说明

为了避免完整 DiT 与文本编码器同时在显存中造成 OOM，启动命令中的 `--dit-cpu-offload true` **仅用于初始加载到 CPU**。实验 hook 在编码结束后将其他模型转到 CPU、清理缓存、将完整 DiT 转到 GPU，并在进入任何计时前将 `server_args.dit_cpu_offload=False`。它断言所有 DiT 参数均在 CUDA，且 layerwise offload/FSDP 均关闭。此后四轮去噪中 DiT 保持常驻，不把首次搬入时间算进结果。

这不是把卸载方案冒充常驻方案，而是为隔离 DiT 性能使用的分阶段准备方式。它也不意味着现有服务仅修改一个启动参数就能永久常驻并连续处理新请求；生产化还需解决后续请求的编码器显存安排。本轮没有改动生产代码。

## 测量结果

| 配置 | 预热 s | 正式三次 s | 中位数 s | 相对单卡加速 | PyTorch 峰值分配 GiB/卡 |
|---|---:|---|---:|---:|---:|
| single | 150.323 | 146.773 / 147.393 / 147.489 | 147.393 | 1.000× | 32.591 |
| ulysses2 | 96.556 | 94.808 / 95.606 / 94.548 | 94.808 | 1.555× | 31.703 |

## 证据与复现

本地实验目录 `.local/dit-resident-benchmark/` 保存 `run.py`、`hooks/sitecustomize.py`、`hooks/resident_hook.py`、`summarize.py`、各变体 `command.json` / `server.log` / `rank*.json` / 请求结果和逐秒资源采样。实验 hook 仅通过测试进程的 PYTHONPATH 注入，不修改仓库中的服务源文件。

GPU 映射为物理 1、2 的测试容器存在且已启动、内部端口 30000 空闲时，依次运行（同名结果目录需先归档，避免覆盖）：

```sh
docker exec -u root videoedit_bench_12 python3 /sgl-workspace/sglang/.local/dit-resident-benchmark/run.py single
docker exec -u root videoedit_bench_12 python3 /sgl-workspace/sglang/.local/dit-resident-benchmark/run.py ulysses2
python3 .local/dit-resident-benchmark/summarize.py
```

脚本等待后端退出后才能运行下一组。容器沿用前一轮的模型、仓库路径挂载及镜像；`single` 将 CUDA_VISIBLE_DEVICES 设为 0，`ulysses2` 设为 0,1。请求只写本地输出，不发送回调。

## 输出校验与资源收尾

单卡与双卡各生成的视频均为 41 帧，解码后的 framemd5 清单哈希相同；详细记录在 `validation.json`。四次去噪（含预热）的最终 latent 在单卡和两个双卡 rank 之间也全部逐字节一致。

包括前置编码和最终解码在内的整次实验请求，容器内存采样峰值分别约 50.30 GiB、98.95 GiB；nvidia-smi 显存采样峰值分别为 38843 MiB/卡、37081 MiB/卡。这些是整次请求的采样值，不是表格中的去噪 PyTorch 分配峰值，也不包含模型启动阶段。

测试容器 `videoedit_bench_12` 已停止，释放 GPU 1、2。原有 5403/5402 服务健康检查均 HTTP 200、normal=true，保持原有 normal-only 状态。
