# VSR 模型卸载

VSR 的卸载默认关闭，包括通用服务在小显存 GPU 上的自动卸载逻辑。未指定开关时，保持 VAE 和 DiT 常驻 GPU。

## 开关与执行方式

| 开关 | 行为 |
| --- | --- |
| `--vae-cpu-offload` | VAE 编码、解码各自执行前加载到 GPU，完成后回 CPU |
| `--dit-cpu-offload` | 每个 tile 的 DiT 前向前加载到 GPU，完成后回 CPU |
| `--dit-layerwise-offload` | Transformer blocks 使用已有 `LayerwiseOffloadManager`，CPU 保存权重，按层预取和释放；非 block 部分常驻 GPU |
| `--dit-offload-prefetch-size` | 默认 0，即预取 1 层；小于 1 的非负值按层数比例换算，1 及以上按层数取整；实际数量不超过总层数 |

DiT 整模块卸载与按层卸载互斥，VAE 卸载可与任一组合。卸载是启动时配置，不能逐请求切换。

开启 VAE 和 DiT 整模块卸载后，每个空间 tile 执行：VAE 编码并卸载 → DiT 前向并卸载 → VAE 解码并卸载。单个时间窗口内有多个 tile，因此每个 tile 都会产生权重传输。按层卸载直接从 CPU 权重初始化，避免初始化时先把完整 DiT 放入 GPU。

独立 CLI `python -m sglang.multimodal_gen.runtime.vsr.cli restore`、其 `--via-pipeline` 路径，以及 `scripts/serve_vsr.py` 支持上述开关。通用 SGLang 服务使用同名 ServerArgs 配置。多卡 tile 模式每个 worker 独立应用同一配置，每个 worker 也独立持有 CPU 权重副本。

在已有启动命令后追加：

```bash
# VAE + DiT 整模块卸载
--vae-cpu-offload --dit-cpu-offload

# VAE 整模块 + DiT 按层卸载
--vae-cpu-offload --dit-layerwise-offload --dit-offload-prefetch-size 0
```

## 缓存、编译与清理

- 每个启用卸载的阶段使用 `finally` 清理；DiT 前向异常后释放已预取的层，VAE 异常后清理时序缓存，再移回 CPU。
- 固定 DiT 条件缓存可以同时开启，但在 DiT 卸载时清空，避免缓存继续持有 GPU 张量。卸载模式不能保留跨 tile 的条件缓存收益。
- VAE compile 可以与卸载组合；该组合显式关闭 Inductor CUDA Graph，避免捕获固定权重地址。未开启 VAE 卸载时保持原编译配置。
- 不逐层调用 `empty_cache()`；GPU 的 allocated 内存降低后，PyTorch 仍可能保留 reserved 内存供下一 tile 复用，`nvidia-smi` 不一定随之下降。
- 空间融合、时间重叠帧、输入输出队列不属于模型权重卸载范围。GPU 后处理开启时，这些 GPU 工作缓冲仍然存在。
- 卸载降低显存压力，但增加 CPU 权重驻留和 PCIe 传输；不能用于减少 CPU 内存。

## 验证方法

`test/registered/multimodal_gen/vsr/test_offload.py` 使用小型真实 diffusers Wan VAE/DiT，覆盖默认值、配置冲突、重复调用、输出比较、异常清理，以及编译 VAE 卸载后的重复执行。与请求参数、并行 tile、条件缓存和编译相关测试一起运行。

`scripts/benchmark_offload.py` 对正式权重和一个固定 tile 分进程测量 resident / whole / layerwise，每个模式执行三次。记录初始化和推理峰值 allocated、完成后的 allocated/reserved、进程峰值 CPU RSS 和每次耗时，保存最终输出张量供逐元素比较。CPU RSS 包括模型加载，不代表仅推理阶段；此脚本不含完整视频的读写、4K 融合缓冲或多卡通信。

## 2026-09-23 实测

GPU 6，A100 80GB，torch 2.10/cu126、diffusers 0.37，正式 EMA 权重，BF16，固定输入 `[1, 3, 33, 320, 640]`。本轮关闭 VAE compile、cuDNN benchmark 等加速选项，开启固定 DiT 条件缓存；whole/layerwise 同时开启 VAE 卸载，layerwise 预取 1 层。单独进程测量，每组第 1 次预热，后 2 次取中位数，样本较少，仅作卸载代价的初步测量。

| 模式 | 推理峰值 allocated / GiB | 完成后 allocated / GiB | reserved / GiB | 预热后单 tile / 秒 | 进程峰值 CPU RSS / GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| resident | 13.302 | 10.960 | 14.436 | 1.294 | 30.947 |
| whole | 9.804 | 0.009 | 12.359 | 4.692 | 30.946 |
| layerwise | 3.842 | 0.177 | 5.502 | 2.527 | 30.947 |

两种卸载模式的最终 tile 输出与常驻模式逐元素完全一致，最大绝对误差为 0。CPU 峰值 RSS 约 30.95 GiB，受加载阶段峰值主导，不能用这个数字断言卸载没有增加推理期间的 CPU 权重占用。显存峰值包括第一次预热，不包括完整视频的输入输出或 GPU 融合缓冲。

相关回归测试 22 项通过；之后新增编译组合测试，卸载专用测试共 5 项、4 个子用例通过。编译组合以同样开启编译的常驻模型作对照，重复执行和解码异常恢复后输出逐元素一致。最初将编译输出直接与 eager 输出按 1e-4 比较失败（最大误差约 0.001），同编译配置对照排除了卸载导致的误差；本轮不宣称编译与 eager 逐元素一致。

产物：`output_results/vsr/offload_validation/` 中的 JSON、输出张量及日志。完整长视频和真实多卡卸载尚未实测；多卡参数传递复用现有 worker 机制，已有并行 tile 回归测试通过。
