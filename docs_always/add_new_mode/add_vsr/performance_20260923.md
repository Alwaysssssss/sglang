# VSR 全部加速性能测试（2026-09-23）

当前单卡、双卡、三卡相对原始实现的整视频稳态加速分别为 **1.64×、2.88×、3.82×**。四卡测试按用户要求停止，不报告四卡性能。

## 最终结果

| 实现 | 物理 GPU | 三次有效耗时（秒） | 中位数（秒） | 相对原始加速 | 耗时减少 |
| --- | --- | --- | ---: | ---: | ---: |
| 原始实现 | 3 | 172.159 / 172.248 / 172.799 | 172.248 | 1.000× | 0.00% |
| 当前单卡 | 3 | 105.120 / 105.089 / 106.011 | 105.120 | 1.639× | 38.97% |
| 当前双卡 | 6,7 | 59.583 / 59.909 / 59.948 | 59.909 | 2.875× | 65.22% |
| 当前三卡 | 3,6,7 | 45.046 / 45.087 / 45.545 | 45.087 | 3.820× | 73.82% |

## 测试条件

- A100-SXM4-80GB；输入 `../vsr/input/input.mp4`，53 帧，输出高 3840 × 宽 2160。
- 相同权重、BF16、tile 33×320×640、时间重叠 5、空间重叠 32、global 颜色参考（64 样本）、CRF 5、读/写队列 2/4。
- 当前实现开启 cuDNN benchmark、channels-last-3d、VAE encoder compile、VAE decoder compile、decoder implicit padding、固定 DiT 条件缓存、GPU 后处理。多卡额外启用空间 tile 并行，每卡一份完整模型。
- 每组在同一进程内完整视频预热一次，随后测量三次取中位数。计时包含视频解码、模型推理、融合、颜色校正和编码，不含模型加载与预热。全部有效样本的各进程编译计数在计时前后不变，无编译样本被剔除。
- 原始实现使用 SwiftVR 的 `infer.models.stage3.Stage3Pipeline` 与 `infer.stream.stream_restore`；当前实现使用 SGLang VSR，原始侧未开启新增优化。
- 两侧均为 torch 2.10.0+cu126；原始环境 diffusers 0.36.0、Python 3.10，当前环境 diffusers 0.37.0、Python 3.11。因此这是各自现有运行环境下的端到端比较，不是严格隔离所有依赖差异的代码微基准。

## 调度与排除记录

最初 GPU 2 跑原始基线，GPU 3 跑当前单卡，GPU 6/7 跑当前双卡，三组并行。GPU 2 后续出现外部计算进程，原始基线耗时为 170.330 / 200.304 / 347.207 秒；该组保留在 `reference/`，不进入最终加速比。

用户随后将四卡测试改为三卡。已停止本次四卡进程及其子进程，保留 `quad/` 的不完整记录；使用 GPU 3/6/7 完成三卡测试，随后独立使用 GPU 3 补测原始基线（`reference_gpu3/`）。最终表格使用该补测基线。

单卡与双卡计时期间共享主机 CPU、内存和存储；三卡与补测基线顺序运行。GPU 7 有约 12 GB 外部驻留显存，测试开始时利用率为 0%。这些条件已保留在各组 JSON 中。结束后 GPU 3/6 显存回落至约 4 MiB，GPU 7 回落至测试前约 12 GB，本次推理进程已退出。

## 复现与范围

```bash
bash docs_always/add_new_mode/add_vsr/scripts/run_benchmark_all.sh \
  output_results/vsr/all_optimizations_repeat
```

该脚本按最终调度先并行测单卡/双卡，再测三卡，最后测 GPU 3 原始基线。命令与计时说明见 [cli.md](cli.md)，脚本见 [benchmark_all.py](scripts/benchmark_all.py)。

原始记录：`output_results/vsr/all_optimizations_20260923/{single,dual,triple,reference_gpu3}/results.json`；汇总：`performance_summary.json`。同目录保留各组日志、预热视频和每次输出视频。

本轮验证所有有效样本均输出 53 帧，且计时期间没有新增编译；未另行执行逐帧 SSIM/MSE/MAE 验收，性能测试完成不等于新增质量验收通过。结论仅覆盖本次输入、分辨率和 tile 配置。
