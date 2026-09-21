# VSR 空间 tile 多卡并行（第一轮）

## 实现

每张卡一个进程、一份完整 VAE + DiT；owner 卡也参与推理并负责原顺序融合。
新增 `runtime/vsr/parallel.py`，通过 `ParallelVSRRestorer.restore_windows()` 暴露有序、受限的 tile iterator。
`blending.py` 的 padding、裁剪、mask、FP32 累加及归一化语义保持不变；`stream.py` 仍按原顺序做颜色校正和时间融合。

第一版每批至多一个 tile/卡，每张卡的模型串行调用，不做空间 batch。
输入转换为模型精度的连续张量，再通过 CUDA IPC 共享句柄；worker 将输入复制到本卡，结果复制回 owner 后才确认释放 IPC 存储。
复制为 GPU 间传输，不将输出转成 CPU/uint8 后再融合。本版尚未实现跨批流水线或专用通信 stream。

直接 CLI 新增 `--tile-devices cuda:0 cuda:1`，编号相对于 `CUDA_VISIBLE_DEVICES`。
需要至少两个不同的 CUDA 设备。单卡不传此选项。
目前与 `--via-pipeline` 组合会明确拒绝，避免静默退化为单卡。原生 server 的多卡接入现已实现，见 `server_api.md`；该直接 CLI 的 `--via-pipeline` 组合限制仍保留。

示例（加载、编译及首次推理不代表稳态性能）：

```bash
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 \
PYTHONPATH=python output_results/vsr/migration_env210/bin/python \
  -m sglang.multimodal_gen.runtime.vsr.cli restore \
  --checkpoint_dir /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300 \
  --wan_root /mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers \
  --input ../vsr/input/input.mp4 --output output_results/vsr/parallel.mp4 \
  --target_resolution 3840x2160 --tile-devices cuda:0 cuda:1 \
  --cudnn-benchmark --channels-last-3d --compile-encoder --compile-decoder \
  --decoder-implicit-padding --gpu-postprocess
```

## 验证方法

`scripts/benchmark_parallel.py` 在同一模型生命周期内运行单卡、双卡、双卡、单卡。
两卡使用同一真实 tile 预热 6 次；排除加载、编译和预热，计时前后读取每个进程的 Dynamo stats，有变化即拒绝该计时样本。
计时包含完整视频解码、模型、融合、颜色校正和输出编码，不开启 dump/profiler。

同卡模型选项一致，启用 encoder/decoder compile、channels-last-3d、cuDNN benchmark、decoder implicit padding、GPU 后处理；未启用固定条件缓存或早期近似常量化。
质量另跑原始 uint8 RGB dump 和 MP4 对照，用户最新阈值 SSIM≥0.985、MSE≤36、MAE≤6，失败帧容忍 0。

第一轮集成发现：有序 iterator 在恰好消费最后一次 yield 后被关闭，GeneratorExit 被误判成推理失败并关闭整个 pool。已修复，增加连续两次调用的回归测试。

## 稳态性能结果

GPU6/GPU7，A100 80GB，torch 2.10.0+cu126。A1 视频 53 帧，输出 H=3840、W=2160，共两个时间窗口、每窗口 56 个空间 tile。

| 模式 | 第一次 | 第二次 | 平均 |
|---|---:|---:|---:|
| 单卡 GPU6 | 106.531 s | 106.045 s | 106.288 s |
| 双卡 GPU6+GPU7 | 60.354 s | 60.491 s | 60.422 s |

双卡相对**本轮同配置单卡**加速 1.759 倍，耗时减少 43.15%。这不是对原始 SwiftVR 的速度比较。
四次计时前后，两进程均保持 calls_captured=2386、unique_graphs=6，无新增编译。
GPU7 当时存在约 13 GB 其他进程的驻留显存；开始前观察到该卡利用率为 0。

结果：`output_results/vsr/parallel_round1/results.json`、`performance_summary.json`。
测试：parallel tile 与 request parameters 共 15 项通过，覆盖小输入 padding、多时间窗口、重复调用、提前关闭 iterator、奇数尾批、有序输出和 IPC 消费确认协议。

## MP4 精度检查

完整 53 帧及结构检查：

| 对照 | SSIM 最低 | MSE 最大 | MAE 最大 | 失败帧 |
|---|---:|---:|---:|---|
| 本轮双卡 vs 本轮单卡 | 0.9903482354 | 2.3471875 | 1.0337926 | 无 |
| 本轮双卡 vs SwiftVR 原始 | 0.9889917495 | 2.6693528 | 1.1164147 | 无 |
| 本轮单卡 vs SwiftVR 原始 | 0.9889910038 | 2.6709003 | 1.1163722 | 无 |

**按用户最新要求 SSIM≥0.985、MSE≤36、MAE≤6，以上三组 MP4 对照均全帧通过。**
原先 0.989 阈值下，单卡与双卡对原始算法都只有最后一帧临界失败。用户随后明确调整到 0.985，报告依据已测逐帧指标重新判定，未改动模型、视频或测量值。
报告：`quality.json`、`single_original_quality.json`。

本轮验证范围为 GPU6+GPU7 双卡及 A1；四卡伸缩性和更多视频/shape尚未验收；后续原生server双卡验收记录见 `server_api.md`。

## 原始 RGB 与重复请求

独立通过新 CLI 双卡入口导出编码前 uint8 RGB（`quality_dump/retired`），与原始 SwiftVR A1 dump 比较：53/53 帧通过，SSIM 最低 0.9980903975，MSE 最大 0.3865180，MAE 最大 0.3125267。报告 `raw_original_quality.json`。该 dump 推理含冷启动，仅用于质量检查，不用于性能结论。

双卡对原始视频的分辨率、帧数、FPS、帧序全部通过。两次单卡输出文件 SHA256 相同，两次双卡输出 SHA256 也相同，验证重复请求未污染因果缓存：

- 单卡：`d35a8a66d26f39695b1c4b5c502f87528caf7501f6679ffa9a3bcaad6d8fa15c`
- 双卡：`50e3ade4a883045d237a8c76422a70250db16812e44fe67c22189b1294ae8d9f`

## 复现稳态性能

```bash
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 \
PYTHONPATH=python output_results/vsr/migration_env210/bin/python \
  docs_always/add_new_mode/add_vsr/scripts/benchmark_parallel.py
```

CLI 质量导出使用已有 `verify.dump_candidate`，需提供 `--vsr-repo ../vsr`；所有性能结论来自独立暖机 benchmark。
所有本轮 GPU 推理任务已结束，worker 正常退出，GPU6/7 显存恢复到测试前水平。
