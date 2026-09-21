# 单卡优化第一轮：cuDNN算法搜索与Conv3d布局

2026-09-20。环境：torch2.10.0+cu126、diffusers0.37.0、A100 80GB、bf16，原权重、tile、overlap、颜色和编码设置不变。

## 性能口径

按用户要求，所有性能对比均排除模型加载、初始化和warmup。GPU7独占本轮性能测试；GPU2、3、6用于独立质量验收。各GPU均单卡推理，多卡仅缩短测试总时间。

微基准使用真实33×320×640 tile，每组独立进程，预热2次、测量6次，中位数如下。原始记录在 `output_results/vsr/optimization_round1/{baseline,benchmark,layout,both}.json`。

| 配置 | tile秒 | 编码秒 | DiT秒 | 解码秒 | 较基线耗时减少 | 本进程峰值allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 | 1.3131 | 0.2624 | 0.1213 | 0.9229 | — | 12.99 |
| cuDNN benchmark | 1.1910 | 0.2300 | 0.1217 | 0.8332 | 9.30% | 18.88 |
| Conv3d channels_last_3d | 1.2564 | 0.2405 | 0.1115 | 0.8980 | 4.32% | 12.78 |
| 两者组合 | 1.1295 | 0.2069 | 0.1116 | 0.8042 | 13.98% | 15.01 |

组合的tile吞吐提升约16.25%（1.1625倍），与“耗时减少13.98%”分母不同。峰值显存包含warmup期间临时工作区；不是GPU整卡占用。cuDNN选核随硬件/版本/负载可能变化，不能将本机收益直接推广到其他环境。

完整A1视频采用同一已加载模型、基线→组合→组合→基线顺序；每次切换布局/算法设置后先预热2个真实tile，再计时完整53帧3840×2160流式处理。关闭验收dump，包含读入、缩放、融合、颜色和编码。脚本 `scripts/benchmark_video_warm.py`，结果 `output_results/vsr/optimization_round1/warm_video/results.json`。两次基线为171.890/171.757秒（中位数171.823秒），两次组合优化为151.089/150.522秒（中位数150.806秒）。完整视频耗时减少12.23%，吞吐提升13.94%（1.1394倍）。这是两次重复的本机实测，不是置信区间。

## 接入方式

独立CLI和`--via-pipeline`均支持在原命令追加：

```bash
--cudnn-benchmark --channels-last-3d
```

Python pipeline配置：

```python
WanVSRPipelineConfig(cudnn_benchmark=True, channels_last_3d=True)
```

两个开关默认False。保留精确基线模式；组合优化有数值变化，按冻结质量门限验收，不声称逐位一致。布局仅转换VAE的Conv3d权重。cuDNN benchmark仅在模型调用期间设置，异常时也恢复原设置。cuDNN设置是进程全局状态，与现有VAE因果缓存一样要求worker内串行执行，不适合多个线程并发调用不同配置的模型。

## 验证

首轮三个候选B1、组合C1全部通过冻结逐帧门限。组合B1最差SSIM0.998303/MSE0.955525/MAE0.477789；C1为0.997508/1.132348/0.544282。原始数据在 `output_results/vsr/optimization_round1/*_B1/frames.json` 和 `both_C1/frames.json`。

正式开关九配置（A1–A3、B1–B3、C1–C3）共649帧，逐帧、mp4与结构三层全部通过，无失败帧；结果目录为 `production/`，汇总为 `output_results/vsr/optimization_round1/report.json`。六处中间张量均通过：输入/window误差0，latent rel_mean=0.0003531（限0.0019），velocity=0.0017269（限0.0067），decoded/spatial_fused=0.0142645（限0.048）。native入口与独立CLI的33帧视频像素完全一致，结构检查通过；结果目录为 `native/`。单元测试7项通过，包含native CLI参数传递和推理失败后恢复cuDNN全局设置。

## 复现

从仓库根目录运行，Python使用 `output_results/vsr/migration_env210/bin/python`，`OMP_NUM_THREADS=8`。

- `scripts/benchmark_window.py --implementation candidate --report <json> [--cudnn-benchmark] [--channels-last-3d] [--dump-output <pt>]`：设`CUDA_VISIBLE_DEVICES=7`，四组依次运行。
- `scripts/run_optimization_round1.py --full`：GPU2/3/6分别串行执行三组配置；比较原生swiftvr基准，冻结帧门限与mp4/结构门限不变。
- `scripts/benchmark_video_warm.py`：设`CUDA_VISIBLE_DEVICES=7`，预热后完整A1交错对照。
- `scripts/validate_optimized_native.py`：等待本轮GPU2的C1结束后验证六处张量及native入口，不与GPU7性能测试争用设备。

脚本和日志中的验收进程总耗时不是性能指标。下一轮优先研究VAE decoder局部编译；本轮不修改算法、tile几何或验收门限。

最终汇总可运行 `scripts/summarize_optimization_round1.py`：缺失任何结果或任一门限失败都会报错。Ruff和git diff --check均通过。
