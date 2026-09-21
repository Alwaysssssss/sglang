# 单卡优化第二轮：编译VAE decoder

最终采用：cuDNN算法搜索 + Conv3d channels_last_3d + decoder编译。环境为torch2.10.0+cu126、Triton3.6.0、diffusers0.37.0、A100 80GB、bf16。性能固定GPU7；根据用户最新要求，后续主要使用GPU6/7。所有性能数字均排除初始化、编译和warmup。

## 使用方式

独立CLI和`--via-pipeline`均支持：

```bash
--cudnn-benchmark --channels-last-3d --compile-decoder
```

Python配置为 `WanVSRPipelineConfig(cudnn_benchmark=True, channels_last_3d=True, compile_decoder=True)`。三个选项默认关闭，保留原精确基线。

仅编译 `vae.decoder.forward`（`fullgraph=True, dynamic=False`），VAE外层时序循环、缓存清理、视频流式读写保持Python执行。权重、dtype、tile几何、overlap、颜色及编码参数不变。

## 稳态性能

真实33×320×640 tile，warmup4次后测6次，报告中位数。原始文件为 `output_results/vsr/optimization_round2/baseline.json` 和 `compiled_typed.json`。

| 配置 | tile秒 | 编码秒 | DiT秒 | 解码秒 |
| --- | ---: | ---: | ---: | ---: |
| 上一轮组合，本轮GPU7复测 | 1.13046 | 0.20764 | 0.11152 | 0.80472 |
| 组合 + 编译decoder | 0.90245 | 0.20805 | 0.11146 | 0.57642 |

相对上一轮，tile耗时再减少20.17%，吞吐提升约25.27%；decoder耗时减少28.37%。本进程峰值allocated显存13.61GiB（含warmup临时工作区），不是整卡占用。

完整A1（53帧3840×2160），同一模型按上一轮组合→编译→编译→上一轮组合顺序交错运行，每次先warmup4次，关闭验收dump。计时包含视频读取/缩放、模型、融合、颜色和编码：

| 配置 | 两次实测秒 | 中位数秒 |
| --- | --- | ---: |
| 上一轮组合 | 151.015 / 151.404 | 151.210 |
| 组合 + decoder编译 | 125.867 / 126.774 | 126.321 |

完整视频耗时再减少16.46%，吞吐提升19.70%（1.1970倍）。四次计时前后编译计数器均未变化。固定tile形成3个图，无graph break。编译组两次MP4 SHA256相同，基线两次也相同；这是同进程重复验证，不代表跨进程逐位一致。两次重复不作为统计置信区间。

结果在 `warm_video/results.json`。对比第一轮初始精确基线171.823秒，累计耗时约减少26.5%；这个累计值跨轮次，直接交错对照的结论以上表为准。

## 精度口径与结果

用户明确允许适当放宽精度对齐，因此最终不要求优化后的不同进程或不同入口逐像素一致。

- 九配置A1–A3、B1–B3、C1–C3共649帧：**仍沿用原有冻结门限**，逐帧、mp4和结构三层全部通过，无失败帧。
- 六处中间张量：原门限不变，输入/window误差0；latent rel_mean=0.0003533（限0.0019），velocity=0.0017434（限0.0067），decoded/spatial_fused=0.0144185（限0.048），全部通过。
- 新增native/CLI 33帧对照：按用户允许的较宽口径，SSIM≥0.99、MSE≤25、MAE≤3.0，失败帧比例仍为0；帧数、分辨率、fps和顺序检查不放宽。实测最差SSIM0.991088、MSE15.3048、MAE2.5015，通过。
- 同一native输出还与原生参考33帧视频按上述口径复核：最差SSIM0.990372、MSE17.2789、MAE2.6148，满足门限。

放宽的是此前给新增native小样本借用的严格B1数值门限，并未修改九配置的冻结门限，也没有丢弃失败帧。旧严格结果保存为 `native/report_strict_initial.json`；最终结果为 `native/report.json` 和总报告 `report.json`。

## 编译兼容与诊断结论

直接编译diffusers原decoder会在 `Tensor == "Rep"` 处失败。`runtime/vsr/compile.py`仅替换当前decoder内upsample3d模块的forward，显式区分None/字符串/张量缓存，保留计算及缓存写入顺序。没有修改安装包，也没有全局替换diffusers类。

缓存单元测试覆盖None→Rep→Tensor、单帧/多帧块、下一视频清空缓存和无缓存输入，输出及缓存逐位一致。最终VSR测试共8项通过。

默认Inductor融合会省略部分BF16中间截断；cuDNN benchmark也可能因搜索结果不同而在不同进程产生不同舍入。诊断中关闭搜索后native/CLI输出可逐像素一致，但tile耗时回到约1.026秒。保留BF16中间转换的实验将固定latent下的decoder误差从0.00663降到0.00309，但单独开启并未解决跨进程严格门限问题。

根据用户允许的精度/速度取舍，最终保留已完成九配置验收的默认融合高速版本；精确对齐实验结果只作诊断记录，不混入最终性能与验收报告。`precision/`、`native_precise_failed/`、`native_heuristic/`、`production_heuristic_partial/`、`warm_video_heuristic_partial/`均非最终候选。临时模型探针已删除。

静态形状编译目前只验证原tile设置；其他tile形状会触发新编译，不承诺任意动态形状都同样加速。换torch/diffusers版本需重新验证。

## 复现

脚本位于本目录 `scripts/`。从仓库根目录，用 `output_results/vsr/migration_env210/bin/python`，设置 `OMP_NUM_THREADS=8`；编译并发可设 `TORCHINDUCTOR_COMPILE_THREADS=4`。

- GPU7微基准：`benchmark_window.py --implementation candidate --cudnn-benchmark --channels-last-3d --compile-decoder --warmup 4 --report <json> --dump-output <pt>`；不加compile参数为上一轮基线。
- GPU7完整视频：`benchmark_video_compile.py`。报告的`baseline`代表上一轮组合，`both`代表组合加编译。
- 质量验收：`run_optimization_round1.py --full --compile-decoder --gpus 6 --output-dir <新目录>`，复用冻结配置。GPU7空闲时可用`--gpus 6,7`，不要与GPU7性能测试并发争用。
- 张量/native：`validate_optimized_native.py --gpu 6 --compile-decoder --output-dir <新目录>`；等完整矩阵结束后运行。`--compare-only`仅复核已有产物，不启动GPU推理、不覆盖原执行命令。
- 汇总：`summarize_optimization_round1.py --output-dir output_results/vsr/optimization_round2`。缺少报告、门限失败或计时阶段新增编译均不能通过。
- 数值诊断：`diagnose_decoder_precision.py`，固定latent比较eager、默认融合和保留精度转换三种decoder。
