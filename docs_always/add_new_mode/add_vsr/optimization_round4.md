# 第四轮单卡优化（2026-09-21）

用户授权五个方向：decoder 搬运与融合、CUDA Graph、DiT 固定条件缓存、扩大 VAE 编译范围、GPU 后处理融合。仍不做空间 tile batch。仅 GPU6/7，torch 2.10/cu126；warmup 后计时，编译/捕获不计入性能；按 SSIM≥0.989、MSE≤36、MAE≤6 检查，记录第三轮已存在的编码后临界失败，不将其隐去。

## 基线与可证伪假设

命令：`CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 output_results/vsr/migration_env210/bin/python docs_always/add_new_mode/add_vsr/scripts/benchmark_window.py --implementation candidate --cudnn-benchmark --channels-last-3d --compile-decoder --compile-encoder --warmup 4 --profile --report output_results/vsr/optimization_round4/profile.json --dump-output output_results/vsr/optimization_round4/baseline.pt`。

已运行，6 次暖机后中位数 0.842540 秒，encode 0.149312、DiT 0.113071、decode 0.573116。真实输入 `[1,3,33,320,640]`，计时没有重编译。速度判定用独立、未开启 profiler 的计时段。Profiler 的 CPU 聚合项与 GPU kernel 可能重叠，不能直接相加；其中 Command Buffer Full 不当作正常推理的 CPU 瓶颈证据。

1. 若时序输出的反复 cat 是重复拷贝来源，改成保存块、最后一次 cat 应降低 decode 时间，且输出逐元素一致。进一步缓存预分配须证明额外收益。
2. 若固定窗口存在发射空隙，捕获整个窗口的 CUDA Graph 应降低墙钟时间。输入缓冲固定，返回输出独立副本，重复请求必须重新覆盖输入。
3. 固定 t=1000 与零文本可缓存 condition_embedder 和 cross-attention K/V；模型/权重/条件不变时应保持输出。当前文本长度为 1，可另试单 token cross-attention 常量输出；数学等价，但不同 GEMM 形状可能改变舍入，必须单独验证。
4. 将保持缓存语义的 decode 时序循环纳入编译应降低调用/拼接开销；需要控制编译时间与内存，不能把冷启动当性能收益。
5. 编译 GPU 颜色归一化和 uint8 转换可减少中间张量及显存扫描；颜色统计须保持原来的 unbiased std 语义。

所有修改先放在 `scripts/round4_experiments.py` 与独立 benchmark，确认速度和精度后才进入生产代码。原始记录：`output_results/vsr/optimization_round4/`。

## 单窗口实验结果

4 次 warmup，6 次计时取中位数；不同卡的微小差异需要完整视频交错对照确认。

| 实验 | 单窗口秒 | decoder 秒 | 说明 |
|---|---:|---:|---|
| profile | 0.842540 | 0.573116 | GPU6 基线；profile 在独立计时段之后 |
| cat_once | 0.847073 | 0.576478 | GPU7，无明显收益，输出逐元素相同 |
| condition | 0.843204 | 0.573931 | GPU6，仅缓存固定条件/KV，无明显整体收益，输出相同 |
| cross_constant | 0.829956 | 0.573263 | GPU6，DiT 约 0.101 秒，数值有变化 |
| graph | 0.836790 | — | GPU6，不到 1% 的收益，输出相同 |
| outer_compile | 0.869703 | 0.597842 | GPU7，反而变慢，不采用 |
| implicit_pad | 0.812212 | 0.543158 | GPU6，decoder 空间 padding 交给卷积 |
| combined | 0.814068 | 0.546030 | GPU7，padding + 单 token 常量化，需同卡端到端确认 |

空间 padding 在真实窗口上通过原生质量门限；有界缓存、卷积步长及时间因果语义另有 CPU 测试。已接入默认关闭的 `--decoder-implicit-padding`，支持直接 CLI 与原生 pipeline。具体实现仅更改单个 VAE decoder 实例，不修改 diffusers 全局类。

GPU 后处理微基准（33 帧 3840×2160，仅颜色归一化与 uint8 转换，不含模型/读写）：约 44.7 ms → 12.3 ms。uint8 最大差 1，像素差异比例约 3.59e-6，MSE/MAE 约 3.59e-6。整片只有少数块调用该段，不能把这个约 3.6 倍加速当作完整视频加速。融合实现及单 token 常量化目前仅作为实验路径，不是生产默认。

扩大编译范围已实际编译并完成计时，没有把编译等待误计为性能下降；其稳态 decoder 约 0.598 秒，比原 0.573 秒更慢。CUDA Graph 已完成捕获和重放，其不足 1% 的初步收益不支持继续引入固定缓冲管理复杂度。


## 完整视频与精度

GPU6，53 帧 3840×2160，按 baseline / padding / combined / combined / padding / baseline 交错计时。baseline 为第三轮 encoder compile + decoder compile + GPU 后处理；padding 仅增加 decoder 隐式空间 padding；combined 再加单 token 常量化和后处理融合。编译计数发生变化的第一个 combined 视频样本（109.54 秒）已丢弃，未混入稳态统计。

单窗口候选均已通过原生门限；cat_once、普通固定条件缓存和 CUDA Graph 与本轮基线逐元素相同。padding、扩大编译范围和单 token 常量化均有数值变化，单窗口通过不能代替编码后视频。

GPU7 单独加载的 padding A1 验证通过；但 GPU6 同配置计时产物 A1 最后一帧 SSIM=0.988976124，combined=0.988985334，均低于 0.989。两组 MSE/MAE 分别至多 2.67122/1.11683、2.66743/1.11622，结构通过。保留跨进程/卡差异，不选择性只引用通过产物，不据此放宽门限。生产开关保持默认关闭。

交错计时已完成：

| 配置 | 两次稳态秒 | 均值秒 | 相对本轮基线耗时减少 |
|---|---|---:|---:|
| baseline | 109.344 / 109.647 | 109.496 | 0.00% |
| padding | 106.084 / 105.648 | 105.866 | 3.31% |
| combined | 104.694 / 104.524 | 104.609 | 4.46% |

以上包含读写、颜色和编码，全部计时前后编译计数稳定。不能与原始算法的单窗口基准直接混算；这是相对第三轮优化候选的额外收益。


## 使用与复现

新增生产开关为 `--decoder-implicit-padding` 和 `--cache-dit-condition`，均默认关闭；对应 pipeline config 字段 `decoder_implicit_padding=True`、`cache_dit_condition=True`。在本轮候选基础上复现：

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 \
  output_results/vsr/migration_env210/bin/python -m sglang.multimodal_gen.runtime.vsr.cli restore \
  --checkpoint_dir /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300 \
  --wan_root /mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers \
  --input ../vsr/input/input.mp4 --output output_results/vsr/round4_example.mp4 \
  --target_resolution 3840x2160 --cudnn-benchmark --channels-last-3d \
  --compile-decoder --compile-encoder --gpu-postprocess --decoder-implicit-padding
```

上述 CLI 单次启动包含加载/编译，不能用它的总墙钟时间复现稳态收益。计时用 `scripts/benchmark_round4_video.py`；profile 与单窗口实验用 `scripts/benchmark_round4.py --experiment ...`；后处理独立测量用 `scripts/benchmark_round4_post.py`。计时产物的视频检查用 `scripts/compare_round4_video.py`。

九配置：`scripts/run_optimization_round1.py --full --compile-decoder --compile-encoder --decoder-implicit-padding --gpu-postprocess --gpus 6,7 --output-dir output_results/vsr/optimization_round4/padding_validation`。本次为了避免与 GPU6 计时争用，先分配 GPU7 的 A1/A2/A3/B1/B2，再在计时结束后安排 GPU6 的 B3/C1/C2/C3；最终汇总逐目录生成，不依赖并行协调器最后写入的单组 marker。

原生入口：`scripts/validate_optimized_native.py --gpu 6 --compile-decoder --compile-encoder --decoder-implicit-padding --gpu-postprocess --skip-matrix-wait --output-dir output_results/vsr/optimization_round4/padding_validation`。

所有诊断代码在 VSR 文档目录的 scripts 中；依赖源码未修改，没有开启 tile batch。10 项单元测试以及涉及变更的 Ruff 检查通过。


### warmup 口径再确认

按用户本轮追问，所有性能数字排除首次初始化、编译和图捕获。单窗口先 warmup 4 次再测 6 次；完整视频切换模式后 warmup 4 个真实窗口，并检查每个视频前后编译计数。新编译样本丢弃后重跑。CUDA Graph 捕获前另外 warmup 3 次，捕获之后才进入正常预热和计时。后处理微基准每组先 warmup 3 次，再测 6 次。质量验收日志中的冷启动耗时不用于性能结论。

## 九配置结果

Padding + encoder/decoder 编译 + GPU 后处理，649 帧编码前全部通过，结构全部通过；编码后 7/9 配置通过。缓存开关不在此矩阵中，另见下述独立验证。

| 配置 | 编码前最低 SSIM | 编码后最低 SSIM | 编码后通过 |
|---|---:|---:|---|
| A1 | 0.998090634 | 0.989005397 | True |
| A2 | 0.998030190 | 0.988974942 | False |
| A3 | 0.998097075 | 0.989073045 | True |
| B1 | 0.998195300 | 0.990241668 | True |
| B2 | 0.997615309 | 0.989710311 | True |
| B3 | 0.996304222 | 0.990083333 | True |
| C1 | 0.997337709 | 0.989434154 | True |
| C2 | 0.997297009 | 0.989606050 | True |
| C3 | 0.997172478 | 0.988860529 | False |

编码前最差 SSIM=0.996304222、MSE=2.261953、MAE=0.764381。A2 第 52 帧、C3 第 63 帧的编码后 SSIM 失败，其余帧通过；这两项上一轮 encoder 候选也有临界失败。完整数据在 `padding_validation/summary.json`。

## 保持原 GEMM 形状的固定条件缓存

`--cache-dit-condition` 复用固定时间/文本条件及单 token cross-attention 输出。首次使用原始完整 query 形状计算，以避免将输出投影从矩阵改成单行时的舍入变化。缓存每模块仅一项，形状、dtype、设备、条件 tensor 或参数版本变化会重新计算；训练、有梯度、非单 token/mask/image 条件或无法跟踪版本的输入会走原始路径。它只适用于 VSRRestorer 固定 t=1000 的私有 DiT，不是通用扩散缓存。

实际生产 helper 已在原图、左右翻转、缩小振幅、再次原图四次调用中验证：所有输出与未缓存基线逐元素相同，最大差 0。GPU7 暖机后 ABBA：基线组中位数均值 0.819049 秒，缓存组 0.806503 秒，额外耗时减少 1.53%。每组 warmup 3 次，再计时 6 次；事先还做模型 warmup 4 次。记录在 `constant_production.json`。

本轮 104.609 秒的完整视频 combined 实验使用的是早期单行常量化与后处理融合，**不能当作最终保持形状缓存开关的完整视频结果**。最终缓存开关只报告上述单窗口交错收益，尚未做其独立完整视频 ABBA/九配置矩阵；两种缓存实验都有明确记录，不能混用。

## 原生入口最终验收

- padding 开关：六处中间张量、原生入口与直接入口比较、结构和对原生视频检查均通过。对原生最低 SSIM 0.990001392、最大 MSE 18.198824、MAE 2.648587。记录：`padding_validation/native/report.json`。
- padding + 固定条件缓存开关：六处中间张量、原生入口与直接入口比较、结构和对原生视频检查均通过。对原生最低 SSIM 0.990001392、最大 MSE 18.198824、MAE 2.648587。记录：`cached_validation/native/report.json`。

本轮所有推理任务已结束；10 项单元测试、Ruff 及 `git diff --check` 通过。新增开关默认关闭，不将通过的单窗口/native 样例扩大解释为所有编码后视频均达标。
