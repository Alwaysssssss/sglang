# 第三轮：encoder 编译与 GPU 后处理

## 范围与方法

按用户最终更正，**方案三（空间 tile batch）不做**；其余方案均进行了实际实验。使用 GPU6/7、torch 2.10.0+cu126、A100，tile 33×320×640，维持原来的 tile/overlap、帧顺序和颜色策略。每组计时先 warmup，编译计数发生变化的样本丢弃；不计首次加载/编译。质量直接对原生参考逐帧比较，SSIM≥0.989、MSE≤36、MAE≤6，要求零失败帧。

本轮基线是第二轮已优化的 cuDNN benchmark + channels_last_3d + decoder compile，不是未经优化的迁移版，也不是原生仓库。下面百分比均是相对本轮基线的额外收益。

## 单窗口筛选（GPU6）

4 次 warmup，6 次计时，中位数；单位秒。

| 实验 | 整体 | encode | DiT | decode | 决策 |
|---|---:|---:|---:|---:|---|
| 本轮基线 | 0.89953 | 0.20824 | 0.11097 | 0.57308 | 对照 |
| encoder compile | 0.84361 | 0.14983 | 0.11313 | 0.57360 | 候选，整体约快 6.2% |
| decoder max-autotune-no-cudagraphs | 0.89654 | 0.20826 | 0.11132 | 0.57023 | 约 0.3%，无明显收益，不采用 |
| VAE FP16、DiT BF16 | 0.90362 | 0.20922 | 0.11125 | 0.57669 | 更慢，不采用 |
| DiT compile | 0.87798 | 0.20816 | 0.08846 | 0.57380 | 有小幅速度收益，组合验证精度未过 |
| encoder + DiT compile | 0.82097 | 0.14924 | 0.09107 | 0.57407 | 原生视频 SSIM 失败，不采用 |

以上单窗口 uint8 RGB 都通过新门限，但这不能代替完整视频验收。encoder + DiT 的 native 33 帧视频对原生最低 SSIM 为 **0.985304**，33 帧均失败；最大 MSE 31.7801、MAE 3.2412，结构和中间张量检查通过。不能只看 MSE/MAE 宣称达标。未把 DiT 编译开关保留在生产入口；独立 benchmark 仍可复现单窗口实验。

## 方案四：4K A1 完整视频（53 帧）

GPU7 顺序 baseline / fusion / prefetch / prefetch / fusion / baseline；GPU6 顺序 baseline / postprocess / postprocess_prefetch / postprocess_prefetch / postprocess / baseline。每个模式 warmup 后计时，所有计时编译计数稳定。

| 实验 | 两次耗时（秒） | 相对同卡对照 | 峰值 allocated 显存 |
|---|---|---|---|
| GPU7 对照 | 124.666 / 125.374 | — | 13.61 GiB |
| 仅 GPU 空间融合 | 124.350 / 124.177 | 约 0.6% | 22.05 GiB |
| 空间融合 + pinned 异步预取 | 125.367 / 123.273 | 约 0.6% | 25.13 GiB |
| GPU6 对照 | 124.479 / 124.131 | — | 13.61 GiB |
| GPU 空间/颜色/时间融合及 uint8 转换 | 116.103 / 115.473 | 约 6.8% | 23.26 GiB |
| 全部后处理 + pinned 异步预取 | 117.231 / 115.133 | 约 6.5% | 25.13 GiB |

异步预取没有稳定额外收益，不采用；保留同步块上传与 GPU 后处理。CPU 队列仍有界，GPU 只保存当前块与未退休的重叠帧，显存不随视频长度增长。GPU6 首个后处理视频对原生最低 SSIM 0.989170，最大 MSE 2.62415、MAE 1.10208，通过但接近 SSIM 下限，完整矩阵仍是必要门槛。

## 生产候选与验收状态

已接入可选 `--compile-encoder`、`--gpu-postprocess`；默认关闭。前者也由 pipeline config 转发；后者由 sampling params 支持逐请求设置。float→uint8 在原设备完成，只传回 uint8。GPU 空间融合、颜色与时间运算保持原循环/加权顺序，没有更改 batch 或 tile 几何。

组合方案完整视频 ABBA 为基线 124.308 / 124.476 秒，候选 109.081 / 108.724 秒；各自均值（两样本中位数）124.392 / 108.903 秒。额外耗时减少 **12.45%**，吞吐提升 **14.22%**。四次计时期间编译计数稳定。

组合方案 native 33 帧入口通过：对原生最低 SSIM 0.990352、最大 MSE 17.8835、MAE 2.64199，六处中间张量和结构通过。但 A1 编码后最后一帧 SSIM **0.988996213 < 0.989**，因此严格按新门限判失败，不把四舍五入后的 0.989 当作通过；其编码前 53 帧全部通过（最低 SSIM 0.998089、最大 MSE 0.38637、MAE 0.31262）。组合矩阵在 A1 失败后停止，没有声称九配置通过。

encoder 单独编译的九配置现已完成：**649 帧编码前全部通过，结构全部通过；编码后 7/9 配置通过，2/9 配置失败**。原始帧最差 SSIM 0.996236、MSE 2.26701、MAE 0.76798，明显满足用户门限。编码后失败如下：

| 配置 | 最低 SSIM | 失败帧（从 0 开始） | 最大 MSE / MAE |
|---|---:|---|---|
| A2：4K、按块颜色校正 | 0.988992113 | 52（最后一帧） | 2.68776 / 1.12107 |
| C3：64 帧横向 4K | 0.988864267 | 63（最后一帧） | 3.00956 / 1.21592 |

A1/A3/B1/B2/B3/C1/C2 三层检查通过，包含 200 帧长视频。算法输出与编码后视频分层报告，不因原始帧通过而忽略视频门限失败。没有额外放宽 SSIM，也没有更改 CRF、参考视频或原算法几何来换取通过。

`--compile-encoder` 和 `--gpu-postprocess` 均保持默认关闭，作为可选优化路径；**本轮没有新增可宣称全部配置通过新视频门限的默认推荐组合**。已有第二轮推荐配置不变。需要使用速度更快的组合时，必须知晓上述失败记录，不能把本报告当成全配置质量通过证明。只开启 encoder 的精确完整视频速度尚未做独立 ABBA，本报告只对它报告单窗口收益；12.45% 是 encoder 与 GPU 后处理组合的完整视频收益。

8 项单元测试通过，涉及修改文件的 Ruff 检查与 `git diff --check` 通过。所有性能测量结束，九配置报告位于 `encoder_validation/summary.json`。encoder 单独开启的原生入口补测也通过：最低 SSIM 0.990316111、最大 MSE 17.777092、MAE 2.651533；中间张量、原生入口与直接入口比较及结构全部通过，记录在 `encoder_validation/native/report.json`。


## 复现与原始记录

- `scripts/benchmark_window.py`：`--compile-encoder`、`--compile-dit`、`--decoder-mode max-autotune-no-cudagraphs`、`--vae-fp16` 独立实验。
- `scripts/benchmark_video_io.py` / `io_experiment.py`：空间融合与全部后处理、异步预取对照；只在实验进程内替换函数。
- `scripts/benchmark_video_round3.py`：第二轮基线与 encoder + 可选 GPU 后处理的 ABBA。
- `scripts/run_optimization_round1.py --full --compile-decoder --compile-encoder --gpu-postprocess --gpus 7`：质量矩阵（质量运行不作为稳态计时）。
- 原始 JSON、视频、日志：`output_results/vsr/optimization_round3/`，包括 `tile_quality.json`、`io/results.json`、`post/results.json`、`combined_validation/native/report.json`、`final_validation/`、`warm_video/`。
