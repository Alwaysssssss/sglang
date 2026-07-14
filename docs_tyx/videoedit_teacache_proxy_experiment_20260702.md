# VideoEdit TeaCache Proxy 对齐实验结果

分析日期：2026-07-03

## 结论

这次重新跑的 TeaCache 参数已经和 no-TeaCache baseline 对齐：`bbox_expand_scale=1.0`、`mask_scale=1.0`、`dilate_px=0`。metadata 校验显示 20 个样例的 `bbox/crop/aligned size/fps/frames/drop_reference_frame/window count` 全部一致，因此这次可以作为有效对比。

端到端耗时上，TeaCache `20/20` 全部成功且全部快于 baseline。baseline 平均耗时 `2492.17s`，TeaCache 平均耗时 `972.04s`，按平均耗时计算 speedup 为 `2.56x`；逐样本 speedup 均值为 `2.51x`，中位数为 `2.51x`，范围为 `2.27x` 到 `2.71x`。

当前结果没有到 `3.8x`。从 trace 看，每个 window 有 `55` 条 TeaCache decision，其中 `36` 条 skipped、`19` 条 computed，skip ratio 为 `65.45%`。只按 decision 数估算，理论上限约为 `55 / 19 = 2.89x`，所以端到端 `2.56x` 是合理的。如果要达到 `3.8x`，需要更高的实际跳步比例，或者之前的 `3.8x` 使用的是不同统计口径。

proxy 决策本身是正确的：`3300` 条 decision 的 replay accuracy 为 `1.0`。但 proxy 作为“真实相邻 step 相似度”的估计器，主要对 cond branch 有较强相关性；整体 skip/compute 分离度一般，说明它更像稳定的 timestep-driven skip schedule，而不是强内容自适应判断。

## 数据路径

No-TeaCache baseline：

```text
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_bbox10/manifest.jsonl
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_bbox10/denoise_traces/
```

Aligned TeaCache：

```text
/home/tyx/workspace/sglang/outputs/teacache_sweep_bbox1_aligned/tc_default_batch_bbox1_ms1_dp0/manifest.jsonl
/home/tyx/workspace/sglang/outputs/teacache_sweep_bbox1_aligned/tc_default_batch_bbox1_ms1_dp0/summary.json
/home/tyx/workspace/sglang/outputs/teacache_sweep_bbox1_aligned/tc_default_batch_bbox1_ms1_dp0/runs/teacache_thr0p3_start5_end1/denoise_traces/
/home/tyx/workspace/sglang/outputs/teacache_sweep/teacache_trace_gpu0.jsonl
```

## 耗时结果

| 指标 | No-TeaCache | TeaCache |
| --- | ---: | ---: |
| 对齐 completed 样例数 | 20 | 20 |
| 平均 elapsed | 2492.17s | 972.04s |
| 中位 elapsed | 1967.17s | 780.82s |
| 按均值 speedup | - | 2.56x |
| 逐样本 speedup 均值 | - | 2.51x |
| 逐样本 speedup 中位数 | - | 2.51x |
| 逐样本 speedup 范围 | - | 2.27x - 2.71x |

最慢和最快的 speedup 都比较稳定，没有再出现未对齐时的异常离散值。

## Trace 跳步结果

| 指标 | 数值 |
| --- | ---: |
| matched requests | 20 |
| total windows | 60 |
| decisions | 3300 |
| skipped | 2160 |
| computed | 1140 |
| skip ratio | 0.6545 |
| replay accuracy | 1.0000 |
| per-window decisions | 55 |
| per-window skipped | 36 |
| per-window computed | 19 |
| decision-level 理论上限 | 2.89x |

每个 window 的 pattern 是固定的：

```text
cond decisions: 40, cond skipped: 28
uncond decisions: 15, uncond skipped: 8
total decisions: 55, total skipped: 36, total computed: 19
```

典型 skipped step：

```text
5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 33, 34, 36, 37, 39
```

## Proxy 与相邻 step 相似度

使用 aligned no-TeaCache denoise trace 作为真实相邻变化参考，统计 TeaCache proxy `rel_l1` 与 normal `noise_pred_*_change.relative_l1` 的关系：

| 分支 | 样本数 | Pearson | Spearman | skipped actual mean | computed actual mean | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cond | 2100 | 0.956 | 0.705 | 0.01381 | 0.01623 | 0.617 |
| uncond | 600 | 0.431 | 0.355 | 0.00991 | 0.00993 | 0.493 |
| all | 2700 | 0.943 | 0.620 | 0.01294 | 0.01483 | 0.588 |

解释：

- cond branch 上 proxy 和真实相邻变化趋势相关性强。
- uncond branch 上 skip 和 compute 的真实变化均值几乎一样，区分能力弱。
- all branch 的 AUC 只有 `0.588`，说明当前 skip/compute 的真实相似度分离不强。
- 因为每个请求的 skip pattern 基本一致，当前策略主要还是 timestep-driven，而不是明显内容自适应。

## 策略判断

`teacache_thresh=0.3/start=5/end=1.0` 是一个稳定有效的加速策略：参数已经对齐，20 个样例全部成功，端到端稳定加速约 `2.5x`，trace 决策回放完全正确。

但它不是“3.8x”级别的策略。按当前 trace 的跳步比例，decision-level 上限约 `2.89x`，端到端 `2.56x` 符合预期。若目标是更高加速，需要继续 sweep 更激进参数，例如提高 threshold，并同步做质量评估。

建议下一步扫：

```text
teacache_thresh = 0.35, 0.4, 0.45, 0.5
teacache_start_skipping = 5
teacache_end_skipping = 1.0
```

每组同时记录 speedup、skip ratio、proxy replay accuracy，并抽帧检查编辑区域质量。
