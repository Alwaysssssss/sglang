# VideoEdit TeaCache Residual Validation Fixed 结果分析

日期：2026-07-08

## 结论

本次重跑目录是有效结果，可以用于判断 `teacache_thresh=0.3 / start=5 / end=1.0` 的 residual 风险。

和之前 TeaCache 最佳参数实验对照，这次 residual validation 的 skip/compute pattern 完全一致：20 个视频、60 个 window、3300 条 trace，每个 window 固定 `55` 条记录，其中 `36` 条 would skip、`19` 条 would compute，整体 skip ratio 为 `65.45%`。这说明本次已经跑到正确的 TeaCache 决策路径，上次 `start=5.0` 被误解释为比例的问题已经规避。

从 residual error 看，当前参数整体是合理的：`would_skip` 的 residual relative L1 明显低于非边界 `would_compute`。但它不是非常保守，少数 skipped step 有较高尾部误差，主要集中在 `1544866964` 和 `85522644`。因此当前参数可以继续作为稳定加速参数使用；如果继续提高 threshold 追求更高速度，必须同步抽查这些高残差样例的画质。

## 对照依据

之前 aligned TeaCache sweep 的结论见：

```text
/home/tyx/workspace/sglang/docs_tyx/videoedit_teacache_proxy_experiment_20260702.md
```

当时有效参数为：

| 参数 | 数值 |
| --- | ---: |
| teacache_thresh | 0.3 |
| teacache_start_skipping | 5 |
| teacache_end_skipping | 1.0 |
| bbox_expand_scale | 1.0 |
| mask_scale | 1.0 |
| dilate_px | 0 |

之前 sweep 的核心结果：

| 指标 | 数值 |
| --- | ---: |
| completed | 20 / 20 |
| TeaCache mean elapsed | 972.04s |
| No-TeaCache baseline mean elapsed | 2492.17s |
| speedup by mean elapsed | 2.56x |
| decisions | 3300 |
| skipped | 2160 |
| computed | 1140 |
| skip ratio | 65.45% |
| replay accuracy | 1.0 |

## 本次输入与输出

用户本次运行命令：

```bash
python3 scripts/batch_videoedit_repair.py \
  --bbox-expand-scale 1.0 \
  --mask-scale 1.0 \
  --dilate-px 0 \
  --output-dir outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0_fixed \
  --teacache-residual-trace-dir outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0_fixed/teacache_residual_traces \
  --teacache-thresh 0.3 \
  --teacache-start-skipping 5 \
  --teacache-end-skipping 1.0 \
  --force
```

本次结果目录：

```text
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0_fixed/manifest.jsonl
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0_fixed/teacache_residual_traces/
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0_fixed/denoise_traces/
```

## 完成情况

`manifest.jsonl` 一共有 `40` 行，不应直接按全量行统计成功率：

| 统计口径 | completed | failed | 说明 |
| --- | ---: | ---: | --- |
| manifest 全量行 | 21 | 19 | 第一轮跑到一半 server connection refused |
| 每个 id 取 latest | 20 | 0 | 第二轮 20 个视频全部完成 |

第一轮中 `10012741` 完成，其余 19 个任务因为服务断连失败；后续重跑后，每个 id 的最新记录均为 `completed`。

按每个 id 最新完成记录统计：

| 指标 | 数值 |
| --- | ---: |
| completed ids | 20 |
| mean elapsed | 2521.72s |
| median elapsed | 1962.16s |
| min elapsed | 650.76s |
| max elapsed | 8038.98s |

注意：本次是 no-TeaCache residual validation，服务端仍会真实计算所有 transformer forward，只是模拟 TeaCache would skip/would compute 并记录 residual error。因此本次 elapsed 不能和 TeaCache 加速耗时直接比较。

## Trace 有效性

本次 residual trace 的关键字段已经正确：

| 字段 | 数值 |
| --- | ---: |
| event | teacache_residual_validation |
| threshold | 0.3 |
| start_skipping | 10 |
| end_skipping | 80 |
| num_inference_steps | 40 |
| trace records | 3300 |
| trace files | 20 |
| windows | 60 |

这里 `start_skipping=10` 是正确结果，因为 CFG 下 raw forward index 是 denoise step 的 2 倍；用户传入整数 `5` 后，服务端边界为前 5 个 denoise step，即 raw index `< 10`。

本次已经出现 `would_skip=true`，与上次错误 run 不同。上次错误 run 中 `teacache_start_skipping` 被客户端传成 `5.0`，服务端按比例解释，导致 `start_skipping=400`，所有 step 都被判为 boundary compute。

## Skip Pattern

整体统计：

| 指标 | 数值 |
| --- | ---: |
| total records | 3300 |
| would skip | 2160 |
| would compute | 1140 |
| skip ratio | 65.45% |
| boundary compute | 600 |
| non-boundary compute | 540 |

按 branch 统计：

| branch | boundary compute | non-boundary compute | skip |
| --- | ---: | ---: | ---: |
| cond | 300 | 420 | 1680 |
| uncond | 300 | 120 | 480 |

每个 window 的 pattern 完全一致：

| 每 window 指标 | 数值 |
| --- | ---: |
| records | 55 |
| skip | 36 |
| compute | 19 |
| boundary compute | 10 |

这和之前 aligned TeaCache sweep 的 `2160 skipped / 1140 computed / 65.45% skip ratio` 对齐，说明本次 residual validation 验证的是同一组最佳参数的真实 residual 风险。

## Residual Error 分布

这里主要看 `residual_error.relative_l1`。它表示如果当前 step 复用缓存 residual，与真实当前 residual 的相对 L1 差异。

整体对比：

| 组别 | n | mean | median | max |
| --- | ---: | ---: | ---: | ---: |
| skip, non-boundary | 2160 | 0.0851 | 0.0732 | 0.3196 |
| compute, non-boundary | 540 | 0.1386 | 0.1236 | 0.3369 |

按 branch 对比：

| 组别 | n | q25 | q50 | q75 | q90 | q95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skip cond | 1680 | 0.0525 | 0.0779 | 0.1166 | 0.1594 | 0.1881 | 0.3196 |
| skip uncond | 480 | 0.0466 | 0.0632 | 0.0829 | 0.1125 | 0.1335 | 0.2059 |
| compute cond | 420 | 0.1078 | 0.1298 | 0.1719 | 0.2280 | 0.2434 | 0.3369 |
| compute uncond | 120 | 0.0855 | 0.1045 | 0.1293 | 0.1670 | 0.2027 | 0.2393 |

判断：

- skipped residual 的均值和中位数低于 non-boundary compute，说明当前 TeaCache 决策整体有区分度。
- cond branch 的尾部更高，质量风险主要来自 cond skipped step。
- uncond branch 更稳定，skip p95 只有 `0.1335`。

## Tail Risk

`would_skip` 的 residual relative L1 尾部分布：

| 阈值 | all skip | cond skip | uncond skip |
| --- | ---: | ---: | ---: |
| >= 0.15 | 220 / 2160 | 205 / 1680 | 15 / 480 |
| >= 0.18 | 122 / 2160 | 118 / 1680 | 4 / 480 |
| >= 0.20 | 51 / 2160 | 48 / 1680 | 3 / 480 |
| >= 0.25 | 13 / 2160 | 13 / 1680 | 0 / 480 |
| >= 0.30 | 2 / 2160 | 2 / 1680 | 0 / 480 |

最高 residual 的 skipped 记录：

| residual_l1 | video_id | window | denoise_step | branch | candidate_accumulated |
| ---: | --- | ---: | ---: | --- | ---: |
| 0.3196 | 1544866964 | 1 | 21 | cond | 0.2790 |
| 0.3030 | 1544866964 | 1 | 20 | cond | 0.2787 |
| 0.2993 | 1544866964 | 2 | 21 | cond | 0.2790 |
| 0.2989 | 1544866964 | 0 | 21 | cond | 0.2790 |
| 0.2979 | 85522644 | 0 | 21 | cond | 0.2790 |

高风险 window 主要集中在：

| video_id | window | mean skipped residual_l1 | max skipped residual_l1 |
| --- | ---: | ---: | ---: |
| 1544866964 | 1 | 0.1474 | 0.3196 |
| 1544866964 | 2 | 0.1482 | 0.2993 |
| 1544866964 | 0 | 0.1388 | 0.2989 |
| 85522644 | 0 | 0.1464 | 0.2979 |
| 85522644 | 1 | 0.1112 | 0.2327 |

这些样例应该作为后续人工画质抽查的优先对象。

## 按视频的 skipped residual 概览

| video_id | n | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10012741 | 108 | 0.0662 | 0.0564 | 0.1255 | 0.1932 |
| 10041495 | 180 | 0.0989 | 0.0915 | 0.1887 | 0.2189 |
| 1009048554 | 72 | 0.0779 | 0.0661 | 0.1464 | 0.1974 |
| 10118975 | 144 | 0.0675 | 0.0594 | 0.1322 | 0.1894 |
| 10120244 | 108 | 0.0699 | 0.0607 | 0.1322 | 0.1921 |
| 11273413 | 108 | 0.0850 | 0.0749 | 0.1715 | 0.1919 |
| 113000356 | 72 | 0.0768 | 0.0662 | 0.1460 | 0.1837 |
| 1207687 | 144 | 0.0815 | 0.0711 | 0.1604 | 0.1973 |
| 13068265 | 72 | 0.0645 | 0.0568 | 0.1256 | 0.1856 |
| 14636683 | 108 | 0.0649 | 0.0547 | 0.1245 | 0.1823 |
| 14640052 | 324 | 0.0790 | 0.0695 | 0.1482 | 0.1925 |
| 1544866964 | 108 | 0.1448 | 0.1356 | 0.2806 | 0.3196 |
| 201436395 | 72 | 0.0912 | 0.0841 | 0.1767 | 0.2047 |
| 205398684 | 72 | 0.0673 | 0.0590 | 0.1306 | 0.1858 |
| 2079239854 | 72 | 0.0943 | 0.0851 | 0.1830 | 0.2000 |
| 2687361644 | 108 | 0.1038 | 0.0934 | 0.2022 | 0.2212 |
| 2795700644 | 72 | 0.0669 | 0.0589 | 0.1268 | 0.2059 |
| 65735554 | 72 | 0.1013 | 0.0912 | 0.1944 | 0.2283 |
| 74832364 | 72 | 0.0816 | 0.0687 | 0.1527 | 0.1883 |
| 85522644 | 72 | 0.1288 | 0.1195 | 0.2327 | 0.2979 |

## 与上次 Proxy 分析的关系

上次 proxy 分析使用的是 denoise trace 里的 `noise_pred_*_change.relative_l1`，衡量相邻 denoise step 的输出变化。这次使用的是 `residual_error.relative_l1`，衡量复用缓存 residual 与真实 residual 的误差。

这两个指标不能直接比较绝对值：

- 上次 `noise_pred` 变化常见数值在 `0.01` 量级。
- 这次 residual error 常见数值在 `0.05` 到 `0.15` 量级。

本次应该重点看两件事：

1. `would_skip` 是否整体低于 `would_compute`。
2. `would_skip` 是否存在不可接受的高误差尾部。

按这两个标准，本次结果支持当前参数继续作为稳定加速配置，但也提示了少数视频的尾部质量风险。

## 建议

当前推荐继续保留：

```text
teacache_thresh = 0.3
teacache_start_skipping = 5
teacache_end_skipping = 1.0
```

使用这个参数时，预期行为仍应按之前 aligned sweep 估计：

```text
skip ratio ~= 65.45%
端到端 speedup ~= 2.5x
```

如果目标是稳定画质，不建议直接升到 `0.35` 或更高。应先人工抽查以下视频：

```text
1544866964
85522644
65735554
2687361644
10041495
```

如果抽查结果显示高 residual window 仍无明显画质问题，再继续 sweep：

```text
teacache_thresh = 0.35, 0.4, 0.45, 0.5
teacache_start_skipping = 5
teacache_end_skipping = 1.0
```

每组都应记录：

```text
speedup
skip ratio
replay accuracy
residual_error.relative_l1 tail
高残差样例抽帧质量
```

