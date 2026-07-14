# VideoEdit TeaCache Residual Validation 结果检查

日期：2026-07-06

## 结论

这次 no-TeaCache residual validation 批量任务本身跑完了，20 个视频全部 `completed`，并且生成了 `teacache_residual_traces/*.jsonl`。但是这次 trace 不能用于判断 TeaCache 跳步是否合理。

原因是 `teacache_start_skipping` 被客户端脚本传成了 float `5.0`，而 TeaCache 边界逻辑会把 float 当作比例处理：

```text
start_skipping = int(num_inference_steps * 5.0) * 2 = 400
end_skipping = int(num_inference_steps * 1.0) * 2 = 80
```

因此本次主 trace 中 `start_skipping=400`、`end_skipping=80`，所有 step 都被判为 boundary compute，没有任何 `would_skip=true`。这就无法评估“TeaCache 本来会跳的 step，其 residual 误差是否可接受”。

## 本次输出路径

```text
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0/manifest.jsonl
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0/teacache_residual_traces/
/home/tyx/workspace/sglang/outputs/erase_data_case_repair_residual_trace_bbox1_ms1_dp0/denoise_traces/
```

## 完成情况

| 指标 | 数值 |
| --- | ---: |
| manifest rows | 20 |
| completed | 20 |
| unique completed ids | 20 |
| mean elapsed | 1400.52s |
| median elapsed | 1086.22s |
| residual trace files | 20 |
| residual trace records | 3306 |

其中 3300 条主记录的边界参数是错误的：

| 字段 | 数值 |
| --- | ---: |
| start_skipping | 400 |
| end_skipping | 80 |
| would_skip | 0 |
| would_compute | 3300 |

还有 6 条 `10012741` 的旧/预热式记录显示 `num_inference_steps=1`、`start_skipping=10`、`end_skipping=2`，不属于本次有效 40-step 批量分析，应该忽略。

## 问题定位

TeaCache 的 `start_skipping` 支持两种语义：

```text
int: 具体跳过前多少个 step 后开始允许 skip
float: 按 num_inference_steps 的比例计算边界
```

当前要表达的是整数 `5`，不是比例 `5.0`。之前 `scripts/batch_videoedit_repair.py` 里参数类型是 `float`：

```text
--teacache-start-skipping 5 -> 5.0
```

这导致服务端按比例解释，得到错误的 `start_skipping=400`。

## 已修复内容

已修改：

```text
/home/tyx/workspace/sglang/scripts/batch_videoedit_repair.py
```

现在 `--teacache-start-skipping 5` 会被解析成 int `5`，而不是 float `5.0`。dry-run 已确认 payload 中为：

```json
"teacache_start_skipping": 5
```

而不是：

```json
"teacache_start_skipping": 5.0
```

## 需要重跑的实验

需要重跑一次 no-TeaCache residual validation。不要使用本次旧目录继续分析。建议写到新的输出目录：

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

这次重跑后，正确的 trace 应该满足：

```text
start_skipping = 10
end_skipping = 80
每个 window 有 would_skip=true 的记录
cond branch 应出现类似当前 TeaCache 的 skip pattern
```

只有看到 `would_skip=true` 后，才能继续判断：

```text
would_skip 的 residual_error 是否小 -> 跳步是否合理
would_compute 的 residual_error 是否小 -> 是否还有 step 可以继续跳
```

## 当前结论状态

本次结果不能支持“当前 TeaCache 跳步合理/不合理”的结论。它只能说明 residual validation trace 写入链路已经打通，但跳步模拟参数传错，导致没有实际验证到任何跳步。

