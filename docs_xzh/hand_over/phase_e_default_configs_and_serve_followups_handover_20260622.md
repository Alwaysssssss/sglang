# Phase E 默认配置收口与 Serve 后续问题交接

日期：`2026-06-22 UTC`

## 1. 本轮结论

当前 `Vivid-VR` 在 `sglang` 中的 `Phase E` 长视频主线已经完成一轮正式 `serve` benchmark 与消融收口，默认配置可以视为已经明确：

- 单卡默认正式配置：`single_gpu_fa_compile`
- 双卡默认正式配置：`dual_gpu_fa_eager_compile`

这里的双卡默认正式配置具体指：

- `SP=2`
- `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
- `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`
- 请求参数 `--attention-backend fa`
- 双卡 `SP` 下运行时有效 backend 为 `fa_sp`
- `--enable-torch-compile`

这条双卡默认配置的定位不是“绝对最快”，而是“当前最快的质量安全配置”。

## 2. 当前已完成状态

### 2.1 长视频质量与性能收口

当前 `130f / 20 step / serve + curl / warmup excluded` 的正式结论已经收口，详见：

- `/home/zhiheng/sglang/docs_xzh/benchmark_results/vividvr_serve_long_130f_20step_benchmark_20260621.md`

核心结果：

- 单卡最佳：`single_gpu_fa_compile`
  - `total_runtime_seconds = 961.31`
  - `model_inference_runtime_seconds = 950.01`
  - `ssim_mean = 0.987081`
- 双卡最佳安全配置：`dual_gpu_fa_eager_compile`
  - `total_runtime_seconds = 540.79`
  - `model_inference_runtime_seconds = 538.43`
  - `ssim_mean = 0.987086`
- 双卡质量安全端到端加速比：`1.7776x`
- 双卡质量安全纯推理加速比：`1.7644x`

同时也已经确认：

- `sdpa` 双卡路径虽然更快，但 `ssim_mean` 会掉到 `0.966x`
- 因此双卡默认不能再以“更快”为理由选 `sdpa`
- `FA + eager_global + pool=1 + compile` 才是当前正式默认路径

### 2.2 默认命令入口已经整理

当前单卡/双卡直接运行命令、单卡/双卡 `serve` 拉起命令、以及 `curl` 请求命令已经整理到：

- `/home/zhiheng/sglang/docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`

这份文档已经明确了：

- 单卡正式 benchmark 必须串行，不能同时跑两个单卡推理进程
- `serve` benchmark 必须先 warmup，再统计第 2 次正式请求
- 双卡默认环境变量必须显式固定为 `eager_global + pool=1`

### 2.3 AGENTS 约束已经同步

当前仓库根目录 `AGENTS.md` 已同步到最新默认口径：

- 单卡默认正式配置写为 `single_gpu_fa_compile`
- 双卡默认正式配置写为 `dual_gpu_fa_eager_compile`
- 当前项目背景中已补充 caption 环境问题与 `serve` 参数契约问题

## 3. 当前仍未完成的问题

### 3.1 原版 caption 只能在原版环境中稳定正确产出

这是当前最明确的未完成问题之一。

结论：

- 原版 `/home/zhiheng/Vivid-VR` 的 caption 输出目前只能在原版自己的 `.venv` 中稳定正确产出
- 在 `sglang/.venv` 中，由于依赖版本差异，caption 输出会有误

这意味着：

- 如果要对一个此前没有做过公平 benchmark 的新视频生成 reference caption，不能直接在 `sglang` 环境里代跑原版 caption
- 当前实际安全做法仍然是：
  - 先用 `/home/zhiheng/Vivid-VR/.venv/bin/python` 跑原版
  - 从原版日志或原版结果中提取每个 temporal clip 的原始 caption
  - 再把它保存成 sidecar caption 文件，供 `sglang` 复用

后续可能需要补一条更明确的桥接方案，例如：

- 通信交换生成 caption
- 或者更直接地固定“caption 生成永远在原版环境做，`sglang` 只消费 sidecar”

### 3.2 Serve 输入参数契约还没有完全收口

当前 `serve` 路径已经能稳定完成 benchmark 和正式推理，但输入参数契约还不能视为完全定型。

需要继续关注的点：

- 外部调用时哪些字段必须由客户端传入
- 哪些字段应该由服务端默认补全
- prompt / caption / reference / output / perf dump 这些路径参数的约束是否需要进一步统一

这条线目前仍属于“可用，但还需要继续按要求打磨”的状态。

### 3.3 `/progress` 目前只有状态位，不是真实进度条

当前 `serve` 的：

- `GET /v1/videos/{task_id}/progress`

已经可以用于查询任务状态，但目前还不能返回真实的逐步百分比进度。

现状是：

- `queued` 时写 `progress=0`
- `running` 时写 `progress=1`
- `completed` 时写 `progress=100`

也就是说当前的 `progress` 更接近“状态占位值”，而不是“真实进度条”。

已经确认的实现现状：

- `Vivid-VR` 推理内部其实有 `runtime_progress`
- 它会在 denoising step 内部持续更新
- 但这个值目前没有同步回 `VIDEO_STORE`
- 所以 `/progress` 接口现在只能回答“任务是否开始 / 是否完成”，不能回答“当前完成了多少 step”

因此下一轮如果要增强服务体验，一个直接且合理的工作项就是：

- 把 `runtime_progress` 持续同步到 `VIDEO_STORE`
- 让 `/progress` 返回真实百分比进度

## 4. 当前推荐口径

如果下一轮继续做正式推理、benchmark、回归或服务验收，建议默认坚持下面的口径：

- 单卡默认：`FA + compile`
- 双卡默认：`FA + eager_global + pool=1 + compile`
- 双卡默认质量判断不能只看 compare gate 是否通过，还要看是否与单卡 baseline 基本一致
- `serve` benchmark 继续固定 `130f / 20 step`
- 单卡正式 benchmark 必须串行

不建议回退到：

- 双卡 `sdpa` 作为默认模式
- 双卡 `deferred_global` 作为默认模式
- `pool>1` 作为默认质量口径

这些路径可以留作历史对比或性能实验，但不应再作为正式默认配置。

## 5. 重要产物位置

benchmark 文档：

- `/home/zhiheng/sglang/docs_xzh/benchmark_results/vividvr_serve_long_130f_20step_benchmark_20260621.md`

默认命令文档：

- `/home/zhiheng/sglang/docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`

验收指标目录：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator`

结果视频目录：

- `/home/zhiheng/sglang/Vivid_Acceptance/result_videos`

本轮用于 `serve` benchmark / 消融的辅助脚本：

- `/home/zhiheng/sglang/Vivid_Acceptance/tmp/run_vividvr_service_benchmark.py`
- `/home/zhiheng/sglang/Vivid_Acceptance/tmp/run_vividvr_service_ablation.sh`
- `/home/zhiheng/sglang/Vivid_Acceptance/tmp/collect_vividvr_benchmark_metrics.py`

## 6. 工作区注意事项

当前工作区不是干净状态，交接时至少包含：

- `AGENTS.md`
- `docs_xzh/run_vivid_benchmark.md`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py`
- 新增的 benchmark / handover / run_command 文档与辅助脚本

因此下一轮默认要求仍然是：

- 不要做工作区清理
- 不要回退现有未提交改动
- 只在明确目标范围内做最小化改动

## 7. 下一轮建议优先级

建议的后续优先级如下：

1. 收口 `serve` 输入参数契约，减少调用歧义
2. 给 `/progress` 打通真实进度同步，而不是继续停留在 `0 / 1 / 100`
3. 明确 caption 环境桥接方案，避免新视频公平 benchmark 时重复踩环境版本问题
4. 在不破坏当前默认配置的前提下，再考虑是否需要扩展更多服务侧验收

## 8. 一句话交接

当前 `Phase E` 已经完成默认配置收口：单卡固定 `single_gpu_fa_compile`，双卡固定 `dual_gpu_fa_eager_compile`；长视频 `serve` benchmark 已完成并形成正式结论，但 caption 仍受原版环境约束，`serve` 输入参数契约和真实进度条还需要后续继续收口。
