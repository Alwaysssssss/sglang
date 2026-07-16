# VividVR 加速实验自动化脚本设计

**日期：** 2026-07-15

**目标：** 新增一个 Python 入口，自动管理 VividVR 加速实验依赖服务和主推理服务的完整生命周期，顺序执行 `analysis_tables.md` 中全部方案，并在每次 warmup 或正式推理结束后生成结构化 JSON；脚本不向 Markdown 结果表写入性能数据。

**基线保护：** 不修改 VividVR Phase C/D/E 推理语义、服务请求契约、默认 attention backend、compile、VAE、offload 或 caption bridge 行为。实验差异只通过已有服务启动参数表达。

## 1. 方案选择

考虑过三种实现方式：

| 方案 | 优点 | 问题 | 结论 |
| --- | --- | --- | --- |
| Shell 串联 `mock_test.md` 命令 | 起步快 | JSON 合并、失败恢复、端口清理和单元测试困难 | 不采用 |
| 仅实现请求和结果采集器 | 对现有服务侵入最小 | 无法保证启动参数、GPU 隔离、warmup 和服务重启一致 | 不采用 |
| Python 生命周期编排器 | 可验证实验注册表、统一 JSON、自动恢复并保持单入口 | 实现量高于 Shell | 采用 |

对外仍是一个脚本入口；内部按职责分为实验注册、服务管理、请求执行、指标合并四个组件，避免把长时流程写成无法测试的单段代码。

## 2. 文件边界

| 文件 | 操作 | 职责 |
| --- | --- | --- |
| `python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py` | 新增 | CLI、实验注册表、tmux 生命周期、FlowCut 请求、GPU 采样、JSON 合并和批次汇总 |
| `python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py` | 新增 | 注册表、命令生成、指标计算、JSON 状态、恢复和失败续跑的无 GPU 单元测试 |
| `docs_xzh/run_command/vividvr_acceleration_benchmark.md` | 新增 | dry-run、全量执行、单方案调试、tmux attach、恢复和产物说明 |
| `docs_xzh/docs_analysis/analysis_tables.md` | 修改 | 只修正固定输入路径、prompt 来源、R6/R7/R8/R9/R99/R100 的真实能力状态；不填写实验结果 |

不修改 `mock_test.md`，它继续作为原始手动服务验收入口。

## 3. CLI 契约

入口固定为：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py COMMAND
```

公开子命令：

| 子命令 | 行为 |
| --- | --- |
| `run-all` | 按固定顺序执行所有方案；这是正式入口 |
| `run-one --scheme R3` | 使用与全量实验相同的完整生命周期执行一个方案 |
| `dry-run` | 校验环境并打印方案、GPU、服务参数和产物路径，不启动服务或推理 |

`run-all` 和 `run-one` 如果在 tmux 外调用，会自动创建形如 `vividvr_accel_20260715T120000Z` 的 detached session，并在终端打印：

```bash
tmux attach -r -t vividvr_accel_20260715T120000Z
```

脚本在 tmux 内通过隐藏的 `_run-batch` 子命令执行实际长时任务。`dry-run` 不要求 tmux。这样既遵守推理必须在 tmux 中运行的仓库规则，也不要求用户手工拼接多组 session。

正式执行默认参数：

| 参数 | 默认值 |
| --- | --- |
| Python | `/home/zhiheng/sglang/.venv/bin/python` |
| GPU | `0,1,2,3`；启动前要求至少四张空闲 GPU |
| 输入 | `/home/zhiheng/input/test_video_long_960x720_130f.mp4` |
| Caption | `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt` |
| Reference | `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4` |
| 推理 | 130 输入帧、121 temporal process frames、20 steps、seed 42、upscale 1.0 |
| 质量门槛 | `min_ssim=0.98`、`max_failed_frame_ratio=0` |
| 服务健康超时 | 1800 秒 |
| 单次请求超时 | 7200 秒 |
| GPU 采样周期 | 1 秒 |

路径、端口和超时允许通过 CLI 覆盖；加速组合本身不提供任意开关覆盖，避免正式方案被临时参数悄悄改变。调试其他组合应新增显式方案或使用现有手动命令。

## 4. 固定实验注册表

仅启用 compile 的可执行方案运行一次 1-step warmup，再运行一次完整 formal；eager 方案直接运行 formal。warmup 也生成 JSON，但不参与速度和质量结论。

| 方案 | GPU | Requested backend | 并行 | Compile | Fusion | 执行状态 | 增益对照 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| R0 | 1 | `sdpa` | `single` | 关 | 关 | 执行 | 基线 |
| R1 | 1 | `fa` | `single` | 关 | 关 | 执行 | R0 |
| R2 | 1 | `fa` | `single` | 开 | 关 | 执行 | R1 |
| R3 | 2 | `fa` | `SP=2` | 开 | 关 | 执行 | R2 |
| R4 | 4 | `fa` | `SP=4` | 开 | 关 | 执行 | R3 |
| R5 | 4 | `fa` | `CFG=2 × SP=2` | 开 | 关 | 执行 | R3、R4 |
| R6 | 1 | `fa` | `single` | 开 | modulation/residual，targets=`transformer,controlnet` | 执行 | R2 |
| R7 | 1 | `fa` | `single` | 开 | 关 | `unsupported` | R2 |
| R8 | 1 | `fa` | `single` | 开 | 关 | `unsupported` | R2 |
| R9 | 1 | `fa` | `single` | 开 | 关 | `unsupported` | R2 |
| R99 | 2 | `fa` | `SP=2` | 开 | 与 R6 相同 | 执行 | R3 |
| R100 | 4 | `fa` | `CFG=2 × SP=2` | 开 | 与 R6 相同 | 执行 | R4、R5 中 formal 更快者 |

执行顺序固定为 `R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R99, R100`。

关键定义：

- `eager` 只表示 `torch.compile` 关闭，不是一个额外服务参数。
- R3/R4 的有效 backend 预期为 `fa_sp`；脚本必须从 perf JSON 验证，不能只记录请求值 `fa`。
- R5/R100 必须验证 `cfg_parallel_enabled=true`、`cfg_world_size=2`、`sp_world_size=2` 和 `vividvr_parallel_mode=cfg_sp`。
- R6/R99/R100 只启用已通过质量验证的 modulation/residual fusion。QKV fusion、QK norm fusion、QK norm + RoPE fusion 和两项 USP collective 实验开关全部保持关闭。
- R7 的 Cache-DiT 目前只接入 diffusers backend，未接入 VividVR 原生 denoise stage。
- R8 的 TeaCache 没有 VividVR 集成。
- R9 只有量化加载参数传递，没有 VividVR CogVideoX 线性层量化实现和匹配的已验证权重。
- R7/R8/R9 不启动主服务和推理；脚本分别生成一份 `unsupported` 方案 JSON，明确 capability、原因和所需实现，禁止产生空白的成功记录。
- R99/R100 的“全部加速”指当前 VividVR 已实现、已通过质量验证且可组合的 attention、并行、compile、modulation/residual fusion；不包含上述 unsupported 能力或无端到端收益的实验开关。

## 5. 服务命令映射

所有主服务共享 `mock_test.md` 中的模型、pipeline、caption bridge、输出目录和网络配置。每组都显式设置以下公共参数：

```text
--model-path /home/zhiheng/ckpts/CogVideoX1.5-5B
--model-id VividVR
--pipeline-class-name CogVideoXVividVRControlNetPipeline
--component-paths.vividvr /home/zhiheng/ckpts/Vivid-VR
--tp-size 1
--ring-degree 1
--dist-timeout 3600
--input-save-path ""
--vividvr-caption-bridge
--vividvr-caption-sidecar-url http://127.0.0.1:31200
```

差异参数映射：

| 拓扑 | `--num-gpus` | `--sp-degree` | `--ulysses-degree` | CFG 参数 | 环境变量 |
| --- | ---: | ---: | ---: | --- | --- |
| single | 1 | 1 | 1 | `--vividvr-parallel-mode single` | 无 connector SP 覆盖 |
| SP=2 | 2 | 2 | 2 | `--vividvr-parallel-mode sp` | `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global` |
| SP=4 | 4 | 4 | 4 | `--vividvr-parallel-mode sp` | 同上 |
| CFG=2 × SP=2 | 4 | 2 | 2 | `--enable-cfg-parallel --vividvr-parallel-mode cfg_sp` | 同上 |

compile 方案添加 `--enable-torch-compile`。fusion 方案添加：

```text
--enable-cogvideox-modulation-fusion
--cogvideox-modulation-fusion-targets transformer,controlnet
```

各组顺序复用物理 GPU 的前 N 张：单卡 `0`、双卡 `0,1`、四卡 `0,1,2,3`。脚本不并发运行两个主推理服务。

## 6. 生命周期状态机

一个批次的状态流为：

```text
preflight
  -> start Moto S3
  -> create bucket
  -> start fixed caption mock
  -> start callback receiver
  -> for each scheme
       -> unsupported JSON, or
       -> assert GPUs idle
       -> start scheme service in tmux
       -> wait /health
       -> run warmup request with GPU sampler
       -> write warmup JSON
       -> run formal request with GPU sampler
       -> download formal S3 result
       -> compare against reference
       -> write formal JSON
       -> stop scheme service
       -> verify ports released and GPU processes exited
  -> write batch summary
  -> stop callback, caption mock and Moto
```

共享依赖服务每批只启动一次；主推理服务必须在每个方案之间重启，因为 backend、compile、并行和 fusion 都是启动级参数。主服务 session 名包含批次 ID 和方案，例如 `vividvr_accel_20260715T120000Z_R3_service`。

脚本只终止自己创建且 session 名和批次 ownership 文件都匹配的 tmux session，不使用 `pkill`、`killall` 或模糊进程匹配，不触碰用户已有服务。预检发现目标端口已占用或 GPU 上有其他计算进程时，批次在启动推理前失败，不主动清理未知进程。

正常退出、异常退出和 `SIGINT` 都执行 owned session 清理。JSON 与批次 manifest 使用临时文件加原子 rename，避免进程中断留下看似完整的半份结果。

## 7. 单次请求数据流

每个请求使用唯一 task ID：

```text
vividvr-accel-<batch-id>-<scheme-lower>-<warmup|formal>-a<attempt>
```

请求包含固定输入、推理参数、输出路径、S3 配置和独立 `perf_dump_path`。caption 不直接放入主请求，而是由固定 caption mock 按 caption bridge HTTP 契约写到请求指定路径，保持 `mock_test.md` 的服务链语义，同时排除 CogVLM2 GPU 占用和随机性。

计时边界：

- `total_runtime_seconds`：客户端开始提交 POST 到任务进入 `completed` 终态，覆盖 caption bridge、pipeline forward、编码和上传。
- `model_inference_runtime_seconds`：服务 perf JSON 的 `total_duration_ms / 1000`，即完整 pipeline forward。
- `denoising_runtime_seconds`：`VividVRMultiClipDenoisingStage.duration_ms / 1000`。
- 下载、compare 和 JSON 合并耗时单独记录，不计入上述三个表格耗时。

compile 方案的 warmup 使用 130f/1-step 请求，formal 使用固定的 130f/20-step 请求。warmup 的作用：

- compile 方案完成图编译以及 CUDA/kernel/allocator 预热；
- eager 方案不额外 warmup；
- 所有模块增益都使用 formal 对 formal。

warmup 不执行 reference compare，但保留输出路径、服务 perf、callback、运行时快照和 GPU 指标。formal 下载 S3 结果并运行 `compare.py`。

## 8. JSON 契约

### 8.1 每次推理 JSON

文件名：

```text
Vivid_Acceptance/indicator/acceleration_analysis/<batch-id>/<scheme>/<task-id>.analysis.json
```

顶层字段固定如下：

```json
{
  "schema_version": "vividvr_acceleration_benchmark.v1",
  "batch_id": "20260715T120000Z",
  "config_fingerprint": "sha256:...",
  "scheme": {},
  "run": {},
  "request": {},
  "timing": {},
  "stages": {},
  "denoising": {},
  "acceleration": {},
  "resources": {},
  "quality": {},
  "environment": {},
  "derived": {},
  "artifacts": {}
}
```

字段内容：

| 对象 | 必含内容 |
| --- | --- |
| `scheme` | ID、名称、control IDs、GPU 数、配置声明、capability 状态 |
| `run` | `run_role`、attempt、`succeeded/failed/quality_failed`、UTC 起止时间、失败阶段、异常类型和消息 |
| `request` | task ID、FlowCut payload、submit/progress/final callback、输入/caption/reference 的路径与 SHA-256 |
| `timing` | 总耗时、模型推理耗时、denoise 耗时、下载耗时、compare 耗时 |
| `stages` | 八个 VividVR stage 秒数、stage 合计、未归类开销、模型推理总计 |
| `denoising` | clip 数、step 数、逐 step 秒数、平均值、排除 step 0 的 steady median、denoise 占比 |
| `acceleration` | requested/effective backend、compile configured/effective、并行 configured/effective、fusion configured/effective、cache 和 quant 状态 |
| `resources` | 每张 GPU 的 index/UUID/name、采样前显存、峰值显存、峰值增量、最大单卡峰值、服务 rank-0 allocator checkpoints |
| `quality` | warmup 为 `not_evaluated: warmup`；formal 为 compare pass、SSIM mean/min、failed frame ratio 和门槛 |
| `environment` | hostname、GPU、driver、CUDA、Python、PyTorch、FlashAttention、sglang commit、checkpoint、dtype、计时和显存方法 |
| `derived` | 累计加速比、模块增量加速比、GPU·秒、R0 资源效率及 control 可用性 |
| `artifacts` | service/callback/client 日志、perf JSON、output、S3 URL、download、compare JSON |

所有表格字段都保留机器精度数值，不提前格式化为带 `×` 或 `GiB` 的字符串。不可可靠获得的字段使用 `null`，旁边必须有 `availability` 或 `reason`，不能用 `0` 冒充。例如普通 perf 没有稳定的 SP/CFG 通信拆分时：

```json
{
  "sp_communication_seconds": null,
  "cfg_communication_seconds": null,
  "communication_timing_availability": "not_profiled"
}
```

steady step 中位数固定为 `denoise_steps_ms` 中排除 `step=0` 后其余 step 的中位数。未归类开销固定为：

```text
model_inference_runtime_seconds - sum(eight_stage_seconds)
```

若浮点同步产生绝对值小于 `1e-6` 秒的负值则归零；更大的负值视为 perf 数据不一致并使该次运行失败。

### 8.2 Unsupported 方案 JSON

R7/R8/R9 路径为：

```text
Vivid_Acceptance/indicator/acceleration_analysis/<batch-id>/<scheme>/<scheme>_unsupported.analysis.json
```

它们使用同一 schema，但 `run.run_role="not_run"`、`run.status="unsupported"`，性能、质量和资源字段为带原因的 `null`。`scheme.capability` 明确写出当前缺失的 VividVR 集成，保证后续看到空指标时不会误认为实验失败或漏采集。

### 8.3 批次汇总 JSON

每完成一次请求就原子更新：

```text
Vivid_Acceptance/indicator/acceleration_analysis/<batch-id>/batch_summary.json
```

汇总包含方案顺序、当前状态、warmup/formal JSON 路径、失败列表、unsupported 列表、formal 结果索引，以及可直接映射 `analysis_tables.md` 的总体、stage、denoise、运行时快照和模块收益对象。它只汇总 JSON，不修改 Markdown。

derived 指标仅引用同一批次、同一配置指纹下成功的 formal：

- 累计加速比：`R0.model_inference / treatment.model_inference`；
- 模块增量加速比：注册表 control 的 model inference 比值；
- R100 control：R4/R5 中 model inference 更短且质量通过者；
- GPU·秒：`gpu_count * model_inference_runtime_seconds`；
- R0 资源效率：`R0 GPU·秒 / treatment GPU·秒`。

control 尚未成功或质量未通过时，相应 derived 值为 `null` 并记录原因。

## 9. GPU 显存采样

服务 perf JSON 的 allocator checkpoint 仅可靠代表 rank 0，不能代替四卡逐卡峰值。formal 请求提交前启动 NVML 采样线程，按 1 秒周期读取本方案物理 GPU：

- `memory.used`；
- GPU UUID 和名称；
- sampling 起始值和请求期间峰值。

采样在任务终态后停止。JSON 同时保存每卡绝对峰值和相对起始值的增量，主表“最大单卡峰值显存”对应绝对峰值。warmup 同样采样并写入自己的 JSON，但正式表只消费 formal。

NVML 不可用时回退到 `nvidia-smi --query-gpu`。两者都不可用时运行失败，因为峰值显存是当前表格的必需字段。

## 10. 失败、清理与断点恢复

失败策略固定为“记录并继续”：

- 依赖服务启动失败或四张 GPU 不满足预检：批次无法安全开始，立即失败并生成 batch summary；
- 某个主服务健康检查失败：生成该方案 `failed` JSON，清理服务，继续下一方案；
- warmup 失败：生成 warmup `failed` JSON，不执行该方案 formal，继续下一方案；
- formal 推理失败：生成 formal `failed` JSON，继续下一方案；
- 推理成功但 compare 不达标：formal 状态为 `quality_failed`，保留所有性能指标，但不允许作为 R99/R100 control 或最终推荐依据；
- 清理未确认完成：为避免 GPU 污染，不启动下一方案，批次终止并返回非零。

只要出现 `failed`、`quality_failed` 或清理失败，`run-all` 最终退出码非零；unsupported 不单独导致非零，因为它是注册表中的已知能力状态。

`--resume <batch-id>` 使用配置指纹保护恢复：

- 指纹覆盖 scheme 注册表、固定请求参数、输入/caption/reference SHA-256、checkpoint 路径、当前 git commit 和影响服务行为的环境变量；
- warmup 和 formal 均成功且指纹一致的方案整体跳过；
- 只有 warmup 成功而 formal 未成功时，必须重启服务并重新运行 warmup，然后再跑 formal，不能复用上一次进程的热态假设；
- 指纹不一致时拒绝在原批次目录续跑，要求创建新批次。

## 11. 预检和运行时真实性校验

`dry-run` 和正式执行共享以下预检：

- `.venv` Python、`sglang`、`moto_server`、`tmux`、`ffmpeg/ffprobe`、输入、caption、reference、模型和 VividVR checkpoint 存在；
- caption 非空且行数与 temporal clip 规划预期一致；
- 四张 GPU 可见且没有未知 compute process；
- 公共端口和主服务端口未被未知进程占用；
- 产物根目录可创建；
- 当前 git commit、dirty paths 和关键包版本可记录。

每次 perf 合并前校验配置真实性：

- requested/effective attention backend；
- transformer/controlnet compile 是否按方案生效；
- SP/CFG world size、parallel mode 和 connector context；
- fusion targets 和实际 fused module 标记；
- stage 名集合、denoise step 数、task ID 和 perf 文件归属。

声明配置与 effective 快照不一致时，该次运行标记失败，禁止只根据启动命令推断实验实际生效。

## 12. 测试与验收

无 GPU 单元测试至少覆盖：

1. R0-R9、R99、R100 顺序和每组精确服务参数；
2. eager/compile、single/SP2/SP4/CFG2×SP2、fusion 命令映射；
3. unsupported 方案不产生服务命令；
4. 八个 stage、unclassified、denoise mean/steady median、GPU·秒和 speedup 计算；
5. 缺失 control、质量失败 control 和 R100 最快合格 control 选择；
6. perf effective 配置不匹配时失败；
7. warmup/formal/failed/quality_failed/unsupported JSON schema；
8. GPU 多卡峰值聚合和 sampler 不可用错误；
9. 单方案失败后继续、清理失败后终止；
10. 配置指纹一致时恢复、指纹变化时拒绝恢复；
11. dry-run 不调用 tmux、HTTP、NVML 或文件下载；
12. 原子 JSON 写入不遗留临时文件。

实现完成后的轻量验收：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py

PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py dry-run
```

重型验收不在脚本实现阶段自动触发。用户明确开始正式实验后，执行 `run-all`，脚本自动进入 tmux 并跑完整批次。正式完成标准：

- 9 个可执行方案各有 warmup 和 formal JSON；
- R7/R8/R9 各有 unsupported JSON；
- 每个成功 formal 都有 perf、视频、S3 download、compare 和逐卡显存；
- batch summary 能覆盖 `analysis_tables.md` 的所有列；
- 任何失败都在 summary 中可定位，最终退出码与状态一致；
- 脚本退出后不存在其拥有的 tmux session、端口或 GPU 进程。

## 13. 明确不在本次范围内

- 实现 Cache-DiT、TeaCache 或 VividVR 量化；
- 自动修改或填写 Markdown 性能结果；
- 启用 PyTorch profiler/Nsight 来强行拆分 SP/CFG 通信耗时；
- 根据本轮结果自动改变 Phase E 默认配置；
- 自动提交、推送代码或启动正式全量推理。
