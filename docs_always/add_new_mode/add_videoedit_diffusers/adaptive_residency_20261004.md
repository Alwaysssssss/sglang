# 按输入尺寸自动选择 DiT 驻留层数（2026-10-04）

本次在已有逐层卸载和阶段缓存上增加尺寸感知预算，让同一个服务按请求选择权重驻留策略。真实模型测试结果见下方验证记录；不把某一素材的结果视为所有输入上的最快或绝对无 OOM 保证。

## 实现

- 去噪前释放上次权重缓存，以当前可用显存和分配器可复用空间重新计算预算。
- 按实际 latent 的时间、空间体积估算激活预留：双卡 1920×1088×49 的基准为 16 GiB，小输入最低 4 GiB，另保留 2 GiB 余量。单卡采用两倍体积项，但本轮实测范围仅双卡。
- 各 SP rank 通过 CPU 通信组取预算最小值，避免两卡选择不同策略。
- 权重全部能容纳时恢复原有全层缓存、四层预取调度；部分驻留时按字节预算选择驻留前缀，采用一层滚动预取，极低预算可降为按需加载。连最大一层都放不下时明确报错。
- 每个尺寸、SP、CFG、TeaCache 组合独立保存历史高水位，最多 128 条，历史数据只增加预留，不降低尺寸基线。激活历史由总峰值减去初始分配和权重容量估算，是不精确的下界，不能替代显存安全证明。
- 去噪结束释放 GPU 权重缓存，为 VAE 解码留空间。CPU 只读原件继续复用。
- 当前自动模式限制 Wan 14B / BF16 / FA / 纯 Ulysses，batch=1 及 1920×1088 对齐尺寸。窗口帧数不设固定上限：预留按 latent 体积缩放，`infer_len` 越大驻留层数越少；连一层权重都放不下时在去噪前报错，并给出同裁剪可支持的帧数。2026-10-08 的修订见下文。

## 配置

GPU 4/5 专用配置显式设置 `SGLANG_VIDEOEDIT_AUTO_RESIDENCY=1`，改为 `0` 可关闭。通用 `start.sh` 默认关闭，以免影响不支持此预算模型的四卡或其他 attention 配置。直接启动后端时，应显式传入此环境变量。以下配置用于本轮双 L40S 测试：

```bash
NUM_GPUS=2
SP_DEGREE=2
ULYSSES_DEGREE=2
RING_DEGREE=1
ATTENTION_BACKEND=fa
DIT_LAYERWISE_OFFLOAD=true
DIT_CPU_OFFLOAD=false
DIT_OFFLOAD_PREFETCH_SIZE=4
PIN_CPU_MEMORY=true
SGLANG_VIDEOEDIT_AUTO_RESIDENCY=1
SGLANG_VIDEOEDIT_ACTIVATION_RESERVE_MB=4096
SGLANG_VIDEOEDIT_FULL_FRAME_RESERVE_MB=16384
SGLANG_VIDEOEDIT_RESIDENCY_MARGIN_MB=2048
SGLANG_VIDEOEDIT_WEIGHT_BUDGET_MB=0
SGLANG_VIDEOEDIT_RESIDENT_LAYERS=40
SGLANG_VIDEOEDIT_RELEASE_AFTER_DENOISE=1
SGLANG_VIDEOEDIT_ENCODER_CPU_MASTER=1
SGLANG_VIDEOEDIT_OVERLAP_PREFETCH=0
SGLANG_VIDEOEDIT_FUSED_QKV=0
SGLANG_VIDEOEDIT_USP_CHUNKS=1
SGLANG_VIDEOEDIT_ASYNC_STAGE_LOAD=0
SGLANG_VIDEOEDIT_PREFETCH_IMAGE_ENCODER=0
SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB=0
SGLANG_VIDEOEDIT_SHARED_CPU_DIR=
```

新增 `FULL_FRAME_RESERVE_MB` 和 `RESIDENCY_MARGIN_MB`；升级已有 `AUTO_RESIDENCY` 的选择逻辑。自动模式会覆盖固定 `RESIDENT_LAYERS`，局部驻留时内部启用滚动预取。`WEIGHT_BUDGET_MB=0` 表示不额外限制权重预算；非零值作为上限。减小预留可能多驻留几层，但也会提高 OOM 风险。

## 测试与复现

独立容器使用 `sglang-videoedit-dev-v2:l40s`，GPU 4、5，正常模型 40 步、动态 CFG、TeaCache 0.3、seed 42。小尺寸实际模型输入 544×480×49；1080p 实际模型输入 1920×1088×49；源视频和最终输出均为 1920×1080、41 帧。服务输出尺寸不代表 DiT 输入尺寸。

固定配置：小尺寸预热一次、正式三次。自适应配置：同一进程执行“小尺寸预热 → 小 → 大 → 小 → 大 → 小”。两次大全帧分别报告，第一次包含该形状首次执行，不称为三次预热后中位数。

```bash
python3 scripts/videoedit_benchmark/run_adaptive_residency.py \
  --out .local/adaptive-residency-new --gpus 4,5 \
  --image sglang-videoedit-dev-v2:l40s
python3 scripts/videoedit_benchmark/report_adaptive_residency.py \
  .local/adaptive-residency-new \
  --full-reference .local/teacache-sp/full-v2/u2_prefetch1/timed1-result.json
```

输出目录必须不存在。基准脚本不接业务队列、不发送回调，结束后删除所属容器。报告检查真实尺寸、双卡预算与哈希一致、大小尺寸分别重复一致、小尺寸恢复 40 层缓存，以及阶段结束后权重缓存释放；只在所有请求成功后生成通过结论。

单元与回归验证：最终代码的 27 项测试全部通过，包含真实 CUDA 权重调度切换和逐元素输出一致性；两 rank Gloo 测试确认取最小预算，包括零预算情形。记录位于 `.local/adaptive-residency-validation/tests-final-root.log` 与 `consensus.log`。

### 真实模型结果：已完成

最终运行目录：`.local/adaptive-residency-20261004-v2`。固定配置 4 次、自适应配置 6 次，共 10 次完整请求全部成功，无 OOM。

| 输入 / 配置 | 端到端耗时 | DiT 耗时 | 驻留 / 预取层数 |
| --- | ---: | ---: | --- |
| 小尺寸，固定全层缓存，三次中位数 | 53.111 秒 | 33.903 秒 | 40 / 4 |
| 小尺寸，自适应，三次中位数 | 51.455 秒 | 33.064 秒 | 40 / 4 |
| 1080p，自适应，首次该形状 | 609.740 秒 | 534.101 秒 | 27 / 1 |
| 1080p，自适应，间隔小尺寸后的第二次 | 606.835 秒 | 532.295 秒 | 27 / 1 |

端到端取双 rank 完整 pipeline 时间的最大值，包含读入、编码、权重搬运、推理、解码和本地视频输出；HTTP 轮询耗时另存 `client_seconds`。小尺寸正式请求固定组为 51.029 / 53.111 / 53.144 秒，自适应组为 52.095 / 51.455 / 50.834 秒。自动组中位数低约 3.12%，但只有三次且范围重叠，应解释为保留了已有较快的小尺寸路径，不能宣称稳定提速 3.12% 或所有配置中的绝对最快。

两次 1080p 的每卡完整请求显存采样峰值均为 **37.278 GiB**，相对设备报告的约 44.395 GiB 可用总容量，采样峰值下余量约 **7.116 GiB**。去噪活跃分配峰值为每卡 **32.076 GiB**。采样间隔一秒，可能漏掉瞬时峰值；设备占用含分配器缓存，不等同于活跃张量大小。小尺寸完整请求采样最大达到 44.099 GiB，包含可回收的分配器缓存，本轮全部成功，不能将这一采样值当作权重预算。

验证结果：

- 自适应序列驻留层数为 **40 → 40 → 27 → 40 → 27 → 40**；大小尺寸切换后均恢复预期调度。
- 20 个 rank latent 记录检查通过；8 个小尺寸输出与固定基线完全一致，两次全帧输出完全一致，并与历史双卡全帧参考的解码帧哈希一致。各尺寸的 TeaCache 决策也一致。
- 各请求两个 rank 的共同预算、驻留数和预取数一致；每个去噪阶段结束后 GPU 权重缓存均为 0。
- 测试容器已删除，GPU 4/5/6 均为 0 MiB 占用；没有修改现有业务服务。

历史全帧参考为 `.local/teacache-sp/full-v2/u2_prefetch1/timed1-result.json`。历史完整请求中位数 623.004 秒与本轮的代码、采样和预热条件不同，不能据此归因自动驻留带来的加速；本轮证明的是同一服务兼顾小尺寸全层缓存与全帧部分驻留。

测试过程中曾在首次开发运行的小尺寸预热处发现日志变量未定义，已修复并完整重跑上述对照。失败证据保留在 `.local/adaptive-residency-20261004`，最终成功数据单独保存在 `-v2` 目录；没有发生 OOM 后缩图重试。

完整结果：[逐请求报告](../../../.local/adaptive-residency-20261004-v2/mixed-summary.md)、[机器可读明细](../../../.local/adaptive-residency-20261004-v2/mixed-summary.json)。本机 `.local` 产物通常不随 Git 分发，核心数据已收录于本文。

### 2026-10-08 修订：窗口帧数随 infer_len 自适应

原实现把 49 帧（13 个 latent 帧）写成形状白名单，`infer_len` 大于 49 的请求在编码前直接失败（`Automatic residency profile supports batch 1, at most 1920x1088 and 49 frames`）。业务请求的 `infer_len` 会随素材变化，因此改为按参数自适应：

- 形状校验只保留结构性约束：batch=1、16 个 latent 通道、16 像素对齐（latent 空间维度为偶数）以及单帧不超过 1920×1088 对齐裁剪。帧数不再有固定上限。
- 激活预留继续按 latent 体积线性缩放（基准 1920×1088×49 = 16 GiB，下限 4 GiB，另留 2 GiB 余量）。窗口越长，留给权重的预算越少、驻留层数越少，权重全量驻留只在短窗口出现。
- 预检阶段（窗口校验）以设备总容量作为乐观上界，只拒绝物理上不可能的窗口；去噪阶段用释放权重后的实测可用显存重新规划，并要求预算至少能容纳最大的一层（滚动加载下限）。不满足时抛出的错误包含请求帧数、latent 形状、预留、可用显存，以及同一裁剪在该显存下可支持的帧数，替代原来的固定文案。
- 判定仍是线性外推的估算，不是显存安全证明：`SGLANG_VIDEOEDIT_FULL_FRAME_RESERVE_MB`、`SGLANG_VIDEOEDIT_RESIDENCY_MARGIN_MB` 仍按历史实测配置，`SGLANG_VIDEOEDIT_WEIGHT_BUDGET_MB` 非零时优先作为上限。

2026-10-08 已在 GPU 4/5 的 5403 服务上做两次窗口实测（重启容器后提交，均未预热、未重复）。同一素材、同一 tight 裁剪 1744×624、40 步、动态 CFG、TeaCache 关闭、seed 42，只改 `infer_len`：

| `infer_len` | latent 形状 | 激活预留 | 驻留层数 | 各窗口 DiT | 接口计时 | 每卡峰值 | 最小余量 |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 105 | `[1,16,27,78,218]` | 17.31 GiB | 20 / 19 | 1951.2 / 1942.7 秒 | 4030.59 秒 | 42.07 GiB | 2.92 GiB |
| 81 | `[1,16,21,78,218]` | 13.46 GiB | 25 / 25 | 1285.1 / 1283.5 秒 | 2676.21 秒 | 39.87 GiB | 5.12 GiB |

- 两次都由 2 个窗口组成：`infer_len=105` 时窗口 0 覆盖全部 105 帧、窗口 1 只提交第 104 帧；`infer_len=81` 时窗口 0 提交 0–79 帧、窗口 1（start 76、有效 30 帧、51 帧反射填充）提交 80–104 帧。旧实现会在窗口校验阶段直接拒绝这两个请求。
- 更短的窗口同时放开两件事：token 数减少，且权重预算变大——`infer_len=81` 的驻留多 5 层，单窗口 DiT 从 1951 秒降到 1285 秒（比值 0.659），低于 token 比 0.778，说明驻留层数的收益也计入了。
- 显存：请求开头的条件编码与权重装载阶段是峰值所在。`infer_len=105` 峰值 42.07 GiB、余量 2.92 GiB；`81` 峰值 39.87 GiB、余量 5.12 GiB。两次都无 OOM、无缩图、无重试，输出均为 H.264 1920×1080 105 帧。
- 画质差异（仅记录，未做验收）：两版插入文字的亮度不同，第 52 帧文字区域平均亮度 `infer_len=105` 为 38.06、`81` 为 28.12（0–255）。窗口构成不同会改变生成结果，文字观感需要用户按素材确认。
- 线性关系的旁证：planner 记录的激活残差与 latent 帧数成正比——13 帧 5.36 GiB、21 帧 8.63 GiB、27 帧 11.14 GiB，即 0.412 / 0.411 / 0.413 GiB 每 latent 帧；三处都低于按体积线性外推的预留，说明该估算在 49–105 帧区间偏保守。

这两次各只有 1 个请求，不构成预热后中位数，也不代表其他素材、并发请求、其他裁剪或更长窗口无 OOM；它证明的是修订后的规划在这两个形状上可用。证据在 `.local/videoedit-l40s.45-u2-5403/infer105-check/` 与 `infer81-check/`，服务侧记录见[5403 服务交接文档](../../../docs_handoff/videoedit_service_gpu45_u2_5403.l40s.md)。

### 结论边界

本轮完成了双 L40S、BF16、FA、正常模型 40 步、49 帧、单请求串行的大小尺寸交错测试。GPU 4/5 专用配置开启自动模式，下次启动生效；已有运行进程需要重启才会应用。不能据此承诺任意素材、并发请求、其他模型/后端或更多帧数绝不 OOM；也未穷举每个中间尺寸的最快配置。需要更广覆盖时应扩充素材和形状测试，不能直接下调显存预留。

## 代码入口

- [预算与形状历史](../../../python/sglang/multimodal_gen/runtime/utils/videoedit_memory.py)
- [层驻留与预取切换](../../../python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py)
- [VideoEdit 阶段接入](../../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py)
- [策略与 CUDA 切换测试](../../../scripts/videoedit_benchmark/test_videoedit_residency.py)
- [真实模型基准](../../../scripts/videoedit_benchmark/run_adaptive_residency.py)、[结果校验](../../../scripts/videoedit_benchmark/report_adaptive_residency.py)

[返回优化汇总](./memory_optimization_20261004.md)
