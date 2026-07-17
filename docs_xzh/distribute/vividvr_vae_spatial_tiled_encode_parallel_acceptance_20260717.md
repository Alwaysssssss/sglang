# Vivid-VR VAE Spatial Tiled Encode 并行验收记录

## 结论

本轮实现和三种真实拓扑的 bitwise 正确性验收已完成，但正式性能验收**未通过**，因此 `--vae-encode-sp` 继续保持默认关闭的实验性能力，不进入 `single_gpu_fa_compile`、`dual_gpu_fa_eager_compile` 或 `dual_gpu_sdpa_eager_compile` 默认配置。

- SP2、SP4、CFG2×SP2 的完整 posterior moments、等价 generator sampled latents 和 rank-divergent 输入检查均为 `torch.equal`。
- R99 treatment 未达到 1.5× Long Clip Preparation 门槛，且 Decode/Trim 回归超过 3%。
- R100 treatment 达到 Long Clip Preparation 门槛，但 Decode/Trim 回归超过 3%。
- R101 treatment 未达到 2.5× Long Clip Preparation 门槛，且 Decode/Trim 回归超过 3%。
- 三组 treatment 的模型推理耗时均改善，Denoise 回归均在 3% 以内。
- 历史 decode-only Control 未重跑；正式运行前后 SHA-256 与 `mtime_ns` 完全一致。

## 实现提交

| Commit | 内容 |
| --- | --- |
| `c25671d75` | 配置、CLI 与 pipeline wiring |
| `a3638976e` | tiled encode plan、worker、merge 与 transport 原语 |
| `24f4c51d2` | SP subgroup tiled encode dispatch |
| `693c20b25` | encode 统计、长视频聚合与服务 perf 透传 |
| `4069d92eb` | 真实 VAE bitwise 分布式验证工具 |
| `4a73aa98c` | 三个 treatment benchmark 注册与派生门槛 |
| `b66549cfa` | Control 防改、有效配置和 runner 验收保护 |
| `194ee2131` | 本轮代码格式化 |
| `36ca416c9` | rank-divergent 验收 JSON schema 对齐 |

## Bitwise 正确性

三次真实 GPU 验证均在 `tmux` 中运行。输入为 `bfloat16`、`[1, 3, 17, 720, 960]`，并显式覆盖 non-contiguous tensor；moments 形状为 `[1, 32, 5, 90, 120]`，sampled latents 形状为 `[1, 16, 5, 90, 120]`。

| 拓扑 | 验收 JSON | SHA-256 | `overall_pass` | moments exact | sampled latents exact | rank-divergent moments exact | rank-divergent latents exact | non-contiguous | SP subgroup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SP2 | `Vivid_Acceptance/indicator/vae_encode_sp_sp2_seed42_20260717T041530Z.json` | `58afe70162dea391f97fcf375469e196ae49b1421e25fae68aa74be2cf5c5c40` | `true` | 全 rank `true` | 全 rank `true` | 全 rank `true` | 全 rank `true` | `true` | `[0, 1]` |
| SP4 | `Vivid_Acceptance/indicator/vae_encode_sp_sp4_seed42_20260717T041620Z.json` | `fa9f42cf4d61eef9e9616eadbb7420051f07dc2785a91bd98a7cfcad2d4d3352` | `true` | 全 rank `true` | 全 rank `true` | 全 rank `true` | 全 rank `true` | `true` | `[0, 1, 2, 3]` |
| CFG2×SP2 | `Vivid_Acceptance/indicator/vae_encode_sp_cfg2_sp2_seed42_20260717T041708Z.json` | `69b30b7e92bd8091f1ce16bd84b199a1ff3cdd3aaba5025993b710903f920875` | `true` | 全 rank `true` | 全 rank `true` | 全 rank `true` | 全 rank `true` | `true` | `[0, 1]`、`[2, 3]` |

对应日志：

- `Vivid_Acceptance/logs/vae_encode_sp_sp2_20260717T041530Z.log`
- `Vivid_Acceptance/logs/vae_encode_sp_sp4_20260717T041620Z.log`
- `Vivid_Acceptance/logs/vae_encode_sp_cfg2_sp2_20260717T041708Z.log`

CFG2×SP2 使用两个独立 subgroup；第一组 seed 为 42，第二组 seed 为 43。组内 root input 与 seed 一致，组间 root input 不同，未发生 tensor 或 marker 混组。

## 历史 Control 不变性

三个 Control 都是已经存在的 **decode-only** 正式 record，仅作为只读输入。前后指纹日志分别为：

- `Vivid_Acceptance/logs/vae_encode_sp_control_fingerprints_before_20260717.log`
- `Vivid_Acceptance/logs/vae_encode_sp_control_fingerprints_after_20260717.log`

两份日志经 `cmp` 检查逐字节一致。

| Control | Record | SHA-256（前后） | `mtime_ns`（前后） |
| --- | --- | --- | ---: |
| R99_VAE_SP | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716/records/R99_VAE_SP_formal.json` | `ddad7e3e305dd1b17578f026dded44287ff30fad17bed9d3b319607d669a2104` | `1784204506115949677` |
| R100_VAE_SP | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716/records/R100_VAE_SP_formal.json` | `46aa80b02194e1da8259fcc53e796ff8660691010ff6108a470d56db53db8497` | `1784206024900878220` |
| R101_VAE_SP4 | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717/records/R101_VAE_SP4_formal.json` | `049c9b8a40c3b818fb08dc646b645a8e67b13c0e0ec19a280cba7eebc5f7c8c0` | `1784251333118153815` |

## 正式 Treatment 产物

三条 compile treatment 均只执行一次 1-step warmup 和一次 20-step formal，并串行运行。tmux session 已正常结束。

### R99：SP2

- Batch：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717`
- tmux：`vividvr_accel_batch_vividvr_vae_encode_sp_r99_20260717`
- Record：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717/records/R99_VAE_ENCODE_SP_formal.json`
- Video：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717/requests/vividvr_vae_encode_sp_r99_20260717-R99_VAE_ENCODE_SP-formal/downloaded.mp4`
- Perf：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717/requests/vividvr_vae_encode_sp_r99_20260717-R99_VAE_ENCODE_SP-formal/perf.json`
- Compare：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717/requests/vividvr_vae_encode_sp_r99_20260717-R99_VAE_ENCODE_SP-formal/compare.json`
- Callback：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717/logs/callbacks.jsonl`
- Service log：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717/logs/R99_VAE_ENCODE_SP_service.log`

### R100：CFG2×SP2

- Batch：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r100_20260717`
- tmux：`vividvr_accel_batch_vividvr_vae_encode_sp_r100_20260717`
- Record：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r100_20260717/records/R100_VAE_ENCODE_SP_formal.json`
- Video：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r100_20260717/requests/vividvr_vae_encode_sp_r100_20260717-R100_VAE_ENCODE_SP-formal/downloaded.mp4`
- Perf：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r100_20260717/requests/vividvr_vae_encode_sp_r100_20260717-R100_VAE_ENCODE_SP-formal/perf.json`
- Compare：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r100_20260717/requests/vividvr_vae_encode_sp_r100_20260717-R100_VAE_ENCODE_SP-formal/compare.json`
- Callback：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r100_20260717/logs/callbacks.jsonl`
- Service log：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r100_20260717/logs/R100_VAE_ENCODE_SP_service.log`

### R101：SP4

- Batch：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp4_r101_20260717`
- tmux：`vividvr_accel_batch_vividvr_vae_encode_sp4_r101_20260717`
- Record：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp4_r101_20260717/records/R101_VAE_ENCODE_SP4_formal.json`
- Video：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp4_r101_20260717/requests/vividvr_vae_encode_sp4_r101_20260717-R101_VAE_ENCODE_SP4-formal/downloaded.mp4`
- Perf：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp4_r101_20260717/requests/vividvr_vae_encode_sp4_r101_20260717-R101_VAE_ENCODE_SP4-formal/perf.json`
- Compare：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp4_r101_20260717/requests/vividvr_vae_encode_sp4_r101_20260717-R101_VAE_ENCODE_SP4-formal/compare.json`
- Callback：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp4_r101_20260717/logs/callbacks.jsonl`
- Service log：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp4_r101_20260717/logs/R101_VAE_ENCODE_SP4_service.log`

## 正式性能与质量指标

Control 与 treatment 均为相同的 130f / 20-step 口径。Long Clip、Denoise、Decode/Trim 为各自 stage 秒数；回归比为 treatment 相对 Control。GPU·秒使用物理 GPU 数量计算，CFG2×SP2 为 4 卡而不是 SP subgroup 大小 2。

| Treatment | Topology | Total（s） | Model（s） | Long Clip（s） | Long Clip speedup | Denoise（s / 回归） | Decode/Trim（s / 回归） | GPU·秒 | SSIM mean | SSIM min | failed ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R99_ENCODE_SP | SP2 | 491.181 | 486.321 | 42.176 | 1.4441× | 380.798 / -0.10% | 61.428 / +4.23% | 982.362 | 0.984562 | 0.976291 | 1.5385% |
| R100_ENCODE_SP | CFG2×SP2 | 310.903 | 309.705 | 47.257 | 1.5464× | 195.837 / -1.47% | 64.287 / +6.83% | 1243.610 | 0.984504 | 0.978161 | 1.5385% |
| R101_ENCODE_SP4 | SP4 | 270.743 | 265.806 | 28.748 | 2.2414× | 203.575 / +0.30% | 30.963 / +3.22% | 1082.971 | 0.984169 | 0.977092 | 1.5385% |

| Treatment | Long Clip gate | Model improvement gate | Denoise ≤3% gate | Decode/Trim ≤3% gate | Performance 总门槛 | Quality vs Control |
| --- | --- | --- | --- | --- | --- | --- |
| R99_ENCODE_SP | **FAIL**：1.4441× < 1.5× | PASS：1.0334× | PASS：-0.10% | **FAIL**：+4.23% | **FAIL** | **FAIL**：mean +0.000101、min -0.003312、failed ratio +0.0000 |
| R100_ENCODE_SP | PASS：1.5464× ≥ 1.5× | PASS：1.0797× | PASS：-1.47% | **FAIL**：+6.83% | **FAIL** | **FAIL**：mean -0.000099、min +0.000678、failed ratio +0.0000 |
| R101_ENCODE_SP4 | **FAIL**：2.2414× < 2.5× | PASS：1.1248× | PASS：+0.30% | **FAIL**：+3.22% | **FAIL** | **FAIL**：mean -0.000462、min -0.002689、failed ratio +0.007692 |

三组 `vae_encode_sp_effective=true`，tile 分配分别为每 clip `[8, 8]`、`[8, 8]`（每个 CFG subgroup）和 `[4, 4, 4, 4]`，没有 fallback 或 collective 错误。派生 speedup、regression 和 GPU·秒已由 Control/Treatment 原始 JSON 独立复算，与 record 的绝对差均小于 `1e-9`。

质量比较使用 runner 的严格 “not worse than Control” 口径；三个历史 Control 自身的严格 `control_quality_passed` 也是 `false`，但这不豁免 treatment 的质量 gate。本轮不以 SSIM 替代 bitwise 门槛，也不把质量失败改写为通过。

## 失败分析与环境说明

- R99 formal 的 Long Clip stage 比同批 warmup 慢约 4.75 秒；encode gather/merge 合计仅约 0.03 秒，没有证据表明 collective 是主要瓶颈。
- 三组模型总推理耗时均改善，但 Decode/Trim 的跨运行波动导致回归超过固定 3% 门槛；R101 超出约 0.22 个百分点。
- 正式执行时 GPU 0–3 存在其他 PID namespace 留下的 idle CUDA context。runner 使用既有 `--allow-idle-gpu-processes` 保护选项，仅在所选 GPU 即时利用率严格为 0% 时继续；没有终止或修改这些外部进程。该环境偏差是后续复核性能时需要保留的风险。
- 按计划不修改门槛、不重跑历史 Control，也不因结果失败挑选或重复 treatment。

## 默认语义、服务契约与回滚

- `--vae-sp` 仍只控制 tiled decode；`--vae-encode-sp` 只控制 tiled encode，默认值为 `False`。
- Phase C/D/E 的 clip、caption、denoise、decode/trim、color fix、stitch 语义没有改变。
- 三条正式默认配置和 FlowCut 请求、回调、取消、对象存储、输入清理、进度查询契约均未改变，因此不修改 `AGENTS.md`。
- 回滚实验能力只需从启动命令移除 `--vae-encode-sp`；是否保留 `--vae-sp` 由原 decode 配置独立决定。
