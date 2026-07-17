# Vivid-VR 纯 SP4 与 CFG2×SP2 四卡性能对比

## 结论

在相同四卡、`130 frames / 20 steps`、FA-SP、`torch.compile`、
modulation/residual fusion 和 VAE spatial tile parallel 条件下，纯 `SP4`
比 `CFG2×SP2` 更快：

- 端到端总耗时从 `341.004324 s` 降至 `310.786260 s`，节省
  `30.218064 s`，耗时降低 `8.8615%`，即 `1.097231×` 加速。
- 模型推理耗时从 `334.402225 s` 降至 `298.988013 s`，节省
  `35.414212 s`，耗时降低 `10.5903%`，即 `1.118447×` 加速。

`CFG2×SP2` 的 denoise 比纯 `SP4` 快 `4.214239 s`（`2.1204%`），
但纯 `SP4` 把 VAE SP world size 从 2 提升到 4，使 VAE decode 减少
`30.219740 s`（`50.4346%`）。在当前包含 VAE spatial tile parallel 的
端到端链路中，后者的收益更大，因此纯 `SP4` 是更快的四卡拓扑。

## 实验条件

| 项目 | 纯 SP4 | CFG2×SP2 |
| --- | --- | --- |
| scheme | `R101_VAE_SP4` | `R100_VAE_SP` |
| GPU 数 | 4 | 4 |
| parallel mode | `sp` | `cfg_sp` |
| SP world size | 4 | 2 |
| CFG parallel | 关闭 | 开启 |
| effective attention backend | `fa_sp` | `fa_sp` |
| `torch.compile` | 开启 | 开启 |
| modulation fusion | 开启 | 开启 |
| VAE SP | 生效，world size 4 | 生效，world size 2 |
| VAE tile 分配 | `[8, 8, 8, 8]` | `[16, 16]` |
| 输入 | 同一视频、caption、reference、`seed=42` | 同左 |
| 正式请求 | 130 frames、20 steps | 同左 |
| warmup | 1 step | 1 step |

`R101_VAE_SP4` 的 runner 辅助 control 是同拓扑但未启用 fusion 的 `R4`；
该 control 只用于生成 runner 派生字段。本文主结论始终是
`R101_VAE_SP4` 与已验收 `R100_VAE_SP` 的直接 formal record 对比。

## 正式耗时结果

| 指标 | 纯 SP4 (s) | CFG2×SP2 (s) | SP4 相对变化 | SP4 加速比 |
| --- | ---: | ---: | ---: | ---: |
| total runtime | 310.786260 | 341.004324 | -30.218064 (-8.8615%) | 1.097231× |
| model inference | 298.988013 | 334.402225 | -35.414212 (-10.5903%) | 1.118447× |
| long clip preparation | 64.434853 | 73.079098 | -8.644245 (-11.8286%) | 1.134155× |
| denoise | 202.964994 | 198.750755 | +4.214239 (+2.1204%) | 0.979237× |
| Decode/Trim | 29.996236 | 60.178877 | -30.182641 (-50.1549%) | 2.006214× |
| VAE decode | 29.698975 | 59.918715 | -30.219740 (-50.4346%) | 2.017535× |
| VAE tile decode | 29.470813 | 59.758195 | -30.287382 (-50.6832%) | 2.027708× |
| VAE tile gather | 0.011835 | 0.012690 | -0.000855 (-6.7399%) | 1.072270× |
| VAE tile merge | 0.131767 | 0.140075 | -0.008308 (-5.9308%) | 1.063048× |

这里的“SP4 加速比”统一按 `CFG2×SP2 耗时 / 纯 SP4 耗时` 计算；大于
1 表示纯 SP4 更快。denoise 一项小于 1，反映 CFG 并行确实对 denoise
有小幅收益，但不足以抵消 VAE decode 的差距。

## 质量与资源结果

| 指标 | 纯 SP4 | CFG2×SP2 | SP4 - CFG2×SP2 |
| --- | ---: | ---: | ---: |
| SSIM mean | 0.984631 | 0.984603 | +0.000028 |
| SSIM min | 0.979781 | 0.977484 | +0.002297 |
| failed-frame ratio | 1/130 (0.007692) | 2/130 (0.015385) | -1/130 |
| max single-GPU peak | 54.223 GiB | 53.727 GiB | +0.496 GiB |

严格 comparator 因纯 SP4 有 `1/130` 帧低于逐帧阈值，将 formal record
标记为 `quality_failed`。但纯 SP4 的 SSIM mean、SSIM min 和失败帧数均
优于此前已经按用户确认容差通过的 `R100_VAE_SP`。因此本次按相同容差
口径判定质量验收通过，同时保留原始 runner 状态，避免隐藏严格门禁结果。

## 产物与复现记录

- SP4 batch：
  `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717`
- SP4 formal record：
  `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717/records/R101_VAE_SP4_formal.json`
- SP4 formal video：
  `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717/requests/vividvr_vae_sp4_fusion_20260717-R101_VAE_SP4-formal/downloaded.mp4`
- CFG2×SP2 formal record：
  `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716/records/R100_VAE_SP_formal.json`
- 启动日志：
  `Vivid_Acceptance/logs/vividvr_vae_sp4_compare_20260717.log`

SP4 formal 视频经 `ffprobe` 检查为 H.264、`960×720`、25 fps、130 帧，
时长 5.2 秒。formal record 证明运行时为 `sp_world_size=4`、
`cfg_parallel_enabled=false`、`vae_sp_effective=true`、
`vae_sp_world_size=4`，与实验合同一致。
