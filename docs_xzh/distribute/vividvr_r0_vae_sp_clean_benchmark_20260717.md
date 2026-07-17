# Vivid-VR R0 基线 VAE SP2/SP4 纯净服务测试验收

## 结论

2026-07-17 按历史 R0 的 `130f / 20 step / seed 42 / SDPA eager` 服务口径，串行完成 SP2、SP4 两组正式请求。两组仅新增 SP 拓扑及 VAE tiled encode/decode 空间并行；`torch.compile`、modulation fusion、CFG parallel、cache 和 quantization 均未启用，也没有 eager warmup 请求。

| 方案 | raw VAE encode | encode 所在准备阶段 | 相对 R0 | raw VAE decode | decode/trim 阶段 | 相对 R0 | denoise | 相对 R0 | 模型推理 | 相对 R0 | 端到端 | 相对 R0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0，1 GPU | 未采集 | 60.2926 s | 1.000× | 未采集 | 111.9847 s | 1.000× | 928.8722 s | 1.000× | 1102.8278 s | 1.000× | 1111.8279 s | 1.000× |
| SP2 | 30.4623 s | 36.6628 s | 1.6445× | 57.4646 s | 57.7200 s | 1.9401× | 471.7694 s | 1.9689× | 567.5467 s | 1.9431× | 571.1127 s | 1.9468× |
| SP4 | 15.5375 s | 22.7383 s | 2.6516× | 28.7739 s | 29.0414 s | 3.8560× | 258.6430 s | 3.5913× | 311.9901 s | 3.5348× | 320.7514 s | 3.4663× |

相对 R0，SP2 的 encode 所在准备阶段、decode/trim、模型推理和端到端耗时分别下降 39.19%、48.46%、48.54% 和 48.63%；SP4 分别下降 62.29%、74.07%、71.71% 和 71.15%。

历史 R0 record 生成时还没有 `vae_encode_seconds` / `vae_decode_seconds` 埋点，因此不能严谨地给出 raw VAE kernel 相对 R0 的直接加速比。表中的 R0 可比口径是现有三份 record 都具备的 `VividVRLongClipPreparationStage` 和 `VividVRMultiClipDecodeTrimStage`；前者除 encode 外还包含准备开销，后者除 decode 外还包含 trim 开销。SP2/SP4 的 raw VAE 数值来自新增 runtime 埋点。

## 测试矩阵与运行时确认

- R0：单卡、`requested_backend=sdpa`、`effective_backend=sdpa`、eager、无 VAE SP，使用历史 record，不重跑。
- SP2：GPU 0–1、`SP=2`、`requested_backend=sdpa`、`effective_backend=sdpa_sp`、`vae_sp_effective=true`、`vae_encode_sp_effective=true`。
- SP4：GPU 0–3、`SP=4`，其余运行时语义与 SP2 相同。
- 两个 treatment 均确认 `torch_compile_applied=false`、`modulation_fusion_applied=false`、`cfg_parallel_enabled=false`、`cache=disabled`、`quantization=disabled`。
- `sdpa_sp` 是多卡 SP 拓扑下的有效 distributed joint-attention backend，不是额外打开的 benchmark 加速选项。多卡相对 R0 的模型/端到端结果包含当前 SP 拓扑本身带来的 denoise 并行收益。
- 两组 eager treatment 各执行一次 formal 请求，`warmup_record=null`，不存在 warmup record。

## 质量与服务链验收

| 方案 | record 状态 | SSIM mean | SSIM min | failed frame ratio | callback / 上传 |
| --- | --- | ---: | ---: | ---: | --- |
| R0 | `quality_failed` | 0.98448865 | 0.97894608 | 2/130 | 历史基线 |
| SP2 | `quality_failed` | 0.98430713 | 0.97819230 | 3/130 | succeeded |
| SP4 | `quality_failed` | 0.98457901 | 0.97764460 | 2/130 | succeeded |

两组服务请求均成功完成推理、对象存储上传、进度查询和最终 callback；`quality_failed` 是严格逐帧质量门槛的结果，不是服务或性能采集失败。SP2 相对 R0 的 mean/min SSIM 分别变化 -0.00018152/-0.00075378；SP4 分别变化 +0.00009036/-0.00130147。

结果对象：

- SP2：`http://127.0.0.1:4566/flowcut/acceleration-benchmark/vividvr_r0_vae_sp2_clean_20260717-R0_VAE_SP2-formal.mp4`
- SP4：`http://127.0.0.1:4566/flowcut/acceleration-benchmark/vividvr_r0_vae_sp4_clean_20260717-R0_VAE_SP4-formal.mp4`

## 产物与可复现性

- R0 record：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/R0_formal.json`
- R0 SHA-256：`82855272cfa5e759ca70fdc915d6e6fbd81c3f4d2c1c3cf627c179cd719099a5`
- R0 mtime_ns：`1784167925220287168`；正式运行前后保持一致。
- SP2 batch：`Vivid_Acceptance/acceleration_benchmark/vividvr_r0_vae_sp2_clean_20260717`
- SP4 batch：`Vivid_Acceptance/acceleration_benchmark/vividvr_r0_vae_sp4_clean_20260717`
- SP2 tmux：`vividvr_accel_batch_vividvr_r0_vae_sp2_clean_20260717`
- SP4 tmux：`vividvr_accel_batch_vividvr_r0_vae_sp4_clean_20260717`

正式运行前 GPU 0–3 利用率均为 0%，但存在其他 PID 命名空间中的空闲 CUDA context，故沿用 benchmark runner 的 `--allow-idle-gpu-processes`；启动前仍要求被选 GPU 利用率严格为 0%。SP2 完全退出且 GPU 利用率回到 0% 后才启动 SP4，未并发运行两组正式请求，也未终止外部进程。

## 复算公式

```text
speedup = R0 seconds / treatment seconds
GPU seconds = treatment GPU count * treatment total seconds
resource efficiency vs R0 = R0 GPU seconds / treatment GPU seconds
```

SP2 的 GPU·秒为 1142.2255，资源效率为 R0 的 97.34%；SP4 的 GPU·秒为 1283.0056，资源效率为 R0 的 86.66%。SP4 绝对延迟更低，但单位 GPU 资源效率低于 SP2。
