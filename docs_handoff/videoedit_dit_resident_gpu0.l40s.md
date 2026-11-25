# GPU 0 单卡 DiT 不卸载测试

日期：2026-09-29。物理 GPU 0，NVIDIA L40S，46068 MiB。镜像 `sglang-videoedit-dev-v2:l40s`。正常模型，40 步，seed 42，guidance scale 5.0，TeaCache 开启、阈值 0.3。复用历史请求 `4d187b56-62b7-4121-8d7b-7528150e35d9` 的本地输入，关闭云端上传及回调。

## 结果

直接使用 `--dit-cpu-offload false --dit-layerwise-offload false` 的完整服务请求失败，没有有效的端到端耗时。模型启动成功，但第一个请求在 `VideoEditTextEncodingStage` 将文本编码器搬入 GPU 时 OOM：进程已占用 44.37 GiB，剩余 17.25 MiB，无法再申请 80 MiB。文本、图像编码器及 VAE 的 CPU offload 均已开启，但文本编码阶段仍会将文本编码器搬入 GPU。

为单独测量常驻 DiT 的去噪性能，补充运行分阶段实验：DiT 初始加载到 CPU；输入编码完成后，将其他模型转到 CPU，再一次性将 DiT 转到 GPU。计时前设置 `dit_cpu_offload=False`，断言全部 DiT 参数在 CUDA、逐层卸载及 FSDP 关闭。预热一次后正式计时三次，期间 DiT 保持常驻。每轮恢复初始 latent、调度器及 TeaCache 状态，计时前后 CUDA synchronize。

| 项目 | 结果 |
| --- | --- |
| 预热去噪 | 48.809 秒 |
| 正式去噪三次 | 46.064 / 45.998 / 46.125 秒 |
| 去噪中位数 | **46.064 秒** |
| DiT 参数显存 | 30.539 GiB |
| 去噪 PyTorch 峰值分配 | 32.970 GiB |
| 整个实验请求 nvidia-smi 采样峰值 | 39193 MiB（约 38.27 GiB） |
| 初始 latent 形状 | `[1,16,13,60,68]` |

四轮最终 latent 哈希完全相同。实验请求最终成功生成视频。

**46.064 秒只包含去噪，不包括编码、首次搬入 DiT、解码、输出。** 实验请求的 228.558 秒包含四轮去噪，不能当作普通单次请求的端到端耗时。此方案也没有解决连续请求中常驻 DiT 与文本编码器争用显存的问题。

此前逐层预取测试的去噪中位数为 45.573 秒，与本次常驻结果接近；因不是本轮同时重测的受控对照，不能据此认定卸载更快或量化性能差异。

## 证据及收尾

- 直接常驻服务的启动命令、OOM 日志及失败请求结果：`.local/gpu0-dit-resident-benchmark/`。
- 分阶段常驻测试脚本及 hook：`.local/gpu0-dit-resident-denoise/run.py`、`hooks/`。
- 三次计时与输出哈希：`.local/gpu0-dit-resident-denoise/single/rank0.json`。
- 成功请求、输出视频及逐秒采样：`.local/gpu0-dit-resident-denoise/single/`。

两个测试容器均已退出，GPU 0 显存恢复为 0 MiB。GPU 7 的业务服务未重启，测试后 normal、DMD 健康检查均通过。未修改服务源代码，未限制测试容器 CPU 或内存。
