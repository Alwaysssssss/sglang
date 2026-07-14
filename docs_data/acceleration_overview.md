# 加速总览数据表

## 填写约定

| 项目 | 约定 |
| --- | --- |
| 耗时单位 | 秒，建议保留 2 位小数 |
| 显存单位 | GiB，建议分别记录每张 GPU；如果只填一个值，默认表示最大单卡占用 |
| 总耗时 | 从请求开始到结果文件写完 |
| stage 耗时 | 以 pipeline / perf dump 记录为准；长视频多窗口时建议记录所有窗口之和 |
| DiT 耗时 | `VideoEditDenoisingStage` 内 transformer denoise 主循环耗时 |
| 加速比 | baseline 总耗时 / 当前方案总耗时 |

## 实验方案定义

| 方案编号 | 方案 | 模型权重 | GPU 数量 | TeaCache | torch.compile | 量化 | 推理步数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 原生无加速 |  |  |  |  |  |  | baseline |
| B | 双卡加速 |  |  |  |  |  |  |  |
| C | 双卡 + TeaCache 加速 |  |  |  |  |  |  |  |
| D | 双卡 + TeaCache + torch.compile 加速 |  |  |  |  |  |  |  |
| E | 双卡 + TeaCache + torch.compile + 量化 |  |  |  |  |  |  |  |
| F | 新模型权重，4 步推理 |  |  |  |  |  | 4 | 不额外叠加 TeaCache / torch.compile / 量化 |

## 总体结果

| 方案编号 | GPU 类型 | 1080 总耗时 | 1080 加速比 | 1080 起服务显存 | 1080 运行显存 | 1080 峰值显存 | 720 总耗时 | 720 加速比 | 720 起服务显存 | 720 运行显存 | 720 峰值显存 | 输出质量备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A |  |  |  |  |  |  |  |  |  |  |  |  |
| B |  |  |  |  |  |  |  |  |  |  |  |  |
| C |  |  |  |  |  |  |  |  |  |  |  |  |
| D |  |  |  |  |  |  |  |  |  |  |  |  |
| E |  |  |  |  |  |  |  |  |  |  |  |  |
| F |  |  |  |  |  |  |  |  |  |  |  |  |

## 输入规格

| 视频规格 | 输入路径 | mask 路径 | reference image 路径 | 原始分辨率 | 运行 crop 分辨率 | 输入帧数 | infer_len | overlap | 窗口数 | 输出帧数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1080p |  |  |  |  |  |  |  |  |  |  |  |
| 720p |  |  |  |  |  |  |  |  |  |  |  |

## 1080 Stage 耗时明细

| Stage | 说明 | A 耗时 | B 耗时 | C 耗时 | D 耗时 | E 耗时 | F 耗时 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 全局准备 | 读视频 / mask、扫描 bbox、构造窗口、准备全局上下文 |  |  |  |  |  |  |  |
| `VideoEditWindowValidationStage` | 校验窗口帧数、mask 数量、分辨率对齐 |  |  |  |  |  |  |  |
| `VideoEditTextEncodingStage` | prompt / negative prompt 文本编码 |  |  |  |  |  |  |  |
| `VideoEditImageEncodingStage` | CLIP reference image 编码；未开启 `use_clip` 时可填 0 或 N/A |  |  |  |  |  |  |  |
| `VideoEditConditionEncodingStage` | VAE encode masked video / raw video，准备 condition latent |  |  |  |  |  |  |  |
| `VideoEditLatentPreparationStage` | 初始化随机噪声、generator、latent shape |  |  |  |  |  |  |  |
| `VideoEditTimestepPreparationStage` | scheduler timesteps 准备 |  |  |  |  |  |  |  |
| `VideoEditLatentInitStage` | noise 或 add_noise 初始化 latents |  |  |  |  |  |  |  |
| `VideoEditDenoisingStage` | DiT denoise 主循环，通常是最耗时部分 |  |  |  |  |  |  |  |
| `VideoEditDecodingStage` | VAE decode latents 到视频帧 |  |  |  |  |  |  |  |
| `VideoEditWindowPostprocessStage` | 窗口输出检查和 metadata 记录 |  |  |  |  |  |  |  |
| 窗口结果合并 | `_commit_window_output()` 合并窗口到全局输出 buffer |  |  |  |  |  |  |  |
| 最终后处理与保存 | paste-back / crop-only、写视频文件、清理临时状态 |  |  |  |  |  |  |  |
| 其他 / 同步开销 | CUDA synchronize、进程通信、日志、无法归类的 IO |  |  |  |  |  |  |  |
| 总计 | 所有 stage 合计 |  |  |  |  |  |  |  |

## 720 Stage 耗时明细

| Stage | 说明 | A 耗时 | B 耗时 | C 耗时 | D 耗时 | E 耗时 | F 耗时 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 全局准备 | 读视频 / mask、扫描 bbox、构造窗口、准备全局上下文 |  |  |  |  |  |  |  |
| `VideoEditWindowValidationStage` | 校验窗口帧数、mask 数量、分辨率对齐 |  |  |  |  |  |  |  |
| `VideoEditTextEncodingStage` | prompt / negative prompt 文本编码 |  |  |  |  |  |  |  |
| `VideoEditImageEncodingStage` | CLIP reference image 编码；未开启 `use_clip` 时可填 0 或 N/A |  |  |  |  |  |  |  |
| `VideoEditConditionEncodingStage` | VAE encode masked video / raw video，准备 condition latent |  |  |  |  |  |  |  |
| `VideoEditLatentPreparationStage` | 初始化随机噪声、generator、latent shape |  |  |  |  |  |  |  |
| `VideoEditTimestepPreparationStage` | scheduler timesteps 准备 |  |  |  |  |  |  |  |
| `VideoEditLatentInitStage` | noise 或 add_noise 初始化 latents |  |  |  |  |  |  |  |
| `VideoEditDenoisingStage` | DiT denoise 主循环，通常是最耗时部分 |  |  |  |  |  |  |  |
| `VideoEditDecodingStage` | VAE decode latents 到视频帧 |  |  |  |  |  |  |  |
| `VideoEditWindowPostprocessStage` | 窗口输出检查和 metadata 记录 |  |  |  |  |  |  |  |
| 窗口结果合并 | `_commit_window_output()` 合并窗口到全局输出 buffer |  |  |  |  |  |  |  |
| 最终后处理与保存 | paste-back / crop-only、写视频文件、清理临时状态 |  |  |  |  |  |  |  |
| 其他 / 同步开销 | CUDA synchronize、进程通信、日志、无法归类的 IO |  |  |  |  |  |  |  |
| 总计 | 所有 stage 合计 |  |  |  |  |  |  |  |

## 1080 DiT Denoise 细分

| 方案编号 | 窗口数 | 推理步数 | CFG cond forward 数 | CFG uncond forward 数 | transformer forward 总数 | TeaCache 命中 / 跳过数 | DiT 总耗时 | DiT 占总耗时比例 | 平均每 step 耗时 | 平均每 forward 耗时 | scheduler step 耗时 | 多卡通信耗时 | torch.compile 编译 / warmup 耗时 | DiT 峰值显存 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 720 DiT Denoise 细分

| 方案编号 | 窗口数 | 推理步数 | CFG cond forward 数 | CFG uncond forward 数 | transformer forward 总数 | TeaCache 命中 / 跳过数 | DiT 总耗时 | DiT 占总耗时比例 | 平均每 step 耗时 | 平均每 forward 耗时 | scheduler step 耗时 | 多卡通信耗时 | torch.compile 编译 / warmup 耗时 | DiT 峰值显存 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F |  | 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 窗口级耗时明细

如果同一个样例被切成多个 81 帧窗口，这里记录每个窗口的 stage 耗时，方便排查首窗口、末窗口、compile warmup、尾部 padding 等差异。

| 视频规格 | 方案编号 | 窗口编号 | 有效帧范围 | crop 分辨率 | text encode | image encode | condition encode | latent prep | timestep prep | latent init | DiT denoise | decode | postprocess | commit | 窗口总耗时 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1080p | A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1080p | B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1080p | C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1080p | D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1080p | E |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1080p | F |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 720p | A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 720p | B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 720p | C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 720p | D |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 720p | E |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 720p | F |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 1080 与 720 对比

| 方案编号 | 1080 总耗时 | 720 总耗时 | 1080 / 720 总耗时比 | 1080 DiT 耗时 | 720 DiT 耗时 | 1080 / 720 DiT 耗时比 | 1080 峰值显存 | 720 峰值显存 | 1080 / 720 显存比 | 结论备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A |  |  |  |  |  |  |  |  |  |  |
| B |  |  |  |  |  |  |  |  |  |  |
| C |  |  |  |  |  |  |  |  |  |  |
| D |  |  |  |  |  |  |  |  |  |  |
| E |  |  |  |  |  |  |  |  |  |  |
| F |  |  |  |  |  |  |  |  |  |  |

## 实验环境记录

| 项目 | 值 |
| --- | --- |
| 机器型号 |  |
| GPU 型号与数量 |  |
| CUDA 版本 |  |
| Driver 版本 |  |
| PyTorch 版本 |  |
| sglang commit |  |
| 模型路径 |  |
| transformer 路径 |  |
| quant 权重 / 配置路径 |  |
| prompt |  |
| negative prompt |  |
| dtype |  |
| seed |  |
| attention backend |  |
| decode mode |  |
| mask downsample mode |  |
| bbox expand scale / padding |  |
| 是否 paste-back |  |
| 是否保存 crop-only |  |
| 统计显存方式 |  |
| 统计耗时方式 |  |
| perf dump 路径 |  |
| stage dump 路径 |  |
