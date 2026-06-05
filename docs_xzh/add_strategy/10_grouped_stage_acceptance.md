# 5 个大阶段验收目标与执行后校验清单

这份文档用于后续代码修改后的阶段验收。

- 细粒度验收源文档：`08_stage7_execution_roadmap.md`
- 执行顺序源文档：`09_code_mod_order.md`
- 本文目标：把 `8` 个小 phase 的验收要求汇总成 `5` 个大阶段的可执行检查单，方便每完成一阶段代码后立即判断是否可以继续。

## 1. 总使用规则

### 1.1 验收口径

- `8` 个小 phase 的细化标准仍以 `08_stage7_execution_roadmap.md` 为准。
- 本文只做大阶段汇总，不替代小 phase 的技术细项。
- 任何大阶段只要存在一个阻断项未关闭，就不应进入下一阶段。

### 1.2 每个阶段结束后必须保留的验收证据

- 本阶段实际修改的文件列表
- 本阶段对应的 smoke / forward / e2e 测试结果
- 关键日志或关键 shape 记录
- 若涉及视频输出，保留输出文件路径
- 若涉及 reference 对齐，保留完整指标：
  - `ssim_mean`
  - `ssim_min`
  - `mse_mean`
  - `mse_max`
  - `mae_mean`
  - `mae_max`
  - `failed_frames`

### 1.3 Reference 阈值口径

逐帧对齐的统一 reference：

- 原始推理命令：`/home/zhiheng/Vivid-VR/xzh_docs/run.md`
- 测试输入目录：`/home/zhiheng/Vivid-VR/input/720p`
- 集成阶段固定 caption 文件：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- reference 输出：
  - `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`

默认宽松基线：

- `min_ssim = 0.90`
- `max_mse = 150.0`
- `max_mae = 8.0`
- `allow_frame_count_delta = 1`
- `max_failed_frame_ratio = 0.05`

strict 档位：

- `min_ssim = 0.96`
- `max_mse = 60.0`
- `max_mae = 5.0`
- `max_failed_frame_ratio = 0.0`

## 2. 大阶段 A：方案冻结 + 工程入口

### 2.1 对应范围

- 对应小阶段：`Phase 1 + Phase 2`

### 2.2 本阶段验收目标

- MVP 范围、非目标、reference 口径、checkpoint 契约已经冻结
- `VividVR` family 已能被 `sglang.multimodal_gen` 正确识别
- `config / sampling / registry` 工程入口已经稳定

### 2.3 必须完成的检查项

- `Vivid-VR` 的任务定义、MVP 边界、非目标在文档中无歧义
- 已明确第一版不接入：
  - auto caption
  - 在 `sglang` 环境中实时运行 `CogVLM2`
  - TextFixer
  - EasyOCR
  - ESRGAN
  - 长视频时间聚合
  - 多卡 TP/SP
- 已明确集成阶段统一读取：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- `VividVRPipelineConfig` 可实例化，默认字段完整
- `VividVRSamplingParams` 可实例化，基本字段校验正常
- `SamplingParams.from_user_kwargs()` 可运行
- `registry.py` 新增 detector 后，不影响已有模型 family
- 给定 `VividVR` 模型名关键字时，能正确解析到：
  - `pipeline_config_cls = VividVRPipelineConfig`
  - `sampling_param_cls = VividVRSamplingParams`

### 2.4 验收通过标准

- `import / py_compile / registry smoke test` 通过
- 无已有 family 回归
- 团队对范围、reference、阈值口径无开放分歧

### 2.5 不通过时禁止进入下一阶段的阻断项

- family 识别不稳定
- config 无法构造
- sampling params 合同未定
- 文档范围仍有争议

## 3. 大阶段 B：核心模型底座迁移

### 3.1 对应范围

- 对应小阶段：`Phase 3 + Phase 4`

### 3.2 本阶段验收目标

- `CogVideoX` base transformer / VAE / scheduler 已可原生运行
- `VividVR` 增量 transformer / controlnet 已可叠加在 base 上工作
- 组件级 forward 合同已稳定

### 3.3 必须完成的检查项

- `CogVideoX` transformer 类能被 `ModelRegistry` 发现
- 可基于 `ckpts/CogVideoX1.5-5B/transformer/config.json` 完成初始化
- VAE 能完成：
  - `encode([B,C,T,H,W])`
  - `decode(latents)`
- VAE 关键配置与 reference 一致：
  - `scaling_factor`
  - `temporal_compression_ratio`
  - `latent_channels`
- scheduler 至少完成单步语义对照：
  - `pred_original_sample` 数值范围合理
  - `restoration_guidance_scale=-1` 行为一致
  - `restoration_guidance_scale>0` 不报错且输出合理
- 三组增量权重可独立加载：
  - `connectors.pt`
  - `control_feat_proj.pt`
  - `control_patch_embed.pt`
- controlnet 可加载：
  - `ckpts/Vivid-VR/controlnet/`
- `control_hidden_states` 层数与 block 数对应
- CFG 分支开启时，正负 prompt batch 拼接不报错
- 组合前向在固定输入下无：
  - NaN
  - Inf
  - shape mismatch
  - dtype mismatch

### 3.4 验收通过标准

- base 组件和增量组件都能独立完成一次稳定前向
- VividVR 模型链可独立完成一次 noise prediction
- 核心 shape 合同稳定，可供 pipeline 调用

### 3.5 不通过时禁止进入下一阶段的阻断项

- scheduler 语义仍不确定
- VAE encode/decode 不稳定
- controlnet 与 transformer 接口不稳定
- 存在 NaN / shape mismatch / dtype mismatch

## 4. 大阶段 C：单 clip MVP + Reference 对齐

### 4.1 对应范围

- 对应小阶段：`Phase 5`

### 4.2 本阶段验收目标

- 建立单 clip 原生端到端 pipeline
- 在固定 seed 下稳定输出视频
- 单 clip 结果达到 reference 对齐门槛

### 4.3 必须完成的检查项

- `runtime/pipelines/vividvr_pipeline.py` 可完成单 clip 调度
- `before_denoising / denoising / decoding` 主链闭环成立
- `runtime/vividvr/preprocess.py` 和 `runtime/vividvr/tiling.py` 的输入输出合同稳定
- caption 来源固定为：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 输出视频的帧数、分辨率、dtype、保存路径都正确
- 固定 seed 下可重复
- 关键中间 shape 有日志或可追踪记录
- 对 reference mp4 完成逐帧比较，并输出：
  - `ssim_mean`
  - `ssim_min`
  - `mse_mean`
  - `mse_max`
  - `mae_mean`
  - `mae_max`
  - `failed_frames`

### 4.4 验收通过标准

- 单 clip 端到端输出稳定
- 满足默认宽松基线：
  - `min_ssim >= 0.90`
  - `max_mse <= 150.0`
  - `max_mae <= 8.0`
  - `frame_count_delta <= 1`
  - `failed_frame_ratio <= 0.05`

### 4.5 不通过时禁止进入下一阶段的阻断项

- 单 clip 仍有崩溃、空输出、明显错帧
- reference 对齐未达宽松基线
- `ssim_min` 和 `mse_max` 未被稳定上报
- 集成过程中误调用了实时 `CogVLM2` caption 链路

## 5. 大阶段 D：长视频能力 + 可选增强

### 5.1 对应范围

- 对应小阶段：`Phase 6 + Phase 7`

### 5.2 本阶段验收目标

- 长视频 clip split / merge / temporal orchestration 稳定
- `caption / postprocess` 作为可选模块接入，且不破坏主链
- 默认长视频验收输入固定，避免 benchmark 口径漂移：
  - 输入视频固定为 `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
  - 对应 prompts / caption sidecar 固定为 `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
  - 验收对比的 reference 视频固定为 `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_6step/videos/test_video_long_960x720_130f.mp4`
  - 上述 reference 是原版 `Vivid-VR` 在 `step=6` 条件下得到的长视频结果，后续默认以这条结果作为公平对比基准

### 5.3 必须完成的检查项

- `runtime/vividvr/windowing.py` 能稳定产生 clip 切分计划
- clip 间的 latent / frame merge 规则与 reference 理解一致
- overlap 区域无明显重复、缺帧、接缝跳变
- 长视频输出帧数与总时长合理
- 默认长视频验收必须使用：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- 默认长视频 reference 必须使用：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_6step/videos/test_video_long_960x720_130f.mp4`
- 长视频 candidate 与 reference 完成逐帧比较
- `captioning.py` 和 `postprocess.py` 可独立启停
- `captioning.py` 当前阶段若存在，仅允许保留占位或文件读取逻辑，不允许默认调用 `CogVLM2`
- 可选模块失败时，主生成链可降级运行，而不是整链崩溃

### 5.4 验收通过标准

- 长视频 merge 稳定，无明显边界伪影
- 长视频结果达到默认宽松基线，或不低于单 clip 阶段的可接受误差级别
- 验收结果指标口径与 `Phase C` 相同；长视频 benchmark 至少要稳定上报与 `Phase C` 相同的一组核心指标：
  - `ssim_mean`
  - `ssim_min`
  - `mse_mean`
  - `mse_max`
  - `mae_mean`
  - `mae_max`
  - `failed_frames`
  - `failed_frame_ratio`
- 若产出 JSON 验收报告，其字段集合应默认与 `Phase C` 验收 JSON 保持一致，只允许在此基础上新增长视频特有调试字段，不应减少既有质量指标或用时字段
- 可选模块开关测试通过，不破坏主链

### 5.5 不通过时禁止进入下一阶段的阻断项

- 长视频 clip merge 不稳定
- overlap 区域出现明显质量断层
- 可选模块无法独立关闭
- 可选模块异常会拖垮主链

## 6. 大阶段 E：性能收口 + 回归验收

### 6.1 对应范围

- 对应小阶段：`Phase 8`

### 6.2 本阶段验收目标

- 默认参数、backend、offload、compile 方案收口
- 建立可重复使用的 regression 套件
- 进入 strict 或接近 strict 的 release gate

### 6.3 必须完成的检查项

- 明确默认推理配置：
  - dtype
  - attention backend
  - VAE tiling / slicing
  - offload 策略
  - 是否启用 compile
- profile 结果可复现，至少包含：
  - 峰值显存
  - 端到端耗时
  - 关键模块耗时分布
- regression 集至少覆盖：
  - 短视频单 clip
  - 长视频多 clip
  - 可选文字区域或复杂内容视频
- 性能优化前后无明显质量回退
- strict 或接近 strict 的 reference 对齐报告已产出

### 6.4 验收通过标准

- 有稳定默认参数组合
- 有可复用 regression 流程
- reference 质量达到 strict 或接近 strict 的发布门槛：
  - 目标值优先参考：
    - `min_ssim >= 0.96`
    - `max_mse <= 60.0`
    - `max_mae <= 5.0`
    - `failed_frame_ratio = 0.0`

### 6.5 不通过时禁止视为完成的阻断项

- 性能结论无法复现
- regression 集不完整
- 优化后出现质量回退
- release gate 指标未达标且无明确豁免说明

## 7. 最终推进门槛

后续所有代码修改建议都以这三个门槛控制是否继续推进：

1. `大阶段 A` 结束后，`registry + config` 必须稳定
2. `大阶段 C` 结束后，单 clip reference 对齐必须达标
3. `大阶段 C` 未通过前，不进入性能优化收口阶段

## 8. 推荐使用方式

后续每完成一个大阶段代码修改后，按下面顺序执行验收：

1. 对照本文定位当前大阶段的必查项
2. 回到 `08_stage7_execution_roadmap.md` 检查对应小 phase 细项
3. 记录测试输出、关键日志、reference 指标
4. 只有通过“验收通过标准”并关闭阻断项，才进入下一阶段
