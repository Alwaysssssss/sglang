# Stage 7: 最终实施路线图与阶段验收细化

## 1. 路线图总览

| Phase | 目标 | 涉及文件 | 预估工作量 | 前置依赖 | 验证方法 | 完成标准 |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 1 | 冻结范围与 checkpoint 契约 | `add_strategy/*` | 小 | 无 | 文档审阅 | 功能边界、目录契约、reference 基准冻结 |
| Phase 2 | 补齐 config / sampling / registry 框架 | `configs/pipeline_configs/vividvr.py` `configs/sample/vividvr.py` `registry.py` | 小到中 | Phase 1 | import / registry smoke test | pipeline family 可识别，参数类可构造 |
| Phase 3 | 移植 CogVideoX base transformer / VAE / scheduler | `runtime/models/dits/cogvideox.py` `runtime/models/vaes/cogvideox.py` `runtime/models/schedulers/cogvideox_dpm_vividvr.py` | 大 | Phase 2 | 组件级前向与单步测试 | 核心 base 组件可加载，可单独前向 |
| Phase 4 | 移植 VividVR transformer / controlnet | `runtime/models/dits/cogvideox_vividvr.py` `runtime/models/dits/cogvideox_vividvr_controlnet.py` | 大 | Phase 3 | controlnet + transformer shape test | control 分支可运行，增量权重可加载 |
| Phase 5 | 建立单 clip MVP pipeline | `runtime/pipelines/vividvr_pipeline.py` `model_specific_stages/vividvr.py` `runtime/vividvr/tiling.py` | 大 | Phase 4 | 短视频端到端测试 + reference 对齐 | 单 clip 输出闭环成立且对齐达标 |
| Phase 6 | 接入长视频 orchestration | `runtime/vividvr/windowing.py` `vividvr_pipeline.py` | 中到大 | Phase 5 | 长视频对照测试 + overlap 检查 | clip merge 稳定，长视频对齐达标 |
| Phase 7 | 接入 caption / 后处理可选模块 | `runtime/vividvr/captioning.py` `runtime/vividvr/postprocess.py` | 中 | Phase 6 | 开关测试 + 子模块对照 | 可选模块可独立启停，不破坏主链 |
| Phase 8 | 性能优化与回归 | compile/offload/backend 相关配置与适配 | 中到大 | Phase 7 | profile + regression | 有稳定默认参数、性能结论、回归集 |

## 2. Reference 质量基准

后续所有“输出达标”类验收，以以下 reference 为主：

- 原始推理命令：`/home/zhiheng/Vivid-VR/xzh_docs/run.md`
- 测试输入目录：`/home/zhiheng/Vivid-VR/input/720p`
- 集成阶段固定 caption 文件：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 现成 reference 输出：
  - `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`

统一比较口径：

1. 逐帧读取 reference mp4 与 SGLang candidate mp4
2. 逐帧计算：
   - `SSIM`
   - `MSE`
   - `MAE`
   - `PSNR`
   - `max_abs_diff`
3. 输出全局统计：
   - `ssim_mean`
   - `ssim_min`
   - `mse_mean`
   - `mse_max`
   - `mae_mean`
   - `mae_max`
   - `failed_frames`

默认“宽松基线”阈值：

- `min_ssim = 0.90`
- `max_mse = 150.0`
- `max_mae = 8.0`
- `allow_frame_count_delta = 1`
- `max_failed_frame_ratio = 0.05`

可选阈值档位：

### smoke

- `min_ssim = 0.80`
- `max_mse = 400.0`
- `max_mae = 15.0`
- `max_failed_frame_ratio = 0.10`

### strict

- `min_ssim = 0.96`
- `max_mse = 60.0`
- `max_mae = 5.0`
- `max_failed_frame_ratio = 0.0`

要求：

- 无论使用哪一档，必须保留 `ssim_min` 和 `mse_max` 上报
- Phase 5 起开始引入逐帧对齐
- Phase 8 进入 strict 或接近 strict 的 release gate

## 3. 分阶段实施细则与细化验收

## Phase 1: 范围冻结

目标：

- 确认第一版只做单 clip 核心闭环
- 冻结 checkpoint 目录约定
- 冻结 reference 数据与比较口径
- 冻结 caption 输入策略

涉及文件：

- 文档层，无代码

验证方法：

- 团队审阅文档
- 明确 reference 文件和阈值档位

细化验收标准：

- `Vivid-VR` 的任务定义、非目标、MVP 范围在文档中无歧义
- 已明确第一版不接入：
  - auto caption
  - 在 `sglang` 环境中实时运行 `CogVLM2`
  - TextFixer
  - EasyOCR
  - ESRGAN
  - 长视频时间聚合
  - 多卡 TP/SP
- 已明确模型目录契约
- 已明确 reference 输入与 reference 输出路径
- 已明确集成阶段使用：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 已明确对齐指标与默认阈值

完成标准：

- 团队对范围、reference、阈值口径无开放分歧

## Phase 2: 框架骨架

目标：

- 让 SGLang 认识一个新的 `VividVR` family

涉及文件：

- `configs/pipeline_configs/vividvr.py`
- `configs/sample/vividvr.py`
- `configs/pipeline_configs/__init__.py`
- `configs/sample/__init__.py`
- `registry.py`

关键实现点：

- `VividVRPipelineConfig`
- `VividVRSamplingParams`
- `register_configs(...)`
- `prompt_file_path` 合同

验证方法：

- import 成功
- detector 命中正确
- `SamplingParams.from_user_kwargs()` 可运行

细化验收标准：

- `python -m py_compile` 或等价 import 检查通过
- `VividVRPipelineConfig` 可实例化，默认字段完整
- `VividVRSamplingParams` 可实例化，基本校验正常
- `prompt` 与 `prompt_file_path` 的输入优先级已明确且可校验
- `registry.py` 新增 detector 后，不影响已有模型 family
- 给定模型名关键字时，能解析到：
  - `pipeline_config_cls = VividVRPipelineConfig`
  - `sampling_param_cls = VividVRSamplingParams`

完成标准：

- config family 可被正确识别，且不破坏现有注册链

## Phase 3: CogVideoX 基座移植

目标：

- 补齐 `CogVideoX` base transformer / VAE / scheduler

涉及文件：

- `configs/models/dits/cogvideox.py`
- `configs/models/vaes/cogvideox.py`
- `runtime/models/dits/cogvideox.py`
- `runtime/models/vaes/cogvideox.py`
- `runtime/models/schedulers/cogvideox_dpm_vividvr.py`

关键实现点：

- base transformer
- base VAE
- scheduler step 语义

验证方法：

- 单组件初始化
- 单步 scheduler 对照
- VAE encode/decode smoke test

细化验收标准：

- `CogVideoX` transformer 类能被 `ModelRegistry` 发现
- 基于 `ckpts/CogVideoX1.5-5B/transformer/config.json` 可完成初始化
- VAE 能完成：
  - `encode([B,C,T,H,W])`
  - `decode(latents)`
- VAE 的以下配置与 reference 一致：
  - `scaling_factor`
  - `temporal_compression_ratio`
  - `latent_channels`
- scheduler 至少通过以下单步对照：
  - `pred_original_sample` 数值范围合理
  - `restoration_guidance_scale=-1` 与原逻辑一致
  - `restoration_guidance_scale>0` 时不报错且输出合理
- 所有 base 组件在固定 shape 下无 NaN / Inf

完成标准：

- 三个 base 组件可以独立前向，shape 合同稳定

## Phase 4: VividVR 模型差异移植

目标：

- 接上 transformer 增量与 controlnet

涉及文件：

- `runtime/models/dits/cogvideox_vividvr.py`
- `runtime/models/dits/cogvideox_vividvr_controlnet.py`

关键实现点：

- connectors
- control_feat_proj
- control_patch_embed
- control hidden states 注入

验证方法：

- controlnet 前向
- transformer 前向
- 组合前向

细化验收标准：

- VividVR transformer 类可从 base config 派生初始化
- 三组增量权重可独立加载：
  - `connectors.pt`
  - `control_feat_proj.pt`
  - `control_patch_embed.pt`
- controlnet 可加载：
  - `ckpts/Vivid-VR/controlnet/`
- `control_hidden_states` 的层数与预期 block 数对应
- transformer 在提供 `control_hidden_states` 时输出 shape 与 reference 一致
- CFG 分支开启时，正负 prompt batch 拼接不报错
- 组合前向在固定输入下无 NaN / shape mismatch / dtype mismatch

完成标准：

- VividVR 增量模型链路可独立完成一次 noise prediction

## Phase 5: 单 clip MVP

目标：

- 建立第一个真正可跑的 SGLang 原生单 clip pipeline

涉及文件：

- `runtime/pipelines/vividvr_pipeline.py`
- `runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `runtime/vividvr/preprocess.py`
- `runtime/vividvr/tiling.py`

关键实现点：

- `before_denoising`
- `denoising`
- `decoding`
- tile 级 merge
- 读取固定 `prompt.txt`，而不是调用 `CogVLM2`

验证方法：

- 短视频端到端
- 中间张量校验
- reference 逐帧对齐

细化验收标准：

- 输入短视频后能输出合法 mp4 或视频 tensor
- 输出满足：
  - 帧数误差 `<= allow_frame_count_delta`
  - 分辨率与目标一致
  - 无整帧黑屏、颜色通道错位、明显尺度错误
- 中间状态至少打印或记录：
  - prompt embed shape
  - control latent shape
  - latents shape
  - timestep count
  - tile count
- 逐帧对齐达到默认宽松基线：
  - `ssim_min >= 0.90`
  - `mse_max <= 150.0`
  - `mae_max <= 8.0`
  - `failed_frame_ratio <= 0.05`
- 若未达到基线，必须能输出 `failed_frames` 列表用于定位

完成标准：

- 单 clip 路径稳定，且与 reference 对齐达标

## Phase 6: 长视频 orchestration

目标：

- 复现 `infer_split_clips()` 的时间聚合逻辑

涉及文件：

- `runtime/vividvr/windowing.py`
- `runtime/pipelines/vividvr_pipeline.py`

关键实现点：

- clip 切分
- latent id 映射
- timestep 级 merge

验证方法：

- 长视频端到端
- overlap 接缝检查
- reference 逐帧对齐

细化验收标准：

- 默认长视频验收输入固定为：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- 默认长视频 reference 固定为：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_6step/videos/test_video_long_960x720_130f.mp4`
- 上述 reference 默认指原版 `Vivid-VR` 在 `step=6` 条件下生成的公平对比结果
- clip 切分数量与 reference 逻辑一致
- `latent_id` 映射表与预期一致，无越界
- overlap 区域无明显时间跳变、重复帧、丢帧
- 解码后拼接视频总帧数与预期一致，容许 `allow_frame_count_delta`
- 对齐结果至少满足默认宽松基线：
  - `ssim_min >= 0.90`
  - `mse_max <= 150.0`
  - `mae_max <= 8.0`
  - `failed_frame_ratio <= 0.05`
- 验收结果的指标口径与 `Phase 5 / Phase C` 保持一致，至少稳定输出：
  - `ssim_mean`
  - `ssim_min`
  - `mse_mean`
  - `mse_max`
  - `mae_mean`
  - `mae_max`
  - `failed_frames`
  - `failed_frame_ratio`
- 若有 JSON 验收报告，默认复用 `Phase C` 的字段结构，只在必要时附加长视频特有调试字段
- 若局部接缝区域不达标，需单独输出 overlap 帧索引区间

完成标准：

- 长视频路径可稳定运行并保持接缝可接受

## Phase 7: 可选前后处理

目标：

- 让 caption / textfix 成为可选增强，而不是主链依赖

涉及文件：

- `runtime/vividvr/captioning.py`
- `runtime/vividvr/postprocess.py`

关键实现点：

- auto caption 开关
- AdaIN / TextFixer 插件化

验证方法：

- 开关行为测试
- 子模块单独对照
- 主链回归测试

细化验收标准：

- `auto_caption=False` 时，主链行为与 Phase 6 保持一致
- `auto_caption=True` 时：
  - tile prompt 数量与 tile plan 一致
  - caption 模型缺失时可清晰报错或降级
- `postprocess=False` 时，不影响主生成链
- `postprocess=True` 时：
  - AdaIN 不引入尺寸变化
  - TextFixer 失败时可回退到无后处理结果
- 开启可选模块后，若质量变差，必须能区分是主链还是后处理引起

完成标准：

- 可选模块可独立启停，失败不拖垮主链

## Phase 8: 性能优化与回归

目标：

- 让实现进入可持续维护状态

涉及文件：

- 与 compile/offload/backend 适配相关文件

关键实现点：

- profile
- 默认参数组合
- regression set

验证方法：

- 固定 seed 对照
- 峰值显存
- 单步耗时
- strict 或接近 strict 的逐帧对齐

细化验收标准：

- 至少产出一组稳定默认配置：
  - dtype
  - tile_size
  - offload 开关
  - backend 选择
- profile 报告至少包含：
  - 峰值显存
  - 平均单步 denoise 耗时
  - VAE encode/decode 耗时
  - 长视频总耗时
- 建立最小回归集：
  - 短视频 1 个
  - 长视频 1 个
  - 可选文字区域视频 1 个
- release gate 建议进入 strict 档或接近 strict：
  - `ssim_min >= 0.96`
  - `mse_max <= 60.0`
  - `mae_max <= 5.0`
  - `failed_frame_ratio = 0.0`

完成标准：

- 有稳定默认运行配置、最小回归集、明确质量门槛

## 4. 后续代码修改阶段的实施顺序建议

## 4.1 推荐主线顺序

不建议按“功能模块并行铺开”推进，建议按“先锁合同，再锁结构，再锁输出”的顺序走：

1. `Phase 2`
   - 先补 config / sampling / registry
   - 目标是让工程骨架和命名空间稳定
2. `Phase 3`
   - 先移植 `CogVideoX` base transformer / VAE / scheduler
   - 目标是把 `sglang` 当前缺失的底座补齐
3. `Phase 4`
   - 再移植 VividVR 增量模型
   - 目标是确保 control 分支和恢复引导语义正确
4. `Phase 5`
   - 再做单 clip MVP
   - 目标是最早拿到可对齐的第一份输出
5. `Phase 6`
   - 单 clip 对齐稳定后再做长视频 merge
6. `Phase 7`
   - 长视频稳定后再接 caption / 后处理
7. `Phase 8`
   - 最后再做 compile / offload / backend / regression

## 4.2 推荐文件修改顺序

更细一层，建议按下面顺序动文件：

1. `configs/pipeline_configs/vividvr.py`
2. `configs/sample/vividvr.py`
3. `configs/pipeline_configs/__init__.py`
4. `configs/sample/__init__.py`
5. `registry.py`
6. `configs/models/dits/cogvideox.py`
7. `configs/models/vaes/cogvideox.py`
8. `runtime/models/schedulers/cogvideox_dpm_vividvr.py`
9. `runtime/models/vaes/cogvideox.py`
10. `runtime/models/dits/cogvideox.py`
11. `runtime/models/dits/cogvideox_vividvr.py`
12. `runtime/models/dits/cogvideox_vividvr_controlnet.py`
13. `runtime/vividvr/preprocess.py`
14. `runtime/vividvr/tiling.py`
15. `runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
16. `runtime/pipelines/vividvr_pipeline.py`
17. `runtime/vividvr/windowing.py`
18. `runtime/vividvr/captioning.py`
19. `runtime/vividvr/postprocess.py`

## 4.3 不推荐的并行拆分

一开始不建议让多人并行做这些块：

- 一人做模型移植，一人做长视频 merge，一人做 caption

原因：

- 这三块依赖的 runtime 合同尚未稳定
- 早并行容易在 shape contract 上反复返工

## 4.4 可接受的并行点

当 `Phase 5` 单 clip 稳定后，才建议并行：

- 线程 A：长视频 orchestration
- 线程 B：caption helper
- 线程 C：postprocess helper

原因：

- 此时主生成链的输入输出合同已稳定
- sidecar 模块可以围绕稳定接口开发

## 4.5 每次提交的推荐粒度

建议按以下 commit/PR 粒度推进：

1. config + registry 骨架
2. CogVideoX base 组件
3. VividVR 增量模型
4. 单 clip MVP pipeline
5. 单 clip 对齐与修正
6. 长视频 orchestration
7. caption / postprocess
8. 性能优化与回归

这样每一步都能独立回归，而不会把问题混在一起。

## 5. 最终交付标准

最终认为接入完成，至少要满足：

- `VividVR` family 可被 `sglang` 正确识别
- 单 clip 端到端稳定
- 长视频路径可跑且接缝稳定
- 可选增强模块可开关
- 有 reference 对齐脚本与统计产物
- 有最小回归样例
- 有默认参数与已知限制说明

## 6. 明确不建议的推进方式

- 先上 generic diffusers wrapper 再长期不回收
- 一开始就做多卡
- 一开始就接 caption 和 textfix
- 一开始就为了 VividVR 改公共 runtime 结构
- 在没有 reference 对齐之前就做 compile / backend 优化
