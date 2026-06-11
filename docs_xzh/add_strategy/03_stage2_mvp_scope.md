# Stage 2: 最小可运行版本（MVP）范围

## 1. MVP 目标

第一版目标不是完整复刻 `Vivid-VR` 全部能力，而是建立一个能进入 SGLang 原生 runtime 的稳定闭环：

- 输入一个视频 clip
- 给定显式 prompt / negative prompt，或从固定 caption 文件读取 prompt
- 完成 VAE encode -> controlnet + transformer denoise -> VAE decode
- 输出视频
- 在固定 seed 下可重复

## 2. 第一版必须支持

### 2.1 功能

- 单视频输入
- 单 clip 推理
- 显式 `prompt`
- 从现成 caption 文件读取 `prompt`
- 显式 `negative_prompt`
- `guidance_scale`
- `num_inference_steps`
- `restoration_guidance_scale`
- 空间 tiling
- 单 batch
- 单卡

### 2.2 组件

- tokenizer
- T5 text encoder
- CogVideoX VAE
- VividVR transformer
- VividVR controlnet
- CogVideoX DPM scheduler vividvr 变体

### 2.3 caption 读取策略

集成阶段固定采用：

- 直接读取：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 不在 `sglang` 环境中实时运行 `CogVLM2`

原因：

- 当前 `/home/zhiheng/Vivid-VR` 自有环境中，`CogVLM2` caption 输出正常
- 当前 `sglang` 环境中，`CogVLM2` 输出存在乱码
- 因此在集成正确性阶段，必须把“caption 生成”与“主恢复链集成”解耦

### 2.4 可验证输出

- 生成的视频帧数正确
- 输出分辨率正确
- 无 NaN / shape mismatch / scheduler step 崩溃

## 3. 第一版明确不支持

- 自动 caption
- 在 `sglang` 环境中实时运行 `CogVLM2` 生成 caption
- 长视频跨 clip timestep 级 merge
- TextFixer
- EasyOCR
- RealESRGAN
- AdaIN color fix
- 多 batch
- 多卡 TP/SP
- compile / cache / offload 的全量优化

## 4. 为什么这样切范围

原因不是“功能少更容易”，而是这几类功能不属于同一层次：

- auto caption 属于前处理增强
- `CogVLM2` caption 乱码问题属于环境兼容问题，不应在主链集成阶段混入排查
- TextFixer / ESRGAN / AdaIN 属于后处理增强
- 长视频 merge 属于外层 orchestration
- TP/SP / compile 属于性能路径

若第一版把它们全部绑定，会让错误源不可分离。

## 5. MVP 输入输出合同

### 输入

- `video_input_path` 或已加载视频 tensor
- `prompt`
- `prompt_file_path`
- `negative_prompt`
- `height`
- `width`
- `seed`
- `guidance_scale`
- `num_inference_steps`
- `tile_size`
- `tile_stride`

### 输出

- 默认输出视频 tensor
- 可选输出文件路径由外部调用侧负责

建议第一版先把“写 mp4”留给调用层，而不是写死在 pipeline 内部。

## 6. MVP SamplingParams 建议

建议新增 `VividVRSamplingParams`，字段分三类：

### 请求字段

- `video_input_path`
- `prompt`
- `prompt_file_path`
- `negative_prompt`
- `height`
- `width`
- `seed`
- `guidance_scale`
- `num_inference_steps`
- `restoration_guidance_scale`
- `enable_spatial_tiling`
- `tile_size`
- `tile_stride`
- `dtype`

### 输入兼容字段

- `control_video_tensor`
- `control_latents`

其中：

- `prompt` 与 `prompt_file_path` 至少提供一个
- 集成阶段默认优先支持 `prompt_file_path=/home/zhiheng/Vivid-VR/input/720p/prompt.txt`

### runtime 字段

- `runtime_prompt_embeds`
- `runtime_negative_prompt_embeds`
- `runtime_control_latents`
- `runtime_latents`
- `runtime_timesteps`
- `runtime_old_pred_original_sample`
- `runtime_ori_height`
- `runtime_ori_width`
- `runtime_num_latent_padding_frames`
- `runtime_tile_plan`
- `runtime_decoded_video`

不要在第一版就把长视频 clip 状态字段塞进来。

## 7. MVP 验证方式

### 7.1 最低要求

- 用一段短视频输入
- 固定 seed
- 跑通端到端
- caption 来源固定为：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`

### 7.2 需要对照的中间量

- prompt embed shape
- control latent shape
- noise latent shape
- timestep 数量
- scheduler step 输出范围
- decode 后视频 shape

### 7.3 对照基准

优先对照原始 `Vivid-VR/VRDiT/inference.py` 的短视频路径。

逐帧视频对照的 reference 输出固定为：

- `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`

## 8. MVP 退出标准

满足以下条件后再进入长视频阶段：

- 单 clip 路径稳定
- 自定义 scheduler 语义正确
- tile 级 denoise + merge 正确
- 输出视频与原始实现达到可接受近似一致
- 集成过程中未误调用 `CogVLM2` 实时 caption 链路
