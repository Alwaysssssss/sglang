# 09. 后处理（RIFE 插帧 + Real-ESRGAN 超分）

> 源码位置：`python/sglang/multimodal_gen/runtime/postprocess/rife_interpolator.py`、`runtime/postprocess/realesrgan_upscaler.py`、`runtime/postprocess/__init__.py`、`runtime/entrypoints/utils.py::save_outputs`、`configs/sample/sampling_params.py`。

扩散模型生成的原始输出往往在两个维度上不够"出片"：

- **帧率低**：多数视频模型推理 16–24 fps，播放感不好；
- **分辨率有限**：推理成本随分辨率平方/立方增长。

`multimodal_gen` 自带两个经典后处理组件，允许在 `save_outputs` 链路里**统一集成**，用户不需要写任何胶水脚本。

## 1. 后处理作为 Pipeline 的一部分

进入点：`runtime/entrypoints/utils.py::save_outputs`。它对每个 sample 调用 `post_process_sample`（约 336–380、400–521 行），**顺序**是：

```text
tensor/numpy 输出
   │
   ▼
转成 uint8 帧列表
   │
   ▼
[可选] RIFE 插帧（仅 DataType.VIDEO 且 len(frames)>1）
   │
   ▼
[可选] Real-ESRGAN 超分
   │
   ▼
imageio 写文件（mp4 / png / gif / webp）
```

**插帧后 fps 自动调整**：`fps = fps * multiplier`（约 467–468 行）。

## 2. RIFE 插帧

文件 `runtime/postprocess/rife_interpolator.py`。

### 2.1 模型加载

`FrameInterpolator._ensure_model_loaded`（约 353–381 行）：

- 默认 HF 仓库：`elfgum/RIFE-4.22.lite`（第 27 行）；
- 模型文件名：`{path}/flownet.pkl`，通过 `load_model`（约 276–303 行）加载；
- 设备：`current_platform.get_local_torch_device()`（约 374 行）——**会自动跟上当前进程的 GPU**；
- **进程级缓存** `_MODEL_CACHE[path]`（第 29–30 行）：同一路径只加载一次。

### 2.2 插帧算法

- `interpolate(frames, multiplier)`（约 412–455 行）：对相邻帧递归调用 `_make_inference`，根据 multiplier 决定插入几帧；
  - multiplier=2 → 插 1 帧（原始 2 帧变 3 帧？实际是 2 帧变 4 帧的常见做法，请以代码为准）；
- `interpolate_video_frames(frames, multiplier)`（约 463–483 行）：对外暴露的薄封装。

### 2.3 触发条件

- 数据必须是 `DataType.VIDEO`；
- `len(frames) > 1`（单帧不插）；
- `enable_frame_interpolation=True`（在 `SamplingParams` 或请求参数里打开）；
- `frame_interpolation_multiplier` 指定倍数。

## 3. Real-ESRGAN 超分

文件 `runtime/postprocess/realesrgan_upscaler.py`。

### 3.1 模型加载

`ImageUpscaler._ensure_model_loaded`（约 319–379 行）：

- 默认 HF：`ai-forever/Real-ESRGAN`（第 27 行），`RealESRGAN_x4.pth`（第 28 行）；
- 路径解析 `_resolve_model_path`（约 399–445 行）：支持本地 `.pth`、`repo_id`、`repo_id:filename` 三种格式；
- 架构自适应 `_build_net_from_state_dict`（约 190–247 行）：根据 state_dict 自动选 `RRDBNet`（原版）或 `SRVGGNetCompact`（轻量版）；
- **进程级缓存** `_MODEL_CACHE[resolved_path]`（约 326–327、371–372 行）。

### 3.2 超分算法

- `UpscalerModel.upscale`：单张图上采样；
- `ImageUpscaler.upscale(frames)`：逐帧调用（约 381–391 行）；
- `upscale_frames`：对外接口（约 453–484 行）。

### 3.3 触发条件

- `enable_upscaling=True`；
- `upscaling_scale` 决定倍数（通常 2× / 4×）；
- `upscaling_model_path` / `upscaling_model_weight` 可自定义模型；
- 视频与图像都适用（不限制 DataType）。

## 4. 请求侧开关

后处理开关**不在 `ServerArgs` 顶层**，而在 `SamplingParams`（`configs/sample/sampling_params.py` 约 114–127 行）：

| 字段 | 含义 |
|------|------|
| `enable_frame_interpolation` | 是否 RIFE |
| `frame_interpolation_multiplier` | 插帧倍数 |
| `frame_interpolation_model_path` | RIFE 模型路径（覆盖默认）|
| `enable_upscaling` | 是否 Real-ESRGAN |
| `upscaling_scale` | 超分倍数 |
| `upscaling_model_path` | 模型仓库 / 本地路径 |
| `upscaling_model_weight` | 具体权重文件名 |

## 5. 数据流穿透

```text
Req.enable_frame_interpolation / enable_upscaling
    │
    ▼
GPUWorker.execute_forward 约 284–304 行：
  if req.save_output and req.return_file_paths_only:
      save_outputs(..., enable_frame_interpolation=req.enable_frame_interpolation, ...)
    │
    ▼
save_outputs 约 336–380、400–521 行：
  for sample in outputs:
      post_process_sample(sample, ...)
          ├── RIFE interpolate
          ├── Real-ESRGAN upscale
          └── imageio.write_video / write_image
```

HTTP `video_api` / OpenAI 兼容 API / `diffusion_generator.py` 都走这一链路。

## 6. 与 Disagg / 并行的关系

- 后处理在 **save 阶段执行**，通常落在 **Decoder worker 或 Server worker**（取决于 `return_file_paths_only` 是否为真）；
- RIFE / Real-ESRGAN 模型**独立于 DiT 显存池**，但会占用当前 rank 的 GPU 显存；
- 多请求共享 `_MODEL_CACHE`，**同一进程只加载一次**；
- **多卡场景** RIFE / Real-ESRGAN 不做切片并行，仅单卡执行；若 DiT 占满显存需谨慎。

## 7. 文件保存路径

`save_outputs` 对保存文件名做：

- 视频：mp4（默认）、webm、gif；
- 图像：png、jpg、webp；
- 帧率经插帧倍数调整后写入容器。

## 8. 常见问题

- **OOM**：Real-ESRGAN 对高分辨率敏感，可分块或减小 batch；
- **RIFE 插帧抖动**：换更重的 RIFE 版本（非 lite）；
- **超分模型找不到**：检查 `upscaling_model_path` 是否 `repo_id:filename` 格式；
- **模型加载慢**：`_MODEL_CACHE` 会进程级缓存，首次请求必然慢；
- **RIFE 对单帧无效**：只有视频（多帧）路径触发；如需对图像做时间插值是不支持的。

## 9. 关键文件索引

| 文件 | 符号 | 作用 |
|------|------|------|
| `runtime/postprocess/rife_interpolator.py` | `FrameInterpolator`、`interpolate_video_frames`、`_MODEL_CACHE` | RIFE 封装 |
| `runtime/postprocess/realesrgan_upscaler.py` | `ImageUpscaler`、`UpscalerModel`、`_resolve_model_path`、`_build_net_from_state_dict` | Real-ESRGAN 封装 |
| `runtime/postprocess/__init__.py` | 对外导出 | — |
| `runtime/entrypoints/utils.py` | `save_outputs`、`post_process_sample` | 串联执行 |
| `configs/sample/sampling_params.py` | `enable_frame_interpolation`、`enable_upscaling` 等 | 请求级参数 |

## 10. 调优建议

- **24fps → 60fps**：`multiplier=3` RIFE（一次插 2 帧），或两次 `multiplier=2`；
- **快速预览**：关闭超分，仅 RIFE 减少延迟；
- **长视频整体超分**：建议分块（`chunk_size` 参数），避免单帧显存爆；
- **CI / 回归测试**：关闭后处理以减少波动，单独跑基准；
- **生产环境**：建议在 **Decoder worker** 上执行（`return_file_paths_only=True`），避免把大视频拖回 Server worker；
- **与 compile 兼容**：RIFE / Real-ESRGAN 未被 `torch.compile` 包裹；首次加载仍有冷启动时间。
