# Stage 3: Pipeline 改造方案与文件级计划

## 1. 最终推荐的文件布局

## 1.1 新增文件列表

### pipeline / stage

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`

### sampling / pipeline config

- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- `python/sglang/multimodal_gen/configs/sample/vividvr.py`

### model config

- `python/sglang/multimodal_gen/configs/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/configs/models/dits/cogvideox_vividvr.py`
- `python/sglang/multimodal_gen/configs/models/vaes/cogvideox.py`

### runtime models

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
- `python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/schedulers/cogvideox_dpm_vividvr.py`

### helper

- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- `python/sglang/multimodal_gen/runtime/vividvr/tiling.py`
- `python/sglang/multimodal_gen/runtime/vividvr/windowing.py`
- `python/sglang/multimodal_gen/runtime/vividvr/postprocess.py`

### 第二阶段再考虑新增

- `python/sglang/multimodal_gen/runtime/vividvr/captioning.py`
- `python/sglang/multimodal_gen/runtime/vividvr/textfix.py`

说明：

- `captioning.py` 在当前阶段不代表接入实时 `CogVLM2`
- 它只预留后续能力位置；若当前需要 caption 相关逻辑，最多只做 `prompt.txt` 文件读取 helper

## 1.2 修改文件列表

### 必改

- `python/sglang/multimodal_gen/registry.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py`
- `python/sglang/multimodal_gen/configs/sample/__init__.py`

### 视团队导出习惯决定

- `python/sglang/multimodal_gen/configs/models/dits/__init__.py`
- `python/sglang/multimodal_gen/configs/models/vaes/__init__.py`

说明：

- `runtime/pipelines` 与 `runtime/models` 的自动发现依赖 `EntryClass`，通常不需要改它们的 `__init__.py`
- 但 config 包当前走显式 import，因此 `pipeline_configs/__init__.py` 和 `sample/__init__.py` 需要同步更新

## 2. 注册入口列表

### 2.1 pipeline 注册

通过：

- `runtime/pipelines/vividvr_pipeline.py`
- `EntryClass = VividVRPipeline`

自动发现。

### 2.2 model 注册

通过以下文件中的 `EntryClass` 自动发现：

- `runtime/models/dits/cogvideox.py`
- `runtime/models/dits/cogvideox_vividvr.py`
- `runtime/models/dits/cogvideox_vividvr_controlnet.py`
- `runtime/models/vaes/cogvideox.py`
- `runtime/models/schedulers/cogvideox_dpm_vividvr.py`

### 2.3 config family 注册

必须在：

- `python/sglang/multimodal_gen/registry.py`

中新增：

- `VividVRPipelineConfig`
- `VividVRSamplingParams`
- 对应 `hf_model_paths` / `model_detectors`

建议 detector 至少包含：

- `"vivid-vr"`
- `"vividvr"`
- `"cogvideoxvividvrcontrolnetpipeline"`

## 3. 推荐调用链

### 3.1 单 clip

```text
request
  -> VividVRSamplingParams.from_user_kwargs()
  -> Req
  -> VividVRPipeline.forward()
      -> executor.execute(stages)
          -> validation
          -> before_denoising
          -> denoising
          -> decoding
  -> batch.output
```

当前单 clip 调用链中的 caption 来源固定为：

- `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`

不在 pipeline 内实时调用 `CogVLM2`。

### 3.2 长视频

```text
request
  -> VividVRPipeline.forward()
      -> prepare_global_inputs()
      -> split clips
      -> for each clip: execute(pre stages)
      -> global timestep loop
          -> for each clip: denoise_one_step()
          -> temporal latent merge
      -> for each clip: decode
      -> stitch frames
```

这里的重点是：

- 长视频不能简单复用 `executor.execute(self.stages)` 一次完成
- 必须像 Wan VideoEdit 一样，在 `pipeline.forward()` 内增加自定义 orchestration

## 4. 每个新增文件的职责边界

| 文件 | 职责 | 不应包含 |
| --- | --- | --- |
| `vividvr_pipeline.py` | 组件组装、短视频流程、长视频 orchestration | 低层模型实现 |
| `model_specific_stages/vividvr.py` | stage 合同与单 clip 执行 | 全局注册逻辑 |
| `configs/pipeline_configs/vividvr.py` | 静态组件配置与默认精度 | 运行时张量状态 |
| `configs/sample/vividvr.py` | 请求字段与 runtime 状态 | 真实推理逻辑 |
| `runtime/models/dits/cogvideox*.py` | 模型结构与前向 | request 级逻辑 |
| `runtime/vividvr/tiling.py` | 空间 tile 规划与 merge 权重 | scheduler 调用 |
| `runtime/vividvr/windowing.py` | 时间 clip 切分与 merge | 文本编码 |
| `runtime/vividvr/postprocess.py` | 后处理 helper | 主 denoise loop |

## 5. MVP 阶段建议的 stage 设计

### `VividVRInputValidationStage`

负责：

- 参数合法性
- 视频尺寸约束
- tiling 约束

### `VividVRBeforeDenoisingStage`

负责：

- 文本编码
- 控制视频预处理
- VAE encode
- 初始 latent 生成
- timestep 初始化前准备

### `VividVRDenoisingStage`

负责：

- scheduler timesteps
- 空间 tile 级 denoise
- controlnet + transformer 调用
- scheduler step

### `VividVRDecodingStage`

负责：

- 去掉 latent padding
- VAE decode
- resize 回原尺寸
- 输出 tensor

## 6. 第一版不建议的文件级改造

- 不新增公共 `runtime/models/controlnet/`
- 不新增公共 `controlnet_loader.py`
- 不改 `ComposedPipelineBase`
- 不改 `DiffusersPipeline` 主逻辑
- 不改 `ModelRegistry` 主流程

## 7. 开放决策

- `controlnet` 是否最终进入 `_required_config_modules`

建议：

- MVP 阶段先在 `initialize_pipeline()` 局部加载
- 如果后续第二个模型也需要同类 controlnet，再考虑抽成正式公共组件类型
