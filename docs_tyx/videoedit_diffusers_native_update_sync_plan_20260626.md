# VideoEdit-diffusers 原生更新同步计划（2026-06-26）

## 结论

本次检查的原生仓库为 `/home/tyx/workspace/zhouhao6/VideoEdit-diffusers`，分支 `online`，HEAD 为 `28b9579 fix bbox expanding`，工作区干净。

原生侧最近更新没有改 `models/transformer_wan.py`、`models/autoencoder_kl_wan.py`、`models/flow_match.py` 等模型结构文件，主要集中在推理编排、预处理、CLIP 条件和后处理：

- `infer.py`
- `utils/preprocess.py`
- `utils/postprocess.py`
- `pipelines/pipeline_wan_edit.py`

因此同步到 SGLang 时，优先改 `python/sglang/multimodal_gen/runtime/videoedit/*`、`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`、`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`、`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py` 和 Video Repair API/CLI 参数，不需要先动 VideoEdit DiT/VAE 架构。

## 原生侧更新位置

以 `7b079c4..28b9579` 看，原生侧总 diff 为：

```text
infer.py                       | 245 ++++++++++++++++++++++++++++++++---------
pipelines/pipeline_wan_edit.py |  82 +++++++++++++-
utils/postprocess.py           |  11 +-
utils/preprocess.py            | 177 +++++++++++++++++++++++++----
```

关键提交：

| 提交 | 日期 | 主要文件 | 更新点 |
| --- | --- | --- | --- |
| `915aa57 Modify infer pipeline for online version.` | 2026-05-27 | `infer.py`, `utils/preprocess.py` | 引入参考图 prepend、长视频滑窗、tail reverse mirror、非首窗口参考帧、完整帧输出逻辑。 |
| `88e7e53 fix diff in diffsync and diffuser` | 2026-06-09 | `infer.py`, `pipeline_wan_edit.py`, `utils/preprocess.py`, `utils/postprocess.py` | 引入 CLIP image encoder、CPU per-window noise、`video_latents=None` 纯噪声起步、mask downsample 改 `nearest`、bbox expand 参数、chunk 选择、paste/crop/color 输出。 |
| `5c72a6e fix overlap bug` | 2026-06-10 | `infer.py`, `pipeline_wan_edit.py`, `utils/preprocess.py` | overlap 从“只替换第 0 帧”改为“替换前 `overlap` 帧”；新增 `clip_preprocess=diffuser/diffsynth`，默认 `diffuser`。 |
| `fc3696f 1.0` | 2026-06-25 | `infer.py` | 默认 `num_inference_steps` 从 20 改 10，`bbox_expand_scale` 从 0.3 改 1.5。 |
| `28b9579 fix bbox expanding` | 2026-06-26 | `infer.py`, `utils/preprocess.py`, `utils/postprocess.py` | bbox expand 语义重写为目标面积扩展，默认 `bbox_expand_scale=2.5`；AdaIN boundary 在 paste-back 中始终调用，`adain_boundary_dilate=0` 时 no-op。 |

注意：原生 `infer.py` 当前 `--chunks` 是 `type=int, default="0"`，默认值仍是字符串；如果不显式传 `--chunks 0`，`parse_chunks()` 里 `count <= 0` 可能触发类型错误。同步时不要复制这个字符串默认值，SGLang 应保留整数或无该参数。

## 当前 SGLang 差异

### 1. bbox expand 语义不一致

原生最新位置：

- `VideoEdit-diffusers/utils/preprocess.py:243-294`
- `VideoEdit-diffusers/infer.py:164-168`
- `VideoEdit-diffusers/utils/preprocess.py:344-463`

原生最新行为：

- `bbox_expand_scale` 默认 2.5。
- `expand_bbox()` 不再是“左右各扩原 bbox 宽度的 scale 倍”，而是把原 bbox 的 `h * scale` 和 `w * scale` 作为目标尺寸。
- 当目标高或宽超过整帧时，先 clamp 超出边，再用目标面积补偿另一个方向，最后居中平移到画面内。

SGLang 当前位置：

- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py:260-276`
- `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py:61-63`
- `python/sglang/multimodal_gen/runtime/videoedit/cli.py:56-58`

SGLang 当前仍是旧的 per-side expansion，且默认 `bbox_expand_scale=1.2`。这是这次最需要同步的预处理变化。

### 2. overlap reference 仍只替换第 0 帧

原生最新位置：

- `VideoEdit-diffusers/infer.py:296-320`
- `VideoEdit-diffusers/utils/preprocess.py:568-638`

原生最新行为：

- 非首窗口从上一窗口生成结果里取 `prev_window_frames[stride:stride + overlap]`。
- 新窗口前 `overlap` 帧全部替换为上一窗口生成帧。
- 同时把这 `overlap` 帧 mask 置黑，保证整个重叠区作为上下文锚点。

SGLang 当前位置：

- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:286-308`
- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:310-326`
- `python/sglang/multimodal_gen/runtime/videoedit/windowing.py:88-97`

SGLang 的 `native_skip` 现在会把 overlap masks 清零，但 `_apply_previous_window_reference()` 只替换 `frames[0]`。这会导致 `frames[1:overlap]` 仍是源视频帧，而原生最新是上一窗口生成帧。

### 3. CLIP preprocess 缺少 `diffuser` 分支

原生最新位置：

- `VideoEdit-diffusers/infer.py:157-160`
- `VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:339-381`
- `VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:722-735`

原生最新行为：

- 新增 `clip_preprocess`，可选 `diffuser` 或 `diffsynth`。
- 默认 `diffuser`：先把 PIL image resize 到 `(width, height)`，再走 `CLIPImageProcessor`。
- `diffsynth`：走手工 `[-1, 1] -> bicubic 224 -> [0, 1] -> CLIP mean/std`。

SGLang 当前位置：

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py:257-334`
- `python/sglang/multimodal_gen/configs/pipeline_configs/videoedit_wan.py:55-68`

SGLang 现在只实现了 DiffSynth 风格手工预处理，没有 `clip_preprocess` 参数，也没有给 VideoEdit stage 加载 `CLIPImageProcessor`。如果要严格对齐原生默认输出，需要补 `diffuser` 路径。

### 4. AdaIN boundary 参数被 SGLang 忽略

原生最新位置：

- `VideoEdit-diffusers/utils/postprocess.py:31-69`
- `VideoEdit-diffusers/utils/postprocess.py:85-133`

原生最新行为：

- `paste_back()` 总是调用 `_adain_boundary()`。
- `adain_boundary_dilate=0` 时 boundary 为空，实际 no-op。
- `color_correct` 参数还在签名里，但最新代码不再用它作为 gate。

SGLang 当前位置：

- `python/sglang/multimodal_gen/runtime/videoedit/postprocess.py:37-88`
- `python/sglang/multimodal_gen/runtime/videoedit/frame_provider.py:274-316`

SGLang 接收 `adain_boundary_dilate` 后直接 `del`，所以传非 0 也不会生效。stream/eager 两条 paste-back 路径都要同步。

### 5. 默认值与服务默认需要分开处理

原生当前默认：

- `num_inference_steps=10`
- `bbox_expand_scale=2.5`
- `dilate_px=0`
- `mask_scale=1`
- `feather_px=0`
- `adain_boundary_dilate=0`
- `clip_preprocess="diffuser"`
- overlap commit 是 native skip 语义
- tail padding 是 native reverse mirror

SGLang 当前默认偏服务化：

- `num_inference_steps=40`
- `bbox_expand_scale=1.2`
- `dilate_px=15`
- `mask_scale=1.2`
- `feather_px=15`
- `overlap_commit_mode="weighted"`
- `tail_padding_mode="reflect"`
- `enable_teacache=True`

建议不要盲目把服务默认全改成原生默认。更稳妥的同步方式是：核心语义先支持原生最新行为，再增加一个明确的 native parity preset 或在 CLI/API 文档里提供严格对齐参数组合。若业务要求默认行为完全跟原生一致，再单独做默认值变更和回归。

## 同步计划

### P0：同步会影响结果对齐的核心语义

1. 替换 bbox expand 实现。
   - 修改 `runtime/videoedit/preprocess.py::expand_bbox()` 为原生最新目标面积算法。
   - `_finalize_bbox_geometry()`、`scan_global_bbox()`、`prepare_global_inputs()` 继续复用同一函数，确保 eager 与 stream bbox 一致。
   - 增加 unit test，覆盖普通 bbox、单边超过 frame、双边超过 frame、靠边 bbox、小 bbox 后续 `expand_bbox_for_small()` 触发逻辑。

2. 修正 `native_skip` 的 overlap reference。
   - 在 `VideoEditWindowSpec` 增加 `reference_prev_local_start`/`reference_prev_local_count`，或把现有 `reference_prev_local_idx` 扩展为 range 语义。
   - `native_skip` 下设置 range 为 `[stride, stride + overlap)`。
   - `_apply_previous_window_reference()` 循环替换 `frames[0:n_ref]`，而不是只替换 `frames[0]`。
   - `_zero_reference_context_masks()` 保持清零前 `overlap` 个 mask。
   - 保留 `weighted` 作为 SGLang 扩展路径，不和原生 native skip 混用；当前 weighted 的 single reference 语义先不强行改。

3. 补齐 AdaIN boundary。
   - 把原生 `_adain_boundary()` 移植到 `runtime/videoedit/postprocess.py`。
   - `paste_back_frame()` 增加 `adain_boundary_dilate` 参数并调用 `_adain_boundary()`。
   - `paste_back()` 和 `WindowFrameProvider.paste_back_frames()` 都透传该参数。
   - 保证 `adain_boundary_dilate=0` 输出与旧 no-op 基本一致，非 0 时 eager 和 stream 结果一致。

### P1：补齐原生可选路径和入口参数

4. 增加 `clip_preprocess` 参数。
   - 在 `WanVideoEditSamplingParams` 增加 `clip_preprocess: str`，可选 `diffuser/diffsynth`。
   - 在 CLI `runtime/videoedit/cli.py` 增加 `--clip-preprocess`。
   - 在 Video Repair request protocol 增加 snake_case 和 camelCase 兼容字段。
   - 在 metadata 中记录 `clip_preprocess`，方便输出追踪。

5. 实现 `diffuser` CLIP preprocess。
   - 给 VideoEdit pipeline 加载或懒加载 `CLIPImageProcessor`，优先使用模型目录下 `image_processor` 子目录。
   - `clip_preprocess="diffuser"` 时，复刻原生：PIL resize 到 `(runtime_width, runtime_height)`，再 `CLIPImageProcessor(images=image, return_tensors="pt")`，再送 `CLIPVisionModel`。
   - `clip_preprocess="diffsynth"` 保留当前手工路径，避免破坏已有质量调参。
   - 如果 image encoder 不存在，保持当前报错；是否像原生一样自动关闭 CLIP 需要另开决策，因为服务端静默降级可能掩盖配置错误。

6. 增加 native parity preset 或文档化参数组合。
   - 若保持服务默认不变，新增文档或 helper preset，要求严格对齐时使用：
     - `num_inference_steps=10`
     - `bbox_expand_scale=2.5`
     - `dilate_px=0`
     - `mask_scale=1`
     - `feather_px=0`
     - `adain_boundary_dilate=0`
     - `clip_preprocess=diffuser`
     - `overlap_commit_mode=native_skip`
     - `tail_padding_mode=native_reverse_mirror`
     - `enable_teacache=false`
     - `decode_mode=eager` 作为第一轮 parity 验证，随后再验证 stream。
   - 若业务要求 SGLang 默认跟原生一致，再把默认值变更拆成单独 PR，并同步更新 `test_videoedit_decode_mode_params.py`、API spec 和用户文档。

### P2：验证与回归

7. 更新单元测试。
   - `test_videoedit_windowing.py`：新增 native skip 会替换前 `overlap` 帧的断言。
   - `test_videoedit_frame_provider.py`：增加 `adain_boundary_dilate > 0` 下 eager/stream paste-back 一致性。
   - `test_videoedit_decode_mode_params.py`：覆盖 `clip_preprocess` 默认、CLI、API request、payload normalize。
   - 新增 `test_videoedit_bbox_expand.py` 或放入现有 bbox 测试，直接对齐原生最新 bbox 算法。

8. 做两轮端到端对齐。
   - 第一轮：`num_inference_steps=1`，关闭 TeaCache、插帧、超分，使用 eager decode，验证窗口 materialize、bbox、mask、CLIP embedding shape、输出帧数。
   - 第二轮：同参数切换到 stream decode，验证 stream bbox、frame provider、paste-back 和 eager 输出一致。
   - 第三轮：恢复业务默认或目标默认，做视觉指标比较和性能记录。

建议的 unit test 命令：

```bash
PYTHONPATH=/home/tyx/workspace/sglang/python \
python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_videoedit_windowing.py \
  python/sglang/multimodal_gen/test/unit/test_videoedit_frame_provider.py \
  python/sglang/multimodal_gen/test/unit/test_videoedit_decode_mode_params.py
```

建议的原生 reference 运行要点：

```bash
cd /home/tyx/workspace/zhouhao6/VideoEdit-diffusers
PYTHONPATH=. python infer.py \
  --num_inference_steps 1 \
  --chunks 0 \
  --bbox_expand_scale 2.5 \
  --clip_preprocess diffuser \
  --dilate_px 0 \
  --mask_scale 1 \
  --feather_px 0 \
  --adain_boundary_dilate 0
```

同步后 SGLang 对齐运行要点：

```bash
PYTHONPATH=/home/tyx/workspace/sglang/python \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --num-inference-steps 1 \
  --bbox-expand-scale 2.5 \
  --clip-preprocess diffuser \
  --dilate-px 0 \
  --mask-scale 1 \
  --feather-px 0 \
  --adain-boundary-dilate 0 \
  --overlap-commit-mode native_skip \
  --tail-padding-mode native_reverse_mirror \
  --decode-mode eager \
  --no-enable-teacache \
  --no-enable-frame-interpolation \
  --no-enable-upscaling
```

## 建议实施顺序

1. 先改 bbox expand 和对应 unit test。这是纯 CPU 几何逻辑，风险低，影响明确。
2. 再改 native skip overlap reference，先只影响 `native_skip`，不碰 `weighted`。
3. 再移植 AdaIN boundary，并保证默认 `adain_boundary_dilate=0` 时 no-op。
4. 最后补 `clip_preprocess` 和 `CLIPImageProcessor`，因为这会引入组件加载和 API 参数扩展，回归面最大。
5. 全部完成后再讨论是否把 SGLang 默认值改到原生最新。默认值变更建议单独提交，避免把 parity fix 和业务策略混在一起。

## 风险点

- bbox 默认从 1.2 改 2.5 会显著扩大 crop，显存、速度和输出画面都会变。
- `clip_preprocess="diffuser"` 与当前 DiffSynth 路径的 embedding 分布不同，输出会变化；这是原生默认，但不一定是现有 SGLang 服务质量最优配置。
- overlap reference 修正后，`native_skip` 多窗口输出会变；这是为了对齐原生最新。`weighted` 应先保持扩展语义，避免影响现有长视频平滑策略。
- AdaIN 非 0 时会改变 paste-back 颜色，stream/eager 两条路径必须同步，否则同一请求不同 decode mode 输出不一致。
- 原生当前 `--chunks` 默认字符串是上游小 bug，SGLang 不应照抄。
