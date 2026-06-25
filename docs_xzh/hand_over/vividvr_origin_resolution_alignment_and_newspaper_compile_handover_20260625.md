# Vivid-VR 官方原版尺寸语义对齐与 newspaper compile 故障交接

日期：`2026-06-25 UTC`

## 背景

本交接承接以下几条最近工作线：

- `vividvr_flowcut_minio_progress_acceptance_handover_20260625.md`
- `vividvr_service_boundary_alignment_handover_20260624.md`
- `flowcut_vividvr_service_compat_handover_20260622.md`

本轮核心工作有两部分：

1. 把 `sglang` 中 Vivid-VR 的输入尺寸语义，继续向官方原版 `/home/zhiheng/Viviv-VR-origin` 对齐，不再只对齐 `upscale` 插值本身，而是对齐到 `pre_denoise_process / pipe(...)` 前的整段 resolution 语义。
2. 在此基础上，重新执行 `upscale=0` 的真实 FlowCut 双卡验收，确认此前 `960x720` 输入在 `upscale=0` 下的失败已经修复。

同时，本轮也发现了一个新的未解决问题：

- 新视频 `/home/zhiheng/Vivid-VR/result/newspaper/videos/newspaper.mp4` 在双卡 `compile` 服务路径下会在模型生成阶段失败。
- 当前证据表明，故障更像是 **单 clip + 新 shape + torch.compile / inductor** 组合问题，而不是 FlowCut 契约、caption bridge、MinIO 或 reference profile 写视频问题。

## 本轮已完成的实现

### 1. 官方原版尺寸语义对齐

这轮不再只保留“`upscale` 的 raw resize 行为”，而是把官方原版 `inference.py` 的尺寸规划语义整体迁入 `sglang`。

当前语义：

- `upscale=0.0`：按官方原版语义把短边缩放到 `1024`
- `upscale=1.0`：不缩放
- `upscale>0 且 != 1.0`：按倍率做输入 resize

新增的关键点是：

- raw resized control video 的尺寸继续记录在 `original_height / original_width`
- 新增官方原版等价的 `gen_height / gen_width`
- 后续主推理 `height / width` 改为消费 `gen_height / gen_width`
- 不再直接把 raw resized 尺寸当成模型生成尺寸

### 2. per-tile RoPE 修正

此前 `upscale=0` 路径在 `960x720` 输入上失败，根因不是 FlowCut 协议，而是：

- RoPE (`image_rotary_emb`) 先按 full latent shape 计算一次
- 后续 spatial tiling 过程中错误地复用到了每个 tile
- tile latent shape 与 full latent shape 不一致时，最终在 denoise 中触发 shape mismatch

本轮已改为：

- `prepare` 阶段不再预存整张图通用的 RoPE 供 tile 复用
- `run_denoising_step(...)` 在 tile loop 内按 `tile_latents` 重新计算 RoPE

这一步是 `upscale=0` 重新通过真实验收的关键修复。

### 3. legacy pipeline 同步对齐

不仅模块化 stage 路径改了，`runtime/pipelines/vividvr_pipeline.py` 的 legacy 路径也同步消费 `gen_height / gen_width`，避免新旧路径尺寸语义再次分叉。

## 本轮改动文件

- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- `python/sglang/multimodal_gen/runtime/vividvr/__init__.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_denoising_stage.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py`
- `docs_xzh/run_command/mock_test.md`

## 已完成验证

### 单测与静态检查

已跑：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_denoising_stage.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py -q
```

结果：

- `27 passed, 11 warnings in 10.00s`

已跑：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_sampling_params.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_phase_d_tool.py -q
```

结果：

- `105 passed, 6 warnings, 12 subtests passed in 11.74s`

已跑：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_runtime_snapshot.py -q
```

结果：

- `8 passed, 5 warnings in 10.39s`

已跑：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m py_compile \
  python/sglang/multimodal_gen/runtime/vividvr/preprocess.py \
  python/sglang/multimodal_gen/runtime/vividvr/__init__.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_denoising_stage.py
git diff --check
```

结果：通过。

## 双卡 FlowCut 验收结果

### 服务与依赖状态

本轮正式验收使用双卡默认服务：

- tmux session：`vividvr_serve_dual_default`
- 健康检查：`http://127.0.0.1:31191/health` 返回 `{"status":"ok"}`

同时依赖以下服务：

- caption sidecar：`vividvr_caption_sidecar_mock`
- callback receiver：`vividvr_flowcut_callback_receiver`
- moto S3：`vividvr_moto_s3`

### warmup

先做了一次双卡 warmup：

- `task_id`：`vividvr-dual-warmup-20260625T060023Z`

结果：

- progress 终态：`status=completed, progress=100`
- callback 终态：`status=succeeded, progress=100`
- result_url：
  `http://127.0.0.1:4566/flowcut/outputs/vividvr-dual-warmup-20260625T060023Z.mp4`
- perf：
  `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-dual-warmup-20260625T060023Z_perf.json`

### 正式验收

正式任务：

- `task_id`：`vividvr-dual-accept-20260625T064044Z`

submit 返回：

```json
{"code":0,"message":"ok"}
```

最终结果：

- progress 终态：

```json
{"id":"vividvr-dual-accept-20260625T064044Z","status":"completed","progress":100,"file_path":null,"url":"http://127.0.0.1:4566/flowcut/outputs/vividvr-dual-accept-20260625T064044Z.mp4","error":null,"callback_status":null,"callback_error":null,"callback_attempts":null}
```

- callback 终态：

```json
{"status":"succeeded","progress":100.0,"reason":"succeeded","output":"{\"result_url\":\"http://127.0.0.1:4566/flowcut/outputs/vividvr-dual-accept-20260625T064044Z.mp4\"}"}
```

- MinIO 对象 URL `HEAD` 返回 `200 OK`
- `Content-Length=10948707`

验收结论：

- 这轮“官方原版尺寸语义对齐 + `upscale=0` 双卡正式验收”已经通过

验收产物：

- callback log：
  `/home/zhiheng/sglang/Vivid_Acceptance/logs/mock_callback_20260625T054126Z.jsonl`
- perf：
  `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-dual-accept-20260625T064044Z_perf.json`

## reference profile 写视频说明

双卡 warmup 和正式验收两次都出现了：

- `Failed to save video with reference profile ... Broken pipe. Falling back to default imageio writer.`

这不是推理失败。它的含义是：

- serve 端在保存输出 mp4 时，先尝试按 `reference_video_path` 的编码 profile 写视频
- 这一步失败后自动回退到默认 `imageio writer`
- 最终视频仍然成功保存、上传并回调成功

结论：

- 这是**输出写视频阶段**的 warning
- 不是当前主阻塞问题

## 新阻塞问题：newspaper.mp4 在双卡 compile 路径下失败

### 复现请求

用户使用以下新视频发起请求：

- 输入视频：
  `/home/zhiheng/Vivid-VR/result/newspaper/videos/newspaper.mp4`
- `upscale=1.0`
- FlowCut 桥接服务：`/v1/videos/repairs/flowcut`

对应任务：

- `task_id`：`vividvr-newspaper-20260625T074111Z`

submit 返回：

```json
{"code":0,"message":"ok"}
```

这说明请求被成功 accept。

### 请求推进到哪里

callback log 可见：

- `accepted`
- `input_ready`
- `caption_ready`

随后失败：

```json
{"status":"failed","progress":0.0,"reason":"Model generation returned no output. Error from scheduler: Error executing request vividvr-newspaper-20260625T074111Z: TypeError: randn() received an invalid combination of arguments - got (Add, device=torch.device, dtype=torch.dtype), ...","output":""}
```

说明：

- 失败点不在 submit
- 不在 caption bridge
- 不在 callback
- 不在 MinIO
- 不在 reference profile 写视频

### 服务日志中的关键证据

日志文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_serve_dual_default_20260625T055902Z.log`

关键日志：

- caption 成功：
  - `VividVR caption bridge generated captions request_id=vividvr-newspaper-20260625T074111Z ... count=1`
- FlowCut accept 成功：
  - `FlowCut video repair accepted task_id=vividvr-newspaper-20260625T074111Z ...`
- 失败堆栈：
  - `Error executing request vividvr-newspaper-20260625T074111Z: TypeError: randn() received an invalid combination of arguments - got (Add, device=torch.device, dtype=torch.dtype)`

最重要的调用栈位置：

- `VividVRDenoisingStage.forward`
- `run_denoising_step`
- `self.transformer(...)`
- 进入 `torch._dynamo / torch._inductor`
- 最终在 `torch._dynamo.testing.rand_strided -> torch.randn(...)` 失败

### 当前对故障的判断

这是目前最重要的分析结论：

1. 这次失败发生在 **模型生成阶段**
2. 堆栈明确显示落在 **torch.compile / torch._inductor autotune** 路径
3. 这次新视频走的是 **单 clip 路径**
   - caption `count=1`
   - stage 也落在 `VividVRDenoisingStage`
4. 它不是前面已通过正式验收的 `MultiClip` 路径问题

当前高概率判断：

- 问题与 **单 clip + 新分辨率 + compile** 组合有关
- 暂时没有证据表明是 FlowCut 契约问题
- 暂时也没有证据表明是 caption 或 MinIO 问题

### 新视频的尺寸信息

通过本地读取与当前 `sglang` 预处理语义确认：

- 原视频 metadata：
  - `size=(1824, 1024)`
  - `fps=25.0`
- 在当前对齐后的 `upscale=1.0` 语义下：
  - `original_height=1024`
  - `original_width=1824`
  - `gen_height=1024`
  - `gen_width=1824`

也就是说，这次不是 `upscale=0` 的特殊输入，而是：

- 一个新的、较大的单 clip 分辨率
- 在 compile 路径中触发了 inductor 的 shape 相关失败

## 给下一个 Codex 的建议排查方向

默认不要先改 FlowCut 协议，也不要先改 callback / MinIO / 输出写视频。

优先做以下隔离：

1. **先用同一条 `newspaper.mp4` 在双卡 eager 服务上重跑**
   - 目的：确认问题是否只存在于 compile 路径
   - 如果 eager 通过，说明主推理语义大概率没问题，compile 是主要嫌疑
   - 如果 eager 也失败，再回头看单 clip 路径的 shape 语义

2. **如果 compile 独有失败，重点看单 clip compiled 路径**
   - `VividVRDenoisingStage`
   - `run_denoising_step(...)`
   - 传给 transformer 的 tensor shape / stride
   - 是否有某个输入张量在新的 `1824x1024` shape 下触发 inductor autotune bug

3. **必要时加最小 shape 诊断日志**
   - 在单 clip compiled 路径中记录：
     - latent shape
     - control latent shape
     - per-tile latent shape
     - rotary embedding shape
     - transformer 输入 tensor shape
   - 目标是确认哪一个输入让 inductor 生成了 `Add` 类型的 `needed_size`

4. **如果只是为了先跑通业务**
   - 可以先用双卡 eager 服务绕过 compile
   - 但不要把这当成最终修复

## 当前工作区注意事项

当前 `git status --short` 不是干净的，存在本轮及前几轮相关修改，包括：

- Vivid-VR `upscale` 接入
- FlowCut 协议暴露 `upscale`
- mock / run_command 文档更新
- 本轮官方原版尺寸语义与 RoPE 修复

下一个 Codex 继续工作前应先读清本轮改动，不要误回退。

## 结论

截至本交接：

- `sglang` 中 Vivid-VR 的官方原版尺寸语义对齐已完成到一个新的稳定点
- `upscale=0` 的双卡真实 FlowCut 正式验收已通过
- 当前剩余主问题是：
  - `newspaper.mp4` 在双卡 `compile` 的 **单 clip** 路径下失败
  - 高概率是 compile / inductor 与新 shape 组合的问题
  - 下一步最合理的是先用双卡 eager 做隔离复现，再决定是否修 compile 路径
