# VividVR Official-Origin Resolution Semantics Alignment 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `sglang` 中 Vivid-VR 的输入视频尺寸处理语义，完整对齐到 `/home/zhiheng/Viviv-VR-origin/VRDiT/inference.py`，覆盖 `upscale`、`gen_height/gen_width` 派生、caption 输入尺寸以及主推理 `height/width` 传递，并用对齐后的代码重新验证 `serve / FlowCut` 的 `upscale=0` 请求。

**架构：** 保留当前 `load_control_video(...)` 的原版 `upscale` 插值行为，但新增一层官方原版等价的 “resolution planner”，把 `raw resized control_video` 与 `gen_height/gen_width` 明确分离。后续 caption、`pre_denoise_process`、主 pipeline 统一消费 `gen_height/gen_width`；postprocess / AdaIN / reference color fix 继续以 raw resized 尺寸作为“原始输出尺寸”语义，避免把两类尺寸混在一个字段里。

**技术栈：** Python、PyTorch、现有 `sglang.multimodal_gen` Vivid-VR preprocess / stage / legacy pipeline、pytest、FlowCut mock 服务、tmux。

---

## 范围与约束

- 本计划的对齐对象是 `/home/zhiheng/Viviv-VR-origin/VRDiT/inference.py`，不是 `/home/zhiheng/Vivid-VR` 的另一份历史代码。
- 必须对齐的是整段尺寸语义，不只是 `upscale` 插值：
  - `upscale == 0.0` 时短边缩放到 `1024`
  - `upscale == 1.0` 时不缩放
  - `upscale != 1.0 and > 0.0` 时按倍率 bicubic resize
  - 用 resize 后的 raw 尺寸计算 `gen_height/gen_width`
  - caption 输入使用 `gen_height/gen_width`
  - `pre_denoise_process / pipe(...)` 的 `height/width` 使用 `gen_height/gen_width`
- 必须保留 raw resized 尺寸与 gen 尺寸的分离：
  - raw resized 尺寸用于 `reference_video`、`runtime_original_height`、`runtime_original_width`
  - gen 尺寸用于运行时生成尺寸
- 不引入新的对外请求字段；沿用当前已暴露的 `upscale`
- 不处理 `enable_upscaling / upscaling_scale`
- 默认 `upscale=1.0` 的已验收 Phase C / D / E 基线不能回归

## 文件结构

- 修改 `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
  - 新增官方原版等价的 resolution planner，输出 raw resized 尺寸与 gen 尺寸
- 修改 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
  - 让 core stage 使用 `gen_height/gen_width` 作为运行时生成尺寸，同时保留 raw 尺寸给 postprocess
- 修改 `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
  - 让 legacy pipeline 与 core stage 使用同一份尺寸语义，避免两条路径继续分叉
- 修改 `python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py`
  - 补 raw 尺寸与 gen 尺寸的单测
- 修改 `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`
  - 补 stage 侧消费 `gen_height/gen_width` 的断言
- 修改 `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
  - 保留请求层 `upscale` 透传断言，并补一条“服务路径最终会带着 `upscale` 走到模型参数”的轻量保护
- 修改 `python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py`
  - 如果已有 dry-run payload snapshot，则补充新尺寸语义相关字段的快照断言
- 修改 `docs_xzh/run_command/mock_test.md`
  - 增加 `upscale=0` 服务验证命令与期望结果

---

### 任务 1：把官方原版尺寸规划逻辑收口到 preprocess

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py`

- [ ] **步骤 1：梳理并命名两类尺寸字段**

在 `preprocess.py` 里明确区分两类尺寸：

```python
"original_height": raw_height
"original_width": raw_width
"gen_height": gen_height
"gen_width": gen_width
```

要求：
- `original_height/original_width` 继续表示 raw resized control video 尺寸
- `gen_height/gen_width` 新增，语义严格对齐官方原版 `VRDiT/inference.py`

- [ ] **步骤 2：新增官方原版等价的 generation-size planner**

在 `preprocess.py` 中新增专用 helper，例如：

```python
def plan_generation_resolution(
    *,
    raw_height: int,
    raw_width: int,
    tile_size: int,
    vae_scale_factor_spatial: int,
) -> tuple[int, int]:
    threshold = tile_size * vae_scale_factor_spatial
    gen_height = (
        8 * math.ceil(raw_height / 8)
        if raw_height < threshold
        else raw_height
    )
    gen_width = (
        8 * math.ceil(raw_width / 8)
        if raw_width < threshold
        else raw_width
    )
    return int(gen_height), int(gen_width)
```

实现要求：
- 公式必须与 `/home/zhiheng/Viviv-VR-origin/VRDiT/inference.py:433-435` 一致
- 不加入“额外修正”“更安全对齐”“自定义 rounding”

- [ ] **步骤 3：让 `load_control_video(...)` 只负责 raw resize，不猜运行时生成尺寸**

把 `load_control_video(...)` 保持为：

```python
def load_control_video(video_path: str, *, upscale: float = 1.0) -> dict[str, object]:
```

但输出结构中至少新增占位字段或保留接口，供后续 stage / pipeline 填充 `gen_height/gen_width`。如果这里拿不到 `tile_size` 和 `vae_scale_factor_spatial`，不要硬算，留给拥有完整上下文的上层。

- [ ] **步骤 4：补 preprocess 单测，锁死 raw resize 语义**

在 `test_vividvr_preprocess.py` 补充或改写测试：

```python
def test_plan_generation_resolution_matches_official_origin_formula():
    gen_h, gen_w = plan_generation_resolution(
        raw_height=1024,
        raw_width=1365,
        tile_size=128,
        vae_scale_factor_spatial=8,
    )
    assert gen_h == 1024
    assert gen_w == 1365
```

还要保留现有断言：
- `upscale=0.0` 对短边 1024 的 raw resize 语义
- `upscale=1.0` 不缩放
- `upscale=2.0` 倍率缩放

- [ ] **步骤 5：运行定向单测**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py -q
```

预期：
- PASS
- 新增测试明确证明 `1024x1365` 在官方原版 planner 下不会被自动改成 `1368`

- [ ] **步骤 6：Commit**

```bash
git add \
  python/sglang/multimodal_gen/runtime/vividvr/preprocess.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py
git commit -m "refactor: align vividvr resolution planning with official origin"
```

### 任务 2：让 core stage 用 `gen_height/gen_width` 跑模型，同时保留 raw 尺寸给 postprocess

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`

- [ ] **步骤 1：在 core stage 中计算官方原版 `gen_height/gen_width`**

在拥有以下信息的地方计算：
- `control_video_info["original_height"]`
- `control_video_info["original_width"]`
- `params.tile_size`
- `vae.config.block_out_channels`

目标等价于官方原版：

```python
vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
gen_height, gen_width = plan_generation_resolution(
    raw_height=int(control_video_info["original_height"]),
    raw_width=int(control_video_info["original_width"]),
    tile_size=int(params.tile_size),
    vae_scale_factor_spatial=vae_scale_factor_spatial,
)
```

- [ ] **步骤 2：把 `_sync_runtime_resolution(...)` 从 raw 尺寸切到 gen 尺寸**

预期改成：

```python
params.height = int(control_video_info["gen_height"])
params.width = int(control_video_info["gen_width"])
```

同时保留：

```python
params.runtime_original_height = int(control_video_info["original_height"])
params.runtime_original_width = int(control_video_info["original_width"])
```

要求：
- 运行时生成尺寸和 postprocess 输出尺寸不能再共用一组字段

- [ ] **步骤 3：确认 caption / reference / prepared inputs 继续按官方原版语义分流**

检查并最小化修改以下语义：
- caption 输入 resize 到 `gen_height/gen_width`
- `reference_video` 仍然保持 raw resized、未 padding 的 control video
- postprocess 仍然回到 `runtime_original_height/runtime_original_width`

如果 stage 中已有 prepared dict，把新字段显式带下去，例如：

```python
"gen_height": gen_height,
"gen_width": gen_width,
```

- [ ] **步骤 4：补 stage 单测**

在 `test_vividvr_preprocess.py` 或更合适的 stage 单测中增加：

```python
def test_core_stage_syncs_runtime_resolution_from_generation_dims():
    params = VividVRSamplingParams(video_input_path="/tmp/input.mp4", upscale=0.0)
    stage._sync_runtime_resolution(
        params,
        {
            "original_height": 1024,
            "original_width": 1365,
            "gen_height": 1024,
            "gen_width": 1365,
        },
    )
    assert params.height == 1024
    assert params.width == 1365
    assert params.runtime_original_height == 1024
    assert params.runtime_original_width == 1365
```

在 `test_stage_d_vividvr_temporal_orchestration.py` 增加一条轻量断言，保证长视频路径不会偷偷退回 raw 尺寸。

- [ ] **步骤 5：运行定向单测**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py -q
```

预期：
- PASS
- `upscale=1.0` 的既有测试不回归

- [ ] **步骤 6：Commit**

```bash
git add \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py
git commit -m "fix: use official vividvr generation dimensions in core stage"
```

### 任务 3：让 legacy pipeline 与 core stage 共享同一套尺寸语义

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py`

- [ ] **步骤 1：查明 legacy pipeline 当前使用 raw 尺寸的位置**

重点核对这些调用点：
- `_resolve_input_video_info(...)`
- `params.runtime_original_height / runtime_original_width`
- 构造 `input_video_info` / prepared info 的位置
- 任何直接把 `original_height/original_width` 当成生成尺寸传下去的地方

把所有“生成尺寸”来源统一收口到 `gen_height/gen_width`。

- [ ] **步骤 2：让 legacy pipeline 与 core stage 共享 planner 结果**

实现要求：
- 不重复写第二套 `gen_height/gen_width` 公式
- 复用 `preprocess.py` 的 planner helper
- cache key 继续只区分输入文件与 `upscale`，不要把 `gen_height/gen_width` 再拼进去，因为它是确定性派生值

- [ ] **步骤 3：补 legacy pipeline 轻量测试**

在现有 `test_vividvr_preprocess.py` 中补一条覆盖 legacy cache / resolve 路径的断言，例如：

```python
def test_legacy_pipeline_resolves_generation_dims_from_official_origin_formula():
    info = {
        "original_height": 1024,
        "original_width": 1365,
    }
    # 调用实际 helper 或最小包装后的接口
    assert resolved["gen_height"] == 1024
    assert resolved["gen_width"] == 1365
```

- [ ] **步骤 4：运行定向单测**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py -q
```

预期：
- PASS
- 证明 core stage 和 legacy pipeline 不再各算各的尺寸语义

- [ ] **步骤 5：Commit**

```bash
git add \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py
git commit -m "refactor: share official vividvr resolution semantics across pipelines"
```

### 任务 4：补服务路径保护测试，并更新 mock 验证文档

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py`
- 修改：`docs_xzh/run_command/mock_test.md`

- [ ] **步骤 1：补请求层保护测试**

在 `test_flowcut_video_repair_api.py` 保留现有 `upscale` 透传测试，并补一条最小保护：

```python
def test_build_vividvr_kwargs_keeps_original_upscale_for_official_origin_alignment(...):
    kwargs = build_vividvr_kwargs(...)
    assert kwargs["upscale"] == 0.0
```

如果 acceptance tool 的 dry-run payload 当前会回显尺寸相关字段，补断言；如果不会，不要为了测试去扩协议。

- [ ] **步骤 2：更新 mock 测试文档**

在 `docs_xzh/run_command/mock_test.md` 增加一节：
- `upscale=0` 服务验证步骤
- 说明预期：
  - 请求应被接受
  - caption sidecar 正常返回
  - 不应再在当前 rope shape mismatch 处失败
  - 最终通过 callback 或 progress 拿到结果视频

把“如何用 callback 中的 `result_url` 下载视频”的命令放在同一节，避免用户重复排查。

- [ ] **步骤 3：运行轻量测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py -q
```

预期：
- PASS

- [ ] **步骤 4：Commit**

```bash
git add \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py \
  docs_xzh/run_command/mock_test.md
git commit -m "test: document and protect vividvr upscale service path"
```

### 任务 5：做最小有效回归，并用服务实测 `upscale=0`

**文件：**
- 不新增实现文件；运行验证并记录结果

- [ ] **步骤 1：跑完整针对性单测集合**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_preprocess.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py -q
```

预期：
- PASS

- [ ] **步骤 2：跑静态校验**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m py_compile \
  python/sglang/multimodal_gen/runtime/vividvr/preprocess.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py
git diff --check
```

预期：
- 无语法错误
- 无 whitespace / conflict marker 问题

- [ ] **步骤 3：在 tmux 中重启 mock 服务并重跑 `upscale=0`**

使用现有文档中的 tmux 命令重启：
- moto S3
- caption sidecar
- bridge service
- callback receiver

然后提交与用户当前一致的请求，只改 `TASK_ID`：

```bash
export TASK_ID=vividvr-bridge-mock-$(date -u +%Y%m%dT%H%M%SZ)
NO_PROXY=* curl -sS -X POST "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "upscale": 0,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json",
  "minioConfig": {
    "endpoint": "${MOTO_S3_ENDPOINT}",
    "bucket_name": "${MOTO_S3_BUCKET}",
    "access_key": "${MOTO_S3_ACCESS_KEY}",
    "secret_key": "${MOTO_S3_SECRET_KEY}",
    "secure": false,
    "region": "us-east-1"
  }
}
JSON
```

要求：
- 这一步必须在 `tmux` 环境下完成服务运行
- 在最终总结中给出 session 名和只读 attach 命令

- [ ] **步骤 4：确认验收结果**

验收标准：
- submit 返回 `{"code":0,...}`
- progress 不再出现当前的 rope shape mismatch
- callback 最终出现：

```json
{"status":"succeeded","progress":100.0,"reason":"succeeded","output":"{\"result_url\":\"...\"}"}
```

- MinIO 桶里出现对应 `outputs/${TASK_ID}.mp4`

- [ ] **步骤 5：Commit**

```bash
git add -A
git commit -m "fix: align vividvr resolution semantics with official origin"
```

---

## 自检

- 规格覆盖度：
  - 官方原版 `upscale` 插值：任务 1
  - 官方原版 `gen_height/gen_width`：任务 1、2、3
  - stage / pipeline 一致性：任务 2、3
  - 服务路径回归与文档：任务 4、5
- 占位符扫描：
  - 本计划未使用 TODO / 待定 / “类似任务 N” 等占位写法
- 类型一致性：
  - raw 尺寸统一叫 `original_height/original_width`
  - 生成尺寸统一叫 `gen_height/gen_width`
  - 不混用两类字段

