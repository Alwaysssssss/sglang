# VividVR Pool And SP Semantics Cleanup 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 删除 VividVR 双卡路径中的 connector control pooling 和错误的 `native` SP 语义，同时让用户接口继续只接受 `fa` / `sdpa`，并在 `130f / 20 step` 的 `serve` 长视频口径下完成单卡、双卡和双卡 `sdpa` 兼容入口的正式质量验收。

**架构：** 对外继续保留 `--attention-backend fa|sdpa` 这组 kernel 选择，对内把运行时分成两层：`SP=1` 走 local 语义，`SP>1` 统一走 Ulysses sequence-parallel joint-attention 语义。`fa` 和 `sdpa` 都必须接到同一条正确的 SP 语义骨架上；`fa` 不可用时只能 fallback 到 `sdpa` kernel，不能再回退到旧 `native` 本地注意力路径。Connector 侧彻底删除 control pool，`eager_global` 固定为 full global gather。

**技术栈：** Python、PyTorch、diffusers、现有 `sglang.multimodal_gen` VividVR pipeline/runtime、pytest、`tmux`、`sglang serve`、`curl`、`ffprobe`、现有 `compare_videos` 工具。

---

## 范围与约束

- 这轮只收口 VividVR 专项语义，不修改全仓库通用 DiT `torch_sdpa` 能力。
- 用户接口保持不变：
  - `--attention-backend fa`
  - `--attention-backend sdpa` 或 `torch_sdpa`
- 内部运行时语义必须收口为：
  - `SP=1` -> local semantics
  - `SP>1` -> Ulysses SP semantics
- 这轮删除的是：
  - `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE`
  - `eager_global` 中先 pool 再 gather 的控制分支
  - `SP>1 + sdpa/native -> 本地 native attention` 这条错误路径
- 这轮保留的是：
  - 单卡 local + `fa`
  - 单卡 local + `sdpa`
  - 双卡 SP + `fa`
  - 双卡 SP + `sdpa`
- 这轮正式验收必须使用 `serve` 主口径，不以 direct inference 作为主验收。
- 长视频正式验收固定：
  - 输入：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
  - caption：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
  - prompt：`/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
  - reference：`/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
  - `num_inference_steps=20`
  - `seed=42`
  - `num_temporal_process_frames=121`

## 文件结构

- 修改 `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
  - 将 backend 解析从“单字符串选 processor”改成“语义层 + kernel 层”，删除双卡错误 native 语义，补齐 SP + `sdpa` 的正确 kernel 路径。
- 修改 `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
  - 负责用户输入 `fa|sdpa` 的解析、`SP=1/SP>1` 的语义决策、`fa -> sdpa` kernel fallback 和统一日志输出。
- 修改 `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
  - 删除 control pool 的 env/helper/分支/日志，让 `eager_global` 固定为 global gather。
- 修改 `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
  - 把 alias、processor 安装、pipeline backend 分流和 fallback 断言改到新语义契约。
- 修改 `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py`
  - 删除 pool 相关假设，锁死 `eager_global` 只走 full global gather。
- 修改 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
  - 去掉 pool env，更新双卡 `sdpa` 入口语义说明。
- 修改 `docs_xzh/benchmark_results/vividvr_serve_long_130f_20step_benchmark_20260621.md`
  - 标注旧 `dual_gpu_sdpa_*` 结论属于已删除历史语义，并为本轮重新验收留出更新入口。
- 修改 `AGENTS.md`
  - 删除 `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1` 作为默认契约的说明，更新双卡 backend 说明为“对外只选 `fa|sdpa`，SP 语义自动选择”。
- 使用 `Vivid_Acceptance/tmp/run_vividvr_service_benchmark.py`
  - 作为正式 `serve` 验收和 SSIM 计算入口，不新增重复脚本。

---

### 任务 1：先用单测锁死新 backend 契约

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

- [ ] **步骤 1：改 alias 预期，先让测试表达新契约**

把当前 alias 断言从：

```python
self.assertEqual(normalize_cogvideox_attention_backend("torch_sdpa"), "native")
```

改成类似下面的目标：

```python
self.assertEqual(normalize_cogvideox_attention_backend("fa3"), "fa")
self.assertEqual(normalize_cogvideox_attention_backend("flash"), "fa")
self.assertEqual(normalize_cogvideox_attention_backend("torch_sdpa"), "sdpa")
self.assertEqual(normalize_cogvideox_attention_backend("sdpa"), "sdpa")
```

- [ ] **步骤 2：补 processor 安装测试，表达 `fa|sdpa` 仅是 kernel 选择**

新增或改写测试，要求至少覆盖：

```python
def test_set_attention_backend_replaces_processors_for_fa_and_sdpa(self):
    module = _DummyCogVideoXAttentionModule()

    set_cogvideox_attention_backend(module, "fa")
    self.assertEqual(inspect_cogvideox_attention_backend(module), "fa")

    set_cogvideox_attention_backend(module, "sdpa")
    self.assertEqual(inspect_cogvideox_attention_backend(module), "sdpa")
```

这里的关键不是 class 名，而是避免再把 `torch_sdpa` 解释成 `native`。

- [ ] **步骤 3：补 pipeline 分流测试，锁死 `SP=2` 时的语义决策**

新增测试覆盖以下 3 组：

```python
pipeline._apply_attention_backend(
    SimpleNamespace(attention_backend="fa", sp_degree=2, ulysses_degree=2)
)
pipeline._apply_attention_backend(
    SimpleNamespace(attention_backend="sdpa", sp_degree=2, ulysses_degree=2)
)
pipeline._apply_attention_backend(
    SimpleNamespace(attention_backend="fa", sp_degree=1, ulysses_degree=1)
)
```

断言目标：
- 单卡 `fa` -> effective backend 仍是本地 `fa`
- 双卡 `fa` -> effective backend 进入 SP 正确语义
- 双卡 `sdpa` -> effective backend 进入 SP 正确语义，但 kernel 是 `sdpa`

- [ ] **步骤 4：补 `fa -> sdpa` kernel fallback 测试**

用 `unittest.mock.patch` 模拟 `fa` kernel 初始化或执行失败，要求 fallback 仍留在 SP 正确语义中：

```python
with patch(
    "sglang.multimodal_gen.runtime.models.dits.cogvideox_attention_backend.flash_attn_func",
    side_effect=RuntimeError("flash init failed"),
):
    pipeline._apply_attention_backend(
        SimpleNamespace(attention_backend="fa", sp_degree=2, ulysses_degree=2)
    )
    debug = pipeline._build_runtime_acceleration_debug(...)
    self.assertEqual(debug["attention_backend_transformer"], "sdpa_sp")
```

如果最终实现不使用 `sdpa_sp` 这个 inspect 值，必须把断言改成最终真实字符串，但测试意图不变：
- fallback 发生
- fallback 后不是 `native`
- fallback 后仍是 SP 正确语义

- [ ] **步骤 5：运行测试确认当前代码先失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py -q
```

预期：
- FAIL
- 失败点至少包括 `torch_sdpa` 仍解析成 `native`，以及 pipeline 仍只认识 `fa_sp`

- [ ] **步骤 6：Commit 测试骨架**

```bash
git add python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py
git commit -m "test: lock vividvr backend semantics split"
```

### 任务 2：实现 attention backend 的“语义层 + kernel 层”收口

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

- [ ] **步骤 1：在 backend 文件里把 `sdpa` 从 `native` alias 中拆出来**

将 alias map 从：

```python
"native": "native",
"torch_native": "native",
"torch_sdpa": "native",
"sdpa": "native",
```

改成：

```python
"native": "sdpa",
"torch_native": "sdpa",
"torch_sdpa": "sdpa",
"sdpa": "sdpa",
```

要求：
- 对外兼容旧 `torch_sdpa`
- 内部不再把 `sdpa` 叫做 `native`

- [ ] **步骤 2：引入显式的 backend 解析结果类型**

在 `cogvideox_attention_backend.py` 中新增轻量解析结果，例如：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CogVideoXAttentionRuntimeChoice:
    semantics: str  # "local" | "ulysses_sp"
    kernel: str     # "fa" | "sdpa"
```

并新增 helper：

```python
def resolve_cogvideox_attention_runtime_choice(
    *,
    requested_backend: str | None,
    sp_enabled: bool,
) -> CogVideoXAttentionRuntimeChoice:
    ...
```

要求：
- `requested_backend is None` 时默认 kernel = `fa`
- `sp_enabled=False` -> `semantics="local"`
- `sp_enabled=True` -> `semantics="ulysses_sp"`

- [ ] **步骤 3：把现有 SP processor 提升成通用骨架，按 kernel 分派实现**

不要继续把 SP processor 和 `fa_sp` 强绑定。目标结构类似：

```python
class CogVideoXSPAttnProcessor:
    def __init__(self, kernel: str):
        self._semantics = "ulysses_sp"
        self._kernel = kernel
```

在 `__call__` 里：
- 仍然复用 Ulysses replicated-prefix 语义
- 当 `kernel == "fa"` 时走 flash kernel
- 当 `kernel == "sdpa"` 时走 SP 语义下的 `torch.nn.functional.scaled_dot_product_attention`

关键要求：
- `sdpa` 必须共享和 `fa` 相同的 text-prefix replicated + video-suffix sharded 语义
- 不允许再回退到 `CogVideoXAttnProcessor2_0` 的本地 concat 路径

- [ ] **步骤 4：删除公开的 `native` 运行时入口**

把：

```python
def build_cogvideox_attention_processor(backend: str) -> object:
```

改成能接收语义层 + kernel 层，例如：

```python
def build_cogvideox_attention_processor(
    *, semantics: str, kernel: str
) -> object:
```

支持集合应收口为：

```python
("local", "fa")
("local", "sdpa")
("ulysses_sp", "fa")
("ulysses_sp", "sdpa")
```

错误信息必须明确写出不再支持 `native` 作为 VividVR 运行时 backend。

- [ ] **步骤 5：重写 pipeline 入口分流和日志**

在 `vividvr_pipeline.py` 中，把当前：

```python
resolved_backend = normalize_cogvideox_attention_backend(requested_backend)
if resolved_backend == "fa" and sp_enabled:
    resolved_backend = "fa_sp"
```

改成调用新的解析 helper：

```python
runtime_choice = resolve_cogvideox_attention_runtime_choice(
    requested_backend=requested_backend,
    sp_enabled=sp_enabled,
)
component.set_attention_backend(runtime_choice)
```

日志至少要包含：

```python
logger.info(
    "Applied VividVR attention runtime choice: requested_backend=%s semantics=%s kernel=%s effective_backend=%s",
    requested_backend,
    runtime_choice.semantics,
    runtime_choice.kernel,
    applied_backend,
)
```

- [ ] **步骤 6：实现 `fa -> sdpa` kernel fallback，只允许在同一语义层内发生**

在 SP 路径下，`fa` kernel 不可用或执行失败时，fallback 代码必须类似：

```python
try:
    return run_fa_kernel(...)
except Exception:
    if self._kernel == "fa":
        return run_sdpa_kernel_same_sp_semantics(...)
    raise
```

禁止 fallback 到：

```python
CogVideoXAttnProcessor2_0()
CogVideoXNativeAttnProcessor()
```

- [ ] **步骤 7：运行 backend 单测确认通过**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py -q
```

预期：
- PASS
- 不再出现任何断言期待 `native` 或 `fa_sp`

- [ ] **步骤 8：Commit runtime backend 收口**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py
git commit -m "refactor: unify vividvr sp semantics across fa and sdpa"
```

### 任务 3：删除 connector control pooling，并锁死 eager_global 只走 full gather

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py`

- [ ] **步骤 1：删除 pool env 和 helper**

从 `cogvideox_vividvr_common.py` 中删除以下元素：

```python
_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_ENV
_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE_DEFAULT
get_vividvr_connector_control_pool_size()
_pool_control_state_2d(...)
```

删除后，文件顶部只保留和 `SP_CONTEXT_MODE` 有关的契约。

- [ ] **步骤 2：把 `build_vividvr_connector_control_states(...)` 收口为 eager_global 固定 full gather**

将 eager_global 路径的行为固定为：

```python
global_states = restore_vividvr_connector_global_control_states(
    local_states,
    shard_state,
)
```

然后再组装 connector context：

```python
connector_states.append((local_state, global_state))
```

要求：
- 不再读取 pool size
- 不再出现“先 pool 再 gather”
- 日志中不再出现 pool size / pooled shape

- [ ] **步骤 3：改 sequence-shard 单测，去掉 pool 假设**

在 `test_stage_e_vividvr_sequence_shard.py` 中保留和新增以下断言：

```python
def test_build_connector_control_states_restores_global_states_in_eager_mode(self):
    ...
    gather_mock.assert_called_once()
    self.assertTrue(torch.equal(connector_states[0][1], gathered_states[0]))
```

再补一条环境变量忽略测试，锁死旧 env 不再生效：

```python
def test_legacy_control_pool_env_is_ignored(self):
    with patch.dict(
        environ,
        {
            "SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE": "eager_global",
            "SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE": "2",
        },
        clear=False,
    ):
        ...
        gather_mock.assert_called_once()
```

预期行为：
- 旧 env 即使存在，也不改变 full gather 结果

- [ ] **步骤 4：运行 sequence-shard 单测**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py -q
```

预期：
- PASS
- 无 pool 相关断言失败

- [ ] **步骤 5：Commit connector pool 删除**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py
git commit -m "refactor: remove vividvr connector control pooling"
```

### 任务 4：同步文档并完成 130 帧 `serve` 正式验收

**文件：**
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 修改：`docs_xzh/benchmark_results/vividvr_serve_long_130f_20step_benchmark_20260621.md`
- 修改：`AGENTS.md`
- 运行：`Vivid_Acceptance/tmp/run_vividvr_service_benchmark.py`

- [ ] **步骤 1：更新默认命令文档，删除 pool env**

将文档中的双卡命令从：

```bash
export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && \
...
--attention-backend fa
```

改成：

```bash
export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
...
--attention-backend fa
```

并新增说明：
- 用户只传 `fa|sdpa`
- `SP>1` 自动走 Ulysses SP 正确语义
- `sdpa` 在双卡下不再表示旧 `native` 路径

- [ ] **步骤 2：更新 benchmark 文档，对历史 `dual_gpu_sdpa_*` 结果加“旧语义已废弃”说明**

在 `docs_xzh/benchmark_results/vividvr_serve_long_130f_20step_benchmark_20260621.md` 的术语或结论部分补充：

```markdown
- 2026-06-30 之后，双卡 `sdpa` 不再表示旧 `native` SP 路径。
- 本文中 `dual_gpu_sdpa_*` 的 0.966x 质量结果仅代表历史语义，不再代表当前实现。
```

- [ ] **步骤 3：更新 `AGENTS.md` 的默认双卡契约**

把当前默认双卡约束从：

```markdown
SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1
--attention-backend fa
双卡 SP 下运行时有效 backend 记为 fa_sp
```

改成：

```markdown
不再支持 control pooling
双卡 SP 自动进入 Ulysses 正确语义
用户请求侧只使用 --attention-backend fa 或 --attention-backend sdpa
```

- [ ] **步骤 4：在 `tmux` 中启动单卡 `serve` 基线**

运行：

```bash
tmux new-session -d -s vividvr_serve_single_cleanup \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 1 --tp-size 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1 \
    --enable-torch-compile \
    --host 127.0.0.1 --port 31190 --master-port 30190 --scheduler-port 56190 --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_single_cleanup_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看：

```bash
tmux attach -r -t vividvr_serve_single_cleanup
```

- [ ] **步骤 5：运行单卡 `serve` 正式 benchmark 并记录 SSIM**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  Vivid_Acceptance/tmp/run_vividvr_service_benchmark.py \
  --base-url http://127.0.0.1:31190 \
  --label single_gpu_fa_compile_cleanup
```

预期：
- 生成 JSON 指标到 `Vivid_Acceptance/indicator`
- 生成结果视频到 `Vivid_Acceptance/result_videos/service_benchmark`
- `ssim_mean` 接近历史 `single_gpu_fa_compile`

- [ ] **步骤 6：在 `tmux` 中启动双卡 `serve` 默认链路**

运行：

```bash
tmux new-session -d -s vividvr_serve_dual_cleanup \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
    --enable-torch-compile --dist-timeout 3600 \
    --host 127.0.0.1 --port 31191 --master-port 30191 --scheduler-port 56191 --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_cleanup_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看：

```bash
tmux attach -r -t vividvr_serve_dual_cleanup
```

- [ ] **步骤 7：运行双卡 `fa` 正式 benchmark 并检查日志里的 effective backend**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  Vivid_Acceptance/tmp/run_vividvr_service_benchmark.py \
  --base-url http://127.0.0.1:31191 \
  --label dual_gpu_fa_eager_compile_cleanup
```

额外检查服务日志：

```bash
rg -n "requested_backend|effective_backend|semantics|kernel" Vivid_Acceptance/logs/vividvr_serve_dual_cleanup_*.log
```

预期：
- `ssim_mean` 接近单卡 `fa` 基线
- 日志明确显示 `SP` 正确语义
- 无 pool 相关日志

- [ ] **步骤 8：运行双卡 `sdpa` 兼容入口 benchmark，确认不再落入旧坏簇**

先重启双卡服务为 `sdpa` 请求入口：

```bash
tmux kill-session -t vividvr_serve_dual_cleanup
tmux new-session -d -s vividvr_serve_dual_cleanup_sdpa \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend sdpa \
    --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
    --enable-torch-compile --dist-timeout 3600 \
    --host 127.0.0.1 --port 31191 --master-port 30191 --scheduler-port 56191 --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_cleanup_sdpa_$(date -u +%Y%m%dT%H%M%SZ).log'
```

再运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  Vivid_Acceptance/tmp/run_vividvr_service_benchmark.py \
  --base-url http://127.0.0.1:31191 \
  --label dual_gpu_sdpa_eager_compile_cleanup
```

再查日志：

```bash
rg -n "requested_backend|effective_backend|semantics|kernel|native" Vivid_Acceptance/logs/vividvr_serve_dual_cleanup_sdpa_*.log
```

预期：
- `requested_backend=sdpa`
- effective runtime 是 SP 正确语义，不是旧 `native`
- `ssim_mean` 明显高于历史 `0.966x`，并接近双卡 `fa` 结果

- [ ] **步骤 9：汇总三组正式 JSON 与关键指标**

运行：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("/home/zhiheng/sglang/Vivid_Acceptance/indicator")
labels = [
    "single_gpu_fa_compile_cleanup",
    "dual_gpu_fa_eager_compile_cleanup",
    "dual_gpu_sdpa_eager_compile_cleanup",
]
for label in labels:
    matches = sorted(root.glob(f"vividvr-service-benchmark-long-130f-20step-{label}-*.json"))
    path = matches[-1]
    payload = json.loads(path.read_text())
    print(label, path.name, payload["compare"]["ssim_mean"], payload["model_inference_runtime_seconds"])
PY
```

预期：
- 三组 JSON 均存在
- 三组都有 `compare.ssim_mean`
- 双卡 `sdpa` 不再是历史低 SSIM 簇

- [ ] **步骤 10：Commit 文档与验收结论**

```bash
git add \
  AGENTS.md \
  docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
  docs_xzh/benchmark_results/vividvr_serve_long_130f_20step_benchmark_20260621.md
git commit -m "docs: update vividvr sp semantics and validation contract"
```

## 自检

- 规格覆盖度：
  - 删除 pool -> 任务 3
  - 删除双卡 native 错误语义 -> 任务 1 + 任务 2
  - `fa -> sdpa` fallback -> 任务 1 + 任务 2
  - `serve` 主口径 130 帧验收 -> 任务 4
  - 单卡 / 双卡 / 双卡 `sdpa` 三组 SSIM -> 任务 4
- 占位符扫描：
  - 本计划不包含 `TODO`、`待定`、`后续实现`、`类似任务 N`
- 类型一致性：
  - 用户可见 backend 统一写作 `fa|sdpa`
  - 内部语义统一写作 `local|ulysses_sp`
  - 双卡 fallback 统一定义为“同语义层 kernel fallback”，不再使用 `native`
