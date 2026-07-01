# Vivid-VR 接入 SGLang 加速与服务学习文档执行计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 产出一份只聚焦“Vivid-VR 如何接入 SGLang 的加速能力与服务能力”的项目学习文档，保存到 `docs_xzh/project_learning_guide.md`，覆盖 native pipeline 注册、运行时参数、attention backend、算子融合、`torch.compile`、SP/Ulysses 并行和 FlowCut 服务链路，不展开算法流程。

**架构：** 采用“按运行链路展开、按加速主题穿插”的写法，从文档总览和注册入口开始，顺着 `serve CLI -> server args -> pipeline 初始化 -> acceleration hooks -> service API` 组织内容。最终文档只修改 `docs_xzh/project_learning_guide.md`，所有说明都必须引用具体代码路径与关键函数/类名，不改任何功能代码。

**技术栈：** Markdown、`rg`、`sed`、Git、SGLang multimodal runtime、FastAPI、PyTorch

---

## 实施前必读

- `AGENTS.md`
- `docs_xzh/add_strategy/README.md`
- `docs_xzh/add_strategy/06_stage5_acceleration_adaptation.md`
- `docs_xzh/add_strategy/08_stage7_execution_roadmap.md`
- `docs_xzh/add_strategy/10_grouped_stage_acceptance.md`
- `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`
- `docs_xzh/add_strategy/12_phase_e_sp_native_acceleration_plan.md`
- `docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`
- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- `docs_xzh/hand_over/phase_e_default_configs_and_serve_followups_handover_20260622.md`

## 文件结构

- 创建：`docs_xzh/project_learning_guide.md`
  - 最终学习文档；只讲 Vivid-VR 如何接入 SGLang 加速与服务，不讲算法流程。
- 修改：`.codex/plans/project-learning-doc.md`
  - 当前执行计划；记录章节拆分、必读文件、验证命令和写作边界。

## 写作边界

- 只允许修改文档文件：`docs_xzh/project_learning_guide.md`
- 不允许修改任何功能代码、测试代码、运行脚本、配置默认值或服务协议实现
- 每个章节都必须至少引用 2 个具体代码路径；涉及关键挂接点时要落到函数、方法或类
- 引用代码时优先使用以下格式：
  - 文件路径：``python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py``
  - 符号名：``VividVRPipeline.initialize_pipeline()``
  - 命令示例：``rg -n "initialize_pipeline|_apply_attention_backend" python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py``
- 文档中明确排除：
  - Vivid-VR 算法原理
  - diffusion / control 流程推导
  - benchmark 结果复现细节
  - 非 Vivid-VR 模型的通用 multimodal pipeline 介绍

## 章节映射

- 任务 1 -> `## 1. 项目整体做什么`
- 任务 2 -> `## 2. 整体架构与核心模块`
- 任务 3 -> `## 3. Vivid-VR 如何注册为 SGLang Native Pipeline`
- 任务 4 -> `## 4. 加速与服务参数如何从 Serve CLI 进入运行时`
- 任务 5 -> `## 5. Pipeline 初始化阶段如何挂接加速能力`
- 任务 6 -> `## 6. Flash Attention / SDPA Backend 接入`
- 任务 7 -> `## 7. 算子融合与 Torch Compile 接入`
- 任务 8 -> `## 8. SP / Ulysses 并行接入`
- 任务 9 -> `## 9. 服务接入：FlowCut API 与 Caption Bridge`
- 任务 10 -> `## 10. 推荐学习顺序、重点文件与阅读方法`

### 任务 1：撰写“项目整体做什么”

**文件：**
- 创建：`docs_xzh/project_learning_guide.md`
- 阅读：`docs_xzh/add_strategy/README.md`
- 阅读：`docs_xzh/hand_over/phase_e_default_configs_and_serve_followups_handover_20260622.md`
- 阅读：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 阅读：`python/sglang/multimodal_gen/registry.py`
- 阅读：`python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`

- [ ] **步骤 1：提取“native 集成 + 加速能力”主叙事**

运行：

```bash
sed -n '1,220p' docs_xzh/add_strategy/README.md
sed -n '1,220p' docs_xzh/run_command/vividvr_default_run_and_serve_commands.md
rg -n "VividVRPipelineConfig|VividVRSamplingParams|vividvr" python/sglang/multimodal_gen/registry.py python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py
```

预期：拿到“Vivid-VR 被作为 native pipeline 注册进 `sglang.multimodal_gen`，并使用 SGLang runtime 加速与服务化”的一手材料。

- [ ] **步骤 2：编写第 1 章正文**

将以下骨架写入 `docs_xzh/project_learning_guide.md`：

```markdown
# Vivid-VR 接入 SGLang 加速与服务学习指南

## 1. 项目整体做什么

- 本项目的目标不是在运行时依赖原版 `/home/zhiheng/Vivid-VR`，而是把 Vivid-VR 作为 native pipeline 接入 `sglang.multimodal_gen`。
- 接入后的重点收益是复用 SGLang 的运行时能力，包括 attention backend、算子融合、`torch.compile`、SP/Ulysses 并行和 `serve` 服务化。
- 学习本文时应关注工程接入链路，而不是算法流程或模型原理。

关键代码入口：
- `python/sglang/multimodal_gen/registry.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
```

- [ ] **步骤 3：补充代码路径与符号级引用**

在本章中加入以下明确引用：

```markdown
- `registry.py` 中的 `ModelRegistry.register(...)` 负责把 `VividVRPipelineConfig` 和 `VividVRSamplingParams` 挂进统一注册表。
- `configs/pipeline_configs/vividvr.py` 中的 `VividVRPipelineConfig` 固化了 Vivid-VR 在 SGLang 内部的 pipeline 语义边界。
```

- [ ] **步骤 4：验证本章已包含代码路径**

运行：

```bash
rg -n "^## 1\\.|registry.py|pipeline_configs/vividvr.py|VividVRPipelineConfig|VividVRSamplingParams" docs_xzh/project_learning_guide.md
```

预期：输出第 1 章标题和至少 4 处精确代码路径/符号引用。

### 任务 2：撰写“整体架构与核心模块”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/registry.py`
- 阅读：`python/sglang/multimodal_gen/configs/sample/vividvr.py`
- 阅读：`python/sglang/multimodal_gen/runtime/server_args.py`
- 阅读：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 阅读：`python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
- 阅读：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`

- [ ] **步骤 1：按职责归纳核心模块**

运行：

```bash
rg -n "class VividVRSamplingParams|class ServerArgs|class VividVRPipeline|create_app|APIRouter" \
  python/sglang/multimodal_gen/configs/sample/vividvr.py \
  python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/runtime/entrypoints/http_server.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py
```

预期：能把模块分成“注册/配置、runtime 装配、模型加速适配、HTTP 服务接入”四层。

- [ ] **步骤 2：编写第 2 章正文**

将以下结构追加到 `docs_xzh/project_learning_guide.md`：

```markdown
## 2. 整体架构与核心模块

本文只关心四类模块：
- 注册与配置层：决定 Vivid-VR 如何成为 SGLang 可识别的 pipeline
- 运行时装配层：决定 `serve` 参数如何变成真实 pipeline 实例
- 加速适配层：决定 backend、fusion、compile、SP 如何挂接
- 服务入口层：决定外部 HTTP 请求如何进入 Vivid-VR 生成链路

关键代码入口：
- `python/sglang/multimodal_gen/registry.py`
- `python/sglang/multimodal_gen/runtime/server_args.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
```

- [ ] **步骤 3：补充“模块 -> 责任 -> 为什么必须读”三列表述**

至少写出以下四个模块：

```markdown
- `registry.py`：native pipeline 注册入口
- `server_args.py`：加速与服务参数的统一入口
- `vividvr_pipeline.py`：attention backend、fusion、compile、parallelism 真正挂接的位置
- `vividvr_flowcut_api.py`：FlowCut 服务协议进入 Vivid-VR sampling params 的入口
```

- [ ] **步骤 4：验证本章覆盖核心模块**

运行：

```bash
rg -n "^## 2\\.|server_args.py|vividvr_pipeline.py|vividvr_flowcut_api.py|registry.py" docs_xzh/project_learning_guide.md
```

预期：第 2 章出现 4 个核心模块路径，且每个模块有一句职责描述。

### 任务 3：撰写“Vivid-VR 如何注册为 SGLang Native Pipeline”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/registry.py`
- 阅读：`python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- 阅读：`python/sglang/multimodal_gen/configs/sample/vividvr.py`

- [ ] **步骤 1：定位注册表、pipeline config 和 sampling params 的连接点**

运行：

```bash
rg -n "VividVRPipelineConfig|VividVRSamplingParams|register\\(" \
  python/sglang/multimodal_gen/registry.py \
  python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py \
  python/sglang/multimodal_gen/configs/sample/vividvr.py
```

预期：能明确回答“Vivid-VR 在哪里被注册、请求参数对象在哪里定义、pipeline 默认语义在哪里固定”。

- [ ] **步骤 2：编写第 3 章正文**

追加以下骨架：

```markdown
## 3. Vivid-VR 如何注册为 SGLang Native Pipeline

- `registry.py` 负责把 Vivid-VR 接入统一的 multimodal pipeline 注册表。
- `configs/pipeline_configs/vividvr.py` 中的 `VividVRPipelineConfig` 定义 pipeline 默认行为边界。
- `configs/sample/vividvr.py` 中的 `VividVRSamplingParams` 定义服务请求最终会落成什么运行时参数对象。

关键代码入口：
- `python/sglang/multimodal_gen/registry.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- `python/sglang/multimodal_gen/configs/sample/vividvr.py`
```

- [ ] **步骤 3：增加“注册链路”小节**

用 3 个编号点写清楚：

```markdown
1. `registry.py` 注册 Vivid-VR 对应的 config 与 sampling params。
2. `VividVRPipelineConfig` 约束了文本长度、reference 输入和其他 pipeline 级默认语义。
3. `VividVRSamplingParams.from_user_kwargs(...)` 把服务请求与 server args 合并成最终执行参数。
```

- [ ] **步骤 4：验证本章具备类名与方法名引用**

运行：

```bash
rg -n "^## 3\\.|VividVRPipelineConfig|VividVRSamplingParams|from_user_kwargs|registry.py" docs_xzh/project_learning_guide.md
```

预期：第 3 章包含类名、方法名和代码路径，而不是只有概念描述。

### 任务 4：撰写“加速与服务参数如何从 Serve CLI 进入运行时”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py`
- 阅读：`python/sglang/multimodal_gen/runtime/server_args.py`
- 阅读：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`

- [ ] **步骤 1：抽取 CLI 参数入口与关键 flag**

运行：

```bash
rg -n "from_cli_args|add_cli_args|attention_backend|sp_degree|ulysses_degree|enable_torch_compile|enable_cogvideox" \
  python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py \
  python/sglang/multimodal_gen/runtime/server_args.py
sed -n '1,240p' docs_xzh/run_command/vividvr_default_run_and_serve_commands.md
```

预期：整理出 `serve` 如何暴露 `--attention-backend`、`--sp-degree`、`--ulysses-degree`、`--enable-torch-compile`、fusion flags 和 caption bridge flags。

- [ ] **步骤 2：编写第 4 章正文**

追加以下骨架：

```markdown
## 4. 加速与服务参数如何从 Serve CLI 进入运行时

- `entrypoints/cli/serve.py` 只做一件事：从 CLI 创建 `ServerArgs`，再启动 server。
- `runtime/server_args.py` 是 Vivid-VR 加速参数和服务参数的统一入口。
- 默认运行命令中的 backend、SP、compile、caption sidecar 配置，都会先汇入 `ServerArgs`。

关键代码入口：
- `python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py`
- `python/sglang/multimodal_gen/runtime/server_args.py`
```

- [ ] **步骤 3：逐项列出关键参数与对应处理点**

至少覆盖：

```markdown
- `attention_backend`
- `sp_degree`
- `ulysses_degree`
- `enable_torch_compile`
- `enable_cogvideox_qkv_fusion`
- `enable_cogvideox_qk_norm_fusion`
- `enable_cogvideox_qk_norm_rope_fusion`
- `enable_cogvideox_modulation_fusion`
- `vividvr_caption_bridge`
- `vividvr_caption_sidecar_url`
```

并为每个参数补一句“在 `server_args.py` 的哪个方法里被校验或调整”。

- [ ] **步骤 4：验证本章具备参数到方法的映射**

运行：

```bash
rg -n "^## 4\\.|_adjust_parallelism|_adjust_attention_backend|_validate_vividvr_caption_bridge|add_cli_args" docs_xzh/project_learning_guide.md
```

预期：第 4 章能看到参数名和 `ServerArgs` 方法名的一一对应。

### 任务 5：撰写“Pipeline 初始化阶段如何挂接加速能力”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`

- [ ] **步骤 1：提取初始化顺序与 acceleration hooks**

运行：

```bash
rg -n "initialize_pipeline|_maybe_initialize_model_parallel_runtime|_apply_attention_backend|_apply_qkv_fusion|_apply_qk_norm_fusion|_apply_qk_norm_rope_fusion|_apply_modulation_fusion|_apply_torch_compile" \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py
```

预期：得到一条清晰的初始化顺序链：并行运行时初始化 -> 模型装载 -> backend 应用 -> fusion 应用 -> `torch.compile`。

- [ ] **步骤 2：编写第 5 章正文**

追加以下骨架：

```markdown
## 5. Pipeline 初始化阶段如何挂接加速能力

- `vividvr_pipeline.py` 是 acceleration hooks 的真正落点。
- `VividVRPipeline.initialize_pipeline()` 负责在模型装载后依次应用 backend、fusion 和 `torch.compile`。
- 本章应该用“初始化顺序图”的写法解释接入点，而不是解释模型算法。

关键代码入口：
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
```

- [ ] **步骤 3：补充“初始化顺序”编号列表**

使用以下顺序：

```markdown
1. `_maybe_initialize_model_parallel_runtime(...)`
2. 装载 text encoder / transformer / controlnet
3. `_apply_attention_backend(...)`
4. `_apply_qkv_fusion(...)` / `_apply_qk_norm_fusion(...)` / `_apply_qk_norm_rope_fusion(...)`
5. `_apply_modulation_fusion(...)`
6. `_apply_torch_compile(...)`
```

- [ ] **步骤 4：验证本章已引用所有 hook 名称**

运行：

```bash
rg -n "^## 5\\.|_apply_attention_backend|_apply_qkv_fusion|_apply_modulation_fusion|_apply_torch_compile" docs_xzh/project_learning_guide.md
```

预期：第 5 章列出全部关键 hook，而不是只说“进行了优化”。

### 任务 6：撰写“Flash Attention / SDPA Backend 接入”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- 阅读：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`

- [ ] **步骤 1：定位 backend 规范化、解析和应用点**

运行：

```bash
rg -n "normalize_cogvideox_attention_backend|resolve_cogvideox_attention_runtime_choice|inspect_cogvideox_attention_backend|set_cogvideox_attention_backend|set_attention_backend" \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py
```

预期：能说明 `fa` / `sdpa` 如何被规范化，以及双卡 SP 时为什么会解析成 `fa_sp` / `sdpa_sp`。

- [ ] **步骤 2：编写第 6 章正文**

追加以下骨架：

```markdown
## 6. Flash Attention / SDPA Backend 接入

- backend 的“字符串语义”集中定义在 `cogvideox_attention_backend.py`。
- backend 的“真正应用”发生在 `vividvr_pipeline.py` 的 `_apply_attention_backend(...)`。
- controlnet 侧也暴露了 backend setter，因此 transformer 与 controlnet 可以保持同一套 attention 语义。

关键代码入口：
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
```

- [ ] **步骤 3：增加“backend 解析表”**

至少写清楚：

```markdown
- `fa`：单卡/非 SP 的 Flash Attention 语义
- `sdpa`：单卡/非 SP 的 PyTorch SDPA 语义
- `fa_sp`：SP/Ulysses 场景下的 Flash Attention kernel 语义
- `sdpa_sp`：SP/Ulysses 场景下的 SDPA kernel 语义
```

并补一句：`resolve_cogvideox_attention_runtime_choice(...)` 负责从“用户请求 backend”推导到“实际 runtime choice”。

- [ ] **步骤 4：验证本章覆盖 backend 名称与应用函数**

运行：

```bash
rg -n "^## 6\\.|fa_sp|sdpa_sp|resolve_cogvideox_attention_runtime_choice|_apply_attention_backend" docs_xzh/project_learning_guide.md
```

预期：第 6 章同时出现 backend 名称、解析函数和 pipeline 应用函数。

### 任务 7：撰写“算子融合与 Torch Compile 接入”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`
- 阅读：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 阅读：`python/sglang/multimodal_gen/runtime/server_args.py`

- [ ] **步骤 1：定位 fusion 开关与 compile 开关**

运行：

```bash
rg -n "enable_cogvideox_qkv_fusion|enable_cogvideox_qk_norm_fusion|enable_cogvideox_qk_norm_rope_fusion|enable_cogvideox_modulation_fusion|enable_torch_compile" \
  python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py
rg -n "LayerNormScaleShift|ScaleResidualLayerNormScaleShift|MulAdd" \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py
```

预期：能把“开关来自 server args、应用发生在 pipeline、实现位于 operator fusion 文件”三者串起来。

- [ ] **步骤 2：编写第 7 章正文**

追加以下骨架：

```markdown
## 7. 算子融合与 Torch Compile 接入

- fusion flags 定义在 `server_args.py`，真正执行发生在 `vividvr_pipeline.py`。
- `cogvideox_operator_fusion.py` 提供 Vivid-VR / CogVideoX 相关的 fused operator 实现。
- `torch.compile` 的开关与编译时机同样由 `vividvr_pipeline.py` 统一管理。

关键代码入口：
- `python/sglang/multimodal_gen/runtime/server_args.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`
```

- [ ] **步骤 3：分别写出“开关层 / 应用层 / 实现层”**

至少使用以下三段小标题：

```markdown
### 7.1 开关层：ServerArgs
### 7.2 应用层：VividVRPipeline
### 7.3 实现层：CogVideoX Operator Fusion
```

并在 `torch.compile` 段落中点名 `_apply_torch_compile(...)` 与 `_maybe_torch_compile_module(...)`。

- [ ] **步骤 4：验证本章出现 fusion 类名与 compile 方法**

运行：

```bash
rg -n "^## 7\\.|LayerNormScaleShift|MulAdd|_apply_torch_compile|_maybe_torch_compile_module" docs_xzh/project_learning_guide.md
```

预期：第 7 章具备具体 fused class 名称和 compile 方法名。

### 任务 8：撰写“SP / Ulysses 并行接入”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/runtime/server_args.py`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
- 阅读：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 阅读：`docs_xzh/add_strategy/12_phase_e_sp_native_acceleration_plan.md`
- 阅读：`docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`

- [ ] **步骤 1：抽取 SP 维度、Ulysses 语义和 connector 适配点**

运行：

```bash
rg -n "sp_degree|ulysses_degree|ring_degree|_adjust_parallelism|_validate_parallelism" \
  python/sglang/multimodal_gen/runtime/server_args.py
rg -n "ulysses_sp|USPAttention|SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE|eager_global|distributed_local|shard_vividvr_video_tokens|gather_vividvr_video_tokens" \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py
```

预期：能解释清楚“SP 参数如何进入 runtime、为什么双卡统一进入 Ulysses joint-attention 语义、Vivid-VR connector 如何适配 sequence parallel”。

- [ ] **步骤 2：编写第 8 章正文**

追加以下骨架：

```markdown
## 8. SP / Ulysses 并行接入

- SP 并行参数从 `server_args.py` 进入，并在那里完成合法性校验和默认推导。
- backend 解析在 SP 打开后不再停留在 `fa` / `sdpa` 的单卡语义，而会进入 Ulysses 对应的 runtime choice。
- Vivid-VR 特有的 connector/control 路径适配集中在 `cogvideox_vividvr_common.py`。

关键代码入口：
- `python/sglang/multimodal_gen/runtime/server_args.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
```

- [ ] **步骤 3：增加“单卡 vs 双卡 SP”对比段**

至少写清楚：

```markdown
- 单卡正式配置：`single_gpu_fa_compile`
- 双卡正式配置：`dual_gpu_fa_eager_compile`
- 双卡兼容口径：`dual_gpu_sdpa_eager_compile`
```

并补一句：这些配置的服务命令维护在 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`。

- [ ] **步骤 4：验证本章出现 SP 核心术语**

运行：

```bash
rg -n "^## 8\\.|USPAttention|eager_global|ulysses|single_gpu_fa_compile|dual_gpu_fa_eager_compile" docs_xzh/project_learning_guide.md
```

预期：第 8 章同时覆盖配置名、并行术语和 connector 代码路径。

### 任务 9：撰写“服务接入：FlowCut API 与 Caption Bridge”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`python/sglang/multimodal_gen/runtime/launch_server.py`
- 阅读：`python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
- 阅读：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 阅读：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py`
- 阅读：`python/sglang/multimodal_gen/configs/sample/vividvr.py`

- [ ] **步骤 1：梳理 HTTP server 到 sampling params 的调用链**

运行：

```bash
rg -n "launch_server|launch_http_server_only|create_app" \
  python/sglang/multimodal_gen/runtime/launch_server.py \
  python/sglang/multimodal_gen/runtime/entrypoints/http_server.py
rg -n "ensure_vividvr_caption_file|build_vividvr_repair_kwargs|from_user_kwargs|read_vividvr_runtime_progress" \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py \
  python/sglang/multimodal_gen/configs/sample/vividvr.py
```

预期：拿到一条完整链路：`serve` 启动 -> FastAPI 路由注册 -> FlowCut handler -> caption bridge/shared helpers -> `VividVRSamplingParams.from_user_kwargs(...)`。

- [ ] **步骤 2：编写第 9 章正文**

追加以下骨架：

```markdown
## 9. 服务接入：FlowCut API 与 Caption Bridge

- 服务启动入口位于 `launch_server.py` 和 `http_server.py`。
- Vivid-VR 的业务服务入口位于 `vividvr_flowcut_api.py`，它不是一个泛化的 video API 包装，而是独立的 FlowCut 服务链。
- caption sidecar / caption bridge 的服务适配通过 `video_repair_shared.py` 进入请求预处理，再落到 `VividVRSamplingParams`。

关键代码入口：
- `python/sglang/multimodal_gen/runtime/launch_server.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py`
```

- [ ] **步骤 3：用编号列表写清服务数据流**

使用以下顺序：

```markdown
1. `serve.py` / `launch_server.py` 启动 runtime 和 HTTP server
2. `http_server.py` 注册 `vividvr_flowcut_api` 路由
3. `vividvr_flowcut_api.py` 处理请求清洗、progress、callback 和任务提交
4. `video_repair_shared.py` 处理 caption bridge 与 request kwargs 组装
5. `VividVRSamplingParams.from_user_kwargs(...)` 生成最终运行参数
```

- [ ] **步骤 4：验证本章具备服务链路关键节点**

运行：

```bash
rg -n "^## 9\\.|launch_server.py|create_app|vividvr_flowcut_api.py|video_repair_shared.py|from_user_kwargs" docs_xzh/project_learning_guide.md
```

预期：第 9 章完整出现服务链路上的每个关键文件和方法。

### 任务 10：撰写“推荐学习顺序、重点文件与阅读方法”

**文件：**
- 修改：`docs_xzh/project_learning_guide.md`
- 阅读：`docs_xzh/add_strategy/README.md`
- 阅读：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 阅读：`python/sglang/multimodal_gen/runtime/server_args.py`
- 阅读：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`
- 阅读：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
- 阅读：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`

- [ ] **步骤 1：整理一条“先主链后专题”的阅读路径**

运行：

```bash
printf '%s\n' \
  docs_xzh/add_strategy/README.md \
  python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/registry.py \
  python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py \
  python/sglang/multimodal_gen/configs/sample/vividvr.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py
```

预期：得到一条可以直接照着读的文件顺序，而不是松散文件清单。

- [ ] **步骤 2：编写第 10 章正文**

追加以下骨架：

```markdown
## 10. 推荐学习顺序、重点文件与阅读方法

- 推荐先读“文档总览 + server args”，再读“注册/配置”，之后进入 `vividvr_pipeline.py`，最后再读模型层与服务层。
- 阅读方法应遵循“先问入口参数从哪来，再问它在哪个 hook 被应用，最后问服务是怎么把请求送进来”的顺序。

重点文件：
- `python/sglang/multimodal_gen/runtime/server_args.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
```

- [ ] **步骤 3：增加“重点文件导读表”**

表格至少包含三列：

```markdown
| 文件 | 为什么必须读 | 重点关注符号 |
| --- | --- | --- |
| `runtime/server_args.py` | 所有加速与服务参数的入口 | `add_cli_args`, `_adjust_parallelism`, `_adjust_attention_backend` |
| `runtime/pipelines/vividvr_pipeline.py` | acceleration hooks 真正落地 | `initialize_pipeline`, `_apply_attention_backend`, `_apply_torch_compile` |
```

- [ ] **步骤 4：运行全篇自检**

运行：

```bash
rg -n "^## " docs_xzh/project_learning_guide.md
rg -n "python/sglang/multimodal_gen/.+\\.py" docs_xzh/project_learning_guide.md | wc -l
rg -n "TODO|待定|后续补充|算法细节" docs_xzh/project_learning_guide.md
```

预期：
- 一共出现 10 个二级章节
- 至少出现 20 处具体代码路径引用
- 不出现 `TODO`、`待定`、`后续补充` 等占位符

## 完成定义

- `docs_xzh/project_learning_guide.md` 已创建
- 文档包含 10 个章节，且每个章节对应本计划中的一个任务
- 每个章节至少引用 2 个具体代码路径
- 文档只讨论接入 SGLang 加速与服务的工程链路，不展开算法流程
- 未修改任何功能代码、测试代码或运行脚本

## 执行注意事项

- 只在 `docs_xzh/project_learning_guide.md` 中写内容；不要顺手修代码注释、默认参数或服务实现
- 引用代码时优先写“文件路径 + 符号名 + 作用”，不要只贴大段代码
- 如果某个函数名在当前分支与计划略有偏差，先用 `rg` 校准，再更新文档正文，不要臆测
- 如果发现文档与当前代码有历史偏差，以当前代码和 `AGENTS.md` 为准，并在文档中显式说明“当前实现”
