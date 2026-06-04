# 后续代码修改阶段实施顺序建议

这个文档比路线图更偏执行顺序，目的是减少返工。这里强调按阶段推进，不做按周节奏约束。

## 1. 总原则

后续代码改造必须按下面顺序推进：

1. 先固定工程识别链
2. 再补足 `CogVideoX` 缺失底座
3. 再叠加 `VividVR` 私有结构
4. 再做“固定 caption 输入”的单 clip 输出
5. 再做 reference 对齐
6. 再做长视频
7. 再做增强模块
8. 最后做性能

不要反过来。

## 2. 推荐执行顺序

### Step 1: 工程骨架先行

先改：

- `configs/pipeline_configs/vividvr.py`
- `configs/sample/vividvr.py`
- `registry.py`
- `configs/pipeline_configs/__init__.py`
- `configs/sample/__init__.py`

理由：

- 先把命名、探测、配置入口稳定下来
- 后续模型文件有地方可挂
- 同时把 `prompt_file_path` 合同先冻结，避免后面误接 `CogVLM2`

### Step 2: scheduler 先于完整 pipeline

优先做：

- `runtime/models/schedulers/cogvideox_dpm_vividvr.py`

理由：

- scheduler 是最高风险之一
- 它的 step 合同会反向影响 denoising stage 设计

### Step 3: VAE 与 transformer base

再做：

- `runtime/models/vaes/cogvideox.py`
- `runtime/models/dits/cogvideox.py`

理由：

- `sglang` 当前没有 `CogVideoX` 原生底座
- 没有它们，后面所有 pipeline 都是空中楼阁

### Step 4: VividVR 增量模型

再做：

- `runtime/models/dits/cogvideox_vividvr.py`
- `runtime/models/dits/cogvideox_vividvr_controlnet.py`

理由：

- 只有 base 模型稳定后，增量差异才容易定位

### Step 5: 单 clip helper + stages

再做：

- `runtime/vividvr/preprocess.py`
- `runtime/vividvr/tiling.py`
- `runtime/pipelines_core/stages/model_specific_stages/vividvr.py`

理由：

- 这一步开始建立单 clip 合同
- 这一步开始固定 caption 读取来源为：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`

### Step 6: 单 clip pipeline

再做：

- `runtime/pipelines/vividvr_pipeline.py`

理由：

- 最早拿到第一份端到端输出
- 尽快进入逐帧对齐阶段
- 避免把 `CogVLM2` 乱码问题混入主链调试

### Step 7: 先修到对齐达标，再做长视频

对 Phase 5 输出先做：

- shape 修正
- scheduler 修正
- tile merge 修正
- reference 对齐

reference 固定对照：

- `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`

只有单 clip 达标后，才继续：

- `runtime/vividvr/windowing.py`
- `runtime/pipelines/vividvr_pipeline.py` 长视频 orchestration

### Step 8: caption / postprocess 后置

最后再做：

- `runtime/vividvr/captioning.py`
- `runtime/vividvr/postprocess.py`

理由：

- 它们不是核心生成链
- 过早接入只会扩大问题面
- 当前 `sglang` 环境中 `CogVLM2` caption 有乱码问题，因此这里只保留占位，不在当前集成阶段实际接入实时 caption

### Step 9: compile / offload / backend / 回归

最后做：

- 性能分析
- compile
- offload
- attention backend 评估
- regression suite

## 3. 按大阶段落地建议

- `大阶段 A：方案冻结 + 工程入口`
- 包含：`Step 1 + Step 2`
- 退出条件：`registry + config` 稳定，scheduler 单步语义可验证

- `大阶段 B：核心模型底座迁移`
- 包含：`Step 3 + Step 4`
- 退出条件：`CogVideoX` base 与 `VividVR` 增量模型可独立前向

- `大阶段 C：单 clip MVP + reference 对齐`
- 包含：`Step 5 + Step 6 + Step 7` 的单 clip 修正部分
- 退出条件：单 clip 输出稳定，并达到 reference 对齐门槛

- `大阶段 D：长视频能力 + 可选增强`
- 包含：`Step 7` 的长视频部分 + `Step 8`
- 退出条件：长视频 merge 稳定，可选模块可独立启停且不破坏主链

- `大阶段 E：性能收口 + 回归验收`
- 包含：`Step 9`
- 退出条件：默认参数收口、性能结论明确、回归集可复用

## 4. 什么时候可以并行

只有在“单 clip reference 对齐达标”后，才建议并行做：

- 长视频 orchestration
- caption helper
- postprocess helper

在这之前不建议并行，因为基础合同还没稳定。

## 5. 最关键的门槛

后续推进是否继续，建议以这三个门槛控制：

1. `registry + config` 稳定
2. 单 clip 输出稳定
3. 单 clip reference 对齐达标

只要第 3 条还没过，不应该进入性能优化阶段。
