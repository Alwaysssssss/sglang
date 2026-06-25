# Stage 6: 风险分析

## 1. 风险总表

| 模块 | 风险等级 | 原因 | 解决方案 |
| --- | --- | --- | --- |
| CogVideoX transformer 原生移植 | 高 | `sglang` 当前无现成实现，结构复杂 | 先移植 base，再叠加 VividVR 增量 |
| CogVideoX VAE 原生移植 | 高 | 需要 encode/decode、tiling、temporal ratio 正确 | 先做最小功能版并建立 shape 对照 |
| VividVR controlnet | 高 | 当前无公共 controlnet 组件类型 | 第一版局部私有实现，不先公共化 |
| Scheduler step 合同 | 高 | 多出 `old_pred_original_sample` 和 restoration 语义 | 独立 scheduler 文件，单步测试先行 |
| latent shape / padding | 高 | `patch_size_t` 和 `8k+1` padding 容易出错 | 文档化 shape 合同，写单元测试 |
| 长视频 temporal merge | 高 | 逻辑不在 pipeline 内部，而在 orchestration | 先做单 clip，再做全局 timestep merge |
| spatial tiling prompt 对齐 | 中高 | tile 数量要与 prompt list 一一对应 | 把 tile plan 显式持久化到 runtime |
| auto caption | 中 | 引入第二个模型族，扩大调试面 | 延后为可选前处理 |
| TextFixer / OCR / ESRGAN | 中 | 依赖复杂，不属于核心扩散路径 | 延后并插件化 |
| attention backend 兼容性 | 中高 | `CogVideoX` 尚未接入 SGLang 高性能 backend | 第一版用保守后端，后续再优化 |
| compile 稳定性 | 中 | tile shape、control 分支可能影响编译收益 | 数值稳定后再接 |
| registry / detector 识别 | 低 | 机制成熟 | 按既有 `register_configs()` 模式加 detector |

## 2. 最高风险的四个点

## 2.1 Scheduler 语义风险

这是最高风险之一，因为：

- VividVR 的恢复引导不是简单 CFG
- `old_pred_original_sample` 会跨 step 传递
- 一旦实现偏差，最终结果会明显漂移

建议：

- 先写 scheduler 单步对照测试
- 再接入 denoise loop

## 2.2 Latent shape / temporal padding 风险

关键风险来自：

- `patch_size_t = 2`
- `temporal_compression_ratio = 4`
- `8k+1` 帧数假设
- 首帧 latent padding 特殊处理

建议：

- 单独写 `shape contract` 文档和测试
- 任何 helper 都以 `[B,T,C,H,W]` 为统一输入输出

## 2.3 长视频 temporal merge 风险

原实现不是平均融合，而是按照映射关系做覆盖式替换。

风险：

- 若 merge 规则理解错，clip 接缝会不稳定
- 若 merge 放错层，会让 pipeline 状态难以维护

建议：

- 先在 `runtime/vividvr/windowing.py` 独立实现
- 再让 pipeline orchestration 调用

## 2.4 controlnet 局部实现风险

风险不在模型本身，而在工程组织：

- 若过早公共化，会拖累整个 runtime 结构
- 若完全硬编码在 pipeline，又会失去可维护性

建议：

- 第一版先局部但模块化
- 待第二个同类模型出现后再考虑抽象层

## 3. 中风险项

### 3.1 auto caption

风险：

- 引入 CogVLM2 另一个巨型模型
- tile prompt 数量要与 tiling 计划严格一致

建议：

- 先从显式 prompt 模式开始
- caption 结果缓存为 sidecar，避免重复推理
- 当前直接使用：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 在 `sglang` 环境里的 `CogVLM2` 乱码问题未解决前，不进入实时 caption 集成

### 3.2 后处理链

风险：

- OCR / ESRGAN 依赖额外 checkpoint
- 错误时很难判断是生成问题还是后处理问题

建议：

- 不纳入主生成路径验收
- 后处理失败可跳过，不阻塞主结果

## 4. 低风险项

### 4.1 registry / detector

这部分照现有 `registry.py` 模式增加即可，风险较低。

### 4.2 pipeline 自动发现

只要写好 `EntryClass`，风险较低。

## 5. 风险降低顺序

正确顺序应是：

1. 单步 scheduler
2. 单 clip shape contract
3. 单 clip 端到端
4. spatial tiling
5. 长视频 temporal merge
6. caption
7. 后处理
8. 性能优化

## 6. 开放决策

- 是否允许 MVP 阶段先不做“与原 Vivid-VR 数值高度一致”，而只要求结构闭环成立。

建议：

- MVP 阶段要求“结构正确 + 基本数值稳定”
- 第二阶段再做更严格的回归对齐
