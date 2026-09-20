# VSR 迁移实施计划

状态：第一阶段实施计划。依据 [`requirements.md`](requirements.md)（需求与验收）与
[`environment.md`](environment.md)（环境、权重、命令行）。**验收判据一律以 `requirements.md` 为准，本文不另立标准。**

最近更新：2026-09-20。

**进度：第一阶段 M0–M6 全部完成。GPU7 全矩阵 9 组 / 649 帧通过，详见 [2026-09-20 验收记录](acceptance_20260920.md)。第二阶段尚未实施。**

当前默认 VSR 环境已切换至 **torch2.10/cu126**，并重跑9组/649帧、单窗口逐阶段和原生pipeline验收，全部与原始基线一致；见 [runtime210.md](runtime210.md)。

## 1. 阶段范围

### 1.1 做

- 在 SGLang 内落地 VSR 的原生 pipeline、CLI 与流式核心。
- 模型第一阶段直接用 diffusers 的 `AutoencoderKLWan` 与 `WanTransformer3DModel`（`requirements.md` §1）。
- 逐阶段张量对齐 + 完整视频验收（§4.1、§4.2、§4.3）。

### 1.2 不做（已确认后置）

| 后置项 | 依据 |
| --- | --- |
| 内存验收 | §2，与加速阶段 2.1 / 2.2 合并 |
| 目录批处理 | §5.1 |
| 音频 | §5.1 |
| HTTP 服务 | §5.2 |
| SGLang 原生 Wan VAE / DiT | §1，第二阶段 |
| 加速（tile 批处理 / 多 GPU / compile / 量化） | §5.2，第二阶段 |
| decord → cv2 解码器替换 | §5，须先单独验证逐帧一致性 |

## 2. 里程碑

依赖关系是线性的；M1 与 M2 可并行（M1 只碰基线侧，M2 只碰 SGLang 侧）。

### M0：环境与基线复现 ✅ 已完成（2026-09-18）

| | |
| --- | --- |
| **目标** | 确认基线能在 SGLang 环境下跑通，取得差异下限 `ε_torch` |
| **结果** | ① `decord 0.6.0` 已装入 uv 环境；② 基线在 SGLang 环境下完整跑通，输出 `53 帧 / 3840×2160 / 25 fps` 正确；③ `ε_torch` = `ssim_min 0.988793` / `mse_max 2.7160` / `mae_max 1.1283`，**53/53 帧通过 `0.97 / 25.0 / 2.5`** |
| **关键发现** | · 基线在 `swiftvr` 下两次运行**产物逐字节相同**（md5 `4e6576d4c594e8eadd3ef8c8a453281a`）→ 下限纯来自环境差异，不含 GPU 随机性<br>· **`mae` 是最紧指标**，`ε_torch` 已占 45% 预算，实现自身只能再吃约 `1.37`<br>· `sglang` 硬性要求 `diffusers==0.37.0`，diffusers 版本差异**无法消除** |
| **产物** | `output_results/vsr/M0_reference_{sglangen,swiftvr_a,swiftvr_b}.mp4`；`reports/M0_*.json`；日志 `M0_baseline_*.log` |
| **判据达成** | 53 帧跑完且帧数正确 ✔ ｜ diffusers 版本问题由 `ε_torch` 实测覆盖并定性为不可消除 ✔ |

详细数据见 `requirements.md` §4.1.2 与 `environment.md` §1.2 / §1.3 / §3.1。

### M1：对照工具链 ✅ 已完成（2026-09-18）

| | |
| --- | --- |
| **目标** | 建好验收所需的一切工具，并**冻结 `requirements.md` §4.1.2 的容差** |
| **工具** | `runtime/vsr/verify/` 下 6 个：`dump_baseline.py`（monkey-patch 基线，5 个采样点 + 帧 dump）、`compare_frames.py`、`compare_tensors.py`、`structural_check.py`、`encode_frames.py`、`dumps.py`。<br>**生产者工具不依赖 `sglang` 可导入**——基线侧在 `swiftvr` 环境里没有 sglang，这条是踩过坑后加的约束。 |
| **结果** | ① §4.2.3 编码器检查 **通过**：同帧经两环境编码，md5 相同，且与基线两次实跑产物逐字节一致 → 顺带端到端验证了 dump 工具；② §4.3 结构检查通过（分辨率/帧数/帧率/帧序全过，`best_lag=0`）；③ 帧层与张量层 `ε_torch` 已测并冻结门限（`requirements.md` §4.1.2） |
| **⚠️ 关键发现** | **`ε_torch` 只由目标分辨率相对源的方向与倍率决定**（A1 对照组精确复现 M1 已知值，扫描方法学成立）：上采样最小（mp4 `mae_max 1.13–1.23`），下采样越狠越大（`1.39 → 1.66 → 2.76`）。色彩模式与窗口数**均无关**。<br>→ **C1 / C2 的 mp4 层门限本身不可达**（`ε_torch` 已达 `2.76 / 2.86`，失败 `25/64` 与 `117/200`）——这是环境下限，不是实现缺陷。<br>→ C1 / C2 的几何已改用近似原生的 `1920×1080` 重测（它们是为覆盖多窗口路径，`320×640` 只是当时为省算力选的）：`mae_max` 由 `2.76 / 2.86` 降到 `1.49 / 1.55`，**全矩阵 mp4 层零失败帧**。 |
| **产物** | `output_results/vsr/dumps/`（各配置 dump）、`reports/EPS_*.json`、9 配置 × 2 环境共 18 次参考实跑 |
| **补测** | A3 / C2 / C3 首轮因同卡邻居进程导致 CUDA OOM 失败，已加等待+重试后补齐（见 `verify/README.md` 运维注意事项） |

### M2：SGLang 侧最小链路 ✅ 已完成（2026-09-18）

| | |
| --- | --- |
| **目标** | 打通 pipeline / stage / 注册表 / CLI，跑通最简配置 |
| **结果** | ① `runtime/vsr/` 七个模块全部落地；② `verify/compare_ops.py` 把全部纯函数与基线逐一对拍，**61/61 逐位相同**（含 `tiled_restore_rect` 用假 `restore_window_fn` 验空间融合整条路径）；③ **直连 CLI 端到端逐字节相同**；④ pipeline 已注册（`WanVSRPipeline`，注册表 31 条，VideoEdit 未受影响）；⑤ **原生 pipeline 路径端到端逐字节相同**（`--via-pipeline`，与同环境参考产物 md5 一致，耗时 `71 s`） |
| **判据达成** | 采样点 1–3 由端到端逐字节相同覆盖（强于逐点对齐）✔ |
| **框架改动** | ① `registry._get_config_info` 新增 `pipeline_class_name` 参数并跳过 `model_index.json`（`requirements.md` §5.3，用户采纳方案 (a)）；② `sglang/utils.py` 的 `KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS` 注册 `"swiftvr": "WanVSRPipeline"`；③ `ModelTaskType` 新增 `VSR` |
| **踩坑记录** | 接入新模型族的固定成本见 `requirements.md` §5.4。其中 **"stage 忽略请求里的几何/tile 参数"** 这个 bug 只有真跑起来才暴露——只做到"注册成功"会一直漏过，见下 |

**一个只有真跑才会暴露的 bug**：stage 最初只读 `PipelineConfig` 的几何与 tile 参数，完全忽略请求里传的
`target_resolution`。请求传 `320x640`、配置默认 `None`，于是回退到 `long_edge=3840` 输出了 4K —— 
管线"跑通了"、`exit=0`、有产物，但产物是错的。已改为请求优先、配置兜底。

### M3：空间切块与融合 ✅ 已完成（2026-09-18）

| | |
| --- | --- |
| **目标** | 多 tile 的空间 feather 与累积归一化，端到端验证 |
| **配置** | B1（`480×832`，4 tile）、B2（`512×512`，2 tile，`W=512 < tile_w=640` 命中单轴补齐）、B3（`768×1280`，9 tile，双轴带重叠） |
| **结果** | 三个配置全部**逐字节相同**（与同环境参考产物对比）：`2ccf8ce6…` / `412a5648…` / `cd9630af…`。几何与补齐路径均在日志中确认 |
| **意义** | `tiled_restore_rect` 从「纯函数逐位相同」升级为「端到端逐字节相同」，包含单轴小于 tile 的 replicate 补齐 |

### M4：时间窗口与融合 ✅ 已完成（2026-09-20）

| | |
| --- | --- |
| **目标** | 流式核心完整形态：reader / writer 线程、有界队列、增量 retire |
| **配置** | C1（`T=64`，三窗口归一化）、C2（`T=200`，多窗口 retire 交接）、C3（`2160×3840`，横屏） |
| **产出** | `stream.py` 完整实现；本次 C1/C2/C3 编码前帧与同环境基线完全一致，mp4 SHA256 相同；细粒度三窗口全部张量完全一致 |
| **判据** | §4.1.3 采样点 4 的时间部分对齐；输出帧数、帧序零容差（`emitted == T`） |
| **依赖** | M3 |

**这是最容易写错的一步。** C1 是 `requirements.md` §7-4 点名的用例：有帧被 3 个窗口覆盖，
`live_acc` / `live_w` 的**归一化**语义与"双侧 cross-fade"在此处**不相等**，必须照搬归一化，
不能按 cross-fade 重写。

### M5：颜色校正与输出 ✅ 已完成（2026-09-20）

| | |
| --- | --- |
| **目标** | 三种 `color_ref` 模式与最终输出 |
| **配置** | A2（`chunk`）、A3（`none`）、A1（`global`）的颜色部分 |
| **产出** | `color.py` 与 `to_uint8_hwc`；A1/A2/A3 三种颜色模式的编码前帧和 mp4 与同环境基线完全一致；编码器确定性产物校验一致 |
| **判据** | §4.1.3 采样点 5 对齐；§4.2.1 编码前 uint8 帧对齐；§4.2.3 编码器检查 |
| **依赖** | M4 |

两个必须照搬的点：颜色校正施加在时间融合**之前**、按 chunk 施加，且统计量覆盖整块（含随后会被
融合掉的 overlap 区）（§7-5）；最终量化是**向零截断**而非四舍五入（§7-7）。

### M6：完整验收 ✅ 已完成（2026-09-20）

| | |
| --- | --- |
| **目标** | 跑满 §3.2 的 A1–A3 / B1–B3 / C1–C3 全矩阵，出报告 |
| **判据** | §4.2.1 帧层对齐 + §4.2.2 mp4 门限 `0.97 / 25.0 / 2.5` + §4.3 结构检查 |
| **产出** | `reports/migration_20260920_acceptance.json`；逐帧报告、命令、权重和源码校验齐全。九组与同环境参考逐位一致；与 swiftvr 参考的帧层、mp4、结构门限全部通过，零失败帧 |
| **依赖** | M5 |

## 3. 代码落点

### 3.1 新增（`python/sglang/multimodal_gen/`）

| 文件 | 职责 |
| --- | --- |
| `runtime/vsr/__init__.py` | 导出主要入口 |
| `runtime/vsr/geometry.py` | `H×W` 解析、`resize_to_long_edge`、`pad_to_multiple(32)`、`reflect_pad_time`、`compute_tile_positions`（含尾窗回退） |
| `runtime/vsr/blending.py` | 一维 feather 权重与 3D mask、时间权重、累积与归一化 |
| `runtime/vsr/color.py` | `color_stats` / `match_color_to_stats` / `scan_color_reference`（float64 累加器） |
| `runtime/vsr/video_io.py` | decord 读取（probe / 窗口解码）、imageio 写出、`to_uint8_hwc`（**截断**） |
| `runtime/vsr/model.py` | diffusers 组件加载（EMA 优先规则）、latent 归一化、单窗口前向 |
| `runtime/vsr/stream.py` | 流式核心：双线程 + 队列 + retire。**不得依赖 CLI 参数对象**（§5.2） |
| `runtime/vsr/cli.py` | 独立 CLI，`restore` 子命令，参数集见 §5.1 |
| `runtime/vsr/verify/` | 验收工具：基线包装 dump、结构检查、编码器检查、报告生成 |
| `runtime/pipelines/wan_vsr_pipeline.py` | `WanVSRPipeline`，底部 `EntryClass` |
| `runtime/pipelines_core/stages/model_specific_stages/vsr.py` | 单个 stage，调用 `runtime/vsr/stream.py` |
| `configs/pipeline_configs/vsr.py` | `WanVSRPipelineConfig`，`task_type = ModelTaskType.VSR` |
| `configs/sample/vsr.py` | `WanVRSamplingParams`，跨阶段状态放 `runtime_*` 字段 |

`verify/` 放在 `runtime/vsr/` 下是跟随 VideoEdit 的先例（`runtime/videoedit/compare.py`）。
它不会被 `_discover_and_register_pipelines()` 扫到——那个遍历的是 `runtime/pipelines`。

### 3.2 改动的既有文件（尽量少）

| 文件 | 改动 |
| --- | --- |
| `configs/pipeline_configs/base.py` | `ModelTaskType` 加一个成员（`VIDEO_EDIT` 在第 52 行的先例） |
| `registry.py` | 加 `register_configs(...)` 调用与对应 import |
| `configs/pipeline_configs/__init__.py`、`configs/sample/__init__.py` | 导出新类 |

注意：VideoEdit 文档把 `runtime/models/registry.py` 列为改动文件，但实际代码里那里没有任何
`videoedit` 字样，注册只发生在顶层 `registry.py`。**不要照抄那一行。**

## 4. 实现约束清单

这些来自 `requirements.md` §7，写错任何一条都会静默产生错误输出，都属于 §4.1.1 的结构性范畴。

- [x] `encoder_hidden_states` 用 `torch.zeros(1, 1, 4096)`，**不加载文本编码器**，pipeline 的
      `_required_config_modules` 去掉 `text_encoder` / `tokenizer`（§7-3）
- [x] 时间融合按**归一化**语义，不是双侧 cross-fade（§7-4）
- [x] 颜色校正施加在时间融合**之前**、按 chunk，统计量覆盖整块含 overlap 区（§7-5）
- [x] 最终量化**向零截断**，不用 `round` / `np.round`（§7-7）
- [x] `color_ref=global` 的预扫描是语义的一部分，不是可选优化（§7-6）
- [x] 尾窗按 `compute_tile_positions` 回退保持满尺寸，`emit_end` 从计划位置反读而非用 `temporal_overlap`（§7-4 相关）
- [x] 流式核心（`runtime/vsr/stream.py`）不依赖 CLI 参数对象（§5.2）
- [x] 第一阶段**不要**启用 TeaCache / Cache-DiT（§5.2 排除项）

第二阶段切原生模型时再处理 §7-1（VAE decoder 权重加载）与 §7-2（latent 归一化重复施加）。

## 5. 产物与目录

```
output_results/vsr/
├── reference_stream.mp4          # 基线产物
├── sglang_stream.mp4             # SGLang 产物
├── reports/                      # compare.py 的 JSON 报告，按矩阵编号命名
├── dumps/                        # 逐阶段张量与编码前 uint8 帧
└── tiles/                        # --save_tiles_dir 调试输出
```

目录约定来自 `environment.md` §5。日志、临时文件一律写在这里，不写进 `vsr` 仓库。

## 6. 成本估算

按 M0 实测的 `3.0 s/patch`（tile `33×320×640`）估算，单侧：

| 配置 | patch 数 | 估计耗时 |
| --- | --- | --- |
| A1 / A2 / A3（4K，53 帧） | 112 × 3 | 约 `336 s` × 3 |
| B1（480×832） | 8 | 约 `24 s` |
| B2（512×512） | 4 | 约 `12 s` |
| B3（768×1280） | 18 | 约 `54 s` |
| C1（T=64，1920×1080） | 42 | 约 `126 s` |
| C2（T=200，1920×1080） | 98 | 约 `294 s` |
| C3（2160×3840，T=64） | 168 | 约 `504 s` |

全矩阵单侧约 `27 分钟`，两端合计约 `55 分钟`。加上 M0 与 M1，整个第一阶段验收的算力开销
在**两小时量级**——验收本身不是瓶颈，实现才是。

**注意 `1.36 s/patch` 这个旧数字不可用。** 它来自 9月17 的记录，而 M0 当天在**未改动的基线上**
实测为 `3.0 s/patch`（GPU 7，快照时 util 0%）。本机 8 张卡当时被其他任务占用 47–75 GB，
这属于**机器负载差异而非代码差异**——同一天的 SGLang 环境跑同一份代码是 `3.47 s/patch`，
与基线的 `3.01` 只差约 15%，不是早先看到的 2.5×。

因此：**性能数字只在记录 GPU 型号、编号与当时占用的情况下可比。** 这一点对第二阶段的加速验收尤其重要
（`requirements.md` §5.2 要求每项加速记录性能变化），需在验收记录里显式包含 GPU 负载快照。

## 7. 风险与回退

| 风险 | 触发 | 回退 |
| --- | --- | --- |
| diffusers `0.36.0` / `0.37.0` 数值不一致 | M0 的 A/B | 保持 SGLang 要求的 `0.37.0`，按 M0/M1 实测环境差异验收；不降级绕过约束 |
| `decord` 装上但行为与基线不同 | M0 跑基线时 | 逐帧比对 decord 输出；必要时改用同一份轮子 |
| `ε_torch` 过大，`0.97/25/2.5` 不可达 | M1 冻结容差时 | 说明门限需重谈——这是 M1 存在的意义，早发现早处理 |
| 时间融合对齐不上 | M4 | 按 §4.1.5 的差异形态表定位；重点核对 `live_acc` / `live_w` 的累加顺序与 `.clone()` 的防 view 固定 |
| 颜色校正导致整体偏置 | M5 | 检查施加位置、统计量覆盖范围、是否被重复施加（§7-2、§7-5） |
| M0/M1 环境问题 | 任意 | `environment.md` §6 记录了全部核实命令，便于复查 |

## 8. 完成判据

第一阶段完成 = 以下全部成立：

1. `requirements.md` §3.2 的 A1–A3 / B1–B3 / C1–C3 全部跑通并留档；
2. §4.1.3 的 5 个采样点都有对齐记录，结构性项零容差、数值性项在冻结容差内；
3. §4.2.1 编码前 uint8 帧对齐通过；
4. §4.2.2 的 mp4 门限 `0.97 / 25.0 / 2.5` 通过，`failed_frames` 为空；
5. §4.3 结构一致性检查通过；
6. §4.2.3 编码器确定性检查通过；
7. `requirements.md` §3 要求的可复现记录齐全；
8. 验收报告写入 `output_results/vsr/reports/`。
