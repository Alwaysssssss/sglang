# VividVR 加速统计文档修改计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `docs_xzh/docs_analysis/analysis.md` 改写为适配当前 VividVR 加速能力和后续正式实验的统一统计模板。

**架构：** 文档以固定实验口径为前提，用 R0–R9 十组关键方案替代逐开关全排列；结果同时记录延迟与 GPU·秒资源效率，并使用 VividVR 原生长视频 Stage 和统一的 Denoising 明细。所有分辨率差异从表头移除，后续其他输入规格以独立附录复用同一模板。

**技术栈：** Markdown、VividVR runtime report、`stage_metrics_ms`

---

### 任务 1：重写加速统计模板

**文件：**
- 修改：`docs_xzh/docs_analysis/analysis.md`

- [x] **步骤 1：替换实验方案定义**

  写入 R0–R9 十组固定方案，覆盖 SDPA/FA、`torch.compile`、SP=2、SP=4、CFG=2×SP=2、算子融合、Cache-DiT、TeaCache 和通用量化，并明确每组增益对照。

- [x] **步骤 2：替换总体结果与计算口径**

  使用 `model_inference_runtime_seconds` 计算累计加速比和模块增量加速比，使用 GPU 数量乘模型推理耗时计算 GPU·秒，同时记录质量和单卡峰值显存。

- [x] **步骤 3：替换 Stage 与 Denoising 表**

  使用当前长视频 runtime 的八个 `VividVR*Stage`，增加未归类开销；将两套分辨率专用 DiT 表合并为一套 Denoising 核心明细。

- [x] **步骤 4：收敛固定口径、环境和结论表**

  保留 130 帧、20 step、seed 42 等固定条件，记录 requested/effective backend、compile、并行拓扑、缓存和量化配置，删除示例模型遗留字段。

### 任务 2：验证文档结构

**文件：**
- 验证：`docs_xzh/docs_analysis/analysis.md`

- [x] **步骤 1：检查方案完整性**

  运行：

  ```bash
  rg -n '^\| R([0-9]) ' docs_xzh/docs_analysis/analysis.md
  ```

  预期：实验方案表包含且仅包含 R0–R9 十个方案编号。

- [x] **步骤 2：检查 VividVR Stage 完整性与旧标题清理**

  运行：

  ```bash
  rg -n 'VividVR(InputValidation|PromptPreparation|TemporalWindowPlanning|LongClipPreparation|TimestepPreparation|MultiClipDenoising|MultiClipDecodeTrim|TemporalStitchPostprocess)Stage|1080 Stage|720 Stage|1080 DiT|720 DiT' docs_xzh/docs_analysis/analysis.md
  ```

  预期：八个 VividVR Stage 均存在，旧的分辨率专用 Stage/DiT 标题不存在。

- [x] **步骤 3：检查 Markdown 表格列数**

  使用仓库默认 Python 解析每个 Markdown 表格，确认同一表格内每一行的 `|` 分隔列数一致；预期命令退出码为 0。

- [x] **步骤 4：检查实际差异**

  运行：

  ```bash
  git status --short docs_xzh/docs_analysis/analysis.md docs_xzh/hand_over/2026-07-15-vividvr-acceleration-analysis-plan.md
  ```

  预期：只包含本次统计模板和实施计划相关文档；两个文件当前都属于工作区新增文档。

### 任务 3：生成纯表格版统计模板

**文件：**
- 创建：`docs_xzh/docs_analysis/analysis_tables.md`
- 保留：`docs_xzh/docs_analysis/analysis.md`

- [x] **步骤 1：创建纯表格版文档**

  保留主标题、分节标题和九张 Markdown 表格；将计算公式与必要填写约束收进表格单元格，不保留正文段落、代码块或项目符号。

- [x] **步骤 2：验证纯表格约束**

  使用仓库默认 Python 检查：R0–R9 完整、八个 VividVR Stage 完整、所有 Markdown 表格列数一致，并且除空行、标题和表格行外不存在其他内容。

- [x] **步骤 3：确认原文档未被覆盖**

  比较创建前记录的 `analysis.md` SHA-256，确认生成纯表格版后哈希保持不变。

### 任务 4：补充双卡和四卡综合最快方案

**文件：**
- 修改：`docs_xzh/docs_analysis/analysis.md`
- 修改：`docs_xzh/docs_analysis/analysis_tables.md`

- [x] **步骤 1：定义 R99 与 R100**

  R99 使用 R3 双卡 SP=2 基础拓扑，R100 使用 R4/R5 实测更快的四卡基础拓扑；两者只叠加已经验证兼容、质量达标且端到端有正收益的算子融合、量化和缓存配置。

- [x] **步骤 2：补齐全部结果表**

  在总体结果、Stage、Denoising、运行时快照和模块结论表中加入 R99/R100；缓存配置固定为 Cache-DiT、TeaCache、关闭三选一，不默认叠加两个缓存方案。

- [x] **步骤 3：验证两份文档**

  使用仓库默认 Python 检查两份文档均包含 R0–R9、R99、R100，八个 VividVR Stage 完整，所有 Markdown 表格列数一致；额外确认纯表格版除标题、空行和表格行外不存在其他内容。
