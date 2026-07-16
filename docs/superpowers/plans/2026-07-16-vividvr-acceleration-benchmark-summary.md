# VividVR 加速测试耗时总结实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 生成只汇报正式实验用时和最快配置的 VividVR 加速测试 Markdown 总结。

**架构：** 直接读取既有批次的正式 JSON 作为唯一数据源，将 R0–R6、R99、R100 的耗时格式化为一个紧凑表格。文档不修改测试数据，也不重复质量、显存或产物信息。

**技术栈：** Markdown、Python 3.10、VividVR benchmark JSON

---

### 任务 1：提取并核对正式耗时

**文件：**
- 读取：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/*_formal.json`

- [ ] **步骤 1：读取九个可执行方案的正式记录**

运行仓库 `.venv` 中的 Python，依次读取 R0、R1、R2、R3、R4、R5、R6、R99、R100，并输出：

```python
for scheme in schemes:
    record = json.loads((records / f"{scheme}_formal.json").read_text())
    print(
        scheme,
        record["timings"]["total_runtime_seconds"],
        record["timings"]["model_inference_runtime_seconds"],
        record["timings"]["denoising_runtime_seconds"],
        record["timings"]["mean_step_seconds"],
    )
```

预期：九个方案均能读取到非空正数，正式步数均为 20。

- [ ] **步骤 2：计算相对 R0 模型加速比和最快配置**

```python
speedup = r0_model_seconds / current_model_seconds
fastest = min(records_by_gpu_count, key=lambda item: item["model_seconds"])
```

预期：单卡 R2、双卡 R99、四卡 R100 分别为对应卡数的最低模型推理耗时。

### 任务 2：生成耗时总结文档

**文件：**
- 创建：`docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`

- [ ] **步骤 1：写入测试口径**

写明 130 帧、formal 20 step、compile warmup 1 step，以及三个耗时字段定义。

- [ ] **步骤 2：写入总体耗时表**

表格列固定为：

```markdown
| 方案 | 关键配置 | 总耗时（s） | 模型推理耗时（s） | Denoise（s） | 平均 Step（s） | 相对 R0 模型加速比 |
```

所有耗时保留三位小数，加速比保留两位小数。

- [ ] **步骤 3：写入最快方案表**

仅列出单卡 R2、双卡 R99、四卡 R100 的模型推理耗时和相对 R0 加速比。

### 任务 3：验证并提交

**文件：**
- 验证：`docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`

- [ ] **步骤 1：机械核对文档数值**

用 Python 从 Markdown 表格解析方案行，并与正式 JSON 四舍五入后的总耗时、模型耗时、denoise 和平均 step 比较。

预期：九个方案全部一致。

- [ ] **步骤 2：检查范围和格式**

```bash
rg -n 'SSIM|显存|GPU·秒|通信耗时|quality_failed' docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md
git diff --check
```

预期：第一条命令无匹配，`git diff --check` 无输出。

- [ ] **步骤 3：提交正式总结**

```bash
git add docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md \
  docs/superpowers/plans/2026-07-16-vividvr-acceleration-benchmark-summary.md
git commit -m "docs(vividvr): summarize acceleration benchmark timings"
```

预期：提交只包含实现计划和正式总结文档。

### 任务 4：补充完整 Stage 横向总表

**文件：**
- 修改：`docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`
- 读取：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/R{0,1,2,3,4,5,6,99,100}_formal.json`

- [ ] **步骤 1：从正式 JSON 提取完整 Stage 耗时**

使用仓库 `.venv` 中的 Python 读取每条正式记录的 `timings.stage_seconds`，固定检查以下 8 个键：

```python
stage_names = [
    "VividVRInputValidationStage",
    "VividVRPromptPreparationStage",
    "VividVRTemporalWindowPlanningStage",
    "VividVRLongClipPreparationStage",
    "VividVRTimestepPreparationStage",
    "VividVRMultiClipDenoisingStage",
    "VividVRMultiClipDecodeTrimStage",
    "VividVRTemporalStitchPostprocessStage",
]
```

预期：R0–R6、R99、R100 均包含全部 8 个键，且值为非负数。

- [ ] **步骤 2：写入 Stage 名称映射和横向总表**

在“总体耗时”和“最快方案”之间新增“完整 Stage 耗时”章节。先用两列表格说明简化列名到完整类名的映射，再写入以下横向表头：

```markdown
| 方案 | Input Validation | Prompt Preparation | Window Planning | Long Clip Preparation | Timestep Preparation | Multi-Clip Denoising | Decode/Trim | Stitch/Postprocess |
```

九个正式方案各占一行，所有秒数使用三位小数。

- [ ] **步骤 3：机械核对 Stage 表格**

使用 Python 解析“完整 Stage 耗时”横向表，并逐项比较正式 JSON 中 `stage_seconds` 四舍五入到三位小数后的值。

预期：9 个方案 × 8 个 Stage，共 72 个数值全部一致。

- [ ] **步骤 4：检查格式并提交**

```bash
git diff --check
git add docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md \
  docs/superpowers/plans/2026-07-16-vividvr-acceleration-benchmark-summary.md
git commit -m "docs(vividvr): add complete stage timing table"
```

预期：提交只包含实现计划和正式总结文档的 Stage 耗时增量。
