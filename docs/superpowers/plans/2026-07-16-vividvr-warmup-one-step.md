# Vivid-VR benchmark 单步 warmup 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 compile 方案 warmup 请求固定为 1 step，同时保持 formal 请求为 20 step。

**架构：** 为 `build_request_payload` 增加显式 `RunRole` 输入，并在 payload 构造点选择推理步数。`FlowCutRequestExecutor` 将已有 role 原样传入，runner 调度语义不变。

**技术栈：** Python 3.10、pytest、FlowCut Vivid-VR benchmark runner、tmux

---

### 任务 1：角色化请求步数

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] **步骤 1：编写失败的测试**

在请求 payload 测试中显式传入 `RunRole.FORMAL` 并断言 20 step；新增 warmup payload 测试，传入 `RunRole.WARMUP` 并断言 1 step。

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
/home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py -q -k 'request_payload'
```

预期：新增 `role` 参数尚未实现，测试因调用签名或 warmup step 仍为 20 而失败。

- [ ] **步骤 3：编写最少实现代码**

为 `build_request_payload(..., role: RunRole, ...)` 增加角色参数，将 `num_inference_steps` 设置为 `1 if role is RunRole.WARMUP else 20`；executor 调用时传入现有 `role`。

- [ ] **步骤 4：运行测试验证通过**

先重跑 request payload 测试，再运行完整文件：

```bash
/home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py -q
```

预期：全部通过。

- [ ] **步骤 5：提交实现**

只提交规格、计划、runner 与对应测试，提交信息为：

```text
perf(vividvr): reduce benchmark warmup to one step
```

### 任务 2：完整 tmux 验收

**文件：**
- 读取：`docs_xzh/run_command/mock_test.md`
- 产物：`Vivid_Acceptance/acceleration_benchmark/<batch-id>/`

- [ ] **步骤 1：执行 dry-run 与运行时预检**

按脚本 CLI 和既定 GPU 空闲口径检查方案矩阵、依赖、端口与 GPU。

- [ ] **步骤 2：在 tmux 中启动完整批次**

使用清晰 session 名启动全部已支持方案；日志写入批次目录，并向用户提供只读 attach 命令。

- [ ] **步骤 3：持续监督到终态**

轮询 tmux、summary 和 records。每个 formal 记录完成后汇报 `downloaded.mp4`、formal 指标 JSON 与 perf JSON 路径；故障时检查服务、Moto、callback、caption 日志并在任务范围内修复后继续。

- [ ] **步骤 4：最终核对**

确认 summary 为终态、无遗留 owned tmux session，并汇总所有成功、失败、跳过方案及产物位置。
