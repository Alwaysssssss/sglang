# Vivid-VR benchmark 单步 warmup 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 compile 实验的 warmup 推理步数固定为 1，同时保持 formal 请求为 20。

**架构：** 保留固定 formal payload 构造器，在 `FlowCutRequestExecutor` 已知请求角色的位置对 warmup payload 做一次显式覆盖。测试捕获实际提交给服务的 payload，验证两种角色的 step 数。

**技术栈：** Python、pytest、httpx 测试替身

---

### 任务 1：锁定 warmup 与 formal 请求步数

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`

- [ ] **步骤 1：编写失败的测试**

在 executor 单测中捕获 `submit_flowcut_task_with_retry` 收到的 payload，分别调用 `RunRole.WARMUP` 和 `RunRole.FORMAL`，断言 step 数依次为 `1` 和 `20`。

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
/home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py -k warmup_request_uses_one_step -q
```

预期：FAIL，warmup payload 当前仍为 `20`。

- [ ] **步骤 3：编写最少实现代码**

在 `FlowCutRequestExecutor.__call__` 构造 payload 后增加：

```python
if role is RunRole.WARMUP:
    payload["num_inference_steps"] = 1
```

- [ ] **步骤 4：验证单测和完整相关测试**

运行：

```bash
/home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py -q
git diff --check
```

预期：测试全部通过，差异检查退出码为 0。

- [ ] **步骤 5：提交并推送**

只提交上述规格、计划、实现与测试文件，然后使用普通非强制 push 同步 `sglang_Vivid`。
