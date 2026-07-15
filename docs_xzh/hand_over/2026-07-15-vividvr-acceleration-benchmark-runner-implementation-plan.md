# VividVR 加速实验自动化脚本实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现一个可测试的 Python 入口，自动管理 tmux 服务、顺序运行 VividVR 加速实验，并为每次请求生成覆盖统计表字段的 JSON。

**架构：** 单个公开脚本内使用不可变实验注册表、纯函数指标层和有边界的生命周期类。单元测试覆盖所有无 GPU 行为；正式 `run-all` 才进入 tmux、启动服务和运行重型推理。

**技术栈：** Python 3.10、argparse、dataclasses、httpx、boto3、NVML/nvidia-smi、tmux、pytest。

---

### 任务 1：实验注册表与服务命令

**文件：**
- 创建：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] **步骤 1：编写注册表和命令映射失败测试**

  测试导入 `SCHEMES`、`SchemeStatus` 和 `build_service_command`，断言顺序为 `R0..R9,R99,R100`，R7/R8/R9 不可执行，并逐组验证 backend、GPU、SP、CFG、compile、fusion 参数。

```python
def test_scheme_registry_has_fixed_order_and_capabilities():
    assert list(SCHEMES) == [
        "R0", "R1", "R2", "R3", "R4", "R5", "R6",
        "R7", "R8", "R9", "R99", "R100",
    ]
    assert {key for key, value in SCHEMES.items() if not value.executable} == {
        "R7", "R8", "R9",
    }

def test_r100_command_enables_cfg_sp_compile_and_modulation(tmp_path):
    command = build_service_command(SCHEMES["R100"], make_config(tmp_path))
    assert "--enable-cfg-parallel" in command
    assert command[command.index("--sp-degree") + 1] == "2"
    assert "--enable-torch-compile" in command
    assert "--enable-cogvideox-modulation-fusion" in command
    assert "--enable-cogvideox-qkv-fusion" not in command
```

- [ ] **步骤 2：运行测试验证正确失败**

  运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
```

  预期：测试收集失败，原因是新工具模块不存在。

- [ ] **步骤 3：实现不可变注册表和命令生成最小代码**

  定义 `SchemeStatus`、不可变的 `Scheme`、`BenchmarkConfig`、按固定顺序保存的
  `SCHEMES`，以及返回完整 argv 的 `build_service_command(scheme, config)`。
  注册表逐项写明 R0—R9、R99、R100 的 GPU 数、backend、并行拓扑、compile、
  modulation fusion、control 和 unsupported reason，不从表格文本动态推导。

  `build_service_command` 对 unsupported 方案抛出 `BenchmarkConfigError`，对所有方案显式生成 topology guard，避免依赖 `auto` 默认行为。

- [ ] **步骤 4：运行注册表测试验证通过**

  运行任务 1 步骤 2 的命令；预期相关测试全部通过。

- [ ] **步骤 5：提交任务 1**

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
git commit -m "feat(vividvr): register acceleration benchmark schemes"
```

### 任务 2：Perf 指标和 JSON 契约

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] **步骤 1：编写指标计算失败测试**

  使用内存中的真实形态 perf fixture，断言八个 stage、未归类开销、denoise mean、排除 step 0 的 median、GPU·秒、R0 speedup，以及 effective config 不匹配错误。

```python
def test_summarize_perf_computes_table_metrics():
    summary = summarize_perf(make_perf_fixture())
    assert summary.model_inference_runtime_seconds == 10.0
    assert summary.denoising_runtime_seconds == 8.0
    assert summary.unclassified_seconds == pytest.approx(1.2)
    assert summary.steady_step_median_seconds == pytest.approx(0.4)

def test_validate_effective_config_rejects_wrong_sp_backend():
    with pytest.raises(BenchmarkDataError, match="effective backend"):
        validate_effective_config(SCHEMES["R3"], make_perf_fixture(backend="fa"))
```

- [ ] **步骤 2：运行定向测试验证失败**

  运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -k 'summarize or effective or derived or unsupported_record'
```

  预期：因指标函数尚未定义而失败。

- [ ] **步骤 3：实现纯指标函数和记录构造器**

  增加固定的八个 `VIVIDVR_STAGE_NAMES`，以及 `summarize_perf`、
  `validate_effective_config`、`compute_derived_metrics`、
  `build_unsupported_record` 和 `atomic_write_json`。这些函数只接收显式输入，
  不读取全局运行状态，便于用真实形态的内存 fixture 完整覆盖。

  所有未知值使用 `null + reason`；不产生字符串化的秒、GiB 或加速比。R100 只从质量通过的 R4/R5 中选择更快 control。

- [ ] **步骤 4：运行指标测试验证通过**

  运行任务 2 步骤 2 的命令；预期相关测试全部通过。

- [ ] **步骤 5：提交任务 2**

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
git commit -m "feat(vividvr): build acceleration analysis records"
```

### 任务 3：GPU 采样、tmux ownership 与预检

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] **步骤 1：编写基础设施失败测试**

  注入 command runner 和 sample provider，验证逐卡峰值、NVML fallback、owned session 清理、未知端口/GPU 占用拒绝及 dry-run 零副作用。

```python
def test_gpu_sampler_aggregates_per_device_peaks():
    sampler = GpuMemorySampler(
        [0, 1],
        sample_provider=iter(
            [
                {0: 1000, 1: 900},
                {0: 1100, 1: 1200},
            ]
        ),
    )
    sampler.sample_once()
    sampler.sample_once()
    result = sampler.result()
    assert result["max_single_gpu_peak_mib"] == 1200

def test_tmux_manager_only_kills_owned_sessions(tmp_path):
    manager = TmuxManager(batch_id="batch", ownership_dir=tmp_path, run_command=fake)
    manager.stop("vividvr_accel_batch_R0_service")
    assert fake.calls == []
```

- [ ] **步骤 2：运行定向测试验证失败**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -k 'gpu_sampler or tmux or preflight or dry_run'
```

  预期：基础设施类型尚未定义而失败。

- [ ] **步骤 3：实现可注入基础设施边界**

  实现 `GpuMemorySampler.start/stop/result/sample_once`、
  `TmuxManager.start/stop/cleanup_owned` 和
  `run_preflight(config, check_runtime_resources=...)`。command runner 与 GPU sample
  provider 都通过构造参数注入；生产默认实现才访问 tmux、NVML 或 nvidia-smi。

  默认 GPU provider 使用仓库内官方 NVML wrapper，失败时调用 `nvidia-smi`。ownership 文件必须先成功写入，`stop` 才能 kill 对应 session。

- [ ] **步骤 4：运行基础设施测试验证通过**

  运行任务 3 步骤 2 的命令；预期相关测试全部通过。

- [ ] **步骤 5：提交任务 3**

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
git commit -m "feat(vividvr): manage benchmark runtime resources"
```

### 任务 4：请求执行、失败续跑和恢复

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] **步骤 1：编写编排失败测试**

  使用 fake service manager/client/downloader/comparer，验证所有可执行方案 warmup→formal 顺序、warmup 失败跳过 formal、方案失败继续、清理失败终止，以及不完整恢复重新 warmup。

```python
def test_runner_executes_warmup_then_formal_for_each_executable_scheme(tmp_path):
    runner = make_runner(tmp_path)
    result = runner.run([SCHEMES["R0"], SCHEMES["R1"]])
    assert result.request_order == [
        ("R0", "warmup"), ("R0", "formal"),
        ("R1", "warmup"), ("R1", "formal"),
    ]

def test_resume_reruns_warmup_when_only_previous_warmup_succeeded(tmp_path):
    write_partial_manifest(tmp_path, scheme="R2", warmup="succeeded")
    runner = make_runner(tmp_path, resume=True)
    runner.run([SCHEMES["R2"]])
    assert runner.request_order == [("R2", "warmup"), ("R2", "formal")]
```

- [ ] **步骤 2：运行编排测试验证失败**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -k 'runner or resume or cleanup_failure'
```

  预期：编排器尚未定义而失败。

- [ ] **步骤 3：实现请求和批次编排**

  实现 `BenchmarkRunner.run/run_scheme/run_request`、`build_request_payload`、
  `download_result`、`run_compare` 和 `compute_config_fingerprint`。runner 的服务管理、
  HTTP 请求、下载、质量比较和时钟均通过构造参数注入；生产 CLI 组装真实实现。

  请求复用 `submit_flowcut_task_with_retry` 和 `poll_accepted_task`。formal compare 退出 1 但产出有效 compare JSON 时记录 `quality_failed`，不是丢失性能数据。每次请求结束立刻原子写 analysis JSON 和 batch summary。

- [ ] **步骤 4：运行编排测试验证通过**

  运行任务 4 步骤 2 的命令；预期相关测试全部通过。

- [ ] **步骤 5：提交任务 4**

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
git commit -m "feat(vividvr): orchestrate acceleration benchmark batch"
```

### 任务 5：CLI、文档和统计契约同步

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`
- 创建：`docs_xzh/run_command/vividvr_acceleration_benchmark.md`
- 修改：`docs_xzh/docs_analysis/analysis_tables.md`

- [ ] **步骤 1：编写 CLI 失败测试**

  验证 `dry-run` 输出 12 个方案且不调用 tmux，`run-one` 校验 scheme，tmux 外的 `run-all` 生成 detached session 命令，内部 `_run-batch` 才执行生命周期。

```python
def test_dry_run_prints_plan_without_runtime_side_effects(capsys, tmp_path):
    assert main(["dry-run", *minimal_path_args(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert [item["scheme_id"] for item in output["schemes"]] == list(SCHEMES)
```

- [ ] **步骤 2：运行 CLI 测试验证失败**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -k 'cli or dry_run or launcher'
```

  预期：CLI 子命令尚未实现而失败。

- [ ] **步骤 3：实现 CLI 并编写运行文档**

  `parse_args(argv)` 支持 `run-all`、`run-one`、`dry-run`、隐藏 `_run-batch`，并提供 `--resume`、路径、GPU 和超时覆盖。文档给出以下正式入口：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py dry-run

PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py run-all
```

  `analysis_tables.md` 只修正已验证存在的资源路径、caption 唯一 prompt 来源、能力状态和 R99/R100 固定组合，不填写结果单元格。

- [ ] **步骤 4：运行全部新单测和 CLI dry-run**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py

PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py dry-run
```

  预期：测试 0 failures；dry-run 退出码 0，JSON 列出 12 个方案且不新增 tmux session。

- [ ] **步骤 5：提交任务 5**

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  docs_xzh/run_command/vividvr_acceleration_benchmark.md \
  docs_xzh/docs_analysis/analysis_tables.md
git commit -m "docs(vividvr): document acceleration benchmark runner"
```

### 任务 6：最终回归和需求核对

**文件：**
- 验证：上述全部实现文件

- [ ] **步骤 1：运行新工具完整单测**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
```

- [ ] **步骤 2：运行既有 FlowCut 工具回归**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py
```

- [ ] **步骤 3：运行静态和文档检查**

```bash
/home/zhiheng/sglang/.venv/bin/python -m py_compile \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py
git diff --check HEAD~5..HEAD
```

- [ ] **步骤 4：核对未启动重型任务和工作树状态**

```bash
tmux list-sessions -F '#{session_name}' 2>/dev/null | \
  rg '^vividvr_accel_' || true
git status --short
```

  预期：没有本实现创建的推理 session；工作树无未提交改动。

- [ ] **步骤 5：按 finishing-a-development-branch 流程交付分支**

  汇报测试证据、提交、工作树路径和正式 `run-all` 尚未执行的边界，再由用户选择本地合并、PR、保留分支或丢弃。
