# Vivid-VR R0 基线 VAE SP 纯净加速测试实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `executing-plans` 逐任务实现此计划。用户明确要求不使用子代理。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 以历史单卡 SDPA eager `R0` 为只读基线，分别运行 SP2、SP4 两组仅开启 VAE tiled encode/decode 空间并行的 `130f / 20 step` 服务测试，并报告模型与端到端加速比。

**架构：** 在现有 acceleration benchmark runner 中增加两条不进入默认矩阵的隔离 treatment。两条 treatment 固定请求 SDPA eager，关闭 compile、modulation fusion 和 CFG parallel，只启用 SP 拓扑、`--vae-sp` 与 `--vae-encode-sp`；历史 R0 record 通过 SHA-256 和 mtime 快照只读加载，派生指标不套用旧的同拓扑 encode gate，只计算相对 R0 的阶段、模型和端到端加速比。

**技术栈：** Python 3.10、pytest、SGLang serve、tmux、VividVR FlowCut 服务验收、CUDA SP2/SP4。

---

## 文件结构

- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`——注册两条纯净 treatment，校验历史 R0，并写出 R0 派生指标。
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`——锁定配置纯净性、R0 只读加载和派生计算。
- 创建：`docs_xzh/distribute/vividvr_r0_vae_sp_clean_benchmark_20260717.md`——记录正式运行产物、用时、加速比、质量和环境风险。

### 任务 1：注册纯净 SP2/SP4 treatment

- [x] **步骤 1：编写失败的 registry/command 测试**

新增 `test_r0_vae_sp_clean_treatments_are_isolated_and_have_no_extra_acceleration`，断言 `R0_VAE_SP2`、`R0_VAE_SP4` 只存在于隔离 registry，且字段分别为：

```python
assert scheme.backend == "sdpa"
assert scheme.compile_enabled is False
assert scheme.modulation_fusion is False
assert scheme.parallel_mode == "sp"
assert scheme.cfg_parallel is False
assert scheme.vae_sp is True
assert scheme.vae_encode_sp is True
assert scheme.controls == ("R0",)
```

同时检查 service command 包含 `--vae-sp --vae-encode-sp`，不包含 compile、fusion 或 CFG parallel 开关。

- [x] **步骤 2：运行测试验证红灯**

运行：

```bash
PYTHONPATH=python .venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -k r0_vae_sp_clean_treatments
```

预期：FAIL，原因是两条 treatment/registry 尚未定义。

- [x] **步骤 3：实现最小 registry**

新增隔离字典 `R0_VAE_SP_TREATMENTS`，注册 SP2/SP4，并合入 `ALL_SCHEMES`，不修改 `SCHEMES`。

- [x] **步骤 4：运行测试验证绿灯**

重复步骤 2，预期 PASS。

### 任务 2：支持历史 R0 与纯加速比派生

- [x] **步骤 1：编写失败的历史 R0/派生指标测试**

构造 R0 formal fixture，断言 `load_historical_controls` 接受单卡、无 VAE SP 的 R0；断言纯净 treatment 的派生结果包含：

```python
assert derived["model_inference_speedup"] == pytest.approx(r0_model / treatment_model)
assert derived["total_runtime_speedup"] == pytest.approx(r0_total / treatment_total)
assert derived["long_clip_preparation_speedup"] == pytest.approx(r0_prep / treatment_prep)
assert derived["denoise_speedup"] == pytest.approx(r0_denoise / treatment_denoise)
assert derived["decode_trim_speedup"] == pytest.approx(r0_decode / treatment_decode)
```

- [x] **步骤 2：运行测试验证红灯**

运行对应 `-k 'historical_r0 or r0_vae_sp_derived'`，预期因旧 validator 强制 control 开启 `vae_sp` 或缺少纯 R0 派生字段而 FAIL。

- [x] **步骤 3：实现最小 R0 分支**

历史 control validator 根据 control scheme 校验 `vae_sp`；R0 必须明确不启用 encode/decode VAE SP。`compute_vae_encode_sp_derived_metrics` 在 control 为 R0 时返回阶段、模型、总耗时、GPU·秒和质量差值，不生成旧同拓扑性能 gate。

- [x] **步骤 4：运行单测及 runner 全文件回归**

```bash
PYTHONPATH=python .venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
```

预期：全部 PASS。

- [x] **步骤 5：dry-run 两条正式命令**

两条命令均指定历史 R0 batch、GPU 0–1 或 0–3；检查 eager 只请求 formal、backend 为 SDPA、无其他加速开关。

### 任务 3：串行正式服务测试与验收

- [ ] **步骤 1：记录历史 R0 指纹**

对 `Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/R0_formal.json` 记录 SHA-256 与 mtime_ns。

- [ ] **步骤 2：在 tmux 启动 SP2**

使用 `run-one --scheme R0_VAE_SP2 --control-batch-dir ... --gpu-ids 0,1 --allow-idle-gpu-processes`；eager 方案只能执行一次 formal，不 warmup。

- [ ] **步骤 3：检查 SP2 record**

状态必须为 `succeeded` 或 `quality_failed`；runtime 必须显示 `requested_backend=sdpa`、`effective_backend=sdpa_sp`、compile/fusion/CFG 均 false、VAE encode/decode SP effective 均 true。

- [ ] **步骤 4：SP2 完全退出后在 tmux 启动 SP4**

替换 scheme 为 `R0_VAE_SP4`、GPU 为 `0,1,2,3`，其余口径相同。

- [ ] **步骤 5：检查 SP4 record 并验证 R0 未改**

重复 SP2 runtime 检查；重新计算 R0 SHA-256/mtime_ns，与运行前逐字节一致。

- [ ] **步骤 6：独立复算并记录结论**

从三份 formal JSON 独立计算：

```python
speedup = r0_seconds / treatment_seconds
gpu_seconds = treatment_gpu_count * treatment_total_seconds
```

记录 encode、decode、denoise、model、total、质量指标、callback/MinIO/video/perf 路径和 GPU 环境风险。

- [ ] **步骤 7：最终验证**

运行 benchmark 单测、两份 record schema/runtime 断言、派生值 `1e-9` 一致性检查，以及 `git diff --check`。只有取得新鲜输出后才报告完成。
