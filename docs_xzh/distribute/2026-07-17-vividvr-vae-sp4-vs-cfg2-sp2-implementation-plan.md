# Vivid-VR VAE SP4 与 CFG2×SP2 对比 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task by task.

**Goal:** 新增可复现的纯 SP4 + fusion + VAE SP treatment，完成四卡正式推理，并与已验收的 CFG2×SP2 + fusion + VAE SP 结果做公平性能对比。

**Architecture:** 复用现有 acceleration benchmark runner，只在隔离的 `VAE_SP_TREATMENTS` 中新增 `R101_VAE_SP4`。runner 负责服务生命周期、warmup、formal 请求、配置证据、质量与指标采集；最终报告直接读取两份 formal JSON 计算总耗时和阶段耗时差异。

**Tech Stack:** Python、pytest、tmux、PyTorch distributed、SGLang Vivid-VR benchmark runner。

---

### Task 1: 用测试锁定 R101 实验合同

**Files:**
- Modify: `python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

1. 添加测试，要求 `R101_VAE_SP4` 仅存在于 `VAE_SP_TREATMENTS`，并固定四卡、纯 SP、SP degree 4、compile、fusion、VAE SP 和 `R4` 辅助 control。
2. 添加命令测试，要求出现 `--sp-degree 4`、`--ulysses-degree 4`、`--vividvr-parallel-mode sp`、`--vae-sp`、compile 与 fusion 参数，且不出现 CFG parallel 参数。
3. 添加 effective-config 测试，要求 VAE SP world size 为 4。
4. 运行定向 pytest，确认测试先因 treatment 不存在而失败。

### Task 2: 最小实现 R101 treatment

**Files:**
- Modify: `python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`

1. 在 `VAE_SP_TREATMENTS` 中注册 `R101_VAE_SP4`，不修改默认 `SCHEMES`。
2. 运行定向测试和完整 benchmark runner 单测：
   `/home/zhiheng/sglang/.venv/bin/python -m pytest -q python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`
3. 检查 diff，提交 Task 1–2 的代码和设计/计划文档，不纳入已有脏文件。

### Task 3: 在 tmux 中跑四卡正式推理

**Artifacts:**
- Create: `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717/`
- Create: `Vivid_Acceptance/logs/vividvr_vae_sp4_compare_20260717.log`

1. 用 `nvidia-smi` 和进程检查确认 GPU 0–3 没有活跃计算负载。
2. 在 `tmux` session `vividvr_vae_sp4_compare` 中运行：
   `run-one --scheme R101_VAE_SP4 --batch-id vividvr_vae_sp4_fusion_20260717 --gpu-ids 0,1,2,3 --allow-idle-gpu-processes --control-batch-dir .../vividvr_accel_full_warmup1_20260716`。
3. 监控到 warmup、formal、质量评估和清理全部结束；不得把普通阻塞 shell 当成长推理入口。
4. 验证 formal record 中 `sp_world_size=4`、`cfg_parallel_enabled=false`、`vae_sp_effective=true`、`vae_sp_world_size=4`，并确认输出视频和指标文件存在。

### Task 4: 生成公平对比结论并验收

**Files:**
- Create: `docs_xzh/distribute/vividvr_vae_sp4_vs_cfg2_sp2_benchmark_20260717.md`

1. 读取本次 `R101_VAE_SP4_formal.json` 与
   `vividvr_vae_sp_r100_canonicalized_20260716/records/R100_VAE_SP_formal.json`。
2. 计算两者的 total、model inference、denoise、Decode/Trim、VAE tile
   decode/gather/merge 耗时及相对速度比。
3. 写明哪种拓扑更快、快多少、性能差异来自哪些阶段，并记录质量结果与
   实验限制。
4. 运行最终单测、JSON 字段检查和 `git diff --check`；提交并推送本阶段
   相关改动，保留用户原有未提交文件。
