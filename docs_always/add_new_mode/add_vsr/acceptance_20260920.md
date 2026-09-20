# 第一阶段迁移验收（2026-09-20）

状态：**第一阶段 M0–M6 完成**。2026-09-20 在物理 GPU7 完成全矩阵和边界配置验证。

最终机器报告：`output_results/vsr/reports/migration_20260920_acceptance.json`（`pass=true`）。
9 组矩阵共 649 帧；另有 5 组细粒度配置、61 个张量/帧块对拍以及原生 pipeline 验证。

## 范围

沿用 `plan.md` 的第一阶段：SGLang pipeline/CLI、diffusers VAE/DiT、流式读取与融合、逐阶段及视频对齐。原生模型替换和加速仍属于第二阶段。

## 环境恢复

旧 `/home/root/uv-envs/sglang-llm-diffusion/bin/python` 在本次会话不存在。
本次解释器为 `$SGLANG_REPO/output_results/vsr/migration_env/bin/python`，通过 `.pth` 复用
`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/lib/python3.11/site-packages`，并在独立 venv 内补齐：
`decord==0.6.0`、`imageio-ffmpeg==0.6.0`、`numpy==2.4.6`、`accelerate==1.13.0`。
没有修改原依赖目录。核心版本为 torch `2.9.1+cu128`、diffusers `0.37.0`。

所有推理设置 `CUDA_VISIBLE_DEVICES=7`、`OMP_NUM_THREADS=8`，进程中的 `cuda:0` 对应物理 GPU7。
卡上原有其他任务约占 12.5 GiB；本次不是性能基准测试。

## 验收方法

- 全矩阵重新运行迁移实现，通过 `dump_candidate` 保存编码前帧。
- 逐块检查 dtype、shape 与每个 uint8 像素，要求与 `dumps/EPS_<case>_sglang` 完全相等。
- 计算新旧 mp4 的 SHA256，要求逐字节相同。这比最终视频的宽松数值门限更严格。
- 相等性检查完成后，再由 `compare_fresh.py` 直接比较本次输出与 swiftvr 基线：先做结构检查，再按各配置冻结门限比较编码前帧，最后按 `0.97/25/2.5` 比较 mp4。最终报告使用本次重新计算的逐帧指标；历史 EPS 指标仅保留为交叉核对。
- 对照基线仓库仍为 `e9e46440e043ca50202c768f81f39d02e271a340`，工作树干净，与历史 manifest 一致。
- 小配置重新运行两端：单窗口、三窗口、非32倍数空间补齐、10帧时间反射补齐、单帧重复。保存全部 tile 张量与 chunk 张量，比较结构、有限值、设备记录、颜色统计及每个元素，要求完全相等。
- 原生 `--via-pipeline` 单窗口输出另与直接路径逐字节比较。

完整日志、实际命令 JSON、权重校验、Git 状态与产物位于 `output_results/vsr/migration_20260920/`。
本次发现旧 `M4_C3.log` 中途结束，因此未把旧 C3 产物计作通过。

## 本次代码修正

- 原生 CLI 将 `--dtype` 传给模型加载配置；stage 校验请求精度与已加载模型一致，避免通过回转低精度权重伪装成高精度模型。
- 每次请求重新解析 tile 默认值，防止沿用前一次请求覆盖值；请求的 `long_edge` 优先于配置中的固定分辨率。
- 显式选择 `WanVSRPipeline` 时配置解析不再要求 checkpoint 路径含 `SwiftVR` 或包含 `model_index.json`。
- 增加迁移实现的 dump 入口，复用基线观察钩子；记录张量设备与有限值。
- 张量比较器拒绝空比较以及 NaN/Inf，避免假通过。

## 已执行的 CPU 检查

- 纯函数对拍：61/61 逐位相同（`migration_ops.log`）。
- 请求参数、精度一致性及无效张量回归：6/6 通过（`migration_request_tests.log`）。

## 运行方式

```bash
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8
PY="$PWD/output_results/vsr/migration_env/bin/python"
"$PY" docs_always/add_new_mode/add_vsr/scripts/run_acceptance.py
"$PY" docs_always/add_new_mode/add_vsr/scripts/run_edges.py
"$PY" docs_always/add_new_mode/add_vsr/scripts/compare_fresh.py
"$PY" docs_always/add_new_mode/add_vsr/scripts/summarize_acceptance.py
```

脚本当前使用本机文档中的输入和权重路径。每个实际子进程的完整参数另保存在产物目录的 `*_command.json`。

## 最终结果

以下指标为本次输出与 swiftvr 基线的重新计算结果；编码前帧另外按每个配置的冻结门限检查，全部通过。
同环境基线对照更严格：九组编码前像素均完全一致，mp4 SHA256 均相同。

| 配置 | 帧数 | mp4 SSIM 最小 | mp4 MSE 最大 | mp4 MAE 最大 | 失败帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 53 | 0.988793 | 2.7160 | 1.1283 | 0 |
| A2 | 53 | 0.988794 | 2.7350 | 1.1338 | 0 |
| A3 | 53 | 0.988896 | 2.7947 | 1.1349 | 0 |
| B1 | 54 | 0.989416 | 4.3209 | 1.5083 | 0 |
| B2 | 54 | 0.989424 | 3.8498 | 1.3856 | 0 |
| B3 | 54 | 0.989673 | 5.6770 | 1.6598 | 0 |
| C1 | 64 | 0.989153 | 4.4745 | 1.4878 | 0 |
| C2 | 200 | 0.988732 | 4.7308 | 1.5503 | 0 |
| C3 | 64 | 0.988583 | 3.0955 | 1.2332 | 0 |

结构检查全部通过：分辨率、帧数、fps 与帧序一致，未丢帧、截断或缩放候选视频。
细粒度 single / three / pad / short / one 五组的所有采样张量、有限值、设备记录、颜色统计均完全一致。
原生调度器输出 `native.mp4` 与 `single_candidate.mp4` 逐字节一致。

编码器确定性复核：`dumps/enc_sglang.mp4`、`dumps/enc_swiftvr.mp4` 与 `M0_reference_swiftvr_a.mp4`
的 MD5 均为 `4e6576d4c594e8eadd3ef8c8a453281a`。

主要交付视频：`output_results/vsr/migration_20260920/A1.mp4`（53 帧，H3840×W2160，25 fps，global）。

本次检查：61/61 纯函数对拍、6/6 回归测试、修改文件的 ruff 检查及 `git diff --check` 全通过。
模型权重 SHA256、包含未提交代码的源码 SHA256、Git 状态、环境版本与 GPU 快照见 `migration_20260920/provenance.json`。

本结论只覆盖已约定第一阶段。原生 Wan VAE/DiT 替换、加速、长期内存验收、目录批处理、音频和 HTTP 服务仍按计划后置。
