# GPU7 性能差异诊断（2026-09-20）

结论：此前由历史 A1 日志算出的“慢约20%”不是同条件性能回归结论。新测量将主要差异定位到 PyTorch/CUDA 运行栈下的 VAE 执行路径；同环境中原始与迁移实现基本等速。

## 方法

物理 GPU7、相同权重、同一真实输入 tile `[1,3,33,320,640]`、bf16、`OMP_NUM_THREADS=8`。
预热2次后测量6次，取中位数。整 tile 用同步后的墙钟时间，阶段用 CUDA events；不计模型加载、视频读写或验收 dump。
各组顺序执行，不并发运行本任务的 GPU 推理。主机/GPU仍为共享资源，因此小于1%的差异不解释为确定性开销。

## 实测（秒）

| 实现 / 环境 | torch / CUDA | diffusers | 整 tile | VAE编码 | DiT | VAE解码 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| reference_swiftvr | 2.10.0+cu126 | 0.36.0 | 1.3112 | 0.2623 | 0.1214 | 0.9210 |
| reference_torch210_diffusers037 | 2.10.0+cu126 | 0.37.0 | 1.3135 | 0.2626 | 0.1215 | 0.9227 |
| reference_sglang | 2.9.1+cu128 | 0.37.0 | 3.2586 | 0.8116 | 0.1397 | 2.2991 |
| candidate_sglang | 2.9.1+cu128 | 0.37.0 | 3.2773 | 0.8143 | 0.1399 | 2.3062 |

同环境迁移实现 / 原始实现 = **1.0057×**，相差约0.6%。
保持 diffusers 0.37、Python 3.11 和原始代码不变，切换 PyTorch/CUDA 运行栈的耗时比为 **2.481×**。
相对原始环境的额外时间中，约 **99.0%** 位于 VAE 编码和解码。
两套运行栈报告的 cuDNN 数字版本都是 `91002`；不能把差异简单解释为 cuDNN 版本号不同。

这证明主要因素不是迁移代码，也不是 diffusers 0.36→0.37；尚未单独区分 torch 变更、CUDA库组合和具体算子/算法选择。
单 tile 的约2.5倍耗时不能直接外推为4K整视频端到端比例。历史两端运行负载不同，不能用“20%”定量解释本次定位到的环境差异。

## 复现

脚本：`docs_always/add_new_mode/add_vsr/scripts/benchmark_window.py`。
均在仓库根目录执行，设置 `CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8`：

- 原环境：`/mnt/shanhai-ai/envs/conda/envs/swiftvr/bin/python <脚本> --implementation reference --report <JSON>`。
- SGLang环境原始代码：`output_results/vsr/migration_env/bin/python <脚本> --implementation reference --report <JSON>`。
- SGLang环境迁移代码：同上改 `--implementation candidate`。
- 隔离实验：在SGLang环境原始代码命令前再设置 `PYTHONPATH="$PWD/output_results/vsr/torch210-cu126"`，保留 diffusers 0.37，只使用既有torch2.10/cu126依赖目录。不修改已验收环境或生产代码。

输出、6次原始计时和汇总：`output_results/vsr/performance_20260920/`。
第一次 SGLang 原始代码测量为3.295秒/tile，随后附加 profiler 收尾耗时异常，已停止该诊断进程。
表中采用关闭 profiler 后重新完成的3.259秒/tile。Profiler现在是显式 `--profile` 可选项，不影响正常计时。

本次是诊断，没有更改生产推理实现、权重或迁移验收结论。尚未部署运行栈切换或完成新的全视频性能/精度验收。
