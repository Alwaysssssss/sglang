# 下一步单卡优化顺序

最新实验结果见 [第四轮实测](optimization_round4.md)，前一轮见 [第三轮实测](optimization_round3.md)，门限与方案列表见 [第三轮计划](optimization_round3_plan.md)。用户已更正为跳过方案三，其余执行。下文保留最初的优化路线，已完成项与旧占比以第一、二轮报告为准。

2026-09-20。首轮实验与可选开关已实施，验收结果见 [首轮实测](optimization_round1.md)。decoder编译进展见 [第二轮实测](optimization_round2.md)。其他条目仍为待验证建议。沿用物理GPU7、torch2.10/cu126、bf16、原tile/overlap/颜色/编码语义。

## 基线与优先级依据

`output_results/vsr/runtime210_validation/benchmark.json`：真实33×320×640 tile，预热后中位数1.3119秒。
VAE编码0.2622秒，DiT0.1214秒，VAE解码0.9217秒。VAE约占90.2%，其中解码70.3%。
按其他阶段不变的Amdahl估算：VAE整体快2倍，tile整体约快1.82倍；只把解码加速2倍，整体约快1.54倍；只把DiT加速2倍，整体仅约快1.05倍。以上是条件推算，不是实测承诺。

## 推荐实验

| 顺序 | 实验 | 代码落点/适用理由 | 主要验证事项 |
| --- | --- | --- | --- |
| 1a | cuDNN benchmark算法搜索 | 当前`cudnn.benchmark=False`；固定tile重复处理，适合测试`True` | 按用户要求，性能只比较充分warmup后的稳态速度；算法选择可能改变舍入，不预设逐位一致 |
| 1b | Conv3d的channels_last_3d权重布局 | 优先仅转换VAE Conv3d权重，检查padding/permute是否抵消收益 | 保持bf16；不把整个混合2D/3D模型统一套用2D channels_last；检查布局转换开销 |
| 2 | VAE decoder局部torch.compile，再考虑encoder | 解码占70%；优先固定形状的decoder计算块 | Wan有首帧/后续帧分支、可变Python缓存列表及clear_cache；记录graph break/重编译，不直接编译整个流式控制循环 |
| 3 | 同一时间窗口内的空间tile小批处理，B=2起步 | 当前逐tile串行；可沿batch维组合独立tile | 测tiles/s而非仅批次延迟；每个tile的因果缓存相互独立；按原顺序拆回并融合；避免VAE slicing把批处理又串行化 |
| 4 | 有界异步搬运和CPU后处理流水 | 当前输入`.to(cuda)`，输出逐tile`.to(cpu,float32)`后在CPU融合 | pinned缓冲、独立copy stream和events必须配套；仅non_blocking不保证重叠；保证缓冲生命周期、输出顺序及有限队列 |
| 并行考虑 | 多视频任务复用已加载pipeline | 当前CLI每次加载模型；对短视频的总耗时尤其值得测量 | 分开报告冷启动与热运行；不需要为此引入HTTP服务 |

CUDA Graphs仅在编译/剖析表明CPU发射开销显著时继续测试，不当作GPU卷积计算的通用加速器。
直接替换SGLang原生Wan VAE可作为后续独立实验，但不能凭“原生”名称假设更快；须重跑权重映射、latent归一化及逐阶段验收。

首轮建议只做四组：基线、仅benchmark=True、仅Conv3d布局转换、二者组合，先判断便宜的改动是否有收益。

## 测量与保留条件

1. 微基准关闭视频IO和验收dump，测预热后的分阶段时间、单tile或tiles/s、显存峰值，并记录GPU7其他任务占用。按A/B交错顺序重复，报告中位数和离散范围。
2. 按用户要求，统一warmup后计时，初始化、算法搜索及编译不计入性能对比；原始报告保留warmup记录便于复现。
3. 通过微基准后，测不带dump的完整A1：加载、解码/resize、模型、融合/颜色、编码和总墙钟时间分别记录。原先3分53秒包含验收开销，不能直接当作生产推理基线。
4. 首轮每个候选先比单tile中间张量和帧，再验证B1/C1接缝；仅对有效候选做九配置完整验收。
5. 首选保持精确一致；出现数值差异时按既有requirements中的冻结判据检查并记录，不临时放宽门限。批处理/异步队列同时补测固定配置下短片与长片的内存行为。
6. 每次只启用一项；通过速度、精度、内存三项验证后才组合并设为默认。

不优先做DiT量化、attention后端替换或TeaCache：DiT占比仅9.3%，且当前模型每tile仅一次DiT前向，没有多步去噪循环可供TeaCache复用。
不通过减少overlap、修改tile几何、降低分辨率或换编码器来构造加速，因为这些会改变当前验收条件或算法语义。

## 依据

本地实现：`runtime/vsr/model.py`、`runtime/vsr/blending.py`；安装的diffusers0.37 `autoencoder_kl_wan.py`显示_encode/_decode逐时间块调用encoder/decoder，并维护、清理因果缓存。

- [PyTorch调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)：cuDNN算法搜索、编译及数据搬运调优。
- [PyTorch 2.10 Conv3d布局转换](https://docs.pytorch.org/docs/2.10/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html)：仅转换卷积权重的方式及布局转换成本；文档中的收益不能直接外推为本模型bf16实测值。
- [PyTorch 2.10 torch.compile](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)：编译模式、CUDA Graphs及适用限制。
- [Diffusers推理加速指南](https://huggingface.co/docs/diffusers/main/optimization/fp16)：VAE编译等通用方法；图像模型示例不代表Wan时序VAE可以原样套用。
