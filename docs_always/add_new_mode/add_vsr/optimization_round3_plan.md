# 第三轮单卡优化计划与最新质量门限

用户最新指定：SSIM≥0.989、MSE≤36、MAE≤6。用于后续编码前uint8 RGB帧及编码后视频与原生参考的对比；沿用现有SSIM实现，MSE/MAE按0–255 RGB像素计算。三个条件需同时满足，按逐帧检查，不用全片平均掩盖失败帧。帧数、分辨率、fps、帧顺序和非有限值检查不放宽。中间张量误差继续单独记录，不能套用像素单位的门限。

新门限仅约束后续实验，不改写前两轮历史报告。`run_optimization_round1.py`、`validate_optimized_native.py`默认使用新门限，支持`--min-ssim/--max-mse/--max-mae`显式设置；前者加`--legacy-frame-gates`可复现旧九配置门限，后者用显式数值复现旧native门限。

## 当前瓶颈

第二轮GPU7数据：单tile0.90245秒，编码0.20805秒（23.1%）、DiT0.11146秒（12.4%）、解码0.57642秒（63.9%），其余约0.7%。4K A1稳态中位数126.321秒。以下均为待测方案，不是已获得的加速。

## 顺序

1. **编译VAE encoder。** 复用decoder编译经验，保留外层时序循环及因果缓存；处理首帧/后续块专门化。若编码阶段快30%，tile整体理论仅减少约6.9%；编码快2倍时整体约减少11.5%。不是整体提升30%或2倍。
2. **进一步搜索decoder/encoder的编译配置。** 从`max-autotune-no-cudagraphs`开始，和当前default同卡交错对照；cuDNN benchmark与Inductor autotune是不同层次的搜索，不假设重复开启就有收益。只比较完成编译后的稳态。CUDA Graphs放在确认CPU发射空隙后考虑，当前可变缓存必须验证复用及输入修改问题。
3. **空间tile批处理：用户已明确排除，本轮不执行。** 原候选为B=2，再考虑B=4。 一张卡里批量执行，不是多卡并行。沿batch维堆叠相同形状的tile，保持每个样本独立因果缓存，按原顺序拆回融合。比较tiles/s和每视频耗时；批次延迟不应直接对比单tile延迟。监控峰值显存和workspace，不能按当前13.6GiB线性推断B=4必然可用。
4. **GPU融合/颜色处理和有界异步拷贝。** 当前每tile回传CPU并融合。先剖析读入、D2H、空间/时间融合、颜色统计与编码；测试GPU空间融合后按块回传，或pinned双缓冲+独立copy stream+events。4K整块float32累积会明显增加显存，需分块并维持队列有界。单纯加non_blocking不等于计算/拷贝重叠。
5. **利用新精度预算测试局部FP16。** 先仅VAE，再按模块比较BF16/FP16；保持必要的统计/累计为FP32，并检查溢出、NaN和长序列累计误差。A100上BF16改FP16不保证更快，要靠卷积选核和实测决定。
6. **DiT编译/attention作为次优先级。** 当前仅占12.4%；即便DiT快2倍，tile总耗时也只减少约6.2%。先做低成本编译试验；量化和attention替换须证明端到端收益后再投入。

更大tile/更小overlap可能减少重复计算，但改变上下文、边界和融合语义。只能列作明确改变算法配置的独立近似实验，不能与保持原几何的实现优化混算；宽松像素门限也不能替代接缝和时间稳定性检查。

不优先做TeaCache/多步采样缓存：每tile只有一次DiT前向。当前设备是A100，不把Hopper FP8硬件路径当作可直接套用的收益来源。

## 测试安排

GPU7固定做warmup后的A/B/B/A性能比较，GPU6做编译试验与质量验证；若共用GPU7，先结束该卡质量任务。按用户最终更正跳过方案三；测试encoder编译、编译配置、GPU融合/后处理及异步拷贝、局部FP16与DiT编译；有效候选再做九配置、native入口、短片/长片与重复请求验证。所有近似候选直接对原生参考评估，避免逐轮对前一版比较造成误差累积。

## 官方依据

- [PyTorch 2.10 torch.compile](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)：编译模式、autotune、CUDA Graph限制。
- [PyTorch pinned memory和non_blocking指南](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)：异步搬运的条件和同步要求。具体实现仍以本地torch2.10可用API为准。
