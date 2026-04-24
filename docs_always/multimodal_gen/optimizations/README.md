# `multimodal_gen` 优化技术系列文档

本目录是 [`../07_disaggregation_and_optimization.md`](../07_disaggregation_and_optimization.md) 的**深化版**：把 `python/sglang/multimodal_gen` 涉及的所有性能优化、显存优化、精度优化技术拆成**九篇独立的专题文档**，每篇都聚焦一个模块、给出源码引用、调用链、默认值、环境变量、参数开关与适配矩阵。

## 阅读顺序

建议从 `00_overview.md` 开始，后续按编号顺序浏览；如果只关心具体主题，可直接跳到对应章。

| # | 文档 | 关键词 |
|---|------|--------|
| 00 | [`00_overview.md`](./00_overview.md) | 各优化维度在请求生命周期中的位置、参数入口、阅读建议 |
| 01 | [`01_parallelism.md`](./01_parallelism.md) | TP / CFG Parallel / USP (Ulysses + Ring) / FSDP / DiT 与 VAE 分组 |
| 02 | [`02_quantization.md`](./02_quantization.md) | FP8（dynamic/static/block）/ ModelOpt FP8 / NVFP4 / ModelSlim W8A8 / Nunchaku SVDQuant |
| 03 | [`03_cache.md`](./03_cache.md) | TeaCache（多项式 rescale + 累加阈值）/ cache-dit (DBCache + TaylorSeer + SCM) |
| 04 | [`04_attention_backends.md`](./04_attention_backends.md) | FA / Sage / STA / VSA / VMoBA / SVG2 / SLA / AITer / Ascend FA / XPU |
| 05 | [`05_kernels.md`](./05_kernels.md) | RMSNorm / LayerNorm / SiluAndMul / RoPE / Elementwise / QK Norm+RoPE / 渲染扩展 |
| 06 | [`06_torch_compile.md`](./06_torch_compile.md) | `torch.compile` + Inductor + NVFP4 JIT 预热 + comm overlap + NPU torchair |
| 07 | [`07_offload.md`](./07_offload.md) | `dit_cpu_offload` / `dit_layerwise_offload` / 组件级 offload / VAE tiling / pinned memory |
| 08 | [`08_disaggregation.md`](./08_disaggregation.md) | 四角色拆分 / ZMQ 控制面 / Mooncake RDMA 数据面 / capacity-aware 调度 |
| 09 | [`09_postprocessing.md`](./09_postprocessing.md) | RIFE 插帧 + Real-ESRGAN 超分 |

## 覆盖的源码范围

```text
python/sglang/multimodal_gen/
├── csrc/
│   ├── attn/vmoba_attn/            ← 04
│   └── render/                     ← 05
├── runtime/
│   ├── cache/                      ← 03
│   ├── disaggregation/             ← 08
│   ├── distributed/                ← 01
│   ├── layers/
│   │   ├── attention/              ← 04
│   │   ├── quantization/           ← 02
│   │   ├── rotary_embedding/       ← 05
│   │   ├── layernorm.py            ← 05
│   │   ├── activation.py           ← 05
│   │   ├── elementwise.py          ← 05
│   │   ├── custom_op.py            ← 05
│   │   ├── usp.py                  ← 01
│   │   └── lora/                   ← 02
│   ├── loader/                     ← 02 / 07
│   ├── managers/                   ← 01 / 07
│   ├── pipelines_core/
│   │   ├── stages/denoising.py     ← 03 / 06
│   │   └── executors/              ← 01
│   ├── postprocess/                ← 09
│   ├── platforms/                  ← 04 / 05 / 06
│   └── utils/layerwise_offload.py  ← 07
├── configs/
│   ├── quantization/               ← 02
│   └── sample/                     ← 03 / 09
└── tools/                          ← 02
```

## 与现有文档的关系

- 架构总览仍看 [`../01_architecture_overview.md`](../01_architecture_overview.md) 与 [`../07_disaggregation_and_optimization.md`](../07_disaggregation_and_optimization.md)；
- 模型接入教程 [`../08_skill_add_model.md`](../08_skill_add_model.md)；
- Wan2.1 案例 [`../09_case_study_wan2_1.md`](../09_case_study_wan2_1.md)、[`../10_wan2_1_end_to_end.md`](../10_wan2_1_end_to_end.md)；
- 用户部署指南 [`../wan2_1_guide.md`](../wan2_1_guide.md)。

本系列侧重 **「为什么快 / 为什么省显存 / 怎么开关 / 默认值从哪来」**，与架构/教程正交。
