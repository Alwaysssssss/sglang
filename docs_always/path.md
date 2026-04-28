# SGLang 仓库目录结构详解（至第 5 层）

> 本文档对 `sglang` 仓库根目录下的所有主要目录进行中文解读，层级深度统一到**第 5 层**。每一条目附带一句简要作用说明，方便快速定位源码与理解架构。
>
> 说明：
> - 树形符号 `├──` / `└──` 仅用于展示层级关系。
> - 只列出目录（Python/C++ 源文件在正文中以专门小节说明）。
> - 被列出的条目来自仓库根目录一次性扫描（忽略 `__pycache__` / `.git` / `build` / `*.egg-info` 等构建产物目录）。

---

## 0. 顶层布局速览

| 顶层目录 | 用途 | 关键子项 |
| --- | --- | --- |
| `3rdparty/` | 外部/平台适配代码 | `amd/` (ROCm 专用：profiling/tuning/wheel) |
| `assets/` | 仓库 README/文档使用的静态资源（图片） | — |
| `benchmark/` | 各类端到端 / kernel / 任务 benchmark 脚本 | `kernels/`, `gsm8k/`, `mmmu/`, ... |
| `docker/` | Docker 构建文件与配置 | `configs/opt/`, `diffusion_sm.Dockerfile` |
| `docs/` | 当前版本用户文档（Sphinx 源码） | `basic_usage/`, `advanced_features/`, ... |
| `docs_new/` | 正在迁移中的新文档站（Mintlify 风格） | `cookbook/`, `docs/`, `src/` |
| `docs_zh/` | 中文本地文档（本项目） | `multimodal_gen/`, `path.md` |
| `examples/` | 端到端示例代码 | `frontend_language/`, `runtime/`, ... |
| `proto/` | gRPC Protocol Buffers 定义 | `sglang/runtime/v1/` |
| `python/` | Python 发行包源码（核心） | `sglang/` + `tools/` |
| `rust/` | Rust 相关子工程 | `sglang-grpc/` |
| `scripts/` | CI / 发布 / 调试 / 代码同步脚本 | `ci/`, `release/`, `playground/` |
| `sgl-kernel/` | 重量级 AOT C++/CUDA kernel 扩展包 | `csrc/`, `python/sgl_kernel/` |
| `sgl-model-gateway/` | 模型网关（Rust 路由层 + Golang/Python 绑定 + WASM 插件） | `src/`, `bindings/`, `e2e_test/` |
| `test/` | 仓库级测试集（CI registered / manual / srt） | `registered/`, `manual/`, `srt/` |

---

## 1. `3rdparty/` — 第三方/平台适配

```text
3rdparty
└── amd
    ├── profiling        # AMD/ROCm 平台的性能分析脚本
    ├── tuning           # AMD/ROCm 平台的算子调优资料
    └── wheel
        ├── sglang       # 针对 ROCm 打包 sglang wheel 的辅助脚本
        └── sgl-kernel   # 针对 ROCm 打包 sgl-kernel 的辅助脚本
```

- **amd/**：集中存放 AMD/ROCm 相关的适配与打包资产；主仓核心代码不依赖，但 CI/发布流水线需要。

---

## 2. `benchmark/` — 端到端与 Kernel Benchmark

```text
benchmark
├── asr                              # 语音识别（ASR）服务化 benchmark
├── bench_attention_sink             # Attention Sink 变体性能测试
├── bench_in_batch_prefix            # 同 batch 前缀共享（in-batch prefix cache）
├── bench_linear_attention           # 线性注意力实现对比
├── benchmark_batch                  # 批处理吞吐基准
├── benchmark_vllm_060               # 与 vLLM 0.6.0 的对比 benchmark
├── bench_rope                       # RoPE kernel benchmark
├── blog_v0_2                        # 对应 v0.2 博客的复现脚本
├── boolq                            # BoolQ 数据集准确率 benchmark
├── ceval                            # C-Eval 中文评测
├── deepseek_v3                      # DeepSeek V3 专用 benchmark
├── dspy                             # DSPy 框架集成示例 benchmark
├── fla                              # Flash Linear Attention 对比
├── generative_agents                # 生成式 Agent 场景
├── gpt_oss                          # GPT-OSS 系列模型
├── gsm8k                            # GSM8K 数学题
├── hellaswag                        # HellaSwag 常识评测
├── hf3fs                            # HF 分布式文件系统相关 IO benchmark
├── hicache                          # 分层 KV 缓存（HiCache）压力测试
├── json_decode_regex                # 受限解码：JSON + 正则
├── json_jump_forward                # 受限解码：Jump-Forward 加速
├── json_schema                      # 受限解码：JSON Schema
├── kernels                          # 具体算子级 microbenchmark
│   ├── all_reduce                   # TP all-reduce 性能
│   ├── decoding_attention_triton    # 解码阶段 Triton attention
│   ├── deepep                       # DeepEP 专家并行通信
│   ├── deepseek                     # DeepSeek MoE 专用 kernel
│   ├── elementwise                  # 逐元素算子
│   ├── flashinfer_allreduce_fusion  # FlashInfer 融合 all-reduce
│   ├── fused_moe_triton             # Triton 版 Fused-MoE
│   ├── lora_csgmv                   # LoRA CSGMV 稀疏 GEMV
│   ├── quantization                 # 量化 kernel
│   ├── scheduler_batch              # Scheduler 批处理路径
│   └── sliding_window_attention_triton  # 滑动窗口 attention Triton
├── line_retrieval                   # 超长上下文行检索
├── llava_bench                      # LLaVA 视觉多模态
├── llm_judge                        # LLM-as-judge 评测
├── long_json_decode                 # 长 JSON 解码
├── lora                             # 多 LoRA 并发
├── mmlu                             # MMLU 评测
├── mmmu                             # MMMU 多模态评测
├── mtbench                          # MT-Bench 多轮对话
├── multi_chain_reasoning            # 多链推理（例：Self-Consistency）
├── multi_document_qa                # 多文档问答
├── multi_turn_chat                  # 多轮对话压力
├── prefill_only                     # 仅 prefill 的场景
├── react                            # ReAct Agent
├── reasoning_benchmark
│   └── figure                       # 对应论文/博客里的图表数据
├── tip_suggestion                   # 提示建议场景
├── tree_of_thought_deep             # Tree-of-Thought 深度搜索
└── tree_of_thought_v0               # Tree-of-Thought 初版
```

- **kernels/ 子集**：每个子目录对应一类算子，提供可独立运行的 Triton/CUTLASS 版本对比脚本；在修改相关 kernel 时优先在此跑回归。

---

## 3. `docker/` — 容器构建

```text
docker
└── configs
    └── opt                          # /opt 下的运行时配置（入口脚本、nginx conf 等）
```

- 根下的 `Dockerfile` / `diffusion_sm.Dockerfile` 定义主镜像；`configs/opt/` 用于 COPY 进镜像的常驻配置。

---

## 4. `docs/` — 当前主文档（Sphinx）

```text
docs
├── advanced_features                # 进阶特性（speculative / LoRA / quant / ...）
├── basic_usage                      # 基础使用
├── developer_guide                  # 开发者指南
├── diffusion                        # 多模态生成文档
│   ├── api                          # API 参考
│   └── performance
│       └── cache                    # 缓存相关性能文档
├── get_started                      # 新手入门
├── performance_dashboard            # 性能看板（图表数据）
├── platforms
│   └── ascend                       # 昇腾/Ascend 平台专用说明
├── references                       # 参考资料
│   ├── frontend                     # 前端语言参考
│   └── multi_node_deployment        # 多机部署
│       ├── lws_pd
│       │   └── lws-examples         # LeaderWorkerSet 分离部署示例
│       └── rbg_pd                   # RBG 分离部署
├── release_lookup                   # 版本兼容性查询
├── _static                          # Sphinx 静态资源
│   ├── css
│   └── image
└── supported_models                 # 支持的模型清单
    ├── extending                    # 扩展新模型的指南
    ├── retrieval_ranking            # 检索/重排序模型
    ├── specialized                  # 专用/小众模型
    └── text_generation              # 文本生成模型
```

---

## 5. `docs_new/` — 下一代文档站

```text
docs_new
├── cards
│   └── logos                        # 首页厂商 Logo
├── cookbook                         # 实战 Cookbook
│   ├── autoregressive               # 自回归 LLM 示例
│   │   ├── DeepSeek / Ernie / FlashLabs / GLM / Google
│   │   ├── InclusionAI / InternLM / InternVL / Jina
│   │   ├── Llama / MiniMax / Mistral / Moonshotai
│   │   ├── NVIDIA / OpenAI / Qwen / StepFun / Xiaomi
│   ├── base
│   │   ├── benchmarks               # benchmark 教程
│   │   └── reference                # 参考手册
│   ├── diffusion                    # 扩散模型教程
│   │   ├── FLUX / MOVA / Qwen-Image / Wan / Z-Image
│   ├── omni
│   │   └── FishAudio                # 音频全模态
│   └── specbundle                   # Speculative Decoding 组合包
├── docs                             # 主体正文（与旧 docs/ 结构对齐）
│   ├── advanced_features
│   ├── basic_usage
│   ├── developer_guide
│   ├── get-started
│   ├── hardware-platforms
│   │   └── ascend-npus
│   ├── references
│   │   ├── frontend
│   │   └── multi_node_deployment
│   │       ├── lws_pd
│   │       └── rbg_pd
│   ├── sglang-diffusion
│   │   └── api
│   └── supported-models
├── fonts                            # 字体资源
├── images                           # 截图/示意图
├── logo                             # Logo 原稿
├── scripts                          # 构建 / 预处理脚本
└── src
    ├── generated                    # 由脚本生成的 mdx 片段
    └── snippets                     # 可复用代码片段
        ├── autoregressive
        ├── diffusion
        └── specbundle
```

- `docs_new/` 是 Mintlify 风格的新站，`docs_migration_plan.md` 记录了旧站到新站的迁移计划。

---

## 6. `docs_zh/` — 中文本地文档

```text
docs_zh
├── gateway_and_grpc                 # 网关 + gRPC 相关资料
├── multimodal_gen                   # 多模态生成中文资料
├── sgl_kernel                       # `sgl-kernel` 架构与 PyTorch 集成解析
└── path.md                          # 本文件
```

---

## 7. `examples/` — 端到端示例

```text
examples
├── assets                           # 示例图片/数据
├── chat_template                    # chat template 示例
├── checkpoint_engine                # checkpoint engine 热加载权重
├── frontend_language                # 前端语言（sgl.function 等）
│   ├── quick_start
│   │   └── images                   # 入门教程配图
│   └── usage
│       ├── llava_video              # 视频 LLaVA
│       ├── rag_using_parea          # RAG + Parea 追踪
│       └── triton
│           └── models               # Triton Inference Server 模型配置
├── monitoring
│   └── grafana                      # Prometheus+Grafana 监控
│       ├── dashboards
│       │   ├── config               # provisioning config
│       │   └── json                 # dashboard JSON
│       └── datasources              # 数据源配置
├── profiler
│   └── nsys_profile_tools           # Nsight Systems 分析辅助脚本
├── runtime                          # Runtime API 使用示例
│   ├── engine                       # Engine Python API
│   ├── hidden_states                # 获取 hidden states
│   ├── multimodal                   # 多模态输入
│   └── token_in_token_out           # token-in-token-out 模式
├── sagemaker                        # AWS SageMaker 部署
└── usage                            # 其他常见用法
```

---

## 8. `proto/` — gRPC 接口定义

```text
proto
└── sglang
    └── runtime
        └── v1                       # v1 Runtime gRPC schema (.proto)
```

- Python/Rust/Golang 的 gRPC binding 均从该目录生成。

---

## 9. `python/` — Python 主发行包（**核心**）

```text
python
├── sglang                           # 可导入的 Python 包
│   ├── benchmark                    # 包内 benchmark（bench_serving 等）
│   │   └── datasets                 # benchmark 数据集解析器
│   ├── cli                          # 命令行入口（python -m sglang 子命令）
│   ├── eval                         # 内置评测脚本（lm_eval 封装等）
│   ├── jit_kernel                   # 轻量级 JIT kernel 模块
│   │   ├── benchmark
│   │   │   └── diffusion            # 扩散模型专用 benchmark
│   │   ├── csrc                     # C/C++/CUDA 源码
│   │   │   ├── attention            # attention 相关 JIT kernel
│   │   │   ├── diffusion            # 扩散模型 JIT kernel
│   │   │   ├── distributed          # 分布式通信 JIT kernel
│   │   │   ├── elementwise          # 逐元素 JIT kernel
│   │   │   ├── fast-hadamard-transform  # 快速 Hadamard 变换
│   │   │   ├── gemm                 # GEMM 相关
│   │   │   ├── lora                 # LoRA 相关
│   │   │   ├── moe                  # MoE 相关
│   │   │   ├── ngram_corpus         # n-gram 推测解码语料
│   │   │   └── nsa                  # NSA（Native Sparse Attention）
│   │   ├── diffusion                # 扩散模型 Python 封装
│   │   │   ├── cutedsl              # CuTe DSL 版本
│   │   │   └── triton               # Triton 版本
│   │   ├── include
│   │   │   └── sgl_kernel           # 公共头文件
│   │   ├── tests
│   │   │   └── diffusion            # 扩散模型 kernel 单测
│   │   └── triton                   # 通用 Triton kernel
│   ├── lang                         # 前端语言（sgl.function / Runtime DSL）
│   │   └── backend                  # 前端后端适配（OpenAI / Anthropic / ...）
│   ├── multimodal_gen               # 多模态生成（Diffusion/TTS 等）新引擎
│   │   ├── apps
│   │   │   ├── ComfyUI_SGLDiffusion # ComfyUI 自定义节点
│   │   │   └── webui                # 自带 WebUI
│   │   ├── benchmarks               # 包内 benchmark
│   │   ├── configs                  # YAML 配置
│   │   │   ├── backend              # 后端相关
│   │   │   ├── models               # 模型定义
│   │   │   ├── pipeline_configs     # 流水线配置
│   │   │   ├── post_training        # 后训练（LoRA/蒸馏等）
│   │   │   ├── quantization         # 量化配置
│   │   │   └── sample               # 采样策略
│   │   ├── csrc                     # 多模态专用 C++/CUDA
│   │   │   ├── attn                 # attention kernel
│   │   │   └── render               # 渲染/后处理 kernel
│   │   ├── runtime                  # 运行时
│   │   │   ├── cache                # KV / latent cache
│   │   │   ├── disaggregation       # PD 分离
│   │   │   ├── distributed          # 并行通信
│   │   │   ├── entrypoints          # API 服务入口
│   │   │   ├── layers               # 模型层实现
│   │   │   ├── loader               # 权重加载
│   │   │   ├── managers             # 调度 / 会话管理
│   │   │   ├── models               # 具体模型（Flux/Wan/…）
│   │   │   ├── pipelines            # 扩散 pipeline 实例
│   │   │   ├── pipelines_core       # pipeline 核心原语
│   │   │   ├── platforms            # 硬件平台适配
│   │   │   ├── postprocess          # VAE 解码/图像后处理
│   │   │   ├── post_training        # 训练后处理
│   │   │   └── utils                # 工具
│   │   ├── test                     # 多模态测试
│   │   │   ├── cli
│   │   │   ├── manual
│   │   │   ├── scripts
│   │   │   ├── server
│   │   │   ├── test_files
│   │   │   └── unit
│   │   ├── third_party              # 第三方片段
│   │   └── tools                    # 辅助 CLI
│   ├── srt                          # **SGLang Runtime（LLM 核心）**
│   │   ├── batch_invariant_ops      # 批不变性算子（保证 batch=1/N 结果一致）
│   │   ├── batch_overlap            # 批处理与计算/通信重叠
│   │   ├── checkpoint_engine        # checkpoint engine 接入
│   │   ├── compilation              # torch.compile / Inductor 集成
│   │   ├── configs                  # 模型 config 解析
│   │   ├── connector                # 外部存储/缓存 connector
│   │   │   └── serde                # 序列化/反序列化
│   │   ├── constrained              # 受限解码（JSON/regex/grammar）
│   │   │   ├── torch_ops            # PyTorch op 实现
│   │   │   └── triton_ops           # Triton op 实现
│   │   ├── debug_utils              # 调试工具
│   │   │   ├── comparator           # 张量比较器
│   │   │   ├── schedule_simulator   # 调度模拟器
│   │   │   └── source_patcher       # 源码注入式调试
│   │   ├── disaggregation           # PD（Prefill/Decode）分离部署
│   │   │   ├── ascend               # Ascend 平台实现
│   │   │   ├── base                 # 抽象基类
│   │   │   ├── common               # 通用工具
│   │   │   ├── fake                 # 单机模拟 backend
│   │   │   ├── mooncake             # Mooncake Transfer Engine
│   │   │   ├── mori                 # MoRI backend
│   │   │   └── nixl                 # NIXL backend
│   │   ├── distributed              # 分布式（TP/PP/DP/EP）
│   │   │   └── device_communicators # NCCL/HCCL 等通信器
│   │   ├── dllm                     # Diffusion-LLM（dLLM）推理
│   │   │   ├── algorithm            # 解码算法
│   │   │   └── mixin                # Mixin 组件
│   │   ├── elastic_ep               # 弹性 EP（Expert Parallelism）
│   │   ├── entrypoints              # API 服务端入口
│   │   │   ├── anthropic            # Anthropic 兼容
│   │   │   ├── ollama               # Ollama 兼容
│   │   │   └── openai               # OpenAI 兼容
│   │   ├── eplb                     # EP Load Balancer
│   │   │   ├── eplb_algorithms      # 各种负载均衡算法
│   │   │   └── eplb_simulator       # 仿真器
│   │   ├── function_call            # 函数调用（tool-use）解析
│   │   ├── grpc                     # gRPC server/client
│   │   ├── hardware_backend         # 非 CUDA 后端
│   │   │   ├── mlx                  # Apple MLX
│   │   │   ├── musa                 # 摩尔线程 MUSA
│   │   │   └── npu                  # 华为 NPU
│   │   ├── layers                   # 模型层算子
│   │   │   ├── attention            # 各种 attention 后端
│   │   │   ├── deep_gemm_wrapper    # DeepGEMM 封装
│   │   │   ├── moe                  # MoE 组件
│   │   │   ├── quantization         # 量化层
│   │   │   ├── rotary_embedding     # RoPE 实现
│   │   │   └── utils                # 层工具
│   │   ├── lora                     # LoRA 支持
│   │   │   ├── backend              # 不同 backend 实现
│   │   │   ├── torch_ops            # PyTorch 算子
│   │   │   └── triton_ops           # Triton 算子
│   │   ├── managers                 # 调度器/分词管理器/会话管理器
│   │   ├── mem_cache                # KV 缓存系统（RadixCache、HiCache）
│   │   │   ├── cpp_radix_tree       # C++ Radix Tree
│   │   │   ├── hybrid_cache         # 分层混合缓存
│   │   │   ├── sparsity             # 稀疏缓存
│   │   │   ├── storage              # 外部存储（disk/HF3FS/...）
│   │   │   └── unified_cache_components  # 统一缓存组件
│   │   ├── model_executor           # 模型执行器 / CUDA Graph
│   │   │   └── breakable_cuda_graph # 可分段 CUDA Graph
│   │   ├── model_loader             # 权重加载器
│   │   ├── models                   # 具体 LLM 模型实现
│   │   │   └── deepseek_common      # DeepSeek 系列共享组件
│   │   ├── multimodal               # 多模态输入处理
│   │   │   ├── evs                  # EVS（视觉） backend
│   │   │   └── processors           # 各模型 processor
│   │   ├── multiplex                # 多路复用（多模型同进程）
│   │   ├── observability            # 可观测性（metrics/tracing）
│   │   ├── parser                   # 通用解析器
│   │   ├── platforms                # 平台抽象
│   │   ├── plugins                  # 插件机制
│   │   ├── ray                      # Ray 集成
│   │   ├── sampling                 # 采样 / 惩罚项
│   │   │   └── penaltylib           # 重复/频率等 penalty
│   │   ├── session                  # 会话管理
│   │   ├── speculative              # 推测解码（EAGLE/NGRAM/…）
│   │   │   ├── cpp_ngram            # C++ n-gram 实现
│   │   │   └── triton_ops           # Triton 算子
│   │   ├── tokenizer                # 分词器封装
│   │   ├── utils                    # 通用工具
│   │   │   └── hf_transformers      # HF transformers 兼容工具
│   │   └── weight_sync              # RLHF 权重同步
│   └── test                         # 包内测试工具（非 CI，但被 CI 引用）
│       ├── ascend                   # Ascend 专用
│       ├── attention                # attention 单测
│       ├── ci                       # CI 辅助 case
│       ├── external_models          # 需外部权重的模型测试
│       ├── kits                     # 测试工具包（custom_test_case 等）
│       ├── longbench_v2             # 长上下文评测
│       ├── server_fixtures          # 共享 server fixture
│       └── speculative              # spec 解码测试
└── tools                            # 辅助工具脚本（打包、校验等）
```

### `python/sglang/srt/` 与 `multimodal_gen/` 的区别
- **`srt/`**：LLM / Embedding / Reward 等离散文本推理引擎，是 `sglang`/`python -m sglang.launch_server` 的底座。
- **`multimodal_gen/`**：面向扩散模型、图像/视频生成、TTS 的连续模态推理引擎，拥有**独立的** `entrypoints` / `managers` / `pipelines`。

---

## 10. `rust/` — Rust 子工程

```text
rust
└── sglang-grpc
    └── src                          # Rust 版 gRPC client/server（主要供 router 使用）
```

---

## 11. `scripts/` — CI / 运维 / 实验脚本

```text
scripts
├── ci                               # CI 流水线主脚本
│   ├── amd                          # AMD/ROCm 测试入口
│   ├── cuda                         # NVIDIA CUDA 测试入口
│   ├── musa                         # 摩尔线程测试入口
│   ├── npu                          # 华为 NPU 测试入口
│   ├── slurm                        # SLURM 集群提交脚本
│   └── utils                        # CI 通用工具
│       └── diffusion                # 扩散模型 CI 辅助
├── ci_monitor                       # CI 监控 / bisect 工具
├── code_sync                        # 与上游/镜像仓同步脚本
├── playground                       # 实验性脚本（随手测试）
│   ├── disaggregation
│   ├── lora
│   └── router
└── release                          # 版本发布脚本（build wheel / docker 等）
```

---

## 12. `sgl-kernel/` — 重量级 AOT kernel 扩展

```text
sgl-kernel
├── benchmark                        # 专用 benchmark 脚本
├── cmake                            # CMake 模块
├── csrc                             # C++/CUDA 核心源码
│   ├── allreduce                    # 自定义 all-reduce
│   ├── attention                    # attention kernel
│   │   └── cutlass_sm100_mla        # SM100 架构 MLA
│   │       ├── device               # device 侧代码
│   │       └── kernel               # kernel 实现
│   ├── cpu                          # CPU kernel
│   │   ├── aarch64                  # ARM64 专用
│   │   ├── mamba                    # Mamba CPU 实现
│   │   ├── model                    # 模型层 CPU 实现
│   │   └── x86_64                   # x86 专用（AMX/AVX）
│   ├── cutlass_extensions           # CUTLASS 扩展
│   │   ├── detail
│   │   │   └── collective
│   │   ├── epilogue                 # epilogue 模块
│   │   └── gemm
│   │       └── collective           # 集合通信 + GEMM
│   ├── elementwise                  # 逐元素算子
│   ├── expert_specialization        # 专家特化（EP）
│   ├── gemm                         # 通用 GEMM
│   │   ├── gptq                     # GPTQ 量化 GEMM
│   │   └── marlin                   # Marlin 量化 GEMM
│   ├── grammar                      # 受限解码 kernel
│   ├── kvcacheio                    # KV 缓存 I/O
│   ├── mamba                        # Mamba 状态空间算子
│   ├── memory                       # 内存管理 kernel
│   ├── moe                          # MoE kernel
│   │   └── cutlass_moe
│   │       └── w4a8                 # W4A8 MoE
│   ├── quantization                 # 量化 kernel
│   │   └── gguf                     # GGUF 反量化
│   ├── spatial                      # 空间（视觉）算子
│   └── speculative                  # 推测解码 kernel
├── include
│   └── hip
│       └── impl                     # HIP 平台兼容头
├── python
│   └── sgl_kernel                   # Python 绑定
│       ├── quantization             # 量化相关 Python wrapper
│       └── testing                  # 测试辅助
└── tests                            # C++ 单测 + Python 端测
    ├── spatial
    └── speculative
```

- **与 `python/sglang/jit_kernel/` 的区别**：`sgl-kernel` 是**AOT 编译**的 pip 子包，面向稳定 kernel；`jit_kernel` 走 **Triton / CuTe DSL JIT**，便于快速迭代。

---

## 13. `sgl-model-gateway/` — 模型网关（Rust Router）

```text
sgl-model-gateway
├── benches                          # Rust criterion benchmark
├── bindings                         # 多语言 SDK 绑定
│   ├── golang
│   │   ├── examples
│   │   │   ├── oai_server           # OpenAI 风格 server 示例
│   │   │   ├── simple               # 最小示例
│   │   │   └── streaming            # 流式响应示例
│   │   ├── internal
│   │   │   ├── ffi                  # CGO FFI 封装
│   │   │   ├── grpc                 # gRPC 适配
│   │   │   └── proto                # 生成的 proto
│   │   └── src                      # Go 对外 API
│   └── python
│       ├── src
│       │   └── sglang_router        # Python 包入口
│       └── tests                    # Python 端测试
├── e2e_test                         # 端到端测试
│   ├── benchmarks                   # 端到端 benchmark
│   ├── chat_completions             # chat/completions 接口测试
│   ├── embeddings                   # embedding 接口测试
│   ├── fixtures                     # 公共 fixture
│   ├── infra                        # 基础设施/启动脚本
│   ├── responses                    # OpenAI Responses API 测试
│   └── router                       # router 行为测试
├── examples
│   └── wasm                         # WASM 插件示例
│       ├── wasm-guest-auth          # 鉴权插件
│       │   └── src
│       ├── wasm-guest-logging       # 日志插件
│       │   └── src
│       └── wasm-guest-ratelimit     # 限流插件
│           └── src
├── scripts                          # 构建/部署脚本
├── src                              # Rust 主源码
│   ├── config                       # 配置解析
│   ├── core
│   │   └── steps
│   │       └── worker               # worker 管道步骤
│   ├── observability                # 指标/追踪
│   ├── policies                     # 路由/调度策略
│   ├── routers                      # 路由实现
│   │   ├── conversations            # conversations API
│   │   ├── grpc                     # gRPC 路由
│   │   │   ├── common
│   │   │   ├── harmony              # Harmony 格式专用
│   │   │   └── regular              # 常规场景
│   │   ├── http                     # HTTP 路由
│   │   ├── mesh                     # mesh 模式
│   │   ├── openai                   # OpenAI 兼容层
│   │   │   └── responses            # Responses API
│   │   ├── parse                    # 解析层
│   │   └── tokenize                 # 分词层
│   └── wasm                         # WASM 插件宿主
└── tests                            # Rust 单元/集成测试
    ├── api
    ├── common
    ├── fixtures
    │   └── images                   # 图像 fixture
    ├── reliability                  # 可靠性（retry/失败注入）
    ├── routing                      # 路由策略测试
    ├── security                     # 安全测试
    └── spec                         # 规范一致性测试
```

- 这是一个**独立产品**：用户可以单独部署 `sgl-model-gateway` 作为 OpenAI 兼容的网关+路由，上游对接 `srt` worker。

---

## 14. `test/` — 仓库级测试

```text
test
├── lm_eval_configs                  # lm-eval-harness 的配置集合
├── manual                           # 需要人工/特殊资源运行的测试
│   ├── 4-gpu-models                 # 4 卡模型
│   ├── ascend                       # Ascend 手动测试
│   ├── cpu                          # CPU 测试
│   ├── debug_utils                  # 调试工具手动测试
│   ├── entrypoints
│   │   └── http_server              # HTTP server 手动测试
│   ├── ep                           # Expert Parallel
│   ├── hicache                      # 分层缓存
│   ├── kv_transfer                  # KV transfer
│   ├── lang_frontend                # 前端语言
│   ├── layers
│   │   ├── attention
│   │   │   └── nsa                  # NSA attention
│   │   └── moe
│   ├── lora
│   ├── model_loading
│   ├── models
│   ├── nightly                      # nightly 专用
│   ├── openai_server
│   │   └── features
│   ├── piecewise_cudagraph
│   ├── quant
│   ├── rl                           # RL 训练链路
│   └── vlm                          # 视觉语言模型
├── registered                       # **被 CI 注册的测试**
│   ├── 4-gpu-models
│   ├── 8-gpu-models
│   ├── amd
│   │   ├── accuracy
│   │   │   ├── mi30x
│   │   │   └── mi35x
│   │   ├── disaggregation
│   │   └── perf
│   │       ├── mi30x
│   │       └── mi35x
│   ├── ascend
│   │   ├── basic_function
│   │   │   ├── backends
│   │   │   ├── dllm
│   │   │   ├── HiCache
│   │   │   ├── optimization_debug
│   │   │   ├── parallel_strategy
│   │   │   ├── parameter
│   │   │   ├── quant
│   │   │   ├── runtime_opts
│   │   │   └── speculative_inference
│   │   ├── embedding_models
│   │   ├── interface
│   │   ├── llm_models
│   │   ├── rerank_models
│   │   ├── reward_models
│   │   └── vlm_models
│   ├── attention
│   ├── backends
│   ├── bench_fn                     # benchmark 函数本身的测试
│   ├── constrained_decoding
│   ├── core
│   ├── cp                           # Context Parallel
│   ├── cuda_graph
│   ├── debug_utils
│   │   ├── comparator
│   │   │   ├── aligner
│   │   │   ├── dims_spec
│   │   │   └── tensor_comparator
│   │   └── source_patcher
│   ├── disaggregation
│   ├── distributed
│   ├── dllm
│   ├── ep
│   ├── eval                         # 准确率 eval
│   ├── function_call
│   ├── gb300                        # GB300 专用
│   ├── input_embedding
│   ├── kernels
│   ├── language
│   ├── layers
│   │   └── mamba
│   ├── lora
│   ├── mla                          # Multi-Head Latent Attention
│   ├── model_loading
│   ├── models
│   ├── moe
│   ├── observability
│   ├── openai_server
│   │   ├── basic
│   │   ├── features
│   │   ├── function_call
│   │   └── validation
│   ├── ops
│   ├── perf                         # 性能门禁
│   ├── piecewise_cuda_graph
│   ├── prefill_only
│   ├── profiling
│   ├── quant
│   ├── radix_cache
│   ├── reasoning
│   ├── rl
│   ├── rotary
│   ├── sampling
│   ├── scheduler
│   ├── sessions
│   ├── spec
│   │   ├── dflash                   # DFlash 推测
│   │   ├── eagle                    # EAGLE
│   │   └── utils
│   ├── stress                       # 压测
│   ├── tokenizer
│   ├── unit                         # 纯单元测试
│   │   ├── auto_benchmark
│   │   ├── batch_invariant_ops
│   │   ├── configs
│   │   ├── constrained
│   │   ├── entrypoints
│   │   │   └── openai
│   │   ├── eplb
│   │   ├── function_call
│   │   ├── layers
│   │   ├── managers
│   │   ├── mem_cache
│   │   ├── model_executor
│   │   ├── model_loader
│   │   ├── models
│   │   ├── observability
│   │   ├── parser
│   │   ├── platforms
│   │   ├── plugins
│   │   ├── sampling
│   │   ├── server_args
│   │   ├── spec
│   │   ├── tokenizer
│   │   ├── tools
│   │   └── utils
│   ├── utils
│   └── vlm
└── srt                              # 历史遗留位置（逐步迁入 registered/）
    ├── ascend
    ├── configs
    ├── cpu
    └── xpu
```

- **`registered/` vs `manual/` vs `srt/`**：
  - `registered/` 是 CI 自动注册并跑的；修改/新增 CI 必改这里。
  - `manual/` 需特殊硬件/数据，人手触发。
  - `srt/` 为历史遗留目录，新增测试不建议放入。

---

## 15. 速查：核心路径对照表

| 想改 / 想看什么 | 去哪里找 |
| --- | --- |
| LLM 新模型接入 | `python/sglang/srt/models/` |
| 扩散模型 / 图像生成接入 | `python/sglang/multimodal_gen/runtime/models/` + `pipelines/` |
| KV 缓存策略（Radix/HiCache/sparsity） | `python/sglang/srt/mem_cache/` |
| PD 分离（Prefill/Decode） | `python/sglang/srt/disaggregation/` |
| 投机解码（EAGLE/NGRAM） | `python/sglang/srt/speculative/` |
| 受限解码 (JSON/regex/grammar) | `python/sglang/srt/constrained/` + `sgl-kernel/csrc/grammar/` |
| 添加 Triton / CuTe JIT kernel | `python/sglang/jit_kernel/` |
| 添加 AOT C++/CUDA kernel | `sgl-kernel/csrc/` |
| 路由层 / OpenAI 兼容网关 | `sgl-model-gateway/src/routers/` |
| HTTP/OpenAI 入口 | `python/sglang/srt/entrypoints/openai/` |
| gRPC 入口 | `python/sglang/srt/grpc/` + `proto/sglang/runtime/v1/` |
| CI 测试添加 | `test/registered/**`（对应领域子目录） |
| 中文文档 | `docs_zh/`（本目录） |
| `sgl-kernel` 架构解读（中文） | `docs_zh/sgl_kernel/` |
| 主文档（英文） | `docs/` 或新的 `docs_new/docs/` |
| benchmark / 复现博客实验 | `benchmark/` |

---

> 本文档由目录树快照生成，若目录结构后续有增减，请在本文件基础上做增量维护（搜索对应节 + 更新速查表即可）。
