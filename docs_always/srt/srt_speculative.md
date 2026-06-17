# `python/sglang/srt/speculative` 模块分析

## 定位

`speculative` 实现推测解码，包括 EAGLE/EAGLE3、multi-layer EAGLE、standalone draft model 和 N-gram。它通过 draft worker 生成候选 token/tree，再让 target model 一次 verify 多个候选，以减少昂贵 target forward 次数。

## 关键文件

- `spec_info.py`：`SpeculativeAlgorithm`、`SpecInputType`、`SpecInput` 抽象。
- `eagle_info.py`、`eagle_info_v2.py`：EAGLE draft/verify 输入输出、cache loc 和 accepted token 状态。
- `eagle_worker.py`、`eagle_worker_v2.py`：EAGLE worker 两套实现。
- `multi_layer_eagle_worker.py`、`multi_layer_eagle_worker_v2.py`、`multi_layer_eagle_utils.py`：多层 EAGLE。
- `standalone_worker.py`、`standalone_worker_v2.py`：独立 draft model。
- `ngram_worker.py`、`ngram_info.py`、`cpp_ngram/ngram_corpus.py`：N-gram speculative。
- `eagle_draft_cuda_graph_runner.py`、`eagle_draft_extend_cuda_graph_runner.py`、`multi_layer_eagle_draft_extend_cuda_graph_runner.py`：draft CUDA graph runner。
- `spec_utils.py`、`eagle_utils.py`、`draft_utils.py`：tree mask、cache loc、draft backend factory、debug 检测等工具。

## 运行流程

`ServerArgs` 先规范 speculative 参数，`SpeculativeAlgorithm.create_worker` 选择 worker。EAGLE 类路径通常先用 target prefill 产生 hidden states，再由 draft model extend/decode 生成候选树。target model 对候选树做一次 verify，根据 greedy 或 sampling 规则得到 accepted tokens。之后 speculative 模块更新请求输出、grammar 状态、KV cache loc、draft cache 和下一轮 draft 输入。N-gram 路径不加载 draft model，而是从历史 corpus 匹配候选，再复用 target verify。

## 依赖关系

该模块强依赖 `ModelRunner`、`TpModelWorker`、`ForwardBatch`、`ScheduleBatch`、KV memory pool allocator、attention backend、`sampling`、`constrained`、`distributed` 和 `sgl_kernel`。它也读取大量 `ServerArgs` 限制。

## 设计要点和风险

- 组合限制很多：NGRAM/standalone/DP attention/topk/page_size/attention backend 之间有显式相容性要求。
- cache loc 更新是高风险区域，accepted length、draft token、target cache、page size、topk 分支必须一致。
- grammar + speculative 会增加 CPU tree 遍历和 bitmask 成本。
- v2 路径引入 plan stream、event、record_stream/synchronize，跨 stream 生命周期顺序要求严格。
