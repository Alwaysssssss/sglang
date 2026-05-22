# Phase 6：性能收尾、并行兼容与 Upstream 同步策略

## 1. 阶段目标

本阶段的目标是在功能与结果已经稳定的前提下，完成：

1. 显存和吞吐优化
2. 并行兼容性检查
3. 运行时耦合清理
4. 后续同步 STAR upstream 的维护规范

阶段完成后，应满足：

1. 接入方案已经可维护
2. 运行时不含临时调试分支
3. 已有清晰的同步和升级边界
4. 性能指标已记录

---

## 2. 本阶段范围

### 本阶段处理

1. 单卡显存优化
2. CPU offload / VAE tiling 适配
3. 是否支持 SP/TP 的评估与最小接入
4. 日志、报错、文档收尾
5. upstream 同步流程规范

### 本阶段不处理

1. 回到前面大改接口
2. 大规模重构基础框架
3. 为了追性能牺牲 parity

---

## 3. 性能收尾策略

## 3.1 优化前提

所有性能优化必须满足：

1. 阶段 5 parity 已通过
2. 已知当前瓶颈位置
3. 优化前后可对比同一组样本

## 3.2 优先优化项

建议按如下顺序评估：

1. VAE decode 显存
2. 条件视频 encode 显存
3. transformer forward 显存
4. scheduler loop 额外开销

### 为什么先看 VAE

因为 STAR 的 decode 是分块时序 decode，这通常是最容易出现额外显存波动的地方。

---

## 4. 并行与显存策略

## 4.1 MVP 阶段只要求单卡稳定

不要在 parity 之前就强行把 TP/SP 一次做完。

本阶段开始后，建议按顺序评估：

1. 单卡稳定
2. `vae_cpu_offload`
3. `vae_tiling`
4. 是否可安全接入 SP
5. 是否值得接入 TP

## 4.2 对 Sequence Parallel 的建议

由于 STAR 输入是 5D 视频 latent，后续如需接入 SP，应重点检查：

1. `batch.latents` 的时间维 shard 逻辑
2. `batch.image_latent` 是否能按相同方式 shard
3. 自定义 decode 是否必须在 gather 后执行

建议：

1. Phase 6 只做兼容性设计和最小 smoke check
2. 如果 SP 改动大，可以在本期明确标注“后续优化项”，不要强塞进当前交付

## 4.3 对 Tensor Parallel 的建议

TP 只在以下前提下推进：

1. 单卡性能已定位到 transformer 主体
2. STAR DiT 内部线性层和 attention head 划分规则清晰

如果这两个条件不满足，先不做 TP。

---

## 5. 建议补充的测试与 profiling

建议补充：

1. `python/sglang/multimodal_gen/test/manual/profile_star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/test/unit/test_star_sp_smoke.py`（仅当接入 SP）

## 5.1 profile 脚本需要输出

至少输出：

1. 总时延
2. encode 时延
3. denoise 时延
4. decode 时延
5. 峰值显存

## 5.2 对比口径

至少对比：

1. STAR 原始实现
2. SGLang native pipeline

相同条件：

1. 同权重
2. 同视频
3. 同 prompt
4. 同 seed
5. 同分辨率和步数

---

## 6. 运行时代码清理清单

在阶段 5 过程中可能会留下很多临时逻辑，本阶段需要清理：

1. 临时 debug print
2. 临时 shape assert 文本
3. 临时手工路径
4. 直接写死的视频样本路径
5. 为了调试而绕过的 config 分支

### 必须保留的内容

可以保留但应整理成正式机制：

1. manifest 报告
2. parity 工具
3. profiling 脚本

---

## 7. Upstream 同步策略

## 7.1 同步分类

后续同步 STAR upstream 时，应分成两类：

1. **权重更新**
   - 只需更新转换脚本输入或重新导出资产
2. **结构更新**
   - 需要同步 DiT / VAE / scheduler 代码

## 7.2 推荐维护方式

建议为 STAR 接入新增一个最小维护清单：

1. 当前支持的 STAR upstream commit / 版本
2. 当前转换脚本版本
3. 当前目标模型目录版本
4. 结构兼容说明

建议把这些信息写入：

1. `star_integration_config.json`
2. 或 `manifests/source_assets.json`

## 7.3 当 SGLang 上游升级时的处理原则

如果是 SGLang 自身升级：

1. 先检查 `DenoisingStage` 的 `image_latent` 语义是否变化
2. 再检查 `PipelineConfig` hook 是否有签名变化
3. 再检查 loader / registry / sampling params 体系是否变化

建议每次升级都先跑：

1. unit tests
2. smoke test
3. 一组固定 parity case

---

## 8. 文档与维护收尾

本阶段应补齐：

1. 最终使用说明
2. 模型目录说明
3. 转换脚本说明
4. 典型失败问题排查

建议至少在最终接入附近补一份简洁说明，内容包括：

1. 如何准备转换后的模型目录
2. 如何发起一次推理请求
3. 如何跑 smoke test
4. 如何跑 parity 对比

---

## 9. 阶段验收标准

本阶段完成标准：

1. 已有性能基线记录
2. 已清理运行时代码中的临时耦合
3. 已明确 SP/TP 的当前支持范围
4. 已沉淀 upstream 同步规则
5. 已形成最小维护文档集

---

## 10. 最终交付判断

整个 STAR 接入在本阶段结束后，才可以认为“真正交付完成”。

最终交付的判断标准建议为：

1. 代码不依赖 STAR 原仓库运行
2. 结果已完成 parity
3. 文档能指导他人复现转换、加载和推理
4. 未来升级时知道该改哪里、不该改哪里

如果只能满足“能跑”，但还无法稳定升级和维护，则不应视为真正完成。
