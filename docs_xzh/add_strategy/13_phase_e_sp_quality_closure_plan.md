# 13. Phase E: VividVR Native SP 质量闭环与下一阶段提速计划

本文档承接 [12_phase_e_sp_native_acceleration_plan.md](./12_phase_e_sp_native_acceleration_plan.md)，但会替换本文件旧版本里已经失效的主线判断。

当前结论已经很明确：
- `native SP` 的真实双卡加速已经成立。
- `quality opt_v2` 已经证明：只要恢复正确的 `Connector` 全局 control 语义，质量可以回到接近单卡。
- 之后几轮“把 `v2` 压回 `15s/it` 左右”的尝试虽然都拿到了真实提速，但都没有守住 `v2` 质量合同。
- 后续主线不能再继续“先改一版看视频”，而必须先锁死 `v2` 的张量级真值合同，再做保语义优化。

本文档只讨论：
- 当前失败尝试到底失败在哪。
- 为什么速度会上去，但质量没有对齐 `v2`。
- 下一步应该按什么顺序推进，避免继续踩同一个坑。

本文档不讨论：
- `TP / DP / CFG parallel`
- 新 attention backend 试验
- 改 `Phase D` 长视频主语义
- 为了提速主动接受弱于 `v2` 的 connector 语义

## 1. 当前冻结基线

### 1.1 上游质量基线

`Phase C` 和 `Phase D` 已经正式验收，后续任何 `Phase E` 提速都默认不能破坏这两层语义基线。

当前 `native SP` 质量控制基线不是最早的 fast 版本，而是已经正式验收的 `quality opt_v2`：
- 指标：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v2_130f_20step_compile_metrics_seed42_20260611T134903Z.json`
- 日志：`/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v2_recheck_20260611T134851Z.log`
- `model_inference_runtime_seconds = 539.324976`
- `ssim_mean = 0.9846050631221304`
- `ssim_min = 0.9778964153159052`
- steady-state denoise 约 `19.11s/it`

它的意义不是“最终速度已经满意”，而是：
- 它已经证明 `native SP` 可以在双卡下保住接近单卡的质量。
- 后续所有提速 patch 都必须把它当作新的质量真值，而不是重新和 fast 或 `v1` 妥协。

### 1.2 速度参考线

当前有两条速度参考线：

1. fastest native `SP`
- 指标：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_only_130f_20step_compile_metrics_seed42_20260611T052918Z.json`
- `model_inference_runtime_seconds = 396.745880`
- `ssim_mean = 0.9627860811380421`
- `ssim_min = 0.9152052581958419`
- steady-state denoise 约 `12.06s/it`

2. `quality opt_v1`
- 指标和日志见 `phase_e_e41_native_sp_v2_quality_control_and_next_speedup_handover.md`
- `model_inference_runtime_seconds ≈ 470.0s`
- `ssim_mean ≈ 0.9786`
- `ssim_min ≈ 0.953`
- steady-state denoise 约 `15.6s/it`

当前对这两条线的正确理解是：
- fastest native `SP` 只是速度上界参考，不是可发布方案。
- `v1` 只是“明显快于 `v2` 且质量部分恢复”的中间参考，不是新的质量合同。

### 1.3 runtime-only 对照

双卡 runtime-only `SP-only control` 仍然是重要对照：
- 指标：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_sp_only_130f_20step_compile_metrics_seed42_20260611T041018Z.json`
- `ssim_mean = 0.9845581236659583`
- `ssim_min = 0.9795734797285032`

它说明：
- 双卡环境本身不是质量问题来源。
- `FA / compile / distributed runtime` 本身也不是当前主要根因。

## 2. 之前已经做过哪些失败尝试

后续不要把这些失败版当成互相独立的新方向。它们已经收敛到了同一类坏结果。

### 2.1 chunked / short-lived global restore

代表正式产物：
- `phase_e41_native_sp_speedup_chunk4_fix2_130f_20step_compile_metrics_seed42_20260612T025030Z.json`

结果：
- `model_inference_runtime_seconds = 455.206320`
- steady-state denoise 约 `14.86s/it`
- `ssim_mean = 0.9642658109`
- `ssim_min = 0.9113702957`

思路：
- 不再长期持有完整 `global_control`
- 改成按 chunk 恢复、按 chunk 使用

结论：
- 速度收益真实
- 但 `Connector` 实际拿到的 global control contract 已经不再等价于已验收 `v2`

### 2.2 distributed exact connector attention 原型

代表正式产物：
- `phase_e41_native_sp_quality_opt_v2_dist_exact_formal_130f_20step_compile_metrics_seed42_20260612T034515Z.json`

结果：
- `model_inference_runtime_seconds = 460.144272`
- steady-state denoise 约 `15.22s/it`
- `ssim_mean = 0.9642261291`
- `ssim_min = 0.9107123542`

思路：
- 尝试不显式物化完整 `global_control`
- 直接做分布式 exact attention

结论：
- 这版实现并没有真正达到和 `v2` 数学等价
- 从 formal 结果看，它和 chunked restore 失败版落在同一个质量簇

### 2.3 packed eager-global 一组工程优化尝试

代表正式产物：
- `phase_e41_native_sp_quality_opt_v2_identity_scale_formal_130f_20step_compile_metrics_seed42_20260612T042444Z.json`
- `phase_e41_native_sp_quality_opt_v2_identity_only_formal_130f_20step_compile_metrics_seed42_20260612T044748Z.json`
- `phase_e41_native_sp_quality_opt_v2_packed_local_fix_formal_130f_20step_compile_metrics_seed42_20260612T052907Z.json`
- `phase_e41_native_sp_quality_opt_v2_packed_eager_formal_metrics_seed42_20260612T060316Z.json`

结果都非常接近：
- `model_inference_runtime_seconds ≈ 454.6 ~ 455.1s`
- steady-state denoise 约 `14.9s/it`
- `ssim_mean ≈ 0.9643 ~ 0.9644`
- `ssim_min ≈ 0.9106 ~ 0.9128`

思路：
- 保留 eager-global 大方向
- 继续在 `stack / cast / scale / gather / unbind` 组织方式上做降本

结论：
- 这些版本不是不同成功方向，而是在同一个坏语义附近反复调整实现外观

## 3. 失败结果为什么可以判定为“同一个坑”

### 3.1 formal 指标收敛到了同一个坏簇

所有失败版几乎都落在：
- `model_inference_runtime_seconds ≈ 455s`
- denoise `≈ 15s/it`
- `ssim_mean ≈ 0.964`
- `ssim_min ≈ 0.911`

这不是随机噪声，也不是某一版单独实现失误，更像是：
- 一旦把 `v2` 里某一层关键语义削弱到同一程度
- formal 就会稳定收敛到这一簇

### 3.2 最差时间窗和 fastest native `SP` 基本重合

已确认的坏窗：
- `frames 48-58`
- `frames 93-104`

fastest native `SP`：
- `48-58` 窗最低约 `0.941`
- `93-104` 窗最低约 `0.915`

失败版 `packed_eager`：
- `48-58` 窗最低约 `0.943`
- `93-104` 窗最低约 `0.913`

已验收 `v2`：
- 两个窗都维持在 `0.983 ~ 0.985` 一带

这说明：
- 当前失败版的主问题不是新的随机瑕疵
- 而是又重新掉回了和 fastest native `SP` 同类的语义退化

### 3.3 把 transformer 侧 connector 消费链改回“原始 v2 直传语义”后，formal 结果几乎没变

这一点很关键。

已经做过一次验证：
- 将 transformer block 里 `Connector` 的消费方式收回到更接近原始 `v2` 的直接 tuple 传递
- formal 结果仍然落在 `0.964 / 0.912` 那个坏簇

这意味着：
- 当前主要问题大概率不在 `cogvideox_vividvr.py` 里的 block 内 connector 调用方式
- 更可能在更上游的 control state 构造与 restore 路径本身

换句话说，最应该怀疑的是：
- `build_vividvr_connector_control_states()`
- `restore_vividvr_connector_global_control_states()`
- 以及 packed eager-global 路径里混在一起的 `cast / conditioning_scale / gather / unbind` 顺序

## 4. 为什么速度会提升，但质量没有对齐 `v2`

### 4.1 速度提升是真的，而且来源很清楚

`v2` 慢，主要慢在两层：

1. 显式 global control 恢复
- `all_gather`
- 大张量物化
- `contiguous / view / cast / unbind`

2. connector attention 自身变贵
- local `q`
- 对 full global `k/v`
- `kv_len` 相比 local-only 显著增长

所以只要一个 patch 成功减少了下面任一部分：
- 显式 global tensor 驻留
- `all_gather -> materialize -> read again`
- 重复 `cast / reshape / contiguous`
- attention 读 full global `k/v` 的方式

速度就会明显上去。这也是为什么这么多失败版都能稳定打到 `14.9 ~ 15.2s/it`。

### 4.2 质量对不齐 `v2`，说明省下来的正是 `v2` 最关键的语义成本

`v2` 真正保住质量的合同很具体：
- `Connector` 的 attention 必须吃 `global_control`
- `Connector` 的 `c_mlp` 只吃与本 rank token 对齐的 `local_control`

也就是说，`v2` 的关键不是“最后有个 global shape”，而是：
- 这份 global control 必须和已验收 `v2` 的数值语义一致
- local/global 的构造顺序、dtype、scale 位置、视图关系不能被随意改掉

一旦下面这些环节的顺序被改坏：
- gather 前 cast 还是 gather 后 cast
- `conditioning_scale` 是逐层乘，还是 packed 后统一乘
- local/global 是独立张量，还是来自同一份 packed tensor view
- packed gather 是否与逐层 restore 完全等价

最终 formal 就会重新表现得像 fastest native `SP`，即：
- 局部 guidance 变弱
- seam 邻域重新出问题
- `ssim_min` 掉回 `0.91` 附近

## 5. 当前最可信的根因判断

当前最可信的根因不是“distributed exact 这条路天然不行”，而是：

**我们在尝试优化 `v2` 时，把 control-state 构造/restore 路径里的数值合同改掉了。**

更具体地说，最可疑的区域是：
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`

重点不是函数名本身，而是这几类操作的先后关系：
- `stack`
- `cast`
- `conditioning_scale`
- `all_gather`
- `unbind`
- local/global tuple 的组装方式

当前不太像主根因的方向：
- `Phase D` stitch 主逻辑
- 双卡 runtime
- `FA / compile` 普通数值波动
- 仅仅是 `Connector.forward()` 的 block 内调用方式

## 6. 后续不要再踩的坑

### 6.1 不要再把 `2-step` 和 `20-step reference` 的对比当成正式质量结论

`2-step` 只能做：
- 速度 smoke
- 明显语义崩坏排雷

不能做：
- 正式质量验收
- 最终 `ssim_min` 结论

正式 compare 必须固定：
- `130f / 20 step / seed=42`
- reference 仍是 `20-step` 原版视频

### 6.2 不要把 `pass_compare = true` 当成通过

当前 compare 阈值仍然很宽松：
- `min_ssim = 0.9`

所以 `0.912` 也会过，但这显然不是 `native SP` 质量闭环。

### 6.3 不要再把多种语义变化绑进同一个 patch

后续任何一轮都不应同时混入：
- chunked restore
- packed transport
- cast 位置变化
- `conditioning_scale` 位置变化
- local/global view 共享策略变化
- connector attention 路径变化

否则一旦 formal 掉质，就无法定位是哪个环节破坏了 `v2`。

### 6.4 只要 formal 再次回到 `0.964 / 0.912` 这一簇，就应直接止损

这类结果已经足够说明：
- 当前 patch 还在同一个坏语义上打转
- 不值得继续围绕它堆更多实现细节

## 7. 新的推荐主线

新的主线不是“继续 invent 一个更快的 attention 版本”，而是：

**先把 `v2` 的张量级数值合同锁死，再只优化通信组织和执行调度。**

如果这条线仍然无法把速度压到值得 formal 的范围，再把 distributed exact connector attention 作为第二阶段备线。

## 8. 详细执行计划

### 阶段 A：先建立 `v2` 的张量级真值合同

目标：
- 先回答“什么叫和 `v2` 完全等价”
- 不再靠整片视频猜语义是否对齐

至少要锁死三个 tap point：

1. `controlnet_inter_states` 原始输出
2. `build_vividvr_connector_control_states()` 产出的 `(local_control, global_control)` tuple
3. `Connector.forward()` 在单层上的输出

每个 tap point 都应记录并比较：
- shape
- dtype
- `max_abs_diff`
- `mean_abs_diff`
- 必要时的 cosine / relative error

这一步的验收标准不是“看起来差不多”，而是：
- 先和当前已验收 `v2` 的实现逐点对齐
- 确认哪些差异只是无害重排，哪些差异已经足以改变正式质量

### 阶段 B：把 current packed eager-global 路径拆成原子变量

目标：
- 找到到底是哪一个具体变化破坏了 `v2`

必须逐个单独验证的变量：
- `target_dtype` 是 gather 前 cast 还是 gather 后 cast
- `conditioning_scale` 是逐层单独乘，还是 packed 后统一乘
- `local_control` 是否直接来自 packed tensor 的 `unbind view`
- `global_control` 是否直接来自 packed gather 的 `unbind view`
- `stack + gather + unbind` 是否与逐层 restore 数值完全一致

这里的核心纪律是：
- 一次只改一项
- 每改一项先过张量等价测试
- 不通过就不继续叠加后续优化

### 阶段 C：先做“exact packed transport”，不要先改数学

这是当前最推荐的第一条提速线。

目标：
- 只优化通信组织方式
- 不改变任何算术顺序
- 不改变 `(local_control, global_control)` 的数值语义

正确做法应该是：
- pack 只作为运输容器
- 仍然按已验收 `v2` 的顺序完成 `scale / cast / restore`
- 最终解包出来的每层 local/global tensor 与 `v2` 基线逐点对齐

这一步不要求一次性打到 `15s/it`，但要求：
- 先证明“transport 可以更省，但 tensor 结果不变”

### 阶段 D：在数值合同锁死后，再做异步预取和短生命周期 restore

这一步仍然属于低风险主线。

目标：
- 不减少任何应该看到的 global context
- 只减少等待时间、长期驻留和无意义的同步点

推荐方向：
- 下一 chunk 的 global restore 放到独立通信流预取
- 当前 chunk connector compute 与下一 chunk gather 尽量 overlap
- 用双缓冲或短生命周期 buffer 降低 global tensor 驻留时间

这一阶段允许改：
- 调度时机
- buffer 生命周期
- overlap 方式

这一阶段不允许改：
- `Connector` 的数学合同
- local/global 的张量值

### 阶段 E：只有低风险主线不够时，才进入 distributed exact connector attention

这条线仍然值得做，但不应再作为下一步第一选择。

它的正确目标是：
- local `q`
- 各 rank 保持 local `k/v shard`
- 通过 exact softmax 归约得到与 full-global attention 数学等价的结果
- 不显式物化完整 `global_control`

为什么它要放到第二阶段：
- 调试难度更高
- 一旦没有先锁死 `v2` 真值合同，很容易再次落回 `0.964 / 0.912` 坏簇，却不知道是数学错了还是 transport 错了

进入这条线之前必须满足：
- 阶段 A-D 已经建立了可靠的张量级对齐框架
- 已知低风险主线即使成立，也仍然无法把 denoise 压到目标区间

### 阶段 F：正式验收只在速度 smoke 达标后再启动

当前新的速度 gate 已经放宽为：
- steady-state denoise 快于 `18.0s/it`

执行规则：
- `2-step smoke` 只看速度和明显异常
- 排除 warmup 后，如果持续快于 `18.0s/it`，才值得进入正式 `20-step` 验收
- 如果稳定慢于 `18.0s/it`，默认先不跑 formal，继续优化或换策略

## 9. 推荐的验证顺序

### 9.1 单元 / 张量级验证

优先新增或补强：
- `build_vividvr_connector_control_states()` 等价测试
- `restore_vividvr_connector_global_control_states()` 等价测试
- 单层 `Connector.forward()` 对齐测试
- local/global tuple 的 shape、dtype、数值合同测试

这层验证的目标是：
- 在不跑整片视频的情况下直接发现语义偏移

### 9.2 `2-step` smoke

作用只保留两点：
- 看 steady-state denoise 是否已经快于 `18.0s/it`
- 看日志和中间 snapshot 是否完整

禁止把它当成：
- 正式质量 compare
- 最终 `ssim_min` 结论

### 9.3 `20-step` formal

正式验收仍固定为：
- `130f / 20 step / seed=42`
- reference 不变
- backend 不变
- compile 配置不变

formal 结论必须同时看：
- 速度指标
- 全局质量指标
- seam 专项窗口

## 10. 当前建议的验收门槛

### 10.1 值得跑 formal 的 smoke 门槛

- steady-state denoise `< 18.0s/it`

这是“是否值得正式验收”的门槛，不是最终成功定义。

### 10.2 正式质量门槛

建议至少同时满足：
- `ssim_mean >= 0.980`
- `ssim_min >= 0.970`
- 不重新出现 `frames 48-58` 和 `frames 93-104` 的集中坏簇
- 肉眼不再出现 `2s` 左右突发模糊和 `2s~3s` 间的 seam 跳变

### 10.3 正式速度门槛

建议至少满足：
- `model_inference_runtime_seconds` 明显优于已验收 `v2`
- steady-state denoise 明显快于 `19.11s/it`

当前最现实的阶段目标不是直接回到 fastest native `SP`，而是：
- 在保住 `v2` 质量的前提下，先稳定进入 `< 18.0s/it`

## 11. 文件优先级

后续真正开始改代码时，优先看：

第一优先级：
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`

第二优先级：
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`

第三优先级：
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

这里的含义是：
- 先解决 control-state 构造与 restore 的数值合同
- 再做执行时序和通信组织优化
- 最后才是更激进的 distributed exact attention 原型

## 12. 本文档对应的最终完成定义

当本文档对应工作完成时，至少应满足：

1. 已验收 `v2` 的质量合同被保住，不再掉回 `0.964 / 0.912` 坏簇。
2. 正式 `20-step` compare 回到：
   - `ssim_mean >= 0.980`
   - `ssim_min >= 0.970`
3. `frames 48-58` 与 `frames 93-104` 不再出现集中低 `ssim` 窗口。
4. 肉眼不再看到 `2s` 左右模糊和 seam 跳变。
5. steady-state denoise 进入 `< 18.0s/it`。
6. 总体 `model_inference_runtime_seconds` 明显优于当前已验收 `v2`。

如果只满足 5-6，不满足 1-4，应定义为：
- 提速成立
- 但 `v2` 质量闭环未完成

如果只满足 1-4，不满足 5-6，应定义为：
- 质量基线守住了
- 但本轮提速没有形成可接受收益

只有同时满足两侧，才能把这条路径定义为真正完成的 `Phase E native SP` 收口方案。
