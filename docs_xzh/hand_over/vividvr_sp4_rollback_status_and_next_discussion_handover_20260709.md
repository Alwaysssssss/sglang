# VividVR 四卡 SP 回退状态与下一轮方案讨论交接

更新时间：`2026-07-09 Asia/Shanghai`

## 1. 这份文档覆盖什么

这份 handover 面向下一位继续讨论 `VividVR 4 GPU / SP=4` 优化方案的 Codex，重点说明：

- 这轮四卡 SP 尝试到底做过什么
- 后来为什么认定该方向暂时不能继续沿用
- 当前 git / 分支 / stash 处于什么状态
- 已确认的 caption / `decord` / 复现实验结论
- 下一轮重新讨论 SP 方案时，哪些结论应直接继承，哪些问题仍未解

## 2. 当前总判断

当前不应继续基于上一轮“四卡 SP 优化 WIP”直接往前推。原因很简单：

1. 质量没有完成对齐
2. 速度没有变快，反而出现显著变慢
3. 当时新增的大部分 SP4 WIP 没有进入正式提交，只保存在 `stash`

因此，当前更合理的起点是：

- 站在回退后的稳定基线重新讨论四卡 `SP=4` 方案
- 只把已有的诊断信息、排查结论和验收口径继承下来
- 不默认复用之前那批未提交的 SP4 主修改

## 3. 本轮实际做过什么

### 3.1 已提交到分支的内容

当前本地 `sglang_Vivid` 比 `bc770d52b` 只多一个提交：

- `c7e395ad2 feat: add vividvr sp4 runtime observability`

这个提交只做了观测字段补齐，没有把四卡 SP 主优化正式落地。具体内容：

- 修改 `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
  - 在 runtime snapshot 中新增：
    - `sp_world_size`
    - `sp_rank`
    - `denoise_loop_local_compute_ms`
    - `denoise_loop_sp_comm_ms`
- 修改 `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_runtime_snapshot.py`
  - 增加对应单测 `test_runtime_snapshot_includes_sp4_fields`

这条提交的作用仅限于：

- 帮助后续判断四卡 SP 是算得慢，还是通信慢
- 给正式 benchmark JSON 补可观测字段

### 3.2 没有进入正式提交、只存在于 stash 的 SP4 WIP

当前有一个 stash：

- `stash@{0}: On sglang_Vivid: codex-sp4-wip-20260709`

这个 stash 中包含：

- 计划文档：
  - `docs_xzh/distribute/2026-07-08-vividvr-4gpu-sp-mainline-plan.md`
- 以及一批未提交代码改动，主要涉及：
  - `docs_xzh/run_vivid_benchmark.md`
  - `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
  - `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`
  - `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py`
  - `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py`
  - `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`

这些 stash 内改动不要默认视为“已验证可用”。它们只是上轮实验性 WIP。

## 4. 为什么后面决定回退

用户在这轮中明确给出的判断是：

- 之前的四卡 SP 优化“画质没对齐，时间也变慢了”
- 应该回退到 `bc770d52b` 重新讨论并行方案

这背后的事实判断包括：

1. 四卡产出没有满足当时要求的逐帧一致验收
2. 日志中出现过明显变慢的表现，例如 `23.40s/it`
3. 因此那版优化不能被认定为“只提速、不改语义”的合格主线

后续已经执行过“回退到 `bc770d52b` 再验收”的动作；后来用户又决定：

- 不再要求两次独立四卡推理逐帧完全一致
- 后续质量口径可以放宽到“几乎一致即可”

这个放宽是用户最新决策，下一轮讨论 SP 方案时应按这个新口径理解，不要再默认回到“独立重跑必须逐帧 bitwise 一致”。

## 5. 当前 git 状态

截至本交接文档写入时，当前仓库状态是：

- 当前分支：`sglang_Vivid`
- `git status -sb`：
  - `## sglang_Vivid...origin/sglang_Vivid [ahead 1]`
- 本地 `sglang_Vivid` 指向：
  - `c7e395ad2`
- 远端 `origin/sglang_Vivid` 指向：
  - `bc770d52b`
- `Vivid_VR_online` 也指向：
  - `bc770d52b`

这意味着：

1. 当前工作树是干净的
2. 当前代码不包含那批 stash 里的 SP4 主修改
3. 但当前 `sglang_Vivid` 也不完全等于 `bc770d52b`
4. 它只比 `bc770d52b` 多了一个很小的 observability 提交 `c7e395ad2`

因此，若下一个 Codex 想从“最干净的回退基线”出发，应先明确：

- 是接受当前 `sglang_Vivid` 上的 `c7e395ad2`
- 还是要把 `sglang_Vivid` 也挪回 `bc770d52b`

## 6. 本轮验收口径上需要继承的约束

后续只要再做 `--enable-torch-compile` 的正式 benchmark，必须继续遵守下面这个口径：

1. 不要用第一次完整输入的耗时当正式结论
2. 第一次完整 `20 step` 的运行承担 compile 冷启动开销
3. 真正用于记录的 `total_runtime_seconds` 和 `model_inference_runtime_seconds` 必须取同配置下第二次完整 `20 step` 结果

用户后来还特别澄清过一点：

- 这里不是打开某个 `warmup` 开关跑 1 step
- 而是同一份代码、同一套环境下，先完整跑一轮 `20 step`，再完整跑第二轮 `20 step`
- 第二轮才算正式时延

下一轮如果继续写计划或验收文档，必须把这条直接写进验收标准。

## 7. decord / caption 相关结论

### 7.1 `decord` 报错的真实根因

本轮已确认：

- 主推理环境 `/home/zhiheng/sglang/.venv` 之前缺少 `decord`
- caption sidecar 环境 `/home/zhiheng/sglang/.venv-vividvr-caption` 本来就有 `decord==0.6.0`

因此，先前看到的：

- `No module named 'decord'`

并不是 caption sidecar 环境损坏，而是主环境缺包。

本轮已经做过的环境修复：

- 在 `/home/zhiheng/sglang/.venv` 中安装了 `decord==0.6.0`

注意：

- 这是环境修改，不会体现在 git diff 中
- 下一个 Codex 如果复核环境，请不要因为 `git status` 干净就误以为这一点没有发生

### 7.2 `cartoon.mp4` caption 与 other server 是否一致

本轮已经重新验证过：

- 输入视频：
  - `/home/zhiheng/input/cartoon.mp4`
- 对比文件：
  - `/home/zhiheng/sglang/Vivid_Acceptance/captions/cartoon_caption_comparison_20260708.txt`
- 重新生成 caption 产物：
  - `/home/zhiheng/sglang/Vivid_Acceptance/captions/kernel_probe/cartoon_recheck_decord_20260709.txt`
- 对应日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/cartoon_caption_recheck_20260709T062542Z.log`

验证结论：

1. 强制使用 `decord` 重新生成后，caption 仍然不等于 `Caption from the other server`
2. 新生成 caption 等于同文件里的 `Current sglang caption sidecar`
3. 因此，“decord 回退到 cv2”不是这次跨机器 caption 差异的原因

也就是：

- 当前机器内部 `decord / cv2 / direct path` 的 caption 路径是收敛的
- 但当前机器与 other server 的 caption 仍然不一致

## 8. 当前未解问题

### 8.1 两次独立四卡运行为什么不是逐帧完全一致

这个问题在本轮后半段被用户主动降级了优先级：

- 用户最新判断是不再要求严格逐帧一致，只要求“几乎一致即可”

因此，本轮没有把这个问题继续追到最终根因。不要在下一轮 handover 中误写成“已经确认是 generator 不同导致”。

当前只能说：

- 该根因在本轮没有最终定论
- 但它已经不是下一轮 SP 方案讨论必须先解决的问题

### 8.2 四卡 SP 变慢的根因仍需重新拆分

虽然已经有 `c7e395ad2` 补上的 runtime observability 字段，但这一轮没有在干净基线之上完成新的四卡 SP 提速闭环。

下一轮如果要重新讨论四卡 SP，应优先回答：

1. 变慢主要来自：
   - 本地 compute
   - SP 通信
   - connector/control 的全局 gather
   - tile / local shard 布局不均衡
2. 是否需要延续 stash 中那套“先观测、再缩 gather、再调 tile/shard”的思路
3. 还是应该彻底换一条更保守的 SP 方案

## 9. 下一轮建议工作方式

建议下一位 Codex 先做下面几件事，再开始写新方案：

1. 先确认是否要保留 `c7e395ad2`
   - 如果只想保留干净回退基线，先明确要不要把 `sglang_Vivid` 也挪回 `bc770d52b`
2. 只把旧 plan 当作“思路草稿”
   - 不要默认 `stash@{0}` 里的 WIP 已经被验证过
3. 把验收口径直接改写为用户最新决定
   - 时延取第二次完整 `20 step`
   - 质量允许“几乎一致”，不再强求独立重跑逐帧完全一致
4. 如果还想参考旧计划文档
   - 可以单独从 `stash@{0}` 恢复：
     - `docs_xzh/distribute/2026-07-08-vividvr-4gpu-sp-mainline-plan.md`
   - 但不要直接 `git stash apply` 整包恢复再继续写代码，除非明确要接着做那批 WIP

## 10. 建议优先阅读的文件

- `docs_xzh/add_strategy/README.md`
- `docs_xzh/add_strategy/12_phase_e_sp_native_acceleration_plan.md`
- `docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`
- `docs_xzh/hand_over/phase_e_e41_native_sp_v2_quality_control_and_next_speedup_handover.md`
- `docs_xzh/hand_over/phase_e_e41_sp_quality_regression_deep_dive_handover.md`
- `docs_xzh/run_vivid_benchmark.md`
- `docs_xzh/hand_over/vividvr_sp4_rollback_status_and_next_discussion_handover_20260709.md`

## 11. 一句话总结

这轮四卡 SP 没有形成可继续沿用的主线实现；当前真正可继承的是：

- 回退后的稳定基线
- 一条很小的 observability 提交 `c7e395ad2`
- stash 中尚未验证完成的旧 SP4 plan / WIP
- 以及已经确认的 `decord` / caption 差异排查结论

下一轮应该站在这个状态上，重新讨论四卡 SP 的优化方案，而不是默认沿用之前那批未提交实验改动。
