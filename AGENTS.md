# AGENTS.md

本文件定义 `/home/zhiheng/sglang` 仓库范围内对 Codex 和其他代码代理的统一要求。

## 适用范围

- 除非更深层目录存在新的 `AGENTS.md`，否则这些规则适用于整个仓库。

## 项目背景

- 当前工作的主线不是从零设计一个新模型，而是把原版 `/home/zhiheng/Vivid-VR` 以原生、可维护的方式集成到 `sglang.multimodal_gen` 中。
- 这条线经历过一次仓库误清空和恢复；`Phase A / B / C` 已恢复并建立了稳定基线，当前继续推进的是 `Phase D` 和 `Phase E`。
- 当前稳定基线是 `Phase C` 单 clip 路径；后续所有改动默认都要保护这条基线，避免回归。
- `Phase D` 的重点是长视频 `clip split / merge / temporal orchestration` 以及公平 benchmark；截至目前，这部分代码和 benchmark 流程已具备，并已完成正式验收，作为 `Phase E` 的长视频语义基线。
- `Phase E` 的重点不是再发明新语义，而是在 `Phase D` 语义对齐基础上做性能收口、默认配置收口和回归验收，逐步进入 release gate。
- 当前 `Phase E` 的 `130f / 20 step` 长视频 `serve` benchmark 与加速消融已经完成，默认配置已经收口到单卡 `single_gpu_fa_compile` 和双卡 `dual_gpu_fa_eager_compile`。
- 当前仍未完成的主问题有两条：
  - 原版 `/home/zhiheng/Vivid-VR` 的 caption 目前只能在原版环境中稳定正确产出；在 `sglang` 的 `.venv` 中会因依赖版本差异导致 caption 输出异常，后续可能需要补一条“通信交换生成 caption”的桥接路径。
  - `serve` 服务接口的部分输入参数契约仍需按需求继续收口，当前不能默认视为已经完全稳定。
- 下一步优先任务是解决原版 caption 模型环境不兼容问题；默认方向应是建立独立 caption bridge 或 sidecar 生成路径，而不是直接改坏 `/home/zhiheng/sglang/.venv` 的主推理依赖。
- `Vivid-VR` 在 `sglang` 中必须作为原生模型集成运行；推理时不要依赖原版仓库的运行时代码。
- 允许继续复用原版仓库中的外部资源，例如：
  - checkpoint
  - 输入视频
  - `prompt.txt`
  - caption sidecar 基线文件
  - 原版 reference 视频

## 当前阶段任务

- `Phase C` 代表当前单 clip 稳定基线，目标是把原版 `Vivid-VR` 的单段视频编辑主链语义稳定迁移到 `sglang` 中，并通过现有单 clip 验收。
- `Phase C` 必须守住的关键语义包括：
  - prompt 默认来自 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
  - 不走 live `CogVLM2`
  - `prompt_embed_shape` 保持 `226` 长度，不回退到 `512`
  - VAE tiling 默认值保持 `240 / 360`
  - preprocess 保留未 padding 的 `reference_video`
  - decode / postprocess 保留 `drop first 3 frames + crop padding + AdaIN/reference color fix`
- `Phase D` 不是单纯“优化”，而是要继续对齐原版 `Vivid-VR` 的长视频主语义。
- `Phase D` 的核心任务包括：
  - 长视频 `clip split`
  - 多 clip 的 timestep 级时序编排
  - 跨 clip latent merge
  - clip trim / stitch
  - 使用原版 caption sidecar 的公平 benchmark
- 对 `Phase D` 的默认理解应是“补齐原版长视频语义”，不是随意做一个能跑的近似实现。
- 目前 `Phase D` 代码路径、helper、测试和 benchmark 工具已经具备，且长视频公平验收已完成；后续默认应在该语义基线上推进 `Phase E`，不要破坏 `Phase C` 与 `Phase D` 的已验收结果。
- 如果本轮任务没有明确要求推进 `Phase D`，默认仍应先保护 `Phase C` 已验收结果。
- `Phase E` 代表“性能收口 + 回归验收”阶段，不应被理解成脱离语义基线的随意调参。
- `Phase E` 的核心任务包括：
  - 收口默认推理配置，例如 `dtype`、attention backend、VAE tiling / slicing、offload 策略、是否启用 compile
  - 基于稳定实现做 profile，形成可复用的性能结论
  - 建立可重复运行的 regression 套件
  - 让验收逐步从阶段性对齐进入 strict 或接近 strict 的 release gate
- 当前 `Phase E` 日常 benchmark 默认固定为与 `Phase D` 相同 reference 对象的 `130f / 20 step` 长视频口径；`50 step` 只保留给阶段性最终回归。
- 当前 `Phase E` 单卡默认正式配置固定为 `single_gpu_fa_compile`，也就是 `--attention-backend fa` + `--enable-torch-compile`。
- 当前双卡 `SP` 默认质量口径要求 connector context mode 走 `eager_global`，并保持 `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`；也就是默认恢复 full global control context，且默认不启用 control pooling。只有在明确做历史 `v1` 对比或性能实验时，才显式切回 `deferred_global` 或打开 pool 压缩。
- 当前 `Phase E` 双卡默认正式配置固定为 `dual_gpu_fa_eager_compile`，也就是 `--attention-backend fa` + `SP=2` + `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global` + `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1` + `--enable-torch-compile`；在双卡 `SP` 下运行时有效 backend 记为 `fa_sp`。
- 单卡正式 benchmark 或正式对比时，必须保证同一时刻只有一个单卡推理进程在跑，避免并发占用把单卡耗时拉长，造成不公平对比。
- 推进 `Phase E` 时，默认前提是不能破坏 `Phase C` 已验收基线，也不要用性能优化引入新的长视频语义回归。
- 如果任务明确属于 `Phase E`，先确认当前 benchmark 和验收口径是否已经固定；如果默认参数、后端或回归指标发生变化，必须同步更新文档和 `AGENTS.md`。

## 文档入口与实现指引

- 在开始实现、重构或验收前，优先阅读 `/home/zhiheng/sglang/docs_xzh/add_strategy` 下的文档；这些文档是当前集成路线的规划合同，不要脱离这些文档自行改架构。
- 推荐至少按下面顺序建立上下文：
  - `docs_xzh/add_strategy/README.md`：总览
  - `docs_xzh/add_strategy/02_stage1_sglang_mapping.md`：原版 `Vivid-VR` 到 `sglang` 的语义映射
  - `docs_xzh/add_strategy/03_stage2_mvp_scope.md`：MVP 和阶段边界
  - `docs_xzh/add_strategy/04_stage3_pipeline_mod_plan.md`：pipeline 改造方向
  - `docs_xzh/add_strategy/05_stage4_component_migration.md`：组件迁移范围
  - `docs_xzh/add_strategy/09_code_mod_order.md`：推荐改动顺序
  - `docs_xzh/add_strategy/10_grouped_stage_acceptance.md`：分阶段验收要求
- 新增或更新交接文档时，统一保存在 `/home/zhiheng/sglang/docs_xzh/hand_over` 下；不要把 handover 文档散落到仓库根目录、`docs_xzh/add_strategy` 或临时目录。
- 如果任务与当前实现状态、恢复背景或上一轮未完成问题有关，优先查看 `/home/zhiheng/sglang/docs_xzh/hand_over` 下最新的交接文档，不要忽略历史上下文。
- 当前最重要的交接文档包括：
  - `docs_xzh/hand_over/phase_abc_restore_and_next_stage_handover.md`
  - `docs_xzh/hand_over/phase_d_modular_refactor_and_fair_benchmark_handover.md`
  - `docs_xzh/hand_over/phase_d_acceptance_completion_and_phase_e_benchmark_handover.md`
  - `docs_xzh/hand_over/phase_e_e0_e3_acceptance_and_single_gpu_combo_handover.md`
  - `docs_xzh/hand_over/phase_e_default_configs_and_serve_followups_handover_20260622.md`
  - `docs_xzh/hand_over/flowcut_vividvr_service_compat_handover_20260622.md`
  - `docs_xzh/hand_over/vividvr_service_external_access_and_caption_next_handover_20260622.md`
- 如果任务涉及 modular 化方向，额外参考：
  - `docs_xzh/modular_style/vividvr_modular_refactor_plan.md`
- 如果任务涉及 `Phase D` 长视频对齐，优先参考：
  - `docs_xzh/add_strategy/02_stage1_sglang_mapping.md`
  - `docs_xzh/add_strategy/03_stage2_mvp_scope.md`
  - `docs_xzh/add_strategy/10_grouped_stage_acceptance.md`
  - `docs_xzh/hand_over/phase_d_modular_refactor_and_fair_benchmark_handover.md`
- 如果任务涉及 `Phase E` 性能收口、默认配置、compile / offload / backend 选择或回归门槛，优先参考：
  - `docs_xzh/add_strategy/08_stage7_execution_roadmap.md`
  - `docs_xzh/add_strategy/10_grouped_stage_acceptance.md`
  - `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`
  - `docs_xzh/run_vivid_benchmark.md`
- 如果任务涉及 benchmark、原版公平对比、caption sidecar 或验收命令，额外参考：
  - `docs_xzh/run_vivid_benchmark.md`
- 默认要求是“先对齐文档约束，再动代码”；如果代码现状与文档不一致，应先查明这是已知偏差、未完成阶段，还是新的回归，而不是直接按个人判断改动。

## 文件安全

- 不要随意删除文件或目录。
- 不要因为文件看起来没用就直接删除，必须先检查引用关系并确认删除原因。
- 除非任务明确要求，否则不要用生成内容覆盖或替换大文件。
- 如果确实必须删除文件，优先做最小化改动，并在最终总结中说明原因。
- 对用户创建的文档、prompt、日志、验收产物和本地研究资料默认一律保留，除非用户明确要求移除。

## Git 与工作区安全

- 除非用户明确要求，否则绝不使用 `git reset --hard`、`git clean -fd`、`git checkout --` 这类破坏性命令。
- 不要回退不是自己做出的修改。
- 默认假设工作区可能是脏的；只隔离并处理本次任务相关改动，不碰无关修改。
- 在进行较大范围修改前，先检查 `git status`，明确当前工作区状态。

## 默认 Python 与 uv 环境

- 默认使用仓库根目录的 `/home/zhiheng/sglang/.venv` 作为统一 Python 环境。
- 默认使用 `/home/zhiheng/sglang/.venv/bin/python` 运行脚本、测试和推理验收。
- 默认使用 `uv pip install --python /home/zhiheng/sglang/.venv/bin/python ...` 向该环境补依赖。
- 唯一的 repo 内运行时例外是 Vivid-VR caption sidecar；它必须使用独立环境 `/home/zhiheng/sglang/.venv-vividvr-caption`，并通过 `python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh` 创建。
- 除非用户明确要求，或 `.venv` 本身已损坏且无法继续使用，否则不要切换到临时 `uv run --with ...` 环境、其他 `.venv`，或系统 Python。
- 如果因为环境问题无法继续，先说明问题，再处理环境，不要静默切换到别的解释器。
- 唯一的基准对比例外是“运行原版 `/home/zhiheng/Vivid-VR` 做公平对比”时，必须使用原版本身的 `/home/zhiheng/Vivid-VR/.venv/bin/python`，不要用 `sglang` 的 `.venv` 代跑原版。
- 做原版 `Vivid-VR` 公平对比时，必须优先保证原版 caption 语义正确；如果 `sglang` 环境里的 `transformers` 版本会导致 `CogVLM2` caption 异常，禁止继续用 `sglang` 的环境跑原版。
- 解决 caption 环境不兼容时，禁止为了让 caption 在 `sglang/.venv` 内运行而随意降级或替换主推理依赖；应保持 sidecar 作为 `sglang` 仓库内代码启动的独立 HTTP 服务，并固定使用 `/home/zhiheng/sglang/.venv-vividvr-caption`。
- caption bridge 的输出应统一保存为 sidecar caption 文件，并由 `sglang` 原生 Vivid-VR 推理链消费；不要让 `sglang` 推理运行时直接依赖原版仓库的运行时代码。
- caption bridge 验收至少要确认 sidecar 行数、顺序与 temporal clip 切分一致，并用已有 Phase C/D/E 轻量回归确认没有破坏主推理路径。

## 在 tmux 中做推理验收

- 每次启动推理验证、验收验证或任何长时间生成命令时，必须放在 `tmux` session 或 window 中运行。
- 这样做是为了让用户可以实时 attach 查看进度，这条规则是强制的。
- 当 `tmux` 可用时，不要只用普通阻塞式 shell 命令直接跑推理验收。
- `tmux` session 名称要清晰，能够反映任务，例如 `vividvr_phase_c` 或 `sglang_eval`。
- 启动后要告知用户 session 名和准确的 attach 命令；默认优先给只读查看命令，例如 `tmux attach -r -t vividvr_phase_c`，避免终端误发 `Ctrl-C` 中断推理。
- 如果验证会写日志或产物，放在仓库内可预测的位置，并在最终总结里说明路径。

## 验收产物与标准格式

- `Vivid-VR` 集成线的验收指标统一保存在 `/home/zhiheng/sglang/Vivid_Acceptance/indicator`。
- 对应推理结果视频统一保存在 `/home/zhiheng/sglang/Vivid_Acceptance/result_videos`。
- 当前验收指标 JSON 的标准结构以 `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T090642Z.json` 为基准。
- 后续新产出的验收 JSON 必须至少保持与该基准同等字段集合，并额外包含两个时间字段：
  - `total_runtime_seconds`：程序从开始到结束的总用时。
  - `model_inference_runtime_seconds`：纯模型推理阶段用时，默认按 `pipeline.forward(...)` 对应时段统计。
- 如果后续因为需求调整导致验收 JSON 字段发生变化，必须同步更新本文件中的格式说明。

## 标准推理命令

- `Phase C` 单 clip 回归仍以这条单次验收命令作为标准回归入口。
- 当前 `Phase E` 长视频默认配置已经固定为：
  - 单卡：`single_gpu_fa_compile`
  - 双卡：`dual_gpu_fa_eager_compile`
- 对应的单卡/双卡直接运行命令、`serve` 拉起命令和 `curl` 请求命令统一维护在 `/home/zhiheng/sglang/docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`。
- 在 `tmux` 中启动的推荐命令如下：

```bash
tmux new-session -d -s vividvr_phase_c \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_phase_c_single.py 2>&1 | tee Vivid_Acceptance/logs/phase_c_single_$(date -u +%Y%m%dT%H%M%SZ).log'
```

- attach 命令：

```bash
tmux attach -r -t vividvr_phase_c
```

- 如果后续推理参数、脚本路径、环境路径或日志路径发生变化，必须同步修改本文件中的标准推理命令，避免继续引用过时命令。
- 如果是此前没有做过公平对比的新视频，先用原版 `/home/zhiheng/Vivid-VR/.venv/bin/python` 跑原版结果，再从原版日志中提取每个 temporal clip 的原始 caption，逐行保存到 `/home/zhiheng/Vivid-VR/input/captions/<video_stem>.txt`。
- 之后再让 `sglang` 版本通过 `--caption-file /home/zhiheng/Vivid-VR/input/captions/<video_stem>.txt` 重跑；caption 文件必须一行一个 clip caption，顺序与原版生成顺序一致。

## 验收与提交纪律

- 优先做能覆盖本次改动的最小有效验证，但不能跳过明确要求的验收。
- 没有真实跑通验证前，不要声称验证通过。
- 如果跳过重型验证，必须明确说明原因。
- 对“阶段性修改”这类工作，一旦达到该阶段的验收要求，应当及时提交本阶段改动，并推送到远端。
- 阶段性 `commit` 应只包含本阶段相关改动，避免把无关文件混入同一个提交。
- 在执行 `push` 前，确认验收结果、分支目标和提交内容都正确。

## 改动纪律

- 改动范围要紧贴任务目标。
- 除非为安全完成任务所必需，否则避免顺手做无关重构。
- 除非任务明确要求改变行为，否则要保持现有运行时语义稳定。
- 在分阶段集成、恢复或交接工作中，优先保护最近一次已验收基线，避免回归。

## 沟通要求

- 开始较大改动前，先简要说明准备修改什么。
- 开始长时间推理验收前，先说明会在 `tmux` 中启动。
- 最终总结中要包含：改了哪些文件、运行了哪些验证、是否完成验收，以及需要注意的后续风险。
