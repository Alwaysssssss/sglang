# AGENTS.md

本文件定义 `/home/zhiheng/sglang` 仓库范围内对 Codex 和其他代码代理的统一要求。

## 适用范围

- 除非更深层目录存在新的 `AGENTS.md`，否则这些规则适用于整个仓库。

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
- 除非用户明确要求，或 `.venv` 本身已损坏且无法继续使用，否则不要切换到临时 `uv run --with ...` 环境、其他 `.venv`，或系统 Python。
- 如果因为环境问题无法继续，先说明问题，再处理环境，不要静默切换到别的解释器。

## 在 tmux 中做推理验收

- 每次启动推理验证、验收验证或任何长时间生成命令时，必须放在 `tmux` session 或 window 中运行。
- 这样做是为了让用户可以实时 attach 查看进度，这条规则是强制的。
- 当 `tmux` 可用时，不要只用普通阻塞式 shell 命令直接跑推理验收。
- `tmux` session 名称要清晰，能够反映任务，例如 `vividvr_phase_c` 或 `sglang_eval`。
- 启动后要告知用户 session 名和准确的 attach 命令，例如 `tmux attach -t vividvr_phase_c`。
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

- 当前项目默认以 `Phase C` 单次验收命令作为标准推理命令，后续 Codex 进行 `Vivid-VR` 推理或基线验收时，默认从这条命令开始。
- 在 `tmux` 中启动的推荐命令如下：

```bash
tmux new-session -d -s vividvr_phase_c \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_phase_c_single.py 2>&1 | tee Vivid_Acceptance/logs/phase_c_single_$(date -u +%Y%m%dT%H%M%SZ).log'
```

- attach 命令：

```bash
tmux attach -t vividvr_phase_c
```

- 如果后续推理参数、脚本路径、环境路径或日志路径发生变化，必须同步修改本文件中的标准推理命令，避免继续引用过时命令。

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
