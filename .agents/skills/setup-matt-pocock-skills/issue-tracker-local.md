# 问题跟踪器：本地 Markdown

当前仓库的问题单和规格（也称 PRD）以 Markdown 文件形式存放在 `.scratch/`。

## 惯例

- 每个功能一个目录：`.scratch/<feature-slug>/`
- 规格文件：`.scratch/<feature-slug>/spec.md`
- 实现问题单每张任务单一个文件：`.scratch/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 开始编号，绝不合并成单个任务单文件。
- 分诊状态记录在每个问题单文件顶部附近的 `Status:` 行中，角色字符串参见 `triage-labels.md`。
- 评论和对话历史追加到文件底部的 `## 评论` 标题下。

## 技能要求“发布到问题跟踪器”时

在 `.scratch/<feature-slug>/` 下创建新文件，必要时创建目录。

## 技能要求“获取相关任务单”时

读取所引用路径的文件。用户通常会直接传入路径或问题单编号。

## 路径探索操作

供 `/wayfinder` 使用。**地图**是一个文件，每张任务单对应一个**子文件**。

- **地图：**`.scratch/<effort>/map.md`，正文包含“笔记 / 已有决策 / 迷雾”。
- **子任务单：**`.scratch/<effort>/issues/NN-<slug>.md`，从 `01` 开始编号，正文中写明问题。`Type:` 行记录任务单类型（`research`、`prototype`、`grilling`、`task`），`Status:` 行记录 `claimed` 或 `resolved`。
- **阻塞：**顶部附近使用 `Blocked by: NN, NN` 行。列出的所有文件都达到 `resolved` 后，任务单解除阻塞。
- **前沿：**扫描 `.scratch/<effort>/issues/` 中未关闭、未阻塞且未领取的文件，按编号选择第一项。
- **领取：**任何工作开始前将状态设为 `Status: claimed` 并保存。
- **解决：**在 `## 答案` 标题下追加答案，将状态设为 `Status: resolved`，再向 `map.md` 的“已有决策”追加上下文指针（gist 和链接）。
