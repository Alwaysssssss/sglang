# 问题跟踪器：GitHub

当前仓库的问题单和 PRD 存放为 GitHub Issues。所有操作都使用 `gh` CLI。

## 惯例

- **创建问题单：**`gh issue create --title "..." --body "..."`。多行正文使用 heredoc。
- **读取问题单：**`gh issue view <number> --comments`，通过 `jq` 过滤评论并同时获取标签。
- **列出问题单：**`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，并使用适当的 `--label` 和 `--state` 过滤器。
- **评论问题单：**`gh issue comment <number> --body "..."`
- **添加或删除标签：**`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭：**`gh issue close <number> --comment "..."`

从 `git remote -v` 推断仓库；在克隆目录内运行时，`gh` 会自动完成推断。

## 是否把拉取请求作为分诊入口

**把 PR 作为请求入口：no。**（如果当前仓库把外部 PR 视为功能请求，将其设为 `yes`；`/triage` 会读取此标志。）

设为 `yes` 时，PR 使用与问题单相同的标签和状态，并采用对应的 `gh pr` 命令：

- **读取 PR：**`gh pr view <number> --comments`，并用 `gh pr diff <number>` 获取差异。
- **列出待分诊的外部 PR：**`gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`，之后只保留 `authorAssociation` 为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR` 或 `NONE` 的项，排除 `OWNER`、`MEMBER`、`COLLABORATOR`。
- **评论、添加标签、关闭：**`gh pr comment`、`gh pr edit --add-label`/`--remove-label`、`gh pr close`。

GitHub 的问题单和 PR 共用编号空间，因此单独的 `#42` 可能表示任意一种。先用 `gh pr view 42` 解析，失败后改用 `gh issue view 42`。

## 技能要求“发布到问题跟踪器”时

创建 GitHub 问题单。

## 技能要求“获取相关任务单”时

运行 `gh issue view <number> --comments`。

## 路径探索操作

供 `/wayfinder` 使用。**地图**是单个问题单，**子问题单**作为任务单。

- **地图：**带 `wayfinder:map` 标签的单个问题单，正文包含“笔记 / 已有决策 / 迷雾”。使用 `gh issue create --label wayfinder:map`。
- **子任务单：**作为 GitHub 子问题单链接到地图的问题单，通过子问题单端点调用 `gh api`。未启用子问题单时，将子项加入地图正文的任务列表，并在子项正文顶部写入 `Part of #<map>`。标签为 `wayfinder:<type>`，其中类型是 `research`、`prototype`、`grilling` 或 `task`。领取后，将任务单分配给驱动工作的开发者。
- **阻塞：**使用 GitHub **原生问题单依赖**作为规范且界面可见的表示。通过 `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>` 添加边，其中 `<blocker-db-id>` 是阻塞项的内部数值 ID，通过 `gh api repos/<owner>/<repo>/issues/<n> --jq .id` 获取，而不是 `#number` 或 `node_id`。GitHub 的 `issue_dependencies_summary.blocked_by` 只报告未关闭阻塞项，是实时门禁。依赖不可用时，在子项正文顶部使用 `Blocked by: #<n>, #<n>` 作为后备。所有阻塞项关闭后，任务单解除阻塞。
- **前沿查询：**列出地图中未关闭的子项（`gh issue list --state open`，范围限定到地图子问题单或任务列表），排除存在未关闭阻塞项（`issue_dependencies_summary.blocked_by > 0`，或 `Blocked by` 行中存在未关闭问题单）或已有受理人的项；按地图顺序选择第一项。
- **领取：**运行 `gh issue edit <n> --add-assignee @me`，这是会话中的第一次写操作。
- **解决：**运行 `gh issue comment <n> --body "<answer>"`，再运行 `gh issue close <n>`，然后向地图的“已有决策”追加上下文指针（gist 和链接）。
