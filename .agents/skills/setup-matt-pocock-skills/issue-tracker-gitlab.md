# 问题跟踪器：GitLab

当前仓库的问题单和 PRD 存放为 GitLab Issues。所有操作都使用 [`glab`](https://gitlab.com/gitlab-org/cli) CLI。

## 惯例

- **创建问题单：**`glab issue create --title "..." --description "..."`。多行描述使用 heredoc；传入 `--description -` 可打开编辑器。
- **读取问题单：**`glab issue view <number> --comments`。使用 `-F json` 获取机器可读输出。
- **列出问题单：**`glab issue list -F json`，并使用适当的 `--label` 过滤器。
- **评论问题单：**`glab issue note <number> --message "..."`。GitLab 将评论称为 note。
- **添加或删除标签：**`glab issue update <number> --label "..."` / `--unlabel "..."`。多个标签可以用逗号分隔，也可以重复传入标志。
- **关闭：**`glab issue close <number>`。该命令不接受关闭评论，因此应先用 `glab issue note <number> --message "..."` 发布说明，再关闭问题单。
- **合并请求：**GitLab 将 PR 称为合并请求。使用 `glab mr create`、`glab mr view`、`glab mr note` 等；其形式与 `gh pr ...` 相同，只需以 `mr` 代替 `pr`，以 `note`/`--message` 代替 `comment`/`--body`。

从 `git remote -v` 推断仓库；在克隆目录内运行时，`glab` 会自动完成推断。

## 是否把合并请求作为分诊入口

**把 MR 作为请求入口：no。**（如果当前仓库把外部合并请求视为功能请求，将其设为 `yes`；`/triage` 会读取此标志。）

设为 `yes` 时，MR 使用与问题单相同的标签和状态，并采用对应的 `glab mr` 命令：

- **读取 MR：**`glab mr view <number> --comments`，并用 `glab mr diff <number>` 获取差异。
- **列出待分诊的外部 MR：**`glab mr list -F json`，之后只保留作者不是项目成员或所有者的 MR，即贡献者的 MR，而不是维护者进行中的工作。
- **评论、添加标签、关闭：**`glab mr note`、`glab mr update --label`/`--unlabel`、`glab mr close`。

与 GitHub 不同，GitLab 分别为问题单和 MR 编号，因此只要明确维护者指的是哪种入口，`#42` 就没有歧义。

## 技能要求“发布到问题跟踪器”时

创建 GitLab 问题单。

## 技能要求“获取相关任务单”时

运行 `glab issue view <number> --comments`。

## 路径探索操作

供 `/wayfinder` 使用。**地图**是单个问题单，**子问题单**作为任务单。

- **地图：**带 `wayfinder:map` 标签的单个问题单，正文包含“笔记 / 已有决策 / 迷雾”。使用 `glab issue create --label wayfinder:map`。支持原生 epic 的 GitLab 套餐也可用 epic 承载地图；带标签的问题单适用于所有环境。
- **子任务单：**描述顶部包含 `Part of #<map>` 的问题单，并带 `wayfinder:<type>` 标签，其中类型是 `research`、`prototype`、`grilling` 或 `task`。领取后，将任务单分配给驱动工作的开发者。
- **阻塞：**使用 GitLab **原生阻塞链接**作为规范且界面可见的表示。以 note 形式发布 `/blocked_by #<n>` 快捷操作：`glab issue note <child> --message "/blocked_by #<blocker>"`。原生阻塞链接属于 Premium/Ultimate 功能；免费套餐或不可用时，在描述顶部使用 `Blocked by: #<n>, #<n>` 作为后备。所有阻塞项关闭后，任务单解除阻塞。
- **前沿查询：**运行 `glab issue list -F json`，范围限定到地图子项；排除存在未关闭阻塞项（指向未关闭问题单的原生 `blocked_by` 链接，可通过 `glab api projects/:id/issues/:iid/links` 查询，或 `Blocked by` 行中存在未关闭问题单）或已有受理人的项；按地图顺序选择第一项。
- **领取：**运行 `glab issue update <n> --assignee @me`，这是会话中的第一次写操作。
- **解决：**运行 `glab issue note <n> --message "<answer>"`，再运行 `glab issue close <n>`，然后向地图的“已有决策”追加上下文指针（gist 和链接）。
