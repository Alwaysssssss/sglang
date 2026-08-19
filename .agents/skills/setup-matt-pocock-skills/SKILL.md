---
name: setup-matt-pocock-skills
description: 为工程技能配置当前仓库，包括问题跟踪器、分诊标签词汇和领域文档布局。在首次使用其他工程技能前运行一次。
disable-model-invocation: true
---

# 配置 Matt Pocock 技能

创建工程技能依赖的仓库级配置：

- **问题跟踪器** —— 问题单的存放位置，默认使用 GitHub，也直接支持本地 Markdown；
- **分诊标签** —— 五种规范分诊角色使用的字符串；
- **领域文档** —— `CONTEXT.md` 和 ADR 的位置，以及读取这些文档的使用规则。

这是由提示驱动的技能，不是确定性脚本。先探索并展示发现，获得用户确认后再写入。

## 流程

### 1. 探索

检查当前仓库以了解初始状态。读取实际存在的内容，不要假设：

- `git remote -v` 和 `.git/config`：是否为 GitHub 仓库？具体是哪一个？
- 仓库根目录的 `AGENTS.md` 和 `CLAUDE.md`：是否存在？其中是否已有 `## 智能体技能` 章节？
- 仓库根目录的 `CONTEXT.md` 和 `CONTEXT-MAP.md`；
- `docs/adr/` 及所有 `src/*/docs/adr/` 目录；
- `docs/agents/`：本技能之前是否已经生成过输出？
- `.scratch/`：是否已有本地 Markdown 问题跟踪器惯例？
- 是否安装了 `triage` 技能，即本技能旁边存在 `triage` 技能目录，或可用技能中包含 `triage`。这决定是否执行 B 节。
- 单体仓库信号：`pnpm-workspace.yaml`、`package.json` 中的 `workspaces` 字段，或包含自身 `src/` 的非空 `packages/*`。只有真正的大型多包仓库才会出现；没有这些信号通常表示单上下文，这适用于绝大多数仓库。

### 2. 展示发现并询问

总结现有内容和缺失项，然后按顺序逐节处理：每次只处理一节、获得一个回答，再进入下一节。

每节先给出推荐答案，使用户只需一个词即可接受。只有选项确实会分支时才用一行解释。探索已经确定答案时整节跳过，例如未安装 `triage` 时跳过 B 节，不是单体仓库时才需要询问 C 节。

**A 节：问题跟踪器。**

> 说明：“问题跟踪器”是仓库问题单的存放位置。`to-tickets`、`triage`、`to-spec`、`qa` 等技能会在其中读写，因此必须知道应调用 `gh issue create`、在 `.scratch/` 下写 Markdown 文件，还是遵循用户描述的其他工作流。请选择当前仓库实际跟踪工作的地方。

默认情况下，这些技能针对 GitHub 设计。如果 `git remote` 指向 GitHub，建议使用 GitHub；如果指向 GitLab（`gitlab.com` 或自托管主机），建议使用 GitLab。其他情况或用户另有偏好时，提供：

- **GitHub** —— 问题单位于仓库的 GitHub Issues，使用 `gh` CLI；
- **GitLab** —— 问题单位于仓库的 GitLab Issues，使用 [`glab`](https://gitlab.com/gitlab-org/cli) CLI；
- **本地 Markdown** —— 问题单以文件形式位于仓库的 `.scratch/<feature>/` 下，适合个人项目或没有远程仓库的项目；
- **其他**（Jira、Linear 等）—— 请用户用一个段落描述工作流，技能将其记录为自由格式文字。

将选择记录到 `docs/agents/issue-tracker.md`。GitHub 和 GitLab 模板包含“是否把 PR 作为请求入口”标志，默认**关闭**。保持关闭且不要主动询问；希望把外部 PR 加入分诊队列的用户可稍后直接修改文件。

**B 节：分诊标签词汇。**未安装 `triage` 技能时整节跳过；未安装的技能不需要标签。

如果已安装，只问一个问题：

> 是否保留默认分诊标签？（建议：**是**）

默认值是五种规范角色，每个标签字符串与角色名相同：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。用户回答**是**时原样写入。只有用户回答否时才收集覆盖值，通常因为跟踪器已有其他命名，例如用 `bug:triage` 表示 `needs-triage`，使 `triage` 使用现有标签而不是创建重复项。

**C 节：领域文档。**默认使用**单上下文**：仓库根目录放置一个 `CONTEXT.md` 和 `docs/adr/`。这适合几乎所有仓库，无需询问即可写入。

只有探索发现单体仓库信号时才提供**多上下文**选项：由根目录 `CONTEXT-MAP.md` 指向每个上下文的 `CONTEXT.md`，然后确认用户需要哪种布局。

### 3. 确认并编辑

向用户展示以下草案：

- 要添加到 `CLAUDE.md` 或 `AGENTS.md` 的 `## 智能体技能` 区块，选择规则见第 4 步；
- `docs/agents/issue-tracker.md`、`docs/agents/domain.md` 和 `docs/agents/triage-labels.md` 的内容；最后一个文件仅在安装 `triage` 时展示。

允许用户在写入前修改。

### 4. 写入

**选择要编辑的文件：**

- 存在 `CLAUDE.md` 时编辑它；
- 否则，存在 `AGENTS.md` 时编辑它；
- 两者都不存在时询问用户要创建哪一个，不替用户选择。

`CLAUDE.md` 已存在时绝不创建 `AGENTS.md`，反之亦然；始终编辑现有文件。

如果所选文件中已有 `## 智能体技能` 区块，就地更新内容，不要追加重复区块。不要覆盖周边章节中的用户改动。

区块格式：

```markdown
## 智能体技能

### 问题跟踪器

[用一行总结问题单的跟踪位置]。参见 `docs/agents/issue-tracker.md`。

### 分诊标签

[用一行总结标签词汇]。参见 `docs/agents/triage-labels.md`。

### 领域文档

[用一行总结“单上下文”或“多上下文”布局]。参见 `docs/agents/domain.md`。
```

只有安装 `triage` 且执行过 B 节时，才加入 `### 分诊标签` 子区块并写入 `docs/agents/triage-labels.md`；否则二者都省略。

然后以本技能目录中的种子模板为起点写入文档：

- [issue-tracker-github.md](./issue-tracker-github.md) —— GitHub 问题跟踪器；
- [issue-tracker-gitlab.md](./issue-tracker-gitlab.md) —— GitLab 问题跟踪器；
- [issue-tracker-local.md](./issue-tracker-local.md) —— 本地 Markdown 问题跟踪器；
- [triage-labels.md](./triage-labels.md) —— 标签映射，仅在安装 `triage` 时使用；
- [domain.md](./domain.md) —— 领域文档使用规则和布局。

对于“其他”问题跟踪器，根据用户描述从头编写 `docs/agents/issue-tracker.md`。

### 5. 完成

告知用户配置已完成，并说明哪些工程技能会读取这些文件。提醒用户以后可以直接编辑 `docs/agents/*.md`；只有希望切换问题跟踪器或从头重新配置时才需要再次运行本技能。
