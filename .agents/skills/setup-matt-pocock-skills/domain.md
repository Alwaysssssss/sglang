# 领域文档

工程技能探索代码库时应如何使用当前仓库的领域文档。

## 探索前读取

- 仓库根目录的 **`CONTEXT.md`**；或
- 仓库根目录的 **`CONTEXT-MAP.md`**（如存在），它为每个上下文指向一个 `CONTEXT.md`。读取与当前主题相关的每一个文件；
- **`docs/adr/`**：读取涉及目标工作区域的 ADR。在多上下文仓库中，还要检查 `src/<context>/docs/adr/` 中的上下文级决策。

任何文件不存在时都**静默继续**，不要报告缺失，也不要预先建议创建。`/domain-modeling` 技能会在术语或决策真正得到解决时按需创建；该技能可通过 `/grill-with-docs` 和 `/improve-codebase-architecture` 进入。

## 文件结构

单上下文仓库（大多数仓库）：

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

多上下文仓库（根目录存在 `CONTEXT-MAP.md`）：

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← 系统级决策
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← 上下文专属决策
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## 使用术语表词汇

输出中命名领域概念时，例如问题单标题、重构建议、假设或测试名，使用 `CONTEXT.md` 中定义的术语。不要漂移到术语表明确要求避免的同义词。

所需概念尚未出现在术语表中是一项信号：要么正在发明项目并不使用的语言，应重新考虑；要么确实存在缺口，应记录并交给 `/domain-modeling`。

## 标记 ADR 冲突

输出与现有 ADR 冲突时明确指出，不要静默覆盖：

> _与 ADR-0007（事件溯源订单）冲突，但值得重审，因为……_
