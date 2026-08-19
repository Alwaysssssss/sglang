# HTML 报告格式

架构审查渲染为操作系统临时目录中的单个自包含 HTML 文件。Tailwind 和 Mermaid 均通过 CDN 加载。Mermaid 可靠处理图状关系，手写 div 和内联 SVG 则用于更具编辑感的可视化，例如质量图和剖面图。混合使用二者，不要让所有内容都依赖 Mermaid，以免报告显得千篇一律。

## 脚手架

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>架构审查 — {{仓库名称}}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
      mermaid.initialize({ startOnLoad: true, theme: "neutral", securityLevel: "loose" });
    </script>
    <style>
      /* 为 Tailwind 难以干净覆盖的内容添加少量自定义样式：
         接缝虚线、手绘感箭头等。 */
      .seam { stroke-dasharray: 4 4; }
      .leak { stroke: #dc2626; }
      .deep { background: linear-gradient(135deg, #0f172a, #1e293b); }
    </style>
  </head>
  <body class="bg-stone-50 text-slate-900 font-sans">
    <main class="max-w-5xl mx-auto px-6 py-12 space-y-12">
      <header>...</header>
      <section id="candidates" class="space-y-10">...</section>
      <section id="top-recommendation">...</section>
    </main>
  </body>
</html>
```

## 页头

显示仓库名称、日期和紧凑图例：实线框 = 模块，虚线 = 接缝，红色箭头 = 泄漏，深色粗框 = 深模块。不要写介绍段落，直接进入候选项。

## 候选项卡片

图表承担主要表达任务。文字要少而直白，并自然使用 `/codebase-design` 技能中的术语。

每个候选项使用一个 `<article>`：

- **标题** —— 简短命名深化动作，例如“收拢订单接入流程”。
- **徽章行** —— 推荐强度（`强烈推荐` = 翠绿，`值得探索` = 琥珀，`推测性建议` = 灰蓝），并添加依赖分类标签（`进程内`、`本地可替代`、`端口与适配器`、`模拟`）。
- **文件** —— 等宽字体列表，使用 `font-mono text-sm`。
- **改造前/后图** —— 视觉中心，两列并排，参见下方模式。
- **问题** —— 一句话说明痛点。
- **方案** —— 一句话说明改动。
- **收益** —— 项目符号，每项不超过 6 个词，例如“测试只经过一个接口”“定价逻辑不再泄漏”“删除 4 个浅模块”。
- **ADR 提示**（如适用）—— 琥珀色背景框中的一行文字。

不要写解释性段落。如果图表必须配合一整段文字才能理解，应重画图表。

## 图表模式

选择适合候选项的模式并混合使用。不要让所有图表看起来相同，多样性本身就是目标之一。

### Mermaid 图（依赖与调用流程的主力）

当重点是“X 调用 Y，Y 又调用 Z，结构混乱”时，使用 Mermaid `flowchart` 或 `graph`。将其包装在 Tailwind 样式卡片中，避免突兀。用 `classDef` 将泄漏边设为红色、深模块设为深色。时序图适合表达“改造前往返 6 次，改造后 1 次”。

```html
<div class="rounded-lg border border-slate-200 bg-white p-4">
  <pre class="mermaid">
    flowchart LR
      A[OrderHandler] --> B[OrderValidator]
      B --> C[OrderRepo]
      C -.leak.-> D[PricingClient]
      classDef leak stroke:#dc2626,stroke-width:2px;
      class C,D leak
  </pre>
</div>
```

### 手写方框与箭头（Mermaid 布局不合适时）

使用带边框和标签的 `<div>` 表示模块，在相对定位容器上绝对定位内联 SVG `<line>` 或 `<path>` 表示箭头。需要让“改造后”图呈现一个粗边框深模块，并在其中灰显内部结构时使用；Mermaid 无法渲染出恰当的视觉重量。

### 剖面图（适合分层浅薄结构）

堆叠水平条带（`h-12 border-l-4`）展示调用经过的层。改造前是 6 个几乎无所作为的薄层，改造后是标注合并职责的一个厚条带。

### 质量图（适合“接口与实现同样庞大”）

每个模块使用两个矩形，分别表示接口表面积和实现。改造前接口矩形几乎与实现矩形一样高，表示浅；改造后接口矩形较矮、实现矩形较高，表示深。

### 调用图收拢

改造前将函数调用树渲染为嵌套方框；改造后将同一棵树收拢为一个方框，并在内部淡化显示已变成内部实现的调用。

## 样式指南

- 使用精炼的编辑风格，而不是企业仪表盘风格；保留充足留白。标题可选衬线字体，`font-serif` 与石色/灰蓝色搭配良好。
- 谨慎使用颜色：一种强调色（翠绿或靛蓝），红色表示泄漏，琥珀色表示警告。
- 图表高度保持约 320 像素，使改造前后两图无需滚动即可舒适并排。
- 图中模块标签使用 `text-xs uppercase tracking-wider`，应像示意图，而不是界面控件。
- 脚本只能包含 Tailwind CDN 和 Mermaid ESM 导入。报告其余部分保持静态，不加入应用代码，也不添加 Mermaid 自身渲染之外的交互。

## 首选建议章节

使用一张较大的卡片，只包含候选项名称、一句话理由，以及指向其卡片的锚点链接。

## 语气

使用简洁直白的中文，但架构名词和动词必须直接来自 `/codebase-design` 技能。简洁不能成为术语漂移的借口。

**严格使用：**模块、接口、实现、深度、深、浅、接缝、适配器、杠杆率、局部性。

**不得替换为：**组件、服务、单元（代替模块）· API、签名（代替接口）· 边界（代替接缝）· 层、包装器（本意为模块时）。

**符合风格的表述：**

- “订单接入模块很浅，接口几乎与实现一样复杂。”
- “定价逻辑跨接缝泄漏。”
- “深化：一个接口，一个测试位置。”
- “两个适配器证明接缝合理：生产使用 HTTP，测试使用内存实现。”

**收益项目符号**应使用术语表命名收益，例如*“局部性：缺陷集中在一个模块”*、*“杠杆率：一个接口，N 个调用点”*、*“接口缩小，实现吸收原包装模块”*。不要写*“更易维护”*或*“代码更干净”*，这些词不在术语表中，没有存在价值。

不要含糊其辞，不要铺垫，也不要写“值得注意的是……”。能变成项目符号的句子就改成项目符号，能删除的项目符号就删除。某个术语不在 `/codebase-design` 术语表中时，先寻找已有术语，不要立即发明新词。
