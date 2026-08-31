# Claude Cowork 实战: 《File & Document Tasks》文件与文档自动化处理完全指南

> **课程出处**：Anthropic 官方 Claude Academy (`academy.claude.com/courses/introduction-to-claude-cowork/file-document-tasks`)  
> **课程定位**：掌握如何将 Claude Cowork 作为本地文件与文档的自主处理引擎，覆盖批量操作、多格式文档生成、数据提取与综合整理等核心场景  
> **核心主题**：本地文件直接读写、批量操作、Office 文档生成、PDF 处理、数据提取与汇总、并行子 Agent 加速

---

## 目录
1. [核心范式：从"人工搬运"到"文件自动化处理"](#1-核心范式从人工搬运到文件自动化处理)
2. [文件操作全能力矩阵](#2-文件操作全能力矩阵)
3. [能力一：文件组织与批量管理](#3-能力一文件组织与批量管理)
4. [能力二：多格式文档生成（Word / Excel / PPT / PDF）](#4-能力二多格式文档生成word--excel--ppt--pdf)
5. [能力三：跨文件数据提取与综合汇总](#5-能力三跨文件数据提取与综合汇总)
6. [能力四：批量处理与并行子 Agent 加速](#6-能力四批量处理与并行子-agent-加速)
7. [Microsoft 365 深度集成](#7-microsoft-365-深度集成)
8. [安全操作规范：文件任务的高危风险防范](#8-安全操作规范文件任务的高危风险防范)
9. [典型场景速查 Cheatsheet](#9-典型场景速查-cheatsheet)

---

## 1. 核心范式：从"人工搬运"到"文件自动化处理"

在没有 Cowork 的世界里，处理大量本地文件是典型的"脑力外包但体力密集"的重复劳动：

```mermaid
flowchart LR
    subgraph 传统方式 [😓 传统手工方式]
        A1[逐个打开文件] --> A2[阅读提取信息]
        A2 --> A3[手动复制粘贴整合]
        A3 --> A4[逐个重命名/移动]
        A4 --> A5["耗时：数小时，极度机械重复"]
    end

    subgraph Cowork方式 [⚡ Cowork 文件自动化]
        B1[用自然语言描述目标] --> B2[Claude 自主扫描全量文件]
        B2 --> B3[并行处理 + 提取 + 整合]
        B3 --> B4[直接生成/修改/归档成品文件]
        B4 --> B5["耗时：分钟级，人类仅需终审"]
    end
```

---

## 2. 文件操作全能力矩阵

```mermaid
graph TD
    Cowork["Claude Cowork 文件处理引擎"] --> O1
    Cowork --> O2
    Cowork --> O3
    Cowork --> O4
    Cowork --> O5

    O1["📂 文件组织与批量管理\n(重命名 / 移动 / 分类归档)"]
    O2["📄 文档生成\n(Word / Excel / PPT / PDF)"]
    O3["🔍 数据提取与汇总\n(跨文件扫描 + 汇编)"]
    O4["⚙️ 格式转换\n(批量转 PDF / 图片提取等)"]
    O5["🔀 并行子 Agent 加速\n(多文件同时处理)"]
```

| 能力类型 | 代表任务 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| **文件组织** | 整理混乱下载目录 | 杂乱文件夹 | 按类型/项目/日期分类的有序目录 |
| **文档生成** | 从数据生成分析报告 | 原始数据 / 草稿笔记 | `.docx` / `.xlsx` / `.pptx` 成品文件 |
| **数据提取** | 从多份 PDF 发票提取金额 | 批量 PDF 文件 | 汇总 Excel 表格 |
| **批量操作** | 按规则批量重命名 | 大量文件 | 符合命名规范的整齐文件集 |
| **格式转换** | 将 Word 文档转为 PDF | `.docx` 文件 | `.pdf` 格式文件 |

---

## 3. 能力一：文件组织与批量管理

这是最直观、最能立竿见影提升效率的文件任务类型。

### 典型场景

```mermaid
flowchart TD
    Messy["📁 混乱的文件夹\n(几十上百个未分类文件)"] --> Claude
    
    Claude --> Rule1["按文件类型分类\n(.pdf → 文档/ .xlsx → 数据/ .jpg → 图片/)"]
    Claude --> Rule2["按项目名称归档\n(读取文件名/内容推断所属项目)"]
    Claude --> Rule3["按日期排序整理\n(年/月/周 层级目录结构)"]
    Claude --> Rule4["批量重命名\n(统一前缀/日期格式/去除乱码)"]
```

### 示例任务 Prompt
```text
请整理我的 Working Folder/项目归档/ 目录：
1. 将所有 PDF 文件移至子目录 "文档/"
2. 将所有 Excel 文件移至 "数据/"
3. 对文件名中包含 "draft" 或 "草稿" 的文件统一重命名，在文件名末尾加上 "_待审阅"
4. 整理完成后生成一份目录结构清单 directory-map.md
```

---

## 4. 能力二：多格式文档生成（Word / Excel / PPT / PDF）

Cowork 能直接在本地生成**具备完整格式、公式与结构**的 Office 文档：

```mermaid
flowchart LR
    Input["原始素材\n(数据文件 / 笔记 / 草稿 / 研究结论)"] --> Claude

    Claude --> W["📝 Word (.docx)\n标准格式报告\n含标题层级/目录/页眉页脚"]
    Claude --> E["📊 Excel (.xlsx)\n含公式的数据表格\n数据透视 / 图表 / 格式化"]
    Claude --> P["📑 PowerPoint (.pptx)\n演示文稿结构\n幻灯片布局 / 内容填充"]
    Claude --> PDF["📄 PDF\n文档汇编 / 表单填写"]
```

### 文档生成最佳实践
- **明确结构要求**：在 Prompt 中指定文档的章节结构、表格列名、图表类型，生成质量显著更高。
- **数据源引用**：指定从哪些本地文件中读取数据（如"从 `sales-2026Q2.xlsx` 的 Sheet1 中读取数据"）。
- **格式规范**：如有公司/团队模板，将其放入 Working Folder 并在 Instructions 中注明"参照 `company-template.docx` 的样式规范"。

---

## 5. 能力三：跨文件数据提取与综合汇总

这是 Cowork 文件处理中技术含量最高、价值最大的能力之一。

```mermaid
sequenceDiagram
    actor User
    participant Cowork as Claude Cowork
    participant Files as 本地文件集

    User->>Cowork: "从本月所有 PDF 发票中提取供应商名称、金额和日期，汇总到 Excel"
    Cowork->>Files: 并行扫描所有 .pdf 文件
    Files-->>Cowork: 返回各文件内容
    Cowork->>Cowork: OCR 识别 + 结构化提取关键字段
    Cowork->>Files: 写入 expense-summary-2026-08.xlsx
    Cowork->>User: ✅ 已完成，共处理 47 份发票，汇总表已保存
```

### 高价值提取场景
| 场景 | 原始文件 | 提取内容 | 交付成品 |
| :--- | :--- | :--- | :--- |
| 发票/费用报销 | 批量 PDF 发票 | 供应商、金额、日期、税号 | Excel 费用汇总表 |
| 合同信息管理 | 多份合同 .docx | 甲乙方、合同金额、有效期、关键条款 | 合同台账 Excel |
| 会议纪要整合 | 多份会议 Markdown | 决策事项、行动项、负责人、截止日期 | 待办清单 + 决策日志 |
| 简历筛选 | 批量候选人 PDF | 学历、工作经验、技能关键词 | 候选人对比矩阵表 |

---

## 6. 能力四：批量处理与并行子 Agent 加速

当任务涉及**大量独立文件**时，Cowork 会自动启用**并行子 Agent（Sub-Agent Parallelism）**机制：

```mermaid
flowchart TD
    Task["批量任务：处理 200 份 PDF 报告"] --> Dispatch

    Dispatch{Cowork 调度器} -->|"分批并行分发"| A1["子 Agent 1\n处理第 1-50 份"]
    Dispatch --> A2["子 Agent 2\n处理第 51-100 份"]
    Dispatch --> A3["子 Agent 3\n处理第 101-150 份"]
    Dispatch --> A4["子 Agent 4\n处理第 151-200 份"]

    A1 & A2 & A3 & A4 --> Merge["结果汇并 + 综合整理"]
    Merge --> Final["📊 完整汇总成品文件"]
```

- **速度优势**：并行处理使大批量任务的耗时从线性 O(n) 降低至近似 O(1)。
- **无需手动配置**：Cowork 自动判断任务规模并决定是否启用并行子 Agent。

---

## 7. Microsoft 365 深度集成

通过 Connectors，Cowork 可以直接与 Microsoft 365 套件深度协作：

| Office 应用 | Cowork 支持的操作 |
| :--- | :--- |
| **Word** | 读取 / 编辑 / 生成完整文档，维持样式与格式 |
| **Excel** | 读取数据 / 写入公式 / 生成图表 / 数据透视分析 |
| **PowerPoint** | 生成演示文稿结构 / 填充内容与图表 |
| **Outlook** | 读取邮件摘要 / 整理附件 / 辅助起草回复（需 Connector 授权） |

---

## 8. 安全操作规范：文件任务的高危风险防范

文件操作涉及本地数据的直接修改，需特别注意安全合规：

```mermaid
flowchart TD
    Risk["⚠️ 文件操作高危风险点"] --> R1
    Risk --> R2
    Risk --> R3

    R1["删除操作不可逆"] --> M1["✅ Global Instructions 中明确禁止删除\n只允许移动到 _archived/ 目录"]
    R2["批量覆盖写入"] --> M2["✅ 重要文件操作前要求先备份\n或用 Git 版本控制"]
    R3["文件权限越界"] --> M3["✅ Working Folder 采用最小特权原则\n只授权任务所需的最小目录"]
```

### 推荐的 Safety Rails 配置（写入 Global Instructions）
```markdown
### 文件操作安全守则
- 禁止任何形式的永久删除，所有"删除"操作改为移动至 `_archived/YYYY-MM-DD/` 目录
- 批量修改 10 个以上文件前，必须先生成操作预览清单并等待我确认
- 禁止访问或修改 Working Folder 范围之外的任何目录
- 覆盖已有重要文件前，先在同目录创建 `.bak` 备份副本
```

---

## 9. 典型场景速查 Cheatsheet

```markdown
### 📂 File & Document Tasks 实战 Prompt 模板

#### 📁 文件整理
"整理 [目录路径]，按 [规则：类型/项目/日期] 分类，
对 [特定文件类型] 重命名为 [命名格式]，
完成后生成目录清单 index.md"

#### 📊 数据提取汇总
"扫描 [目录] 中所有 [.pdf/.xlsx/.docx] 文件，
提取 [字段：金额/日期/名称/关键条款]，
汇总到 [output-file.xlsx] 的 Sheet1，
按 [排序字段] 降序排列"

#### 📄 文档生成
"根据 [数据来源文件]，生成一份 [文档类型：Word/Excel/PPT]，
结构包含 [章节/表格列/幻灯片主题]，
参照 [模板文件名] 的样式规范，
保存到 [输出路径]"

#### 🔄 批量重命名
"将 [目录] 中所有符合 [规则：包含/不包含/日期格式] 的文件，
重命名为 [新命名格式：前缀_YYYYMMDD_原文件名]，
操作前生成预览清单供我确认"
```
