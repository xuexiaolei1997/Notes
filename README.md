# 个人技术知识库 (Personal Knowledge Base)

欢迎来到个人技术知识库。本知识库涵盖算法建模、机器学习/AutoML、大语言模型、软件工程架构、物联网与多媒体、运维与基础设施以及职业进阶规划等维度的学习笔记与工程实践沉淀。

---

## 📑 知识库导航目录

```
Notes/
├── 算法与建模/             # 时序分析、因果推断、优化算法与状态估计
├── 机器学习与AutoML/        # AutoML 框架与超参数优化
├── 大语言模型/             # LLM 前沿技术与官方课程笔记
├── 软件工程与架构/         # 《代码大全2》精读、清洁架构与设计模式
├── 物联网与多媒体/         # ThingsBoard 物联网、音视频推流与 PDF 报表
├── 运维与基础设施/         # Linux/Shell、Docker、Git、Ray 与系统环境适配
└── 职业规划/               # 算法工程师技能层级与技术路线
```

---

### 📈 算法与建模

| 笔记/代码 | 说明 | 领域/技术 |
| :--- | :--- | :--- |
| [时间序列预测方法总结](算法与建模/时间序列预测方法总结.md) | 时序预测全流程方案（规则周期、ARIMA、STL分解、LSTM/TCN/Attention 等） | 时间序列 |
| [TIGRAMITE 时序因果分析](算法与建模/tigramite.md) | 基于 PCMCI 框架的时间序列条件独立性与因果图挖掘 | 因果推断 |
| [多元状态估计与层次分析 (MSET+AHP)](算法与建模/MSET_AHP.py) | 工业设备状态监控与故障预警算法实现 | 工业算法 / 状态估计 |
| [遗传算法 (GeneticAlgorithm)](算法与建模/GeneticAlgorithm.py) | 遗传算法基础实现与适应度寻优 | 运筹优化 |
| [粒子群优化 (PSO)](算法与建模/Particle%20Swarm%20Optimization.py) | 粒子群寻优算法实现 | 运筹优化 |

---

### 🤖 机器学习与AutoML

| 笔记 | 说明 | 领域/技术 |
| :--- | :--- | :--- |
| [AutoML 概述](机器学习与AutoML/autoML.md) | 自动化机器学习核心概念与选型 | 自动化建模 |
| [FLAML 框架](机器学习与AutoML/FLAML.md) | 微软轻量级快速 AutoML 框架实践 | 自动调优 |
| [超参数优化](机器学习与AutoML/超参数优化.md) | 基于贝叶斯优化（Bayesian Optimization）的 LightGBM 自动寻优 | 参数调优 |

---

### 🧠 大语言模型

| 模块/笔记 | 说明 | 领域/板块 |
| :--- | :--- | :--- |
| [Claude Academy 学习全集导航](大语言模型/academy_claude/README.md) | Anthropic 官方体系化课程笔记导航（涵盖 6 大核心板块） | 全景导航 |
| ↳ [01. Claude 101](大语言模型/academy_claude/claude_ai/01-claude-101-notes.md) | Claude 基础、核心组件与 4D 框架入门 | Claude.ai |
| ↳ [02. Getting Started with Claude.ai](大语言模型/academy_claude/claude_ai/02-getting-started-with-claude-ai-notes.md) | Claude.ai 交互与效率指南 | Claude.ai |
| ↳ [03. Intro to Projects](大语言模型/academy_claude/claude_ai/03-intro-to-projects-notes.md) | Claude Projects 项目管理与知识库应用 | Claude.ai |
| ↳ [04. Choosing the Right Claude Model](大语言模型/academy_claude/claude_ai/04-choosing-the-right-claude-model-notes.md) | Claude 模型家族选型（Haiku/Sonnet/Opus/Fable）与分层路由 | Claude.ai |
| ↳ [01. Introduction to Claude Cowork](大语言模型/academy_claude/claude_cowork/01-introduction-to-claude-cowork-notes.md) | Cowork 桌面端多步任务委派、工作目录、Task Loop 与成品交付 | Claude Cowork |
| ↳ [02. Getting Set Up](大语言模型/academy_claude/claude_cowork/02-getting-set-up-notes.md) | Cowork 环境搭建、Working Folders 读写配置、Connectors 与权限模型 | Claude Cowork |
| ↳ [03. Scheduled Tasks](大语言模型/academy_claude/claude_cowork/03-scheduled-tasks-notes.md) | Cowork 定时任务、云端远程静默执行、周期性晨报与自动化监控 | Claude Cowork |
| ↳ [04. The Task Loop](大语言模型/academy_claude/claude_cowork/04-the-task-loop-notes.md) | Task Loop 四阶段（Plan/Execute/Steer/Verify）与 Human-in-the-Loop 协作范式 | Claude Cowork |
| ↳ [05. Research & Analysis at Scale](大语言模型/academy_claude/claude_cowork/05-research-analysis-at-scale-notes.md) | 多源并行研究、内置 Chromium 浏览器、结构化分析报告交付 | Claude Cowork |
| ↳ [06. Giving Cowork Context](大语言模型/academy_claude/claude_cowork/06-giving-cowork-context-notes.md) | 四大上下文支柱：Global Instructions / Projects / Skills / Plugins | Claude Cowork |
| ↳ [07. Skills: Teach Claude Cowork your way](大语言模型/academy_claude/claude_cowork/07-skills-notes.md) | Skill 可复用 Playbook、四大构件（SKILL.md/Assets/References/Scripts）、用 Claude 构建 Skill | Claude Cowork |
| ↳ [08. Plugins: Encode your team's expertise](大语言模型/academy_claude/claude_cowork/08-plugins-notes.md) | Plugin 打包（Skills+Connectors+Subagents）、两种形态、Marketplace 安装与定制、/setup-cowork | Claude Cowork |
| ↳ [09. Claude in Chrome](大语言模型/academy_claude/claude_cowork/09-claude-in-chrome-notes.md) | 无 Connector 工具的浏览器桥梁、Chrome 与 Cowork 协同、登录与访问边界 | Claude Cowork |
| ↳ [10. Claude for Microsoft 365](大语言模型/academy_claude/claude_cowork/10-claude-for-microsoft-365-notes.md) | Word/Excel/PPT/Outlook 内嵌 Add-in、跨应用上下文流转、M365 vs Cowork 选型 | Claude Cowork |
| ↳ [11. Best practices for working safely](大语言模型/academy_claude/claude_cowork/11-best-practices-working-safely-notes.md) | 专用工作目录、破坏性动词消歧、执行中三查、不当使用排除清单 | Claude Cowork |
| ↳ [12. Validating skills for plugins](大语言模型/academy_claude/claude_cowork/12-validating-skills-notes.md) | Evals 对照实验（有/无 Skill 基线）、反馈驱动迭代、发布标准与边界情况 | Claude Cowork |
| ↳ [13. Share what you build with your team](大语言模型/academy_claude/claude_cowork/13-share-with-team-notes.md) | 私有 Marketplace 分发、四种安装级别、四条分发路径、插件治理四习惯 | Claude Cowork |
| ↳ [14. Wrap up and next steps](大语言模型/academy_claude/claude_cowork/14-wrap-up-next-steps-notes.md) | 四模块全景回顾、贯穿主线（委派真实工作）、本周行动五选一、进阶路径 | Claude Cowork |
| ↳ 🏆 结课测验 | Quiz on Claude Cowork 满分通过 · 14 课 + 测验全部完成 | Claude Cowork ✅ |

---

### 🏗️ 软件工程与架构

| 模块/文件 | 说明 | 核心内容 |
| :--- | :--- | :--- |
| [《代码大全2》精读笔记 (25章全集)](软件工程与架构/代码大全2/README.md) | 经典软件构建、高质量代码与重构实践 | 奠定基础 / 高质量代码 / 变量 / 语句 / 代码改进 (全书全景导航) |
| [Python 工程实践与高级语言特性](软件工程与架构/python与函数.md) | 虚拟环境/离线依赖、编译加速、动态反射与 Jupyter 配置 | Python 进阶 |
| [pybind11 C++/Python 绑定](软件工程与架构/pybind11.md) | C++ 与 Python 混合编程、NumPy 数组零拷贝加速与模块导出 | 高性能混合编程 |

---

### 📡 物联网与多媒体

| 笔记 | 说明 | 领域/技术 |
| :--- | :--- | :--- |
| [ThingsBoard 平台与网关](物联网与多媒体/thingsboard.md) | ThingsBoard IoT 平台部署与 tb-gateway 配置实践 | 物联网/IoT |
| [FFmpeg 多媒体处理](物联网与多媒体/ffmpeg.md) | 音视频转码、剪辑与常用命令 | 音视频/多媒体 |
| [Linux 流媒体推流](物联网与多媒体/linux推流.md) | Linux 环境下的推流配置与调试 | 流媒体 |
| [ReportLab 报表生成](物联网与多媒体/reportlab.md) | Python 动态 PDF 报表与中文字体支持 | 自动化报表 |

---

### ⚙️ 运维与基础设施

| 模块/文件 | 说明 | 分类 |
| :--- | :--- | :--- |
| [Shell 教程](运维与基础设施/Shell教程) | Shell 变量、控制流、函数、输入输出重定向等全套教程 | 脚本编程 |
| [Docker 容器化](运维与基础设施/docker.md) | Docker 常用命令与容器化实践 | 容器技术 |
| [Git 版本控制](运维与基础设施/Git.md) | Git 核心工作流与操作备忘 | 版本控制 |
| [Linux 基础与运维](运维与基础设施/linux.md) | Linux 常用运维命令与系统配置 | 操作系统 |
| [Ray 分布式计算](运维与基础设施/ray.md) | 分布式任务调度与并行计算框架 | 分布式系统 |
| [TensorBoard 可视化](运维与基础设施/tensorboard.md) | 模型训练指标监控与可视化 | 深度学习工具 |
| [NumPy ARM 架构适配](运维与基础设施/numpy%20arm适配.md) | ARM/鲲鹏架构下的 NumPy 编译与适配 | 硬件与平台适配 |
| [Windows PE 系统维护](运维与基础设施/Windows%20PE.md) | Windows PE 系统维护与应急工具 | 操作系统维护 |
| [Markdown 语法指南](运维与基础设施/markdown.md) | Markdown 排版规范与常用语法 | 写作规范 |
| [start.sh 脚本](运维与基础设施/start.sh) | 服务启动脚本模版 | 运维脚本 |

---

### 🎯 职业规划

- [算法工程师技术路线图](职业规划/算法工程师技术路线图.md)：从工程底座、数据分析、传统机器学习、深度学习、业务落地到架构设计的完整能力进阶图谱。
