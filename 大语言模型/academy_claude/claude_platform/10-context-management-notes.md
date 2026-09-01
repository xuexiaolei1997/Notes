# Claude Platform 101: 《Context management》上下文管理四大模式

> **课程出处**：Anthropic 官方 Claude Academy (`academy.claude.com/courses/claude-platform-101/context-management`)  
> **课程定位**：一百万 token 听起来很多，但真实 Agent 跑起来**比你想的快就用完了**——上下文管理让你既待在窗口内、又不丢掉要紧的东西；官方发布**四大模式**（三个 API 特性 + 一个设计模式）  
> **核心主题**：什么算上下文、JIT 上下文 / 服务端压缩 / Prompt 缓存 / Memory 工具、按故障模式选用  
> **课程时长**：约 6 分钟（第 10/13 课）

---

## 目录
1. [什么算上下文](#1-什么算上下文)
2. [模式一：JIT 及时上下文](#2-模式一jit-及时上下文)
3. [模式二：服务端压缩](#3-模式二服务端压缩)
4. [模式三：Prompt 缓存](#4-模式三prompt-缓存)
5. [模式四：Memory 工具](#5-模式四memory-工具)
6. [组合使用与选型](#6-组合使用与选型)
7. [实战 Cheatsheet](#7-实战-cheatsheet)

---

## 1. 什么算上下文

> Context is **everything Claude sees on a given turn**.

| 组成 | 说明 |
| :--- | :--- |
| 系统提示词 | system prompt |
| 消息历史 | message history |
| 工具定义与工具结果 | tool definitions + tool results |
| 附带的文件与 Skills | attached files & skills |
| 思考块 | thinking blocks（L6 埋的伏笔：思考也占上下文） |

三条铁律：

1. 它是**每次 API 调用的输入**
2. **进收费、出也收费**（you pay for it on the way in, and on the way out）
3. **窗口一满，请求就失败**

> 🎯 目标不是"全都塞进去"，而是 **fit the right things in（把对的东西塞进去）**。

---

## 2. 模式一：JIT 及时上下文

> Don't load everything upfront. Load what the agent needs **now**, and let it pull more in via tools when it asks.

- **四个模式中唯一的设计模式**（API 没有专门开关，纯粹是加载时机的设计抉择）
- 典型：合规审查 Agent **不**把整本建筑规范塞进 system prompt——需要某节时调 `lookup_building_code` 拉取

---

## 3. 模式二：服务端压缩

对话跑长时，Anthropic 的**服务端压缩**把旧轮次**摘要成单个块**：

```python
response = client.beta.messages.create(
    betas=["compact-2026-01-12"],
    model="claude-opus-5",
    max_tokens=1024,
    context_management={
        "edits": [
            {"type": "compact_20260112"}
        ]
    },
    messages=messages,
)
```

- 请求加 **`context_management`** 键（内含一个 edit，type 指定压缩版本）
- 输入**越过触发阈值时 API 自动摘要**——**不用自己追踪对话长度**

> 💡 对应 Claude Code L6 的自动 Compaction：产品端那个"黑盒"，在 API 端是可以显式启用的。

---

## 4. 模式三：Prompt 缓存

把请求里**稳定的部分**——系统提示词、工具定义、长文档——**标记缓存**，跨调用复用，**成本只需零头**。

**账要算清楚**：

> 系统提示词 4,000 token、每小时调 100 次——缓存就是"付得起的账单"与"财务打电话来找你"的区别。

| 场景 | 无缓存 | 有缓存 |
| :--- | :--- | :--- |
| 4000 token × 100 次/小时 | 全价 × 100 | 首次全价 + 后续按缓存价 |

---

## 5. 模式四：Memory 工具

有些上下文需要**跨会话存活**：用户偏好、Agent 的运行笔记、上周的决定——推荐原语是 **Memory 工具**。

工作方式：

```mermaid
flowchart LR
    A["Claude 通过工具调用<br>读写 memory 目录"] <--> B["你在客户端实现<br>存储后端"]
    C["Anthropic 自动注入系统指令：<br>开工前先查 memory 目录"] --> A

    B --> B1["文件系统 /<br>数据库 / 加密存储……<br>（随你）"]

    style C fill:#7B68EE,color:#fff
```

- Claude **通过 tool call 读写**记忆目录
- 存储后端**你自己实现**（文件系统、数据库、加密存储都行）
- Anthropic **自动注入**一条系统指令：告诉 Claude 开工前先查记忆目录

> 💡 与 L7 Client Tools 里的 Memory、Cowork 的记忆机制同源——**跨会话状态**的官方解法。

---

## 6. 组合使用与选型

生产应用里**四个一起上**：合规审查 Agent 缓存 system prompt 与工具定义（省钱），建筑规范靠 `lookup_building_code` 按需拉取（省窗口）。

**每个模式对应一种故障模式**：

| 故障模式 | 对策 |
| :--- | :--- |
| **成本**（重复内容反复计费） | Prompt 缓存 |
| **窗口大小**（长对话撑爆） | 服务端压缩 / JIT |
| **无状态**（跨会话失忆） | Memory 工具 |

> 💡 Managed Agents **默认开启缓存与压缩**——不想手动接线时的现成选项（承接 L11-L12）。

---

## 7. 实战 Cheatsheet

```markdown
### 🗜️ 上下文管理速查

#### 1. 什么算上下文
system + 消息历史 + 工具定义/结果 + 文件/Skills + thinking 块
进收费、出也收费；窗口满 → 请求失败
目标：fit the right things in

#### 2. 四大模式（3 API 特性 + 1 设计模式）
① JIT 及时上下文（设计模式）：
   现在要的现在载，其余靠工具拉
② 服务端压缩：
   context_management = {edits: [{type: "compact_20260112"}]}
   越过阈值自动摘要旧轮次
③ Prompt 缓存：
   标记稳定部分（system/工具定义/长文档）跨调用复用，成本零头
④ Memory 工具：
   tool call 读写 memory 目录；后端你自己实现；
   Anthropic 自动注入"先查记忆"指令

#### 3. 故障模式 → 对策
成本 → 缓存；窗口 → 压缩/JIT；无状态 → Memory

#### 4. 生产心法
四个模式组合拳；Managed Agents 默认带缓存+压缩
```

### 课程衔接

> 🔗 **下一课**：L11《What are managed agents?》——不想自己拥有循环时，把整个 Agent 托管给 Anthropic。
