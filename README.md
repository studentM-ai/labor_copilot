# 职场权益 Copilot

> 所处秋招的同学或者不懂法的人用大白话描述遭遇，它找到对应法条、翻译成人话、该算的帮你算。

**当前进度**：语料管道、数据体检、检索基线、最小 RAG、向量检索、
多工具 Agent、多轮对话与人工确认、混合检索对比、40 条测试集与检索评估、
Ragas 生成评估、合同体检、性能成本测量、FastAPI 接口、SSE 流式输出、多轮会话与持久化、前端页面、合同接口与文件上传、向量库（Milvus）、长期记忆、检索双后端与选型、可观测性、对话历史截断已完成

---

## 这是什么

所处秋招的大学生或者普通人遇到劳动纠纷时，往往不知道该问什么、看不懂法条、不知道能拿多少钱。
搜索引擎给一堆链接，通用大模型则会编造条文号。

这个项目把**官方法律条文**做成可检索的知识库，让用户用大白话提问，得到
「结论 + 法条依据 + 该算的数」。

**能力清单**

- [x] 法条级语料构建（30 部 / 1,251 条）
- [x] 数据体检与检索基线（定位四类失败原因）
- [x] 最小 RAG 链路：检索 + 生成 + 引用校验（答案正确 3/10）
- [x] 向量检索替代字面匹配（答案正确 3/10 → 9/10）
- [x] 多工具 Agent：模型自主改写检索词、自主调用计算工具
- [x] 多轮对话 + 写操作人工确认（SqliteSaver 持久化，支持中断恢复）
- [x] 经济补偿计算（《劳动合同法》第 47 条第一款，规则代码化）
- [x] 混合检索对比（实测混合 10/13 < 纯向量 12/13，决定不引入）
- [x] 40 条分层测试集（口语 18 / 术语 8 / 多步 8 / 超范围 6）
- [x] 检索评估：HitRate@1/3/5 + MRR，按题型分组（向量 HitRate@5 = 0.735）
- [x] 生成评估：Ragas faithfulness 0.919 / answer_relevancy 0.801（34 条，按题型分组）
- [ ] 试用期上限 / 加班费计算
- [x] 合同体检：逐条比对法条输出风险清单（5 份合同 38 条，检出 12/12，误报 0/26）
- [x] 合同体检接入 Agent：三工具协作，工具选择 3/3（该调体检时调、不该调时不调）
- [x] 性能与成本测量：冷启动约 4–9 秒（98% 在加载 bge-m3），检索仅占端到端 7–9%
- [x] FastAPI 接口：`/api/ask` 返回答案 + 引用法条 + 工具调用记录（含 Pydantic 校验与自动接口文档）
- [x] SSE 流式输出：status / tool_call / tool_result / token / done / error 六类事件，心跳保长任务连接
- [x] 多轮会话与持久化：AsyncSqliteSaver 按 thread_id 存档，引用编号跨轮连续，支持新建/查历史/删除会话
- [x] 前端单页：SSE 逐字渲染、markdown（含表格）、引用角标可点并重编号、多会话侧边栏
- [x] 合同接口：列表 / 上传（四道防护 + GBK 兜底 + 内容去重）/ 体检 SSE 逐条推进度
- [x] 工具返回值修正：全量列出风险条款（上限 10 条），不再让用户去看服务器上的文件
- [x] 向量库（Milvus）：1,251 条建 HNSW 索引，实测标量过滤与 `ef` 参数（该规模下 `ef` 不影响召回）
- [x] 长期记忆：每轮抽用户事实存 Milvus（kind 枚举保证覆盖），新会话按 `user_id` 过滤召回 top-3
- [x] 检索双后端：`VECTOR_BACKEND=numpy|milvus` 切换，实测 numpy 快 13 倍、召回 100/100
- [x] 可观测性：每请求一行 JSON（request_id / 耗时 / token / 工具 / 状态），`/api/stats` 汇总 P95 与错误率
- [x] 对话历史截断：续聊超过 6 轮按 `HumanMessage` 边界删，同时用记忆补回关键事实

---

## 数据说明

| 项 | 内容 |
|---|---|
| 来源 | [chinese-law-corpus](https://github.com/lttxzmj/chinese-law-corpus)（CC0 协议） |
| 原始出处 | 国家法律法规数据库（flk.npc.gov.cn）官方文本 |
| 同步时间 | 2026-10-21 |
| 全量语料 | 1,513 部法律 / 56,182 条法条 |
| **本项目主库** | **30 部劳动权益相关法律 / 1,251 条法条** |
| 检索粒度 | **一条法条 = 一个检索单元**，无需切块 |
| 字数分布 | 最短 20 / 中位 86 / 最长 797 字 |
| 覆盖范围 | 国家层面法律、行政法规、司法解释 |

### 筛选方法（五步管道，全部可复现）

```
01_scan.py           扫描全量 1513 部 → _index.json
02_filter.py         标题关键词粗筛   → _keyword_candidates.json   (31 部)
03_content_scan.py   正文密度扫描     → _content_candidates.json   (58 部)
        ↓  合并去重 67 部
04_select.py         自动规则 + 人工剔除 → _selected.json          (30 部)
05_build.py          按名单展开       → articles.jsonl             (1,251 条)
```

**取舍分两步，都可查：**

- **自动规则**：排除非劳动关系领域（公务员/军人/监狱/安全生产）、
  密度 < 10% 的横向基本法、全文不足 3 条的零散批复
- **人工剔除**：13 部，每条理由写在 `04_select.py` 的 `EXCLUDE` 字典里
  （如「《中外合作办学条例》命中的是外籍教师聘用条款」「《船员条例》适用主体特殊」）

**为什么不能只靠规则**：纯规则会跑出 43 部，含明显噪声——规则负责"别漏太多"，
人负责"别收错"。

### 已知局限

- **仅覆盖国家层面法律**：各省《劳动合同条例》等地方性法规不在数据集内
- **部门规章覆盖不全**：人社部通知、批复等未系统收录
- **数据时效**：截至 2026-10-21，之后新增或修订的法规需重新同步
- **施行日期不全**：上游 14 部行政法规的 `effectiveDate` 为空（326 / 1,251 条）。
  保留原样，**未用公布日期顶替**——公布日 ≠ 施行日，冒充会让时效判断出错。
  当前检索与生成均不使用该字段。
- **规模有限**：主库 1,251 条，指标置信区间较宽，故评估更关注
  相对提升而非绝对值
- **没有登录体系**：`user_id` 是浏览器本地生成的 UUID（存 localStorage），
  所以换浏览器 / 无痕窗口就是新用户，同一浏览器换人用又会共享记忆。
  长期记忆因此是「单浏览器级」的，生产必须换成账号体系

---

## 检索与问答现状

### 纯字面检索（BM25）

10 个口语化问题，**纯字面匹配（BM25）只命中 3 个**。失败分四类：

- **零匹配** —— 「年假」在语料里 0 条，所有文档得分相同，排序退化
- **稀有词噪声** —— 「公司」仅 2 条而「用人单位」503 条，idf 过高，拉出无关条文
- **二元组假阳性** —— 「仲裁员」被切成「仲裁/裁员」，导致"被裁员"全部命中仲裁庭条文
- **语义错配** —— 问「多久结案」，答「多久申请」

> 这四类就是下一步的任务：Query 改写解决零匹配，向量检索解决假阳性，Rerank 解决错配。
> 完整数据见 [`data/eval/bm25_baseline_result.md`](data/eval/bm25_baseline_result.md)。

### 端到端问答（简单BM25 检索 + DeepSeek 生成）

13 次调用（10 个劳动法问题 + 3 个超范围问题）：

| 指标 | 结果 |
|---|---|
| 答案完全正确 | **3 / 10** |
| 编造（引用不存在的法条） | **0 次** |
| 引用编号越界 | 0 次 |
| 超范围问题正确拒答 | **3 / 3** |

**关键发现：答案正确率（3/10）= 检索命中率（3/10）。**

生成侧不放大也不衰减 —— 检索命中几个，答案就答对几个，1:1 传导。
**当前瓶颈 100% 在检索**，所以下一步的目标是优化检索，而不是调 prompt。

**模型的边界感（意外收获）：**

- 检索完全失败的 3 题（社保 / 裁员 / 年假），模型没有硬编，
  而是逐条说明「[1] 讲的是职工董事、[2] 讲的是劳务派遣、[3] 讲的是立法目的，
  都回答不了你的问题」，然后建议咨询律师
- 「工伤怎么认定」检索到的是程序条文，模型答完程序后主动补充：
  「如果你问的是『什么情况算工伤』，参考法条中没有相关内容」
- 「房东不退押金」——模型识别出法条里的「押金」指职业中介收取的，
  与租房押金不是一回事

> 完整结果见 [`data/eval/easy_rag_result.md`](data/eval/easy_rag_result.md)

---

### 简单向量检索（bge-m3）

把检索从「字面匹配」换成「语义匹配」，其余环节（prompt / 生成 / 引用校验）完全不动：

| 指标 | 字面匹配（BM25） | 语义匹配（bge-m3） |
|---|---|---|
| 答案完全正确 | 3 / 10 | **9 / 10** |
| 编造 | 0 次 | 0 次 |
| 超范围问题正确拒答 | 3 / 3 | 3 / 3 |

**此前诊断出的四类失败，全部被解决：**

| 失败类型 | 字面匹配 | 语义匹配 |
|---|---|---|
| 零匹配 | 「年假」在语料里 0 条，排序退化 | 命中《职工带薪年休假条例》第 3 条 |
| 稀有词噪声 | 「公司」idf 过高，拉出工会法董事会条款 | 命中《社会保险法》第 63/60/86 条 |
| 二元组假阳性 | 「仲裁员」被切成「仲裁/裁员」 | 命中《劳动合同法》第 47 条 |
| 语义错配 | 问「多久结案」，答「60 日内申请」 | 命中《调解仲裁法》第 43 条，答 45 天 |

**唯一未解决的：加班费怎么算。**

《劳动法》第 44 条原文写的是「延长工作时间」「百分之一百五十」，
**没有「加班费」三个字** —— 语义距离仍然过远，向量也跨不过去。
这类问题需要 Query 改写，把口语词映射到法条的实际表述。

**实现方式**：bge-m3 编码 1,251 条法条，存成 1,251 × 1,024 的向量矩阵（4.9 MB），
检索时用一次矩阵乘法算余弦相似度取 Top5。万级以内的数据不需要向量数据库。

> 完整结果见 [`data/eval/vector_rag_result.md`](data/eval/vector_rag_result.md)

---

### Agent（LangGraph + 工具调用）

把「检索」和「计算」做成两个工具，模型自己决定调不调、用什么词调：

| 工具 | 作用 |
|---|---|
| `search_law` | 法条检索（bge-m3 语义检索），返回带全局编号的条文 |
| `calc_severance` | 经济补偿计算（《劳动合同法》第 47 条第一款，规则代码化） |

**图结构**：模型判断 → 有工具调用就走工具节点 → 回到模型 → 直到不再调用

```
START → agent ──(没有工具调用)──→ END
          │
          └─(有工具调用)→ tools → agent → …
```

**11 题结果：**

| 指标 | 结果 |
|---|---|
| 调用工具 | 10 / 11 |
| 每题检索次数 | 2 次 |
| 计算工具 | 1 次，48000 元（计算正确） |
| 超范围问题（零调用、零引用） | 1 / 1 |
| 引用编号越界 | 0 |

**三个值得记录的行为：**

1. **模型自主改写检索词** —— 用户说「社保」，它检索「社会保险 缴纳 用人单位 义务」；
   用户说「加班费」，它检索「延长工作时间 工资 报酬」。
   此前向量检索唯一失败的那道「加班费」，在 Agent 模式下自动解决。

2. **缺参数时反问而不是猜** —— 问「被裁员能拿多少钱」但没说工作年限时，
   它答完规则后追加「告诉我你在这家公司干了多久、月薪多少，我帮你算」。

3. **不属于劳动法时会克制** —— 问「房东不退押金」，零次工具调用、零条引用，
   直接说明属于房屋租赁纠纷。

**引用池**：多步检索会产生多批条文，如果每批都从 `[1]` 开始编号，
答案里的编号就会有歧义。做法是维护一个全局引用池，所有条文统一编号、
按条文 id 去重。实测同一题两次检索命中同一条时，编号复用了 `[1]`。

> 完整调用链见 [`data/eval/agent_d5_result.md`](data/eval/agent_d5_result.md)

---

### 多轮对话与写操作确认（checkpointer）

**问题**：Agent 的状态只在内存里，`invoke` 一返回就丢了——无法多轮，也无法中途停下等人确认。

**解法**：用 `SqliteSaver` 把状态按 `thread_id` 存进 SQLite，每次节点执行完存一份。

**一个机制解决两件事：**

| 能力 | 怎么来的 |
|---|---|
| **多轮对话** | 同一 `thread_id` 历史自动累积，第二轮能接住第一轮的指代 |
| **人工确认（HITL）** | 图停在 `interrupt()`，状态已落盘；新进程用 `Command(resume=...)` 从这里继续 |

**实测（四个场景全部通过）：**

| 场景 | 结果 |
|---|---|
| 多轮 | 第 1 轮 6 条消息 → 第 2 轮 11 条，「那超过了怎么办」正确接上试用期 |
| 会话隔离 | 新 `thread_id` 从 6 条重新开始，不串历史 |
| 中断 | 写操作前返回 `__interrupt__`，payload 含文件名、字数、内容预览 |
| 恢复 | `Command(resume="approve")` 后文件写入，**1168 字与中断时一致** |
| 持久化 | SQLite 中 25 个检查点，三个 thread 分开存 |

**一个设计取舍：工具只收文件名，内容从对话历史取。**

最初的设计是 `save_document(content=..., filename=...)`，让模型把整份文书作为参数生成。
两个问题：**长文档必然超出输出 token 上限被截断**；更严重的是
**模型会重新生成一遍内容，存下来的可能和用户看到的不是同一份**。

改成 `save_document(filename)`，内容取 `state["messages"]` 里最近一条完整回答：
既省 token，也保证「看到的」和「存下来的」严格一致（实测字数 1168 → 1168）。

**另一个观察**：当信息不足时（只说了索赔金额，没给工作年限和月工资），
模型选择反问而不是编造文书。这个行为是对的，但也说明
**「回答前必须先检索」这类硬约束，光靠 prompt 会被模型按语义绕过**。

---

### 混合检索对比

同一批 13 道可判定题（10 道口语 + 3 道法条原词密集；1 道超范围不参与统计）：

| 检索方式 | 命中 Top5 |
|---|---|
| BM25（字符二元组） | 7 / 13 |
| 向量（bge-m3） | **12 / 13** |
| 混合（RRF 等权融合） | 10 / 13 |

**混合反而比纯向量少 2 题。**

原因：RRF 等权融合假设两路质量相当，而实测 BM25（7/13）远弱于向量（12/13）。
BM25 的 20 条候选里大量无关文档，它们的累计分把向量排第 1 的正确文档挤了下去。
典型例子是「工伤怎么认定」：向量排第 1，混合后直接掉出 Top5。

3 道「法条原词密集」题（如「延长工作时间 百分之一百五十」）三种方式全部打平——
说明在 1,251 条这个规模上，向量对法条原词的理解已经够用，BM25 没有独特优势。

**结论：当前规模不引入混合检索，保留纯向量。**
若语料扩到 5.6 万条，向量在更大空间里更容易混淆，届时需重测，
且应使用加权 RRF 而非等权。

**同时固化了一份测试集**，带 gold 法条 id，作为后续评估的基线（后来扩到 40 条，见下一节）。

> 完整数据见 [`data/eval/hybrid_eval_result.md`](data/eval/hybrid_eval_result.md)

---

### 40 条测试集与检索评估

W1 的 13 条是边做边挑的，命中率会被高估。扩到 40 条后重测：

| 类型 | 条数 | 说明 |
|---|---|---|
| 口语（`kouyu`） | 18 | 用户会怎么问 |
| 术语（`term`） | 8 | 查询词几乎逐字来自法条原文 |
| **多步（`duobu`）** | **8** | 一个问题含两个法律问题，需要多次检索 |
| 超范围（`out_of_scope`） | 6 | 不属劳动法，正确行为是拒答 |

可判定题共 34 条（超范围题不参与命中统计）。

| 检索方式 | HitRate@1 | HitRate@3 | HitRate@5 | MRR@5 |
|---|---|---|---|---|
| BM25（字符二元组） | 0.324 | 0.441 | 0.500 | 0.389 |
| **向量（bge-m3）** | **0.441** | **0.618** | **0.735** | **0.546** |
| 混合（RRF 等权） | 0.353 | 0.529 | 0.647 | 0.459 |

按题型看向量检索：

| 类型 | 条数 | HitRate@5 | MRR@5 |
|---|---|---|---|
| `kouyu` | 18 | 0.778 | 0.571 |
| `term` | 8 | 0.875 | 0.719 |
| `duobu` | 8 | **0.500** | **0.317** |

**多步题明显最差，原因不是「难」，而是一次检索只能覆盖一个问题的一半。**
例如「公司拖欠工资 3 个月，我能直接辞职并要补偿吗」同时涉及「拖欠工资」
和「辞职+经济补偿」，单次检索只抓住了前者。这正是 Agent 需要多轮检索的理由。

9 道未命中的题归为五类原因：

| 类型 | 例子 | 机制 |
|---|---|---|
| 术语落差 | 「调岗降薪」 | 法条写的是「变更劳动合同」，字面零重合 |
| 一问多义 | 「拖欠工资 + 辞职要补偿」 | 一次检索只覆盖一半 |
| 否定表达 | 「没培训也没调岗」 | 法条写「经过培训…仍不能胜任」，向量对否定不敏感 |
| 精确词缺失 | 「能要赔偿金吗」 | 法条写「加付赔偿金」，缺了这个词就搜不到 |
| gold 标注偏窄 | 竞业协议未约定补偿 | 检索到司法解释（更贴题），但 gold 只标了基础法条 |

> 完整数据与每道未命中题的实际 Top5 见
> [`data/eval/retrieval_eval.md`](data/eval/retrieval_eval.md)

---

### 生成质量评估（Ragas）

对 34 条可判定题评估生成质量，裁判模型同为 `deepseek-chat`：

| 指标 | 全部 | 口语 18 | 术语 8 | 多步 8 |
|---|---|---|---|---|
| faithfulness | **0.919** | 0.922 | 0.975 | 0.857 |
| answer_relevancy | **0.801** | 0.865 | 0.776 | 0.680 |

**faithfulness** 衡量答案中的句子能否由给定法条原文推出。34 条中 16 条满分。
剩余差距来自三类内容，都由指标定义决定，**不是编造**：

| 类型 | 例子 | 为什么扣分 |
|---|---|---|
| 算术结果 | q27 的「经济补偿 48000 元」 | 由「每满一年一个月工资」算出，数字不在原文 |
| 法律判断 | q34 的「不合法」 | 由「可以解除的条件」反推而来 |
| 免责声明 | 「未找到相关依据」 | prompt 要求的正确行为，无原文支撑 |

**answer_relevancy** 由答案反推问题、再与原问题比相似度。
34 条中有 3 条（`q13` / `q30` / `q32`）得 0 分，原因是答案含「未找到相关依据」，
被 Ragas 判为 noncommittal 直接给 0——**不是答非所问**；排除这 3 条后为 0.878。

**一个必须接受的局限**：裁判模型的打分带随机性，同一批样本重复评估时
faithfulness 波动约 ±0.015，单条题目的分数波动可达 ±0.1。
因此**单条题目只用于定位线索，不用来下结论**。

> 三层指标共同指向同一个短板——**多步题**：
> 检索 HitRate@5 = 0.500，faithfulness = 0.857，answer_relevancy = 0.680。
> 完整数据见 [`data/eval/ragas_eval.md`](data/eval/ragas_eval.md)

---

### 合同体检

输入一份劳动合同文本，切成条款后逐条检索法条并判断风险，输出带依据的清单。

**为什么逐条处理**：整份合同塞进 prompt 会让模型抓大放小。逐条处理时每条形
成独立的检索与判断，能溯源到具体法条。

**关键设计：判断单条时把整份合同作为背景一起传入。**
有些风险只在条款之间成立——「试用期六个月」对照《劳动法》第二十一条是合法的
（上限就是六个月），但配上第一条的「合同期限一年」就违反《劳动合同法》第十九条
（一年期最多两个月）。只传单条时模型只能报「法条不足」。

**切分以空行为主、编号为辅**：合同天然用空行分隔条款，而编号形式（第X条 / 1. / 一、）
可能混用。先按空行切块，块内若还有多个编号开头再细分；没有空行的合同则回退到按编号切。

**测试结果**（5 份自造合同，格式各异：第X条 / 数字编号 / 中文序号 / 无编号 / 混合）：

| 指标 | 结果 |
|---|---|
| 切分 | 5 / 5 份（8 / 9 / 6 / 5 / 10 条） |
| 检出率 | **12 / 12** |
| 误报率 | **0 / 26**（无正常条款被判为高风险） |

埋入的违法约定包括：试用期超长、放弃社保、加班费按最低工资、无补偿竞业、
限制生育、只签试用期合同、每周 48 小时、只缴工伤险、竞业限制三年等。

**两个要说明的口径**：

- 另有 2 条正常条款被判「无法判断」（工作岗位、劳动报酬），未计入误报——
  这类条款不涉及劳动法的强制规定，检索不到相关法条
- 这些合同是为测试人为设计的，违法点明显；**真实合同上的准确率会低于此数**

> 完整报告见 [`data/eval/contract_check_result.md`](data/eval/contract_check_result.md)，
> 量化评估见 [`data/eval/contract_eval.md`](data/eval/contract_eval.md)

**Agent 接入**：同一套体检逻辑也接进了 Agent（`019_contract_agent.py`）。
用户说「帮我看看 sample_1.txt 有没有问题」时，Agent 自主调用 `check_contract`，
再补几次 `search_law` 给出法条依据；普通法律问题只调检索工具，不会误调体检。

| 用户输入 | 期望工具 | 实际调用 |
|---|---|---|
| 帮我看看 sample_1.txt 这份合同有没有问题 | `check_contract` | ✅ 调了，另补 4 次检索 |
| 公司不给我交社保怎么办 | `search_law` | ✅ 只调检索，未误调体检 |
| 我想知道试用期最长能有多久 | `search_law` | ✅ 只调检索 |

**一个设计考虑**：`check_contract` 收的是**文件名**而不是合同全文——
传全文要模型一字不差转述上千字，容易漏抄且浪费 token；
传文件名则和 `save_document(filename)` 的设计保持一致。

---

### 向量库（Milvus）

前面的向量检索是**暴力检索**：`4.9 MB` 的 `articles_vec.npy` 全量放内存，
一次矩阵乘法算完 1,251 条 —— `4 ms`、召回 100%。

这一节把同一批向量搬进**向量数据库**（Milvus），目的有两个：
**① 学向量库这一块**（企业 JD 里 Milvus / FAISS / 向量数据库出现频率很高）；
**② 为后面的跨会话长期记忆铺路** —— 记忆条目要按用户过滤、要能增删，这两件事 numpy 做不了。

#### 环境：三个容器，用 SeaweedFS 顶替 MinIO

Milvus 单机版也要三个容器，因为它本来就是分布式架构：

| 容器 | 作用 | 端口 |
|---|---|---|
| `milvus-standalone` | 计算（proxy / query / data / index 逻辑都在里面） | 19530（gRPC）、9091（健康检查） |
| `milvus-etcd` | 元数据（collection、schema、索引定义） | 2379 |
| `minio`（实际是 SeaweedFS） | 对象存储（向量段文件、索引文件） | 8333 |

**⚠️ 部署时踩了一个坑：MinIO 的镜像拉不下来。**

```
minio/minio:latest          → denied              （Docker Hub 已限制）
quay.io/minio/minio:latest  → 401 UNAUTHORIZED    （quay.io 要认证）
bitnami/minio:latest        → denied
```

2025 年 MinIO 收紧了镜像分发。**但 Milvus 只需要一个 S3 端点，谁提供都行**，
所以换成 S3 兼容的 **SeaweedFS**（`chrislusf/seaweedfs`）：

```yaml
  minio:
    image: chrislusf/seaweedfs:latest
    command: server -s3 -dir=/data -s3.port=8333
```

**Milvus 那边把 `MINIO_ADDRESS` 指向 `minio:8333` 即可，其他一行不用改。**

**这件事本身就是可讲的点**：对象存储是**可替换的后端**，不是 Milvus 的一部分。

#### 参数

| 项 | 值 |
|---|---|
| 服务端 | `pkg/v2.5.4` |
| 维度 / 度量 | 1024 / `COSINE`（bge-m3 已 L2 归一化，COSINE 与 IP 等价） |
| 索引 | `HNSW`，`M=16`，`efConstruction=200` |
| 查询参数 | `ef=64` |
| 数据量 | 1,251 条 |

#### 实测

| 阶段 | 耗时 |
|---|---|
| 建 collection + 建索引 | 约 1.2 秒 |
| 插入 1,251 条（分批 500） | 约 0.5 秒 |
| `flush` 落盘 | 约 3 秒 |
| 单次查询 | 约 6 ms |

**查询结果和 numpy 版本完全一致**（同一批向量、同一个模型）：

```
0.7617  中华人民共和国劳动法            第二十一条
0.7282  中华人民共和国劳动合同法        第十九条
0.6468  中华人民共和国劳动合同法        第八十三条
0.6302  中华人民共和国劳动合同法        第二十条
0.6292  中华人民共和国劳动合同法实施条例  第十五条
```

#### 标量过滤：numpy 做不到的那件事

```python
client.search(..., filter='law_id == "labor-contract-law"')
```

**加过滤后 5 条全部来自《劳动合同法》** —— 把候选范围圈到一部法内。
numpy 版本要实现这个，得先算出全部 1,251 个分数再手工过滤。

#### `ef` 实测：这个数据量下不起作用

| ef | 平均耗时（50 次） | 与 ef=512 的 top-5 重合 |
|---|---|---|
| 8 | 6.85 ms | 5/5 |
| 16 | 5.57 ms | 5/5 |
| 64 | 6.83 ms | 5/5 |
| 256 | 5.08 ms | 5/5 |

> **耗时在 5–7 ms 之间无规律波动**，与 `ef` 没有单调关系 ——
> 因为本机单次 gRPC 往返就要 5–6 ms，而 1,251 条的 HNSW 搜索本身是亚毫秒级，
> 差异全被 RPC 开销淹没了。
>
> 召回方面 `ef` 从 8 调到 512，top-5 完全一致，
> 说明**这个数据量下 `ef` 不需要调**；数据量上到十万级才会成为真正的召回/延迟旋钮。

#### 四个坑

| 坑 | 现象 | 说明 |
|---|---|---|
| **`insert` 后 `row_count` 是 0** | 以为没插进去 | Milvus 先写内存后落盘，`flush` 之后统计才更新 |
| **`flush` 不会让 `row_count` 减少** | 删了一条，`row_count` 不变 | 删除是**逻辑删除**（打墓碑），只有后台 **compaction** 合并段时才真正剔除，时机不可控 |
| **`row_count` ≠ 有效条数** | 删 1 条 + 插回 1 条 → `row_count` 变 1252 | `row_count` 统计**物理行数**（含墓碑）；要有效条数用 `count(*)` |
| **`VARCHAR` 的 `max_length` 是字符数** | `string exceeds max length` | 中文按一个字算，`text` 设 8192 |

**→ 参数与实测完整记录：[`data/eval/milvus_basics.md`](data/eval/milvus_basics.md)**

---

### 长期记忆（跨会话）

前面所有记忆都是**短期**的：`AsyncSqliteSaver` 按 `thread_id` 存档，只在一个会话里有效。
换个会话就全忘了 —— 用户说过「月薪 12000、工作 3 年半」，下一句追问还是要重新问。

长期记忆做的就是：**把用户的关键条件记下来，跨会话复用。**

#### 三步

```
① 抽取   每轮回答结束后，让模型从这轮对话里抽 0~3 条「关于用户的事实」
② 存储   编码成向量，upsert 进 Milvus（主键 = hash(user_id + kind)）
③ 召回   新会话第一句时，按 user_id 过滤召回 top-3，拼进 SystemMessage
```

**实测（跨会话）：**

```
会话 A：我月薪 12000，在杭州工作，工作 3 年半，被裁了能拿多少
        → 经济补偿金 48000 元

会话 B（新建对话）：我能拿多少补偿
        → 按你给的信息（工作 3 年半、月薪 12000），经济补偿金是 48000 元
        → 没有再问「请问月薪多少」
```

**换个 `user_id` 问同样的话，它会重新追问条件** —— 说明 `filter=user_id` 真的在隔离。

#### 三个关键设计

**① `kind` 必须是枚举，不能自由起名**

```python
# 主键 = hash(user_id + kind)
md5("u1|salary") == md5("u1|salary")     # 覆盖 ✅
md5("u1|月薪")   != md5("u1|工资")        # 存成两条 ❌
```

kind 一旦漂移（这轮写「月薪」、下轮写「工资」），**主键就不同，`upsert` 覆盖直接失效**。
所以 prompt 里强制用 8 个英文枚举（`work_years` / `salary` / `city` / `end_reason` /
`contract` / `social_insurance` / `dispute_type` / `other`），代码里还有 `KIND_ALIASES` 兜底映射。

**② 主键 = `hash(user_id + kind)`，不是随机 id**

随机 id 会让用户说三次「月薪」存三条，互相挤占 top-3。
用 `(用户, 类别)` 做哈希，同一个人的同一类事实**永远只有一条**，用户改口时自然覆盖。

**③ 只在新会话注入，续聊不注入**

续聊时模型已经能从历史里看到条件，再注入一遍是浪费 token；
而且条件可能已经变了（用户后面改口），旧记忆反而误导。

#### 一条铁律：记忆不能影响主流程

| 环节 | 失败时 |
|---|---|
| 召回 | 返回空字符串，照常回答（只是「不记得你」） |
| 抽取 + 存储 | 丢到后台任务（`spawn`），失败只打日志 |

**抽取要调一次模型（1~2 秒），所以必须丢后台——否则用户看到答案后还要再等 1~2 秒。**
后台任务用模块级 `set` 持有引用，否则可能被 GC 回收（Python 官方文档专门提过这个坑）。

#### 已知局限：`user_id` 是前端自己编的

项目**没有登录体系**，`user_id` 是浏览器第一次打开时生成的 UUID，存在 `localStorage`：

```
同一浏览器再打开  → 同一个 uid → 记得你 ✅
换浏览器 / 无痕   → 新 uid    → 记忆从零 ❌
同一浏览器换人用  → 同一个 uid → 共享记忆（隐私问题）❌
```

生产上这块必须换成账号体系（`user_id` 从用户表来）；
另外要有过期/清理策略，否则用户清了浏览器数据后，旧记忆会变成无人认领的孤儿数据。

---

### 对话历史截断

多轮对话的上下文是**累积**的 —— 每轮只传一条 `HumanMessage`，但 checkpointer 会把历史全带上，
模型看到的是整段对话。

**实测（D21 的日志）：**

```
新会话（无工具）   tokens_in = 928      ← 基线：System + 问题
第 2 轮（无工具）  tokens_in = 2,100    ← 多出的约 1,170 是上一轮的问答
```

**每多一轮，历史就多约 1,100 token。10 轮之后一次请求约 12,000 token，
而其中真正有用的法条只有约 1,600。**

#### ⚠️ 难点：不能直接删

长期记忆只在新会话注入，续聊时不注入：

```python
if history:                                   # 续聊
    return {"messages":[HumanMessage(content=question)]}   # ← 不注入记忆
```

**所以如果直接把老对话删掉，模型就彻底不知道用户说过「月薪 12000」了 ——
因为那句话只存在于被删掉的部分里。**

#### 方案：删 + 补

```
历史超过 6 轮
   ↓
按 HumanMessage 边界删掉最早的（省 token）
   ↓
同时把用户记忆注入一条 SystemMessage（补回关键事实）
```

**⭐ 这才是「记忆让截断变安全」——两者是配合关系，不是替代关系。**

#### 三个关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 按什么边界切 | **`HumanMessage`** | 一次工具调用是 `AI(tool_calls) → ToolMessage → AI` 三连，从中间切会让模型报 `tool_calls must be followed by tool messages` |
| 切到几轮 | **6 轮** | 指代型追问通常只需前 2 轮，6 轮是保守值 |
| 什么时候补记忆 | **只在触发截断时** | 每轮都补会浪费 token，且旧信息可能过时 |

#### 实测（9 轮对话）

| 轮 | 消息条数 | system 条数 | 耗时 | 结果 |
|---|---|---|---|---|
| 1 | 5 | 1 | 6.9 s | 48000 元 |
| 2–7 | 11 → 41 | 1 | 4–6 s | 正常回答 |
| **8** | **42** | **2** | 3.8 s | **仍然算出 48000，没反问月薪** |
| 9 | 39 | 3 | 2.6 s | 仍记得 |

**第 8 轮 41 → 42 条（只 +1），说明删掉了一整轮 6 条；但仍能算出 48000。**

**服务端日志是决定性证据：**

```
[memory] 记住 3 条：['salary', 'work_years', 'city']
[deps] 截断 4条历史，用记忆补回：月薪 12000；工作 3 年半；在杭州
[deps] 截断 6条历史，用记忆补回：月薪 12000；工作 3 年半；在杭州
```

**历史里那句「我月薪 12000，工作 3 年半」已经被删掉了，但事实从记忆补了回来。**

> ⚠️ **这组数据是手工连续问 8 轮得到的，仓库里没有附带测试脚本。**
> 要复现：起服务后在同一会话里按「先说条件 → 问 6 个不相关的问题 → 再问补偿」的顺序提问，
> 观察 `/api/sessions/{thread_id}/messages` 的条数是否在第 8 轮停住，
> 以及服务端控制台是否出现 `[deps] 截断 N 条历史，用记忆补回：...`。

#### 为什么不用摘要

摘要会把 `12000` 糊成「一万多」——那就没法算补偿了。
这个场景的本质是「用几个精确数字算结果」，所以**结构化抽取 + 截断**比摘要更合适。
摘要只在长文档问答（信息没法结构化成几个字段）时才必要。

---

### 检索后端选型（numpy vs Milvus）

法条检索做了两套**接口完全一致**的实现，用环境变量切换：

| 后端 | 实现 | 默认 |
|---|---|---|
| `numpy` | `VectorRetriever`：`articles_vec.npy` 全量矩阵乘法 | ✅ 是 |
| `milvus` | `MilvusRetriever`：D18 灌进去的 `law_articles` | 可选 |

```powershell
$env:VECTOR_BACKEND="milvus"    # 不设则用 numpy
```

**「同接口」的意思是三个东西都对齐：**
`retrieve(query, k)` 返回同样的 `list[dict]`、都暴露 `.rows`（引用池重建要用）、
都暴露 `.model`（长期记忆编码要用）。所以 `make_search_law` / `check_contract` /
`deps` / `memory` **一行都不用改**。

**实测**（20 条 query，每条 5 次，首次丢弃）：

| 项 | numpy | Milvus |
|---|---|---|
| 纯检索 P50 | **0.36 ms** | **4.71 ms** |
| 纯检索 P95 | 0.45 ms | 7.76 ms |
| 编码 P50 | 106.51 ms | 107.55 ms |
| 元数据过滤 P50 | 0.43 ms | 5.00 ms |
| 召回一致性（top-5 重合） | 基准 | **100 / 100** |

**结论：默认 numpy。**

- **快 13 倍**（0.36 vs 4.71 ms），差异来自 gRPC 往返而非图搜索
- **召回完全一致** —— 1,251 条这个规模下 HNSW 的近似误差为 0
- **编码占 99.7%** —— 检索差几毫秒，用户根本感觉不到

**那 Milvus 什么时候用？** 不是「检索更快」，而是它独有的能力：

| 能力 | numpy | Milvus |
|---|---|---|
| 元数据过滤 | 全量打分后手写过滤 | `filter=` 一个参数 |
| 增量更新 | 重建整个 `.npy` | `upsert` / `delete` |
| 多租户隔离 | 自己维护 id ↔ 下标映射 | `filter=user_id==...` |
| 横向扩容 | 换更强的机器 | 加 query node |

数据量涨到十万级时需要重新评估，但那时候必须真实扩容后重测——外推不可靠
（numpy 的 `argsort` 是 O(N log N)，HNSW 的图也会随数据量变大）。

**→ 完整数据见 [`data/eval/vector_store_compare.md`](data/eval/vector_store_compare.md)**

---

### 性能与成本

冷启动（只发生一次，两次实测）：

| 步骤 | 耗时 | 说明 |
|---|---|---|
| 读 1,251 条语料 | 约 10 ms | 稳定 |
| 建 BM25 索引 | 约 60 ms | 稳定 |
| **加载 bge-m3 + 读向量矩阵** | **约 4–9 秒** | 取决于模型文件是否在系统磁盘缓存中 |
| 合计 | 约 4–9 秒 | |

单次检索（3 次平均）：

| 检索方式 | 单次耗时 |
|---|---|
| BM25（纯计算，无模型） | 约 4 ms |
| 向量（含一次 bge-m3 编码） | 约 165–205 ms |

Agent 端到端（两次实测）：

| 问题 | 耗时 | 工具调用 | token |
|---|---|---|---|
| 试用期最长能有多久 | 约 3.7–4.3 s | 2 | 约 2,400 |
| 工作 3 年半、月薪 12000 被裁能拿多少 | 约 4.7–5.1 s | 3 | 约 3,100 |

合同体检：一份 8 条的合同 **约 11–13 秒**，每条约 1.3–1.6 秒；
按此推算 20 条约 27–32 秒。

**三个观察**：

- 冷启动几乎全部花在加载 bge-m3（约 2 GB）。服务常驻内存后只有第一次付这个代价；
  时间在 4–9 秒之间波动，取决于模型文件是否还在系统磁盘缓存里
- 向量检索比 BM25 慢约 40 倍，但慢在「把问题编码成向量」，
  矩阵乘本身（1251 × 1024）可忽略
- **检索只占端到端耗时的 7–9%，瓶颈在 LLM 往返**——
  想降延迟，减少 LLM 调用次数比优化检索有效得多

> token 只统计 Agent 节点自身调用（从 `usage_metadata` 累加），
> 工具内部的模型调用未计入。
> 数值为多次实测的典型区间：LLM 侧存在天然抖动，不追求小数点精度。

---

### 可观测性（线上日志与统计）

上面的「性能与成本」是**离线测**的 —— 要跑一遍脚本才有一个数字。
但线上跑了多少、谁慢、慢在哪、花了多少 token，它**答不了**。

可观测性补的就是这一块：**每个请求写一行 JSON，`GET /api/stats` 汇总。**

#### 每请求一行，写在 `data/logs/app.jsonl`

```json
{"ts":"2026-11-25T17:36:48","request_id":"24ef4d65","kind":"ask_stream",
 "thread_id":"12f8...","user_id":"161b...","question_len":11,
 "elapsed_ms":5290,"tokens_in":2552,"tokens_out":551,
 "tools":["search_law","search_law"],"status":"ok","answer_len":840}
```

**不记问题原文，只记 `question_len`** —— 工资、纠纷这些是隐私。

#### 三个必须做对的地方

**① `request_id` 用 `contextvars` 传，不用全局变量**

asyncio 是单线程并发，`await` 的时候事件循环会切去跑别的协程 ——
全局变量会被另一个请求覆盖；`threading.local` 也不行（所有协程在同一个线程里）。
`contextvars` 按协程任务隔离，所以两行日志不会串号。

**② token 只能从 `on_chat_model_end` 拿**

`on_chat_model_stream` 的 chunk 上**没有** `usage_metadata`（那只是一小段文字），
只有生成结束时的事件才有。而且一次请求可能有多轮（工具调用 → 模型再来一次），所以必须**累加**。

**③ 日志写在最外层的 `finally` 里**

最需要排查的恰恰是「客户端中途断开」和「模型报错」这两类请求。
如果写在 `yield done` 之后，这些请求**一条日志都不会留下**。

顺带解决了「断开和正常完成看起来一样」的问题：用一个 `finished` 标志，
在 `yield done` **之前**置位，`finally` 里就能分出三种状态：

| status | 含义 |
|---|---|
| `ok` | 正常发完 `done` |
| `error` | 模型/工具报错 |
| `aborted` | **客户端中途断开** |

#### `GET /api/stats`

只读最近 500 条（全量读在 demo 规模没必要，也会越跑越慢）：

```json
{"window":8,"total":8,"error_rate":0.125,
 "latency_p50_ms":17450.0,"latency_p95_ms":24278.0,"latency_max_ms":24278.0,
 "slow_requests":5,"tokens_in":16632,"tokens_out":3353,"tokens_avg":2498.1,
 "tools":{"search_law":11,"calc_severance":1}}
```

#### 实测 8 个请求，读出三个问题

| 耗时 | 工具调用 | tokens_in | 结果 |
|---|---|---|---|
| 24,278 ms | search_law × 2 | 2,502 | ok |
| 17,450 ms | search_law × 2 | 2,561 | ok |
| 23,207 ms | search_law + calc | 937 | **aborted** |
| 5,290 ms | search_law × 2 | 2,552 | ok |
| 2,495 ms | **无** | 2,100 | ok |
| 1,469 ms | **无** | 928 | ok |

**① 有工具调用慢 3–10 倍**

无工具 1.5–2.5 秒，两次 `search_law` 要 5–24 秒。
因为**每次工具调用都是一整轮模型请求** —— 两次检索 = 三轮模型。

**② token 输入/输出比约 5:1，瓶颈在上下文不在生成**

`tokens_in 16,632` vs `tokens_out 3,353`。钱花在「塞进去多少法条和对话历史」上，
不在「模型写多长」。

**③ `aborted` 那行是 `finally` 存在的证据**

只吐了 7 个字就被掐断，`tokens_in` 只花了 937（正常要 2,500）。
这行日志证明了「断开也留痕」，也顺带量出了「用户等不及会损失多少成本」。

**→ 完整数据见 [`data/eval`](data/eval) 同名日志，或直接看 `data/logs/app.jsonl`**

---
#### 一个被数据抓出来的统计口径 bug

做完截断之后跑 9 轮对话，`tokens_in` 一路涨到 **109,558**，第 4 轮问「试用期最长多久」却记了 **10 次工具调用** —— 明显不对。

原因是 `/api/ask` 里取的是**整个会话的历史**：

```python
tokens_in,tokens_out=observability.usage_of(result["messages"])   # ← 全部历史
tool_calls = extract_tool_calls(result["messages"])               # ← 全部历史
```

`result["messages"]` 是 checkpointer 累积的**全量历史**，而每条历史 `AIMessage` 都带着当时的 `usage_metadata` ——
求和等于把前几轮的消耗又算了一遍。

**修法：先记下「本轮之前」已有的消息 id，事后只统计新增的。**

```python
state=await agent.aget_state(config)
before_ids={getattr(m,"id",None) for m in (state.values or {}).get("messages") or []}
...
new_msgs=[m for m in result["messages"] if getattr(m,"id",None) not in before_ids]
```

**⚠️ 必须用 id 集合判断，不能用 `result["messages"][before:]` 切片 —— 截断会把前面的消息删掉，下标全变了。**

修完对比：

| | tool_calls 三轮 | tokens_in 三轮 |
|---|---|---|
| 修前 | 2 → 4 → 6 | 2,143 → 6,073 → 13,265 → … → 109,558 |
| 修后 | 2 → 2 → 2 | 2,518 → 4,814 → 7,483 |

> `tokens_in` 修后仍会小幅增长，因为每轮模型调用看到的上下文确实变长了 —— 那是**真实值**，不是重复计算。

**⚠️ 流式接口 `/api/ask/stream` 不受影响**：它的 token 和工具都是按事件累加的，本来就只算本轮。


### Web 接口

把 Agent 包成 HTTP 服务，代码在 `app/` 目录。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查，不碰模型 |
| POST | `/api/ask` | 提问，返回答案 + 引用法条 + 工具调用记录 |
| POST | `/api/ask/stream` | 提问，SSE 流式：逐字推送 + 工具调用事件 |
| POST | `/api/sessions` | 新分配一个 thread_id（不写数据库，给前端「新建对话」用） |
| GET | `/api/sessions/{thread_id}/messages` | 取某个会话的历史消息 |
| DELETE | `/api/sessions/{thread_id}` | 删除会话（幂等，不存在也返回 200） |
| GET | `/api/contracts` | 列出 `data/contracts` 下可体检的合同 |
| POST | `/api/contracts/upload` | 上传合同（.txt，≤512 KB），内容相同则复用 |
| POST | `/api/contracts/check/stream` | 体检合同，SSE 逐条推进度 |

启动：

**依赖提示**：多轮会话用到 `AsyncSqliteSaver`，它依赖 `aiosqlite` 和
`langgraph-checkpoint-sqlite`。装完 `requirements.txt` 后先确认一下：

```bash
python -c "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; import aiosqlite; print('OK')"
```

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 流式接口：-N 关掉 curl 自己的缓冲，才能看到实时输出
# curl.exe -N -X POST http://127.0.0.1:8000/api/ask/stream \
#   -H "Content-Type: application/json" --data-binary "@body.json"
# 接口文档：http://127.0.0.1:8000/docs
# 前端页面：http://127.0.0.1:8000/    （浏览器打开就是聊天界面）
```

**请求与响应**：

```json
// POST /api/ask
{"question": "公司不给我交社保怎么办"}
```

```json
// 响应（节选）
{
  "answer": "公司不给你交社保，是违法的… [1]",
  "citations": [{"idx": 1, "law": "中华人民共和国社会保险法", "no": "第六十条", "text": "…"}],
  "tool_calls": [{"name": "search_law", "args": {"query": "社会保险 缴纳"}}],
  "thread_id": "3f2a…", "elapsed_ms": 4100
}
```

**三个设计点**：

- **三种资源三种寿命。** 检索器（bge-m3，加载 4–9 秒）进程级单例；引用池 `RefPool`
  按会话复用——同一会话的引用编号必须连续，否则第二轮会引用错法条；Agent 图每请求新建
  ——图里的工具闭包捕获了引用池，共享一个图会让不同会话的编号互相污染
- **`/api/ask` 用 `async def`。** 因为接了异步 checkpointer（`AsyncSqliteSaver`），
  同步的 `SqliteSaver` 不支持异步图（会抛 `NotImplementedError`），所以必须 `await agent.ainvoke(...)`。
  `ainvoke` 会把图里的同步节点丢到线程池执行，不会阻塞事件循环。
  纯内存的 `/api/health` 和 `/api/sessions` 仍是同步 `def`
- **`citations` 只返回答案里真正引用到的法条。** 一次问答可能检索到 10 条，
  答案只用了 3 条；全返回会让前端多出一堆没人用的依据

> 注意：引用编号是跨多次检索连续的（第二次检索到的新法条从 `[6]` 开始），
> 所以答案里可能出现 `[7]` 而不只是 `[1][2]`。这是 `RefPool` 的预期行为——
> 同一个法条始终对应同一个编号。前端展示时可重新编号为 1..n。

#### 流式输出

`/api/ask/stream` 用 SSE（Server-Sent Events），边跑边推，不用等 4–6 秒。

| 事件 | 内容 | 前端用来做什么 |
|---|---|---|
| `status` | `{"stage": "thinking"}` | 显示「正在思考…」 |
| `tool_call` | `{"name": "search_law", "args": {…}}` | 插一行「🔍 检索：社会保险 缴纳」 |
| `tool_result` | `{"name": …, "preview": "前 100 字"}` | 折叠展示工具返回 |
| `token` | `{"text": "试用期"}` | 追加到答案气泡 |
| `done` | `{"answer", "citations", "tool_calls", "thread_id", "elapsed_ms"}` | 渲染引用角标、结束加载态 |
| `error` | `{"message": …}` | 显示错误（此时 HTTP 头已发出，改不了状态码） |

**三个实现要点**：

- **用「生产者任务 + 队列」而不是 `wait_for(agen.__anext__())`。** 后者超时会取消
  异步生成器的推进，状态被破坏后下次迭代抛 `RuntimeError`。用独立任务把事件塞进
  队列，超时只影响「等队列」这一下，不影响生成器本身
- **心跳用 SSE 注释行 `: keep-alive`。** 合同体检一份 8 条，中间有十几秒没有任何
  事件；代理的空闲超时可能在这期间断开连接。注释行不产生事件、不显示给用户，
  但能让链路「看到有数据流动」
- **按 `metadata.langgraph_node == "agent"` 过滤 token。** 事件是层层冒上来的，
  不过滤会收到重复片段（同一段文字出现两遍）

> 注意：`check_contract` 内部用的是裸 `openai` SDK，不产生 LangChain 事件，
> 所以合同体检那几十次模型调用**不会**逐条推进流里——这是预期行为。

> 另：SSE 的 `data:` 行前面**不能有空格**。写成 ` data:{…}` 会让字段名变成
> `" data"`，符合规范的解析器会整行丢弃——`curl` 看不出问题（它只打印原文），
> 但前端会收到空内容。

#### 多轮会话

Agent 图本身没有记忆，每次调用都是从头开始。多轮靠 **checkpointer**：
每次跑完自动把完整消息历史按 `thread_id` 存进 SQLite，下次用同一个 id 调用时读回来。

```json
// 第一次：不传 thread_id，服务端新建一个并在响应里返回
{"question": "我工作3年半，月薪12000，被裁员了能拿多少经济补偿"}

// 第二次：带上同一个 thread_id，这次故意不提年限
{"question": "那如果公司是违法解除呢", "thread_id": "fcce25c5761f4c83acdb4bc6d4c2d95f"}
```

**实测第二轮的回答：**

> 经济补偿 N = 48000 元（4 个月 × 12000）
> 违法解除赔偿金 2N = 96000 元

**它没有重新问工作年限——第一轮算出的 48000 被带进了第二轮。**

**三个实现要点**：

- **引用池必须按 `thread_id` 复用，不能每轮新建。** 反例：第 1 轮检索到 5 条，
  答案写 `[2]`，历史里于是存着「`[2]` = 第十九条」。第 2 轮若新建池，新法条又从 `[1]` 开始，
  模型再写 `[2]` 就会被映射成第 2 轮的某条法条——**编号没错，法条错了**。
  所以 `_pools` 是个 `thread_id → RefPool` 的字典
- **续聊只传 `HumanMessage`，首轮才传 `SystemMessage`。** 历史由 checkpointer 恢复；
  每轮都传 system 消息的话，聊十轮会攒十条重复的 system，白烧 token 还可能干扰模型。
  判断依据是 `aget_state` 里有没有历史
- **每个请求开一个数据库连接，用完关掉。** 用 `async with AsyncSqliteSaver.from_conn_string(...)`
  包住整个请求；aiosqlite 的连接绑定事件循环，跨请求共享会互相干扰

**已知局限**（都在演示范围外）：

- `_pools` 在内存里，服务重启后老会话的引用编号会与新一轮错位——重启后建议新开会话
- 历史无限增长，聊得越久 input token 越大，暂未做截断
#### 合同体检接口

`/api/contracts/check/stream` 和聊天的流式是**两套事件**——前者推「进度」，后者推「token」：

| 事件 | 内容 | 前端用来做什么 |
|---|---|---|
| `status` | `{"stage": "thinking", "file": "sample_1.txt"}` | 显示「正在读取合同…」 |
| `progress` | `{"done": 3, "total": 8, "clause": "第三条 社会保险…", "risk": "high"}` | 更新「正在检查第 3/8 条」 |
| `done` | `{"file", "total", "high", "medium", "none", "unknown", "report", "report_path", "elapsed_ms"}` | 渲染报告 |
| `error` | `{"message": …}` | 显示错误 |

**实测**（`sample_1.txt`，8 条，13.5 秒）：

```
1/8  none     第一条 合同期限…
2/8  high     第二条 试用期 双方约定试用期为六个月…
3/8  high     第三条 社会保险 乙方自愿放弃社会保险…
4/8  high     第四条 工作时间与加班…
5/8  high     第五条 竞业限制…
6/8  high     第六条 婚育约定…
7/8  none     第七条 工资发放…
8/8  unknown  第八条 工作地点…
```

**三个实现要点**：

- **体检必须丢到线程里跑。** `check_contract_file` 是同步的（同步 openai 客户端），
  一份 8 条的合同要 11–13 秒；直接在 `async` 函数里调用会把事件循环占死，
  期间连 `/api/health` 都不响应。所以用 `asyncio.to_thread`
- **跨线程推进度要用 `loop.call_soon_threadsafe`。** `on_progress` 在工作线程里被调用，
  而 `asyncio.Queue` 只在事件循环线程里安全，不能直接 `put_nowait`
- **上传必须做编码兜底。** Windows 记事本存的 txt **默认是 GBK**，
  直接 `read_text(encoding="utf-8")` 会抛 `UnicodeDecodeError`。
  所以先试 `utf-8-sig`（吃掉 BOM）、再 `utf-8`、再 `gbk`

**上传的四道防护**：

| 防护 | 做法 |
|---|---|
| 路径穿越 | 只取文件名部分，反斜杠也切掉（`../../evil.txt` → `evil.txt`） |
| 超大文件 | 分块读、边读边累加，超 512 KB 立刻拒绝 |
| 非文本 | 先试 utf-8/gbk 能否干净解码，再查控制字符占比是否超过 2% |
| 重复文件 | **先比字节**，内容相同就直接复用已有文件，不新建 |

**重复文件那条值得说**：一开始只按文件名判重，传同一个文件三次就多出
`sample_1_1.txt`、`sample_1_1_1.txt`。改成先比内容（大小 → 字节），
一样的直接返回已有文件名，目录里就不会堆垃圾了。

**工具返回值的坑**：`check_contract` 一开始只把最严重的 3 条给模型，
结果模型回答「另外还有 2 条，建议你打开 contract_agent_sample_1.md 看完整清单」——
而用户**根本打不开服务器上的文件**。现在改成把全部风险条款（上限 10 条）都给模型，
并在 prompt 里明令禁止让用户去看文件。

---

### 前端页面

浏览器打开 `http://127.0.0.1:8000/` 就是完整界面，代码在 `app/static/`（三个文件，约 1229 行）。

**没有用 React/Vue，也没有构建步骤**——项目的卖点是 RAG 和 Agent，不是前端；
引一套构建工具会带来和主线无关的东西。原生 HTML + JS 足够。

**页面能做什么**：

- 输入问题 → 答案**逐字出现**（SSE），工具调用折在气泡里
- 答案里的 `[1]` `[2]` 可点，跳到下方「依据」并闪黄；依据可折叠
- 左侧栏列出会话，可新建 / 切换 / 删除；刷新页面自动恢复上次会话
- 空状态列出 `data/contracts` 下的示例合同，点一下就体检
- 📎 上传合同（或直接拖进来）→ 出现附件条 → 「开始体检」看逐条进度，
  也可以**直接在输入框提问**，比如「这份合同有什么坑」，Agent 会自己调体检工具

**六个必须知道的坑**（都踩过）：

- **不能用 `EventSource`。** 它只支持 GET，而提问要 POST JSON。
  所以用 `fetch` + `response.body.getReader()` 手动读流、按 `\n\n` 切消息
- **`TextDecoder` 必须传 `stream: true`，而且 decoder 要复用。**
  一个汉字 3 个字节，网络 chunk 的边界完全可能把它切成两半，
  不复用就会出乱码
- **中文输入法按回车不能直接发送。** 要判 `e.isComposing`，
  否则打拼音选词时一按回车消息就发出去了
- **流式过程中不能每个 token 都重渲染 markdown。** 一个答案 300 个 token，
  每次都重新解析是 O(n²)，长答案会明显卡。做法：流式时用 `textContent` 显示纯文本 +
  闪烁光标，`done` 到了再一次性渲染
- **转义必须发生在加标签之前。** 先 `escapeHtml` 再替换 `**加粗**`，
  顺序反了就是 XSS 洞
- **改了前端没生效，先怀疑浏览器缓存。** `StaticFiles` 默认不设 `Cache-Control`，
  浏览器会自己猜一个缓存时间——你以为功能写错了，其实是它根本没去取新文件。
  所以在 `app/main.py` 加了个中间件，给 `/` 和 `/static/` 都设上 `Cache-Control: no-cache`
  （不是不缓存，是**用之前先问服务器有没有变**：没变回 304，不重传）

**引用编号会重新排。** `RefPool` 的编号是跨多次检索连续的，答案里可能出现 `[7] [1] [4] [3]`。
前端按「在本答案里出现的顺序」重编成 `1..n`，正文角标和依据列表同步；
原始编号藏在 `data-oidx` 里用来查依据。只有能对上法条的 `[n]` 才做成角标，
所以 `[2024]` 这种方括号和模型幻觉出的编号会保持原文，不会变成点了没反应的死角标。

---

## 快速开始

> **依赖提示**：`requirements.txt` 里的版本已固定并验证可用，直接安装即可。
> `langchain-community` 请保持 `0.3.30`，单独升级会和 `ragas` 不兼容。

**语料已随仓库提交**：`data/clean/articles.jsonl`（0.71 MB / 1,251 条法条）。
clone 下来即可直接使用，**无需下载任何原始文件**。

```bash
pip install -r requirements.txt
```

### 可选：从头重建语料

只有在你需要验证数据管道、或同步最新法规时才执行（约 33 MB）：

```bash
cd data/raw
git clone --depth 1 https://github.com/lttxzmj/chinese-law-corpus.git
cd ../..

python scripts/W1/D1/01_scan.py           # 扫描全量       → _index.json
python scripts/W1/D1/02_filter.py         # 标题关键词候选 → _keyword_candidates.json
python scripts/W1/D1/03_content_scan.py   # 正文密度候选   → _content_candidates.json
python scripts/W1/D1/04_select.py         # 规则 + 人工剔除 → _selected.json（白名单）
python scripts/W1/D1/05_build.py          # 按名单展开     → articles.jsonl

# 数据体检（可选，验证产物）
python scripts/W1/D2/06_check_data.py     # → data/eval/check_report.json

# 检索冒烟测试（可选，验证"口语 vs 法条"的落差）
python scripts/W1/D2/07_bm25_baseline.py  # → data/eval/bm25_baseline_result.md
```

若 GitHub 访问不畅，可直接下载
[仓库 zip](https://github.com/lttxzmj/chinese-law-corpus/archive/refs/heads/master.zip)
解压到 `data/raw/`，再执行上面的命令。

产物：`data/clean/articles.jsonl`，每行一条法条。

### 配置 API Key

问答与 Agent 相关脚本需要 DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）。
在项目根目录建一个 `.env`（已在 `.gitignore` 中，不会被提交）：

```bash
DEEPSEEK_API_KEY=sk-xxxxxx
```

### 运行问答与 Agent

`data/clean/articles_vec.npy`（4.9 MB）已随仓库提交，可直接用。
若要重新生成（需先下载 bge-m3，约 2.2 GB）：

```bash
# 国内网络建议先设镜像
#   PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/W1/D4/09_build_vectors.py
```

然后按需要跑任意一个（前三步产物是后一步的输入）：

```bash
python scripts/W1/D3/08_easy_rag.py       # 字面检索版 RAG（答案正确 3/10）
python scripts/W1/D4/010_vector_rag.py    # 向量检索版 RAG（9/10）
python scripts/W1/D5/011_agent.py         # 多工具 Agent（自主改写检索词 + 计算）
python scripts/W1/D6/012_hitl.py          # 多轮对话 + 写操作人工确认

# 检索对比与测试集（可选）
python scripts/W1/D7/013_hybrid_eval.py   # → hybrid_eval_result.md（13 条子集对比）

# 测试集与检索评估（40 条）
python scripts/W2/D8/014_build_golden_set.py    # → data/eval/golden_set.jsonl
python scripts/W2/D8/015_eval_retrieval.py       # → data/eval/retrieval_eval.md

# 生成质量评估（会调很多次 LLM，先跑 2 条验证）
python scripts/W2/D9/016_ragas_eval.py          # → data/eval/ragas_eval.md

# 合同体检（读 data/contracts 下的 .txt，逐条检索法条并判断风险）
python scripts/W2/D10/017_contract_check.py      # → contract_check_result.md + raw.json
python scripts/W2/D10/018_eval_contract.py       # → contract_eval.md

# 合同体检的 Agent 版（用户说话，Agent 自己决定要不要体检）
python scripts/W2/D11/019_contract_agent.py      # → contract_agent_result.md

# 性能与成本测量（会跑 Agent 和一次合同体检，约 1 分钟）
python scripts/W2/D12/020_benchmark.py           # → data/eval/benchmark.md

# 向量库（需要先 cd milvus && docker compose up -d）
python scripts/W4/D18/021_milvus_basics.py      # → data/eval/milvus_basics.md

# 检索后端对比（需要 Milvus 起着；约 1 分钟）
python scripts/W4/D20/022_vector_store_bench.py  # → data/eval/vector_store_compare.md

# 切换检索后端（不设则用 numpy）
# $env:VECTOR_BACKEND="milvus"

# Web 接口（启动后打开 http://127.0.0.1:8000/docs 试接口）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---
## 目录结构

```
app/                    Web 层（FastAPI）
  main.py                 路由：health / ask / ask-stream / 会话三件套 / 合同三件套 / 首页
  schemas.py              Pydantic 请求/响应模型（自动校验 + 生成接口文档）
  deps.py                 检索器单例 + 按会话复用引用池 + 异步 checkpointer 连接
  streaming.py            SSE 流式生成器（生产者队列 + 心跳 + 事件过滤）
  contract_stream.py      合同体检的 SSE 生成器（线程 + 队列 + 逐条进度）
  observability.py        结构化日志（每请求一行 JSON）+ request_id + 统计汇总
  memory.py               长期记忆：抽取用户事实 → 存 Milvus → 新会话召回
  main_helpers.py         引用提取（两个接口共用，避免循环 import）
  static/                 前端单页（浏览器打开 / 就能用）
    index.html              页面结构：侧边栏 + 对话区 + 输入区
    style.css               样式：气泡、表格、引用角标、折叠、进度圈
    app.js                  全部逻辑：SSE 解析、markdown 渲染、会话管理
milvus/                  Milvus 单机环境（docker compose）
  docker-compose.yml      etcd + SeaweedFS(顶替 MinIO) + milvus-standalone
  volumes/                数据卷（约 400 MB，不提交）
data/
  raw/                  上游原始语料（约 33 MB，不提交；重建方式见上）
    _index.json               全量索引
    _keyword_candidates.json  关键词候选
    _content_candidates.json  正文密度候选
    _selected.json            ★ 最终白名单（由 04_select.py 生成）
  output/               写操作产出（AI 生成并保存的文书）
  agent_checkpoints.db  会话状态库（SQLite，checkpointer 落盘）
  contracts/            待体检的劳动合同（.txt）
    sample_1~5.txt        5 份，格式各异（第X条 / 1. / 一、 / 无编号 / 混合）
    labels.json           人工标注（12 个风险点 / 26 条正常条款）
  clean/
    articles.jsonl      法条级语料，1,251 条（提交，核心资产）
    articles_vec.npy    bge-m3 向量矩阵，1251 × 1024（4.9 MB）
  eval/
    check_report.json         数据体检报告
    bm25_baseline_result.md   冒烟测试结果
    easy_rag_result.md        字面匹配的问答结果
    vector_rag_result.md      向量检索的问答结果
    agent_d5_result.md        Agent 的调用链与答案
    hybrid_eval_result.md     三种检索的排名对比
    retrieval_eval.md         40 条测试集的 HitRate / MRR
    ragas_eval.md             Ragas 生成评估（faithfulness / answer_relevancy）
    ragas_samples.jsonl       34 条评估样本（问题 + 答案 + 检索到的法条）
    contract_check_result.md  合同体检报告（5 份合同）
    contract_check_raw.json   体检的结构化结果（评估脚本读取）
    contract_eval.md          合同体检的量化评估
    contract_agent_result.md  Agent 三组场景的调用链与答案
    contract_agent_sample_1.md Agent 版体检生成的报告（运行后生成，不提交）
    benchmark.md              性能与成本测量（冷启动 / 检索 / 端到端 / 体检）
    milvus_basics.md          Milvus 建库/索引/ef/删除 实测与参数表
    vector_store_compare.md   numpy vs Milvus 的延迟 / 召回 / 过滤对比
    golden_set.jsonl          40 条测试集（含 gold 法条 id）
  src/
    retriever.py           BM25 / 向量 / 混合 / Milvus 四种检索器（后两者同接口，可切换）
    citation.py            引用池 RefPool（多次检索编号不串台）
    contract.py            合同体检核心：切分 + 逐条判断
    agent.py               Agent 图：模型节点 + 工具节点 + 循环判断
    tools/                 Agent 工具集合
      search.py              make_search_law（检索 + 进引用池）
      calc.py                make_calc_severance（第 47 条规则代码化）
      check.py               make_check_contract（调 contract.py，返回摘要 + 写报告）
  scripts/
    W1/                 第 1 周
      D1/               数据管道（五步）
        01_scan.py          扫描全量，生成索引
        02_filter.py        标题关键词粗筛
        03_content_scan.py  正文密度扫描
        04_select.py        ★ 唯一做取舍的地方：规则 + 人工剔除
        05_build.py         按白名单展开成法条级 JSONL
      D2/               数据校验与检索基线
        06_check_data.py    数据体检（五项检查）
        07_bm25_baseline.py 术语落差表 + BM25 冒烟测试
      D3/               最小 RAG 链路
        08_easy_rag.py      字面检索 + 生成 + 引用校验
      D4/               向量检索
        09_build_vectors.py 编码 1,251 条法条 → articles_vec.npy
        010_vector_rag.py   语义检索 + 生成 + 引用校验
      D5/               Agent（LangGraph + 工具调用）
        011_agent.py        search_law / calc_severance 两个工具 + 引用池
      D6/               多轮与人工确认
        012_hitl.py         SqliteSaver + interrupt 写操作确认
      D7/               混合检索对比
        013_hybrid_eval.py  BM25 / 向量 / RRF 三方对比
    W2/                 第 2 周
      D8/               测试集与评估
        014_build_golden_set.py  40 条测试集 + --find 查法条
        015_eval_retrieval.py    检索评估（HitRate / MRR）+ 未命中诊断
      D9/               生成评估
        016_ragas_eval.py    Ragas faithfulness / answer_relevancy
      D10/              合同体检
        017_contract_check.py  切分 → 逐条检索 → 判断 → 报告
        018_eval_contract.py   与 labels.json 比对，算三个指标
      D11/              合同体检接入 Agent
        019_contract_agent.py  三工具 + 三组场景测试
      D12/              性能与成本测量
        020_benchmark.py  冷启动 / 检索 / 端到端 / 合同体检
    W4/                 第 4 周（向量库与检索选型）
      D18/              向量库 Milvus 基础操作
        021_milvus_basics.py  Schema / HNSW 索引 / flush / 标量过滤 / ef 实测
      D20/              检索后端对比
        022_vector_store_bench.py  numpy vs Milvus：延迟 / 召回 / 过滤 / 冷启动
  explain/                开发日志
```

---

## 免责声明

本工具仅提供法律信息检索与初步判断，**不构成法律意见**。
具体案件请咨询执业律师，或当地劳动人事争议仲裁委员会 / 人力资源社会保障部门。

数据来源为公开法律法规文本，如与官方发布有出入，以官方为准。
