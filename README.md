# 职场权益 Copilot

> 所处秋招的同学或者不懂法的人用大白话描述遭遇，它找到对应法条、翻译成人话、该算的帮你算。

**当前进度**：W1 D1 —— 语料管道完成，检索与问答开发中

---

## 这是什么

所处秋招的大学生或者普通人遇到劳动纠纷时，往往不知道该问什么、看不懂法条、不知道能拿多少钱。
搜索引擎给一堆链接，通用大模型则会编造条文号。

这个项目把**官方法律条文**做成可检索的知识库，让用户用大白话提问，得到
「结论 + 法条依据 + 该算的数」。

**能力清单**

- [x] 法条级语料构建（当前阶段）
- [ ] 法条问答：检索 + 生成 + 法条引用
- [ ] 权益计算：经济补偿金 / 试用期上限 / 加班费
- [ ] 合同体检：逐条比对法条，输出风险清单
- [ ] 评估体系：法条引用可程序化校验

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
| 字数分布 | 最短 20 / 中位 87 / P90 210 / 最长 797 字 |
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
- **规模有限**：主库 1,251 条，指标置信区间较宽，故评估更关注
  相对提升而非绝对值

---

## 快速开始

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

python scripts/01_scan.py           # 扫描全量       → _index.json
python scripts/02_filter.py         # 标题关键词候选 → _keyword_candidates.json
python scripts/03_content_scan.py   # 正文密度候选   → _content_candidates.json
python scripts/04_select.py         # 规则 + 人工剔除 → _selected.json（白名单）
python scripts/05_build.py          # 按名单展开     → articles.jsonl
```

若 GitHub 访问不畅，可直接下载
[仓库 zip](https://github.com/lttxzmj/chinese-law-corpus/archive/refs/heads/master.zip)
解压到 `data/raw/`，再执行上面的命令。

产物：`data/clean/articles.jsonl`，每行一条法条。

---

## 目录结构

```
data/
  raw/                  上游原始语料（约 33 MB，不提交；重建方式见上）
    _index.json               全量索引
    _keyword_candidates.json  关键词候选
    _content_candidates.json  正文密度候选
    _selected.json            ★ 最终白名单（由 04_select.py 生成）
  clean/
    articles.jsonl      法条级语料，1,251 条（提交，核心资产）
scripts/
  01_scan.py            扫描全量，生成索引
  02_filter.py          标题关键词粗筛
  03_content_scan.py    正文密度扫描
  04_select.py          ★ 唯一做取舍的地方：规则 + 人工剔除
  05_build.py           按白名单展开成法条级 JSONL
explain/                开发日志
```

---

## 免责声明

本工具仅提供法律信息检索与初步判断，**不构成法律意见**。
具体案件请咨询执业律师，或当地劳动人事争议仲裁委员会 / 人力资源社会保障部门。

数据来源为公开法律法规文本，如与官方发布有出入，以官方为准。
