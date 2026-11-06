"""
做检索评估，只做检索不调模型
四个指标：HitRate@1 / HitRate@3 / HitRate@5 / MRR@5    HitRate命中    MRR平均倒数排名 
按类型分组：整体命中率会掩盖问题比如整体 0.85，但分组后可能发现 duobu（多步题）明显更差。
"""

import json
import sys
from pathlib import Path
#找到根目录
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT)) #添加项目根优先级最高

from src.retriever import BM25Retriever, VectorRetriever, HybridRetriever

CLEAN = ROOT / "data" / "clean"
EVAL = ROOT / "data" / "eval"

TOP_K=5

def load_rows():
  """
  读法条语料
  """
  rows = []
  for line in open(CLEAN / "articles.jsonl", encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows

def load_golden_set():
  """
  读40条测试集，返回列表,大致内容如下
  {
    "id": "q01",
    "question": "公司不给我交社保怎么办",
    "gold_law_ids": ["social-insurance-law-0058", "..."],
    "type": "kouyu",
    "must_refuse": false
  }
  """
  records=[]
  for line in open(EVAL/"golden_set.jsonl",encoding="utf-8"):
    if line.strip():
      records.append(json.loads(line))
  return records

def find_rank(hits,gold_ids):
  """
  在检索结果里找gold，返回它的位次 0表示前五名都没有
  hits:检索返回的法条列表，按相关度从高到低
  gold_ids: 这道题的标准答案 id 列表（可能有多个）
  """
  rank=0
  position=1

  for h in hits:  #检索返回的每一条
    if h["id"] in gold_ids:   #如果在标准答案里 则取位次
      rank=position
      break
    position+=1
  return rank #取位高的


def eval_one_retriever(retriever, records):
  """
  对一种检索器跑完全部题目,整合排名
  retriver:检索器对象
  records：40条测试记录
  返回一个列表，{"id": 题号, "type": 类型, "question": 原问题, "rank": 排名}
  """
  results=[]
  for rec in records:
    #超范围题没有gold，它们的正确行为是“拒答”，不该被检索到，所以跳过
    if len(rec["gold_law_ids"])==0:
      continue

    hits= retriever.retrieve(rec["question"], TOP_K)
    rank=find_rank(hits,rec["gold_law_ids"])

    results.append(
      {
          "id": rec["id"],
          "type": rec["type"],
          "question": rec["question"],
          "rank": rank,
      }
    )
  return results


def compute_metrics(results):
  """
  计算四个指标
  results:eval_one_retriever的返回值（处理后的有排名检索结果）
  返回 {"total": 题数, "hit1": ..., "hit3": ..., "hit5": ..., "mrr": ...}
  """
  total=len(results)
  if total==0:
    return {"total": 0, "hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr": 0.0}

  hit1=0
  hit3=0
  hit5=0
  reciprocal_sum = 0.0
  for r in results:
    rank=r["rank"]

    #HitRate@1:gold排第一
    if rank==1:
      hit1+=1
    
    #HitRate@3：gold 在前 3
    if rank>=1 and rank <=3:
      hit3+=1

    if rank>=1 and rank<=5:
      hit5+=1

    #MRR:命中得1/rank 没命中为0
    if rank>0:
      reciprocal_sum+=1.0/rank
  return {
    "total": total,
    "hit1": hit1 / total,
    "hit3": hit3 / total,
    "hit5": hit5 / total,
    "mrr": reciprocal_sum / total,
    }

def group_by_type(results):
  """
  按题目类型分组
  返回{"kouyu": [...], "term": [...], "duobu": [...]}
  """
  groups={}
  for r in results:
    t=r["type"]
    if t not in groups:
      groups[t]=[]
    groups[t].append(r)
  return groups

def miss_details_markdown(records, retriever, miss_ids):
  """
  生成「未命中题的检索结果」markdown 片段
  records:   40 条测试记录
  retriever: 检索器（用向量那个）
  miss_ids:  未命中的题号列表，比如 ["q04", "q12"]
  
  返回一个字符串列表，每个元素是一行 markdown
  """
  md = []

  # 先把 records 转成 {题号: 记录} 的字典，方便按 id 取
  by_id = {}
  for rec in records:
    by_id[rec["id"]] = rec

  for qid in miss_ids:
    if qid not in by_id:
      continue

    rec = by_id[qid]  #拿到未命中的记录
    hits = retriever.retrieve(rec["question"], TOP_K)

    md.append(f"### {qid} {rec['question']}")
    md.append("")
    md.append(f"**gold**：{rec['gold_law_ids']}")
    md.append("")
    md.append("| # | 检索到 | 摘要 |")
    md.append("|---|---|---|")

    position = 1
    for h in hits:
      snippet = h["text"][:38].replace("\n", " ")
      title = f"《{h['law_title']}》{h['display_no']}"
      md.append(f"| {position} | {title} | {snippet}... |")
      position += 1

    md.append("")

  return md


def main():
  rows=load_rows()
  records=load_golden_set() #测试集

  print(f"语料{len(rows)}条，测试集{len(records)}条")

  #统计有gold的题
  scored=0
  for rec in records:
    if len(rec["gold_law_ids"])>0:
      scored+=1
  print(f"可判定题 {scored} 条（超范围题不参与）")

  #建立三种检索器
  bm25 = BM25Retriever(rows)
  vec = VectorRetriever(rows)
  hyb = HybridRetriever(bm25, vec)

  retrievers=[
    ("BM25", bm25),
    ("向量", vec),
    ("混合", hyb),
  ]

  all_results={}  #名字：结果
  all_metrics={}   #名字：四项指标
  for name ,retriever in retrievers:
    print(f"现在跑 {name} ")
    results=eval_one_retriever(retriever,records)

    all_results[name]=results
    all_metrics[name]=compute_metrics(results)

  #终端输出：整体
  print()
  print("整体指标（只看有 gold 的题）")
  print(f"{'检索方式':<10}{'HitRate@1':>12}{'HitRate@3':>12}{'HitRate@5':>12}{'MRR@5':>10}")
  print("-" * 78)

  for name, _ in retrievers:
    m = all_metrics[name]
    print(f"{name:<10}{m['hit1']:>12.3f}{m['hit3']:>12.3f}"
          f"{m['hit5']:>12.3f}{m['mrr']:>10.3f}")

  #终端输出：按类型分组
  print()
  print("按题型分组（用向量检索看，因为它最强）")
  vec_results=all_results["向量"]   #取出向量检索结果
  groups=group_by_type(vec_results) #按类型分
  print(f"{'类型':<12}{'条数':>6}{'HitRate@5':>12}{'MRR@5':>10}")
  type_metrics = {}   #类型名：指标结果
  for t in ("kouyu", "term", "duobu"):
    if t not in groups:
      continue
    m=compute_metrics(groups[t])
    type_metrics[t]=m
    print(f"{t:<12}{m['total']:>6}{m['hit5']:>12.3f}{m['mrr']:>10.3f}")

  #终端输出：没命中的题
  print()
  print("向量检索没进 Top5 的题")
  miss_list = []
  for r in vec_results:
    if r["rank"]==0:
      miss_list.append(r)
      print(f"[{r['id']}] ({r['type']}) {r['question'][:44]}")
  if len(miss_list)==0:
    print("一道题都没有")

  #md文件
  md=[]
  md.append("# 检索评估（40 条测试集）")
  md.append("")
  md.append(f"- 语料：{len(rows)} 条")
  md.append(f"- 测试集：{len(records)} 条，其中可判定 {scored} 条")
  md.append(f"- 检索都取 Top{TOP_K}")
  md.append("")
  md.append("## 整体指标")
  md.append("")
  md.append("| 检索方式 | HitRate@1 | HitRate@3 | HitRate@5 | MRR@5 |")
  md.append("|---|---|---|---|---|")
  for name, _ in retrievers:     
    m = all_metrics[name]
    md.append(f"| {name} | {m['hit1']:.3f} | {m['hit3']:.3f} | "
              f"{m['hit5']:.3f} | {m['mrr']:.3f} |")

  md.append("")
  md.append("## 按题型分组（向量检索）")
  md.append("")
  md.append("| 类型 | 条数 | HitRate@5 | MRR@5 |")
  md.append("|---|---|---|---|")  

  for t in ("kouyu", "term", "duobu"):
    if t in type_metrics:
      m = type_metrics[t]
      md.append(f"| {t} | {m['total']} | {m['hit5']:.3f} | {m['mrr']:.3f} |")

  md.append("")
  md.append("## 向量检索未命中 Top5 的题")
  md.append("")

  if len(miss_list) == 0:
    md.append("无")
  else:
    for r in miss_list:
      md.append(f"- `{r['id']}` ({r['type']}) {r['question']}")

  md.append("")

  md.append("## 附录：未命中题的检索结果")
  md.append("")

  # 从 vec_results 里挑出没命中的题号
  miss_ids = []
  for r in vec_results:
    if r["rank"]==0:
      miss_ids.append(r["id"])

  # 生成附录内容（每道题一张 Top5 表）
  appendix = miss_details_markdown(records, vec, miss_ids)
  for line in appendix:
    md.append(line)

  md.append("")
  md.append("## 已知的标注问题")
  md.append("")
  md.append("`q31`「竞业协议里没写补偿，这个协议还有效吗」被判为未命中，但检索返回的 Top2 是《最高人民法院关于审理劳动争议案件适用法律问题的解释（一）》第三十六条——该条直接回答「约定了竞业限制但未约定补偿」的处理方式，比 gold 标注的《劳动合同法》第二十三条更贴题。")
  md.append("")
  md.append("这属于 **gold 标注粒度偏窄**，不是检索失败。本次**不做修改**，以免出现「为了让指标好看而调整标准」的嫌疑；后续扩测试集时应把司法解释一并纳入 gold。")
  md.append("")

  output = EVAL / "retrieval_eval.md"
  with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
  print()

if __name__ == "__main__":
  main()
