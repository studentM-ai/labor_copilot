"""
性能与成本测量
主要量（分开量）：
1.冷启动的时间 2.单次检索时间  3.agent端到端的时间与token用量 4.合同体验时间
token统计：Agent 内部的调用可以从每条 AIMessage 的 usage_metadata 里累加。
"""
import json,os,sys,time
from pathlib import Path

# 到项目根
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
  ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent import build_graph
from src.citation import RefPool
from src.contract import check_contract_file,load_rows
from src.retriever import BM25Retriever,VectorRetriever
from src.tools import make_search_law,make_calc_severance
from langchain_core.tools import tool

load_dotenv()
EVAL = ROOT / "data" / "eval"
MODEL = "deepseek-chat"
TOP_K = 5
LIMIT = 20

#检索测几次取平均值
RETRIEVE_ROUNDS=3

#Agent测两个问题，一个简单问答，一个算数
CASES=[
  "试用期最长能有多久",
  "我工作了 3 年半，月薪 12000，被裁员能拿多少",
]

#体验用哪份合同
BENCH_CONTRACT="sample_1.txt"

SYSTEM_PROMPT = """你是劳动法助手，服务对象是不懂法的普通人。

工作方式：
1. 回答劳动法问题前，先调用 search_law 检索法条，不要凭自己的知识回答。
2. 需要算经济补偿金额时，调用 calc_severance，不要自己心算。
3. 引用法条时用 [1] [2] 编号。
4. 用大白话解释。"""

def ms(seconds):
  """
  秒转毫秒，保留一位小数
  """
  return f"{seconds*1000:.1f}ms"

def tokens_of(messages):
  """
  累加一趟agent 调用消耗的token
  返回的 AIMessage 的 usage_metadata 上（输入 + 输出 + 总数）
  """
  total=0
  for m in messages:
    um=getattr(m,"usage_metadata",None)
    if um:
      total+=um.get("total_tokens",0)or 0
  return total

def tool_count(messages):
  """
  这次调用一共触发了几次工具
  """
  n=0
  for m in messages:
    if m.type=="ai" and getattr(m,"tool_calls",None):
      n+=len(m.tool_calls)
  return n

def main():
  api_key=os.getenv("DEEPSEEK_API_KEY")
  md = ["# 性能与成本测量", "", f"- 模型：{MODEL}", "- 语料：1,251 条法条", ""]
  #1.冷启动
  ts=time.perf_counter()  #专门用于测速的方法
  rows=load_rows()
  t_load=time.perf_counter()-ts

  ts=time.perf_counter()
  bm25=BM25Retriever(rows)
  t_bm25=time.perf_counter()-ts   

  ts=time.perf_counter()
  vec=VectorRetriever(rows)
  t_vec=time.perf_counter()-ts

  md.append("## 冷启动（只发生一次）")
  md.append("")
  md.append("| 步骤 | 耗时 |")
  md.append("|---|---|")
  md.append(f"| 读 1,251 条语料 | {ms(t_load)} |")
  md.append(f"| 建 BM25 索引（切词 + 词频 + df） | {ms(t_bm25)} |")
  md.append(f"| 加载 bge-m3 + 读向量矩阵 | **{ms(t_vec)}** |")
  md.append(f"| 合计 | **{ms(t_load + t_bm25 + t_vec)}** |")
  md.append("")
  md.append("> 冷启动的主要点是加载 bge-m3（模型文件约 2 GB）。")
  md.append("> 实际服务会把它常驻内存，只有第一次启动付这个代价。")
  md.append("")

  #2.单次检索
  print("测量单词检索")
  query="试用期最长能有多久"
  ts=time.perf_counter()
  for _ in range(RETRIEVE_ROUNDS):
    bm25.retrieve(query,TOP_K)
  t_bm25_query=(time.perf_counter()-ts)/RETRIEVE_ROUNDS  #取平均值

  ts=time.perf_counter()
  for _ in range(RETRIEVE_ROUNDS):
    vec.retrieve(query,TOP_K)
  t_vec_query=(time.perf_counter()-ts)/RETRIEVE_ROUNDS

  md.append(f"## 单次检索（取 {RETRIEVE_ROUNDS} 次平均）")
  md.append("")
  md.append("| 检索方式 | 单次耗时 |")
  md.append("|---|---|")
  md.append(f"| BM25（纯计算，无模型） | {ms(t_bm25_query)} |")
  md.append(f"| 向量（含一次 bge-m3 编码） | {ms(t_vec_query)} |")
  md.append("")
  md.append("> 向量检索慢在「把问题编码成向量」这一步；")
  md.append("> 矩阵乘本身很快（1251 × 1024 一次算完）。")
  md.append("")

  #3.Agent端到端
  print("测量 Agent 端到端...")
  md.append("## Agent 端到端")
  md.append("")
  md.append("| 问题 | 耗时 | 工具调用次数 | token |")
  md.append("|---|---|---|---|")

  for question in CASES:
    #每个问题用新pool
    pool=RefPool()
    tools=[make_search_law(vec,pool),make_calc_severance()]
    app=build_graph(tools,api_key,model=MODEL)

    ts=time.perf_counter()
    try:
      result=app.invoke(
        {
          "messages":[
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question)
          ]
        },
        config={"recursion_limit":LIMIT}
      )
    except Exception as e:
      print(f"[出错]{e}")
      md.append(f"| {question[:22]} | 出错：{e} | - | - |")
      continue
    t_agent=time.perf_counter()-ts   #同样要减去起点

    n_tool=tool_count(result["messages"])
    n_token=tokens_of(result["messages"])
    print(f"  {question[:22]}  {ms(t_agent)}  {n_tool} 次工具  {n_token} token")
    md.append(f"| {question[:22]} | {ms(t_agent)} | {n_tool} | {n_token} |")

  md.append("")
  md.append("> token 只统计 agent 节点自身的调用（从 `usage_metadata` 累加）；")
  md.append("> 工具内部的模型调用不在这份统计里。")
  md.append("")

  #4.合同体验
  print(f"测量合同体验({BENCH_CONTRACT})")
  ts=time.perf_counter()
  chk=check_contract_file(BENCH_CONTRACT,vec,TOP_K,verbose=False)
  t_contract=time.perf_counter()-ts
  md.append("## 合同体检")
  md.append("")
  if "error" in chk:
    md.append(f"体检失败：{chk['error']}")
  else:
    n_clause = len(chk["clauses"])
    md.append("| 项 | 结果 |")
    md.append("|---|---|")
    md.append(f"| 合同 | {BENCH_CONTRACT}（{n_clause} 条条款） |")
    md.append(f"| 总耗时 | **{t_contract:.1f} s** |")
    md.append(f"| 每条平均 | **{t_contract / n_clause:.1f} s** |")
    md.append("| 每条的开销 | 1 次向量检索 + 1 次 LLM 判断 |")
    md.append("")
    md.append(f"> 按这个速度，一份 20 条的合同约需 {t_contract / n_clause * 20:.0f} 秒。")
    md.append("> 这部分 token 未计入上面的统计（工具内部自己开的 client）。")

  md.append("")

  output = EVAL/"benchmark.md"
  with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

  print()

if __name__=="__main__":
  main()