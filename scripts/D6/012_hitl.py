"""
使用checkpointer(检查点存储器，它会存很多内容，比如state的字段messages，执行游标啥的)用来断点续跑、interrupt 恢复
来多轮对话与关键节点人工确认
解决D5状态只存在内存，invoke一返回就丢失的问题
实现多轮，关键节点中断
"""

import json,os,sqlite3
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langgraph.graph import END,START,MessagesState,StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver  
 #用SQLite文件持久保存Agent的State，写入本地db文件
from langgraph.types import Command,interrupt
 #Command更新state，动态指定下一节点   HITL终端时会携带resume参数回填人工输入

load_dotenv()

CLEAN = Path("data/clean")
EVAL = Path("data/eval")
OUTPUT = Path("data/output")  #写工具调用相关
DB = Path("data/agent_checkpoints.db")  
for d in (EVAL, OUTPUT, DB.parent):
    d.mkdir(parents=True, exist_ok=True)

MODEL = "deepseek-chat"
TOP_K = 5
RECURSION_LIMIT = 16  #节点执行限制

#检索器与引用池（与D5差不多）
class VectorRetriever:
  def __init__(self, rows):
    self.rows = rows
    self.vecs = np.load(CLEAN / "articles_vec.npy")
    self.model = SentenceTransformer("BAAI/bge-m3")

  def retrieve(self, query, k=TOP_K):
    q = self.model.encode([query], normalize_embeddings=True)[0]
    idx = np.argsort(-(self.vecs @ q))[:k]
    return [self.rows[i] for i in idx]

class RefPool:
  """全局引用池：多轮检索下编号不串台，按 id 去重"""

  def __init__(self):
    self.items = [] #存法条对象
    self.by_id = {} #存id ： 编号

  def add(self, rows):
    out = []  #存编号 法条
    for r in rows:
        if r["id"] in self.by_id:
            out.append((self.by_id[r["id"]], r))
        else:
            self.items.append(r)
            nidx = len(self.items)
            self.by_id[r["id"]] = nidx
            out.append((nidx, r))
    return out

#建图
def build_app(retriever:VectorRetriever,pool:RefPool,api_key,checkpointer):
   
  #检索工具
  @tool
  def search_law(query:str,k:int=TOP_K)->str:
      """
      检索劳动法条文。

      注意：检索词要用法律术语，不要直接用用户的口语。
        用户说「加班费」→ 用「延长工作时间 工资 报酬」
        用户说「社保」  → 用「社会保险 缴纳」
        用户说「年假」  → 用「年休假 天数」
        同一个问题可以检索多次，换不同关键词。

      """
      hits=retriever.retrieve(query,k) #检索内容
      numbered=pool.add(hits)  #加入全局引用池，得到编号

      # 注意返回给模型的是文本，每条前面需要带全局编号
      return "\n\n".join(
        f"[{nidx}] 《{r['law_title']}》{r['display_no']}\n{r['text']}" for nidx,r in numbered
      )

  #经济补偿计算工具
  @tool
  def calc_severance(years:float,monthly_salary:float)->str:
      """
      按《劳动合同法》第47条第一款计算经济补偿金（N）。
  
      规则：每满一年支付一个月工资；六个月以上不满一年的按一年计算；不满六个月的支付半个月。
      只在用户明确问了金额、且已知工作年限和月工资时调用。
      缺参数就反问用户，一定不要猜。
      注意：本工具没有考虑特殊情况，如《劳动合同法》第47条第二款的规定。（高薪封顶和年限）
      """
  
      #年限
      if years<0:
        return "工作年限不能为负数，请检查输入。"
      if years<0.5:
        months=0.5   #不满半年则算半个月
      elif years<1:
        months=1     #六个月以上（含）不满一年  按一年
      else:
        whole_years=int(years) #取整
        fraction=years-whole_years  #得到年份差值
        if fraction>=0.5:
          months=float(whole_years+1)   #满半年加一个月
        elif fraction>0:
          months=whole_years+0.5  #不满半年加半个月
        else:
          months=float(whole_years)   #整年
  
      amount=months*monthly_salary    #计算金额
      return (f"经济补偿金 = {months} × {monthly_salary:.0f} = {amount:.0f} 元\n"
              f"依据：《中华人民共和国劳动合同法》第四十七条\n"
              f"限制：未计入第二款的高薪封顶与十二年上限")

  #新增 写操作（需要人工确认）
  @tool
  def save_document(filename:str)->str:
    """
    把对话里最近一条完整回答保存成文件。这是写操作，执行前会请求用户确认。
    只在用户明确要求「保存」「导出」「写成文件」时才调用。
    """
    return f"_NEED_APPROVAL_{filename}"

  tools=[search_law,calc_severance,save_document]
  by_name={t.name:t for t in tools} #得到{工具名：工具} 字典

  client=ChatOpenAI(
    model=MODEL,
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
    max_tokens=1500
  )
  client_with_tools=client.bind_tools(tools)

  def call_model(state:MessagesState):
    return {"messages": [client_with_tools.invoke(state["messages"])]}    

  def should_continue(state: MessagesState) :
    last=state["messages"][-1]  #取最后一条消息
    if getattr(last,"tool_calls",None):  #如果最后一条消息是工具调用的结果，继续调用工具
      return "tools"   #继续调用工具
    return END

  def recent_answer(messages):
    """找最近的一条，跳过空的AI消息，找最后一条有实质内容的"""
    for m in reversed(messages):     #倒着找
      if getattr(m,"type","")=="ai" : #是ai的就提取内容
        text=(m.content or "").strip()
        if len(text)>=80: #太短的不算一份文档
          return m.content
    return ""

  #自定义工具节点，不用ToolNode是因为要在执行save_document前插入interrupt
  def call_tools(state:MessagesState):
    last=state["messages"][-1]  #消息的最后一条
    results=[]
    for tc in last.tool_calls:  #默认是工具调用
      name=tc["name"] #拿名
      args=tc["args"] #拿参数

      #先暂停问用户
      if name=="save_document":
        filename=args.get("filename","output.md")
        content=recent_answer(state["messages"][:-1]) #不要最后一条
        if not content:
          results.append(ToolMessage(
            content="没有找到可保存的内容，请先生成回答。",
            tool_call_id=tc["id"]))
          continue

        #interrupt() 会抛出暂停信号，checkpoint已经存盘(当前状态存进库)，下次invoke会从Command（resume=...）这里继续
        decide=interrupt({
          "action":"save_document",
          "filename":filename,
          "字数":len(content),
          "Preview":content[:200]   #预览
        })

        if decide!="approve":
          results.append(ToolMessage(
            content="用户取消保存操作。",
            tool_call_id=tc["id"]
          ))
          continue

        #如果用户批准则开始写入文件
        path=OUTPUT/filename
        path.write_text(content,encoding="utf-8")
        results.append(ToolMessage(
          content=f"已保存到{path}({len(content)}字)",
          tool_call_id=tc["id"],
        ))
        continue

      #如果是普通工具
      fn=by_name.get(name)  #拿到对应工具
      try:
        out=fn.invoke(args) if fn else f"未知工具{name}"
      except Exception as e:
        out=f"工具出错：{e}"
      results.append(ToolMessage(content=str(out),tool_call_id=tc["id"]))

    return {"messages":results}

  g=StateGraph(MessagesState)
  g.add_node("agent",call_model)
  g.add_node("tools",call_tools)
  g.add_edge(START,"agent")
  g.add_conditional_edges("agent",should_continue,["tools",END])
  g.add_edge("tools","agent")
  return g.compile(checkpointer=checkpointer)

SYSTEM_PROMPT = """你是劳动法助手，服务对象是不懂法的普通人与秋招大学生。
工作方式（必须遵守）：
1. 回答任何劳动法问题前，必须先调用 search_law 检索法条，不要凭自己的知识回答。
2. 检索时用法律术语，不要直接用用户的口语。参考工具说明里的例子。
3. 需要计算经济补偿金额时，必须调用 calc_severance，不要自己心算。
   如果用户没说工作年限或月工资，先问清楚，不要假设一个数字。
4. 每句话后面用 [1] [2] 这样的编号，标注依据的是哪一条。
5. 检索到的法条不足以回答时，直接说「未找到相关依据，建议咨询专业律师」，不许推测。
6. 如果问题不属于劳动法范畴，说明它属于哪个领域并建议咨询，不要调用检索工具。
7. 用大白话解释。可以引用法条原文，但要再解释一遍它是什么意思。
8.用户要求「保存」「导出」「写成文件」时才调用 save_document。
9. 如果拒答（没有调用检索工具），不要标注任何引用编号。
"""

def main():
  rows=[json.loads(l) for l in
        open(CLEAN / "articles.jsonl", encoding="utf-8") if l.strip()]

  retriever = VectorRetriever(rows)
  pool = RefPool()          # 注意：多轮共用同一个池，编号跨轮连续(不像11)因为对话是连续的
  api_key = os.getenv("DEEPSEEK_API_KEY")

  #SqliteSaver 需要连接db文件
  conn=sqlite3.connect(str(DB),check_same_thread=False) #允许跨线程用
  saver=SqliteSaver(conn)   ##这就是一个checkpointer，他会将内容存储到sqlite文件
  app=build_app(retriever,pool,api_key,saver)

  #测试1：多轮对话
  print("测试 1  多轮对话（同一个 thread_id）")
  config={"configurable":{"thread_id":"demo-multi"},"recursion_limit":RECURSION_LIMIT}

  #第一轮
  print("\n[第 1 轮] 试用期最长能有多久")
  round1=app.invoke({"messages":[
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="试用期最长能有多久"),
  ]},config)
  print("答：",(round1["messages"][-1].content or "")[:120],"...")
  print("历史消息数：",len(round1["messages"]))

  #第二轮 只传新的user message 历史从checkpoint读
  print("\n[第 2 轮] 那超过了怎么办？")
  round2=app.invoke({"messages":[
    HumanMessage(content="那超过了怎么办？")
  ]},config)
  print("答：", (round2["messages"][-1].content or "")[:200], "...")
  print("  历史消息数：", len(round2["messages"]))

  #测试2：不同的thread隔离
  print()
  print("测试 2  不同 thread_id 应该互不干扰")
  print("=" * 70)
  config2={"configurable": {"thread_id": "demo-diff"}, "recursion_limit": RECURSION_LIMIT}
  round3=app.invoke({"messages":[
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="年假有几天")
  ]},config2)
  print("新thread 历史消息数：", len(round3["messages"]), "（应该从 3 条重新开始）")

  #测试3：HITL中断
  print()
  print("测试 3  写操作前暂停（HITL）   第一轮：只生成内容")
  print("=" * 70)
  config3={"configurable": {"thread_id": "demo-hitl"}, "recursion_limit": RECURSION_LIMIT}

  round4=app.invoke({"messages": [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="我工作了 3 年半，月薪 12000，"  
                "公司以「不能胜任工作」为由把我辞退，也没提前 30 天通知。"
                "帮我写一份仲裁申请要点，主张经济补偿金。"),
  ]}, config3)

  content=round4["messages"][-1].content or ""
  print(f"3测第一轮生成内容{len(content)}字")
  print(f"历史消息数：{len(round4['messages'])}")
  print(f"开头：{content[:80]}...")

  print("测试 3 第 2 轮：要求保存（这里应该暂停）")
  print("=" * 70)

  round5 = app.invoke({"messages": [
    HumanMessage(content="把上面的内容保存成 claim.md"),
  ]}, config3)

  #如果图暂停了，返回值会有__interrupt__
  interrupts = round5.get("__interrupt__")  #拿到的是一个列表
  if not interrupts:
    print("没有暂停")
    print("  最后一条消息：", (round5["messages"][-1].content or "")[:200])
    conn.close()
    return
  
  if interrupts:
    print("\n图已经暂停，等待确认")
    for it in interrupts:
      #这里的value就是你传的上面那个decide参数字典
      print("待确认内容", json.dumps(it.value, ensure_ascii=False, indent=2)[:400])
    print(f"当前状态已存入{DB}")

    #测试4：恢复执行
    print("测试 4  用户点「确认」后恢复执行")
    print("=" * 70)
    #Command恢复执行 resume就是interrupt返回的值  thread_id要一样
    round6=app.invoke(Command(resume="approve"),config3)  
    print(" 恢复后答案：", (round6["messages"][-1].content or "")[:200])

    files=list(OUTPUT.glob("*.md")) #找到其下的所有md文件
    print("已生成文件：", [f.name for f in files])

  else:
    print("\n图没有暂停（模型可能没调用 save_document）")
    print("最后一条消息：", (round4["messages"][-1].content or "")[:200])

  #断开连接
  conn.close()
  print()
if __name__=="__main__":

  main()



# 测试 1  多轮对话（同一个 thread_id）

# [第 1 轮] 试用期最长能有多久
# 答： 试用期最长是6个月，但不是所有情况都能约定6个月，要看你的劳动合同签多久[1][2]。

# 具体分档是这样的[1]：
# - 合同期限3个月以上不满1年：试用期最多1个月[1]
# - 合同期限1年以上不满3年：试用期最多2个月[1]
# - 合同期限3 ...
# 历史消息数： 6

# [第 2 轮] 那超过了怎么办？
# 答： 试用期被约长了，法律给了你两条路：一是让劳动行政部门责令公司改正，二是要公司赔钱[4]。

# **1. 赔偿金怎么算**

# 如果超期的那段试用期你已经实际干过了，公司要按你**试用期满后的月工资**为标准，就超出的那段时间给你赔钱[4]。

# 举个例子：签1年合同（法定试用期最多2个月），公司却约了3个月，你干满了3个月。那多出来的1个月，公司要按你转正后的月工资再赔你1个月的钱[4]。注意这个赔偿金 ...
#   历史消息数： 11

# 测试 2  不同 thread_id 应该互不干扰
# ======================================================================
# 新thread 历史消息数： 6 （应该从 3 条重新开始）

# 测试 3  写操作前暂停（HITL）   第一轮：只生成内容
# ======================================================================
# 3测第一轮生成内容1168字
# 历史消息数：7
# 开头：# 仲裁申请要点（草稿）

# ## 一、你的情况梳理

# - 工作年限：3 年半
# - 月薪：12000 元
# - 公司理由：不能胜任工作
# - 未提前 30 天书面通知...
# 测试 3 第 2 轮：要求保存（这里应该暂停）
# ======================================================================

# 图已经暂停，等待确认
# 待确认内容 {
#   "action": "save_document",
#   "filename": "claim.md",
#   "字数": 1168,
#   "Preview": "# 仲裁申请要点（草稿）\n\n## 一、你的情况梳理\n\n- 工作年限：3 年半\n- 月薪：12000 元\n- 公司理由：不能胜任工作\n-未提前 30 天书面通知\n\n## 二、核心主张\n\n**1. 公司以「不能胜任工作」解除，必须满足两个前提**\n\n《劳动合同法》第四十条规定，用人单位以「劳动者不能胜任工作」为由解除劳动合同，需要先「经过培训或者调整工作岗位，仍不能胜任工作」，并且要「提前三十日以书"
# }
# 当前状态已存入data\agent_checkpoints.db
# 测试 4  用户点「确认」后恢复执行
# ======================================================================
#  恢复后答案： 已保存到 `data/output/claim.md`，共 1168 字。

# 文件里包含：情况梳理、核心主张（不能胜任工作的解除前提、经济补偿金 48000 元、违法解除赔偿金 96000 元）、证据清单和两点提醒。

# 如果你告诉我所在城市，我可以帮你核对高薪封顶的问题，再更新文件。
# 已生成文件： ['claim.md']
    

