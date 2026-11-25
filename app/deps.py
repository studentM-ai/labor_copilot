"""
共享检索器，每次请求新建agent（图+引用池新建）
按 thread 复用的引用池 + 异步持久化
主要是状态共享
每个请求开一个数据库连接
"""

import os,threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from langchain_core.messages import HumanMessage,SystemMessage,RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  #异步本地文件持久化
from types import SimpleNamespace
import re

from . import memory
from .main_helpers import extract_citations

from src.agent import build_graph
from src.citation import RefPool
from src.contract import load_rows
from src.prompts import QA_PROMPT
from src.retriever import VectorRetriever
from src.tools import make_calc_severance,make_check_contract,make_search_law
from src.retriever import MilvusRetriever

#项目根
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Runtime:
  """
  应用级共享状态（启动时创建一次）
  """
  retriever:object
  api_key:str
  db_path:str
  model:str="deepseek-chat"
  limit:int =20   # 递归上限，约 10 圈循环
  top_k:int=5
  milvus_uri:str="http://localhost:19530"  #长期记忆用的 Milvus 地址

# 模块级单例
_runtime:Runtime | None=None

#会话级状态：thread_id:RefPool
_pools:dict[str,RefPool]={}

# 池子最多存多少个会话。
# 每个池子里是这个会话检索过的法条原文，不封顶的话进程越跑占得越多。
MAX_POOLS=200
_pools_lock=threading.Lock()    #互斥锁

#保留最近的原文条数，设置为6
MAX_HISTORY_TURNS=6

def init_runtime():
  """
  应用启动时调用(在fastapi的lifespan里)
  返回建好的Runtime，失败时终止
  """
  global _runtime

  api_key=os.getenv("DEEPSEEK_API_KEY")
  if not api_key:
    raise RuntimeError("没读到DEEPSEEK_API_KEY，检查 .env")

  rows=load_rows()
  #【检索后端切换】默认 numpy（exact search），设 VECTOR_BACKEND=milvus 就走 Milvus
  backend=os.getenv("VECTOR_BACKEND","numpy").strip().lower()

  if backend=="milvus":
    #延迟导入：不用 Milvus 的时候连这个类都不加载
    retreiver=MilvusRetriever(rows,uri=os.getenv("MILVUS_URI","http://localhost:19530"))
    print("[deps] 检索后端：Milvus（HNSW 近似检索）",flush=True)
  else:
    retreiver=VectorRetriever(rows)
    print("[deps] 检索后端：numpy（向量）",flush=True)

  db_path = os.path.join(ROOT, "data", "agent_checkpoints.db")
  _runtime=Runtime(retriever=retreiver,api_key=api_key,db_path=db_path,
    milvus_uri=os.getenv("MILVUS_URI","http://localhost:19530"))
  return _runtime

def get_runtime()->Runtime:
  """
  取共享状态；没初始化就报错
  """
  if _runtime is None:
    raise RuntimeError("运行时还没初始化，检查lifespan是否执行了")
  return _runtime

def get_pool(thread_id:str)->RefPool:
  """
  取某个会话的引用池，没有就新建
  加锁是因为fastapi用线程池跑同步代码可能会多请求同时来

  需要封顶是因为池子里存着这个会话检索过的法条原文，一个会话几百字节到几 KB。
  下次访问这个会话时会从历史里把编号补回来（见 restore_pool_async）。
  """
  with _pools_lock:
    pool=_pools.get(thread_id)  #拿到池对象
    if pool is None:
      #dict 保持插入顺序，next(iter(...)) 拿到的就是最早建的那个
      while len(_pools)>=MAX_POOLS:
        oldest=next(iter(_pools))
        _pools.pop(oldest,None) #删除最早建的那个
      pool=RefPool()
      _pools[thread_id]=pool
    return pool

def drop_pool(thread_id:str):
  """
  删除会话的时候把池也删了，避免内存泄漏
  """
  with _pools_lock:
    _pools.pop(thread_id,None)

@asynccontextmanager
async def open_agent_async(rt:Runtime,thread_id:str):
  """
  打开一次请求用的agent（连同数据库连接）自动拿到 agent、连接池、会话配置
  异步上下文管理器
  产出（agent，pool，config）

  """
  pool=get_pool(thread_id)  #取池
  tools=[
    make_search_law(rt.retriever,pool,rt.top_k),
    make_calc_severance(),
    make_check_contract(rt.retriever,rt.top_k)
  ]

  #from_conn_string异步上下文管理器，进入时连接，退出关闭
  async with AsyncSqliteSaver.from_conn_string(rt.db_path) as saver: #打开数据库连接，返回checkpointer
    #建表（初始化saver内部的数据库表）
    await saver.setup()
    agent=build_graph(
      tools,
      rt.api_key,
      model=rt.model,
      checkpointer=saver
    )
    config={
      "configurable":{"thread_id":thread_id},
      "recursion_limit":rt.limit
    }

    #服务重启后内存池是空的，但历史还在 sqlite 里。不补的话，这一轮新检索到的法条会从 [1] 重新编号前端点角标会跳到错的条文。
    if len(pool.items)==0:
      await restore_pool_async(rt,agent,config,pool)
    yield agent,pool,config


def trim_history(history,max_turns=MAX_HISTORY_TURNS):
  """
  算出需要删除的消息  
  按HumanMessage边界切，因为一次工具调用是tool_calls+ToolMessage+AI三个连在一起的
  """
  #从后向前数，找到max_turns个HumanMessage的位置
  seen=0  #统计已找到的用户消息条数
  cut=0 #截断位置下标
  
  for i in range(len(history)-1,-1,-1):
    if history[i].type=="human":
      seen+=1
      if seen ==max_turns:
        cut=i
        break

  #cut之前的都删除，但system与没有id的不能删
  drop=[]
  for i in range(cut):
    m=history[i]
    if i==0 and m.type=="system":
      continue
    if not getattr(m,"id",None):
      continue
    drop.append(m)
  return drop

async def build_input_async(rt,agent,config,question:str,user_id:str="anonymous")->dict:
  """
  决定这一轮要往图里塞什么消息

  续聊：只传 HumanMessage（不传 System，否则历史里会重复堆 System）
  新会话：System(QA_PROMPT) + Human(问题)

  长期记忆只在新会话注入
  
  截断会把用户说的条件删掉，所以这个时候必须用记忆补回来
  """
  state=await agent.aget_state(config)
  history=(state.values or {}).get("messages")or []

  if history:
    drop=trim_history(history)

    #没超阈值：走原来的路
    if not drop:
      return {"messages":[HumanMessage(content=question)]}

    #超了就先删除再用记忆补
    msgs=[RemoveMessage(id=m.id) for m in drop]
    block=""
    if user_id and user_id !="anonymous":
      block=await memory.memory_context(rt,user_id,question)

    if block:
      #这条SystemMessage是要把截掉的事实塞回来
      msgs.append(SystemMessage(content="【关于这位用户的已知信息（早期对话已折叠）】\n"+block))
      print(f"[deps] 截断 {len(drop)}条历史，用记忆补回：{block}",flush=True)
    else:
      print(f"[deps] 截断 {len(drop)}条历史（无记忆可补）",flush=True)

    msgs.append(HumanMessage(content=question))
    return {"messages":msgs}

  #新会话：先看有没有这个用户的记忆
  system=QA_PROMPT
  if user_id and user_id!="anonymous":
    block=await memory.memory_context(rt,user_id,question)
    if block:
      system=system+"\n\n【关于这位用户的已知信息】\n"+block
      print("[deps] 注入记忆：",block,flush=True)

  return {"messages":[
    SystemMessage(content=system),
    HumanMessage(content=question)
  ]}

async def read_history(rt:Runtime,thread_id:str)->list[dict]:
  """
  读某个会话的历史消息（给 /api/sessions/{tid}/messages 用）

  返回[{"role":..., "content":..., "citations":[...]}, ...]

  工具消息里本来就存着检索结果的原文，格式是：
    [3] 《中华人民共和国劳动合同法》第十九条
    劳动合同期限三个月以上不满一年的……
  把这些条目抠出来，就得到【编号 ： 法条】，和当初 RefPool 里的内容一致。
  然后复用 extract_citations，按答案里实际出现的 [n] 挑出来。
  """
  async with open_agent_async(rt,thread_id) as (agent,_pool,config):
    state=await agent.aget_state(config)
    history=(state.values or {}).get("messages")or []

  pool_like=restore_pool_from_history(history)

  out=[]
  for m in history:
    text=m.content if isinstance(m.content,str)else str(m.content)

    #工具消息：内容和法条原文重复，截断显示就行
    if m.type=="tool":
      out.append({"role":"tool","content":text[:200],"citations":[]})
      continue

    citations=[]
    if m.type=="ai" and text.strip() and pool_like is not None:
      for c in extract_citations(text,pool_like):
        citations.append(c.model_dump())

    out.append({"role":m.type,"content":text,"citations":citations})
  return out


def restore_pool_from_history(history):
  """
  从历史里的工具消息还原[编号 -> 法条]，做成一个「长得像 RefPool」的对象

  返回 SimpleNamespace 而不是 RefPool：extract_citations 只用到 pool.items 这一个属性。
  造一个只有 items 的对象就够了，不引入真实池会干净一点。

  返回 None 表示历史里一个法条都没有（纯计算类对话），调用方跳过即可。
  """
  #把所有tool消息的内容收集
  found={}
  for m in history:
    if m.type!="tool":
      continue
    text=m.content if isinstance(m.content,str)else str(m.content)

    #工具里多个条目用空行分隔
    for chunk in text.split("\n\n"):
      #五组匹配
      match=re.match(r"^\[(\d+)\] 《([^》]+)》([^\n]+)\n([\s\S]*)$",chunk.strip())
      if not match:
        continue
      idx=int(match.group(1)) #取第一组
      #同一个编号可能在不同轮被多次检索到，保留第一次的即可
      if idx in found:
        continue
      found[idx]={
        "law_title":match.group(2),
        "display_no":match.group(3).strip(),
        "text":match.group(4).strip(),
      }

  if len(found)==0:
    return None

  #编号是 1 起的，中间可能有空洞（比如历史被截断），补占位保持下标对齐
  items=[]
  for i in range(1,max(found.keys())+1):
    items.append(found.get(i,{"law_title":"","display_no":"","text":""}))

  return SimpleNamespace(items=items)


async def restore_pool_async(rt:Runtime,agent,config,pool)->None:
  """
  服务重启后，把某个会话的内存引用池补回来

  因为_pools 是内存里的字典，进程一重启就空了；但会话历史在 sqlite 里没丢。
  不补的话：续聊时新检索到的法条会从 [1] 重新编号，前端点角标会跳到错的条文。

  还原出来的条目只有法律名和条号，没有语料里的 id，而池子判段是否重叠靠 id。
  所以拿 (法律名, 条号) 去全量语料里反查 id：
  查到 -> 以后检索到同一条法条会复用这个编号（不会出现同一条两个编号）
  查不到 -> 只是那一条不会被复用，编号本身仍然是对的
  """
  state=await agent.aget_state(config)
  history=(state.values or {}).get("messages")or []

  restored=restore_pool_from_history(history)
  if restored is None:
    return

  pool.items=list(restored.items)
  pool.by_id={}

  #(法律名, 条号) -> id
  index={}
  for row in rt.retriever.rows: #遍历法条
    index[(row.get("law_title",""),row.get("display_no",""))]=row.get("id") #设置（法律名，条号）：id

  for i,it in enumerate(pool.items):
    key=(it.get("law_title",""),it.get("display_no",""))  #拿到（法律名，条号）用来反查id
    rid=index.get(key)
    if rid is not None:
      pool.by_id[rid]=i+1


async def delete_session(rt:Runtime,thread_id:str)->bool:
  """
  删除一个会话的持久化状态
  返回是否删掉，False表示本来没有
  """
  existed=False
  async with AsyncSqliteSaver.from_conn_string(rt.db_path)as saver:
    await saver.setup()
    tup=await saver.aget_tuple({"configurable":{"thread_id":thread_id}})  #获取完整状态元组
    existed=tup is not None
    if existed:
      await saver.adelete_thread(thread_id) #异步删除这个thread记录

  drop_pool(thread_id)  #删除对应池
  return existed

