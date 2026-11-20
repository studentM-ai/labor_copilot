"""
共享检索器，每次请求新建agent（图+引用池新建）
按 thread 复用的引用池 + 异步持久化
主要是状态共享
每个请求开一个数据库连接
"""

import os,threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  #异步本地文件持久化
from types import SimpleNamespace
import re

from .main_helpers import extract_citations

from src.agent import build_graph
from src.citation import RefPool
from src.contract import load_rows
from src.prompts import QA_PROMPT
from src.retriever import VectorRetriever
from src.tools import make_calc_severance,make_check_contract,make_search_law

#项目根
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Runtime:
  """
  应用级共享状态（启动时创建一次）
  """
  retriever:VectorRetriever
  api_key:str
  db_path:str
  model:str="deepseek-chat"
  limit:int =20   # 递归上限，约 10 圈循环
  top_k:int=5

# 模块级单例
_runtime:Runtime | None=None

#会话级状态：thread_id:RefPool
_pools:dict[str,RefPool]={}
_pools_lock=threading.Lock()    #互斥锁

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
  retreiver=VectorRetriever(rows)
  db_path = os.path.join(ROOT, "data", "agent_checkpoints.db")
  _runtime=Runtime(retriever=retreiver,api_key=api_key,db_path=db_path)
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
  """
  with _pools_lock:
    pool=_pools.get(thread_id)  #拿到池对象
    if pool is None:
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
    yield agent,pool,config

async def build_input_async(agent,config,question:str)->dict:
  """
  决定这一轮要往图里塞入什么消息
  如果是续聊则只穿HumanMessage，不穿System，因为它会作为新消息加入历史末尾，会冗余重复
  利用aget_state拿到thread状态判断是不是续聊  如果是新会话那么values是空字典
  """
  state=await agent.aget_state(config)  #拿到状态
  history=(state.values or {}).get("messages")or [] #拿到内容
  if history: #如果有内容则续聊
    return {"messages":[HumanMessage(content=question)]}  #续聊加问题
  return {"messages":[
    SystemMessage(content=QA_PROMPT),
    HumanMessage(content=question)
  ]}

async def read_history(rt:Runtime,thread_id:str)->list[dict]:
  """
  读某个会话的历史消息（给 /api/sessions/{tid}/messages 用）

  返回[{"role":..., "content":..., "citations":[...]}, ...]

  工具消息里本来就存着检索结果的原文，格式是：
    [3] 《中华人民共和国劳动合同法》第十九条
    劳动合同期限三个月以上不满一年的……
  把这些条目抠出来，就得到「编号 -> 法条」，和当初 RefPool 里的内容一致。
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

