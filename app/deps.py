"""
共享检索器，每次请求新建agent（图+引用池新建）
按 thread 复用的引用池 + 异步持久化

每个请求开一个数据库连接
"""

import os,threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  #异步本地文件持久化

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
  返回[{"role": "human"/"ai"/"tool"/"system", "content": "..."}, ...]
  """
  async with open_agent_async(rt,thread_id) as (agent,_pool,config):
    state=await agent.aget_state(config)  #拿状态
    history=(state.values or {}).get("messages")or []   #拿到历史消息

  out=[]
  for m in history:
    content=m.content if isinstance(m.content ,str)else str(m.content)  #转字符串
    #工具返回可能很长，截断
    if m.type=="tool":
      content=content[:200]

    out.append({"role":m.type,"content":content})
  return out


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

