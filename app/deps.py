"""
共享检索器，每次请求新建agent（图+引用池新建）
"""

import os
from dataclasses import dataclass
from src.agent import build_graph
from src.citation import RefPool
from src.contract import load_rows
from src.retriever import VectorRetriever
from src.tools import make_calc_severance,make_check_contract,make_search_law

@dataclass
class Runtime:
  """
  应用级共享状态（启动时创建一次）
  """
  retriever:VectorRetriever
  api_key:str
  model:str="deepseek-chat"
  limit:int =20   # 递归上限，约 10 圈循环
  top_k:int=5

# 模块级单例
_runtime:Runtime | None=None

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
  _runtime=Runtime(retriever=retreiver,api_key=api_key)

def get_runtime()->Runtime:
  """
  取共享状态；没初始化就报错
  """
  if _runtime is None:
    raise RuntimeError("运行时还没初始化，检查lifespan是否执行了")
  return _runtime

def make_agent(rt:Runtime):
  """
  为一次请求新建agent（连同新的引用池）
  """
  pool=RefPool()
  tools=[
    make_search_law(rt.retriever,pool,rt.top_k),
    make_calc_severance(),
    make_check_contract(rt.retriever,rt.top_k)
  ]
  agent=build_graph(tools,rt.api_key,model=rt.model)
  return agent,pool