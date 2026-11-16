"""
FastAPI 应用
D13只弄 api/health与ask 检查与非流式提问
D14加SSE流式

"""
from contextlib import asynccontextmanager    #异步上下文管理器(进入执行一段，退出执行清理)
from dotenv import load_dotenv
from fastapi import FastAPI,HTTPException
from langchain_core.messages import HumanMessage,SystemMessage
from . import deps
from .schemas import AskRequest,AskResponse,citation,ToolCallInfo
from src.prompts import QA_PROMPT
import time,uuid

load_dotenv()

@asynccontextmanager
async def lifespan(app:FastAPI):
  """
  启动时加载检索器，关闭时不做事
  """
  deps.init_runtime()
  yield

app=FastAPI(
  title="职场权益 Copilot",
  description="劳动法RAG+Agent，回答带发条引用",
  version="0.1.0",
  lifespan=lifespan
)

#工具函数
def extract_citations(answer:str,pool)->list[citation]:
  """
  从答案正文里找出真正用到的引用编号，映射回具体法条
  """
  #按顺序去重
  used=[]
  i=0
  while i<len(answer):
    if answer[i]=="[":
      j=answer.find("]", i + 1)   #从i+1后找到右括号  [xx]
      if j>0:
        inner=answer[i+1:j] #取出[]中间的内容
        if inner.isdigit(): #判断内容是否是数字
          idx=int(inner)  #如果是就转int类型
          if idx not in used:
            used.append(idx)  #加到列表里
          i=j+1 #i从右括号后开始
          continue
    i+=1

  #拿到引用法条数据 
  out=[]
  for idx in used:
    #编号从1开始，而池子下标是从0开始
    if idx >= 1 and idx <= len(pool.items):
      r=pool.items[idx-1]
      out.append(citation(
        idx=idx,
        law=r["law_title"],
        no=r["display_no"],
        text=r["text"],
      ))
  return out

def extract_tool_calls(messages)->list[ToolCallInfo]:
  """
  从消息历史里提取调用过的工具
  """
  out=[]
  for m in messages:
    if m.type =="ai" and getattr(m,"tool_calls",None):
      for tc in m.tool_calls:
        out.append(ToolCallInfo(
          name=tc["name"],
          args=tc.get("args",{})or {},
        ))
  return out

#路由
@app.get("/api/health")
def health():
  """
  健康检查，只检验服务存在，与模型无关
  """
  return {"status":"ok"}

@app.post("/api/ask",response_model=AskResponse)
def ask(req:AskRequest):
  """
  提问（非流式）
  """
  rt=deps.get_runtime()
  agent,pool=deps.make_agent(rt)

  thread_id=req.thread_id or uuid.uuid4().hex
  ts=time.perf_counter()
  try:
    result=agent.invoke(
      {"messages":[
        SystemMessage(content=QA_PROMPT),
        HumanMessage(content=req.question),
      ]},
      config={"recursion_limit":rt.limit}
    )
  except Exception as e:
    raise HTTPException(status_code=500,detail=f"Agent执行失败:{e}")

  elapsed_ms=int((time.perf_counter()-ts) * 1000) #本次耗时

  # 最后一条消息就是最终答案
  answer = result["messages"][-1].content or ""
  return AskResponse(
    answer=answer,
    citations=extract_citations(answer,pool),
    tool_calls=extract_tool_calls(result["messages"]),
    thread_id=thread_id,
    elapsed_ms=elapsed_ms
  )