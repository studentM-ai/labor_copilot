"""
FastAPI 应用
D13只弄 api/health与ask 检查与非流式提问
D14加SSE流式

"""
from contextlib import asynccontextmanager    #异步上下文管理器(进入执行一段，退出执行清理)
from dotenv import load_dotenv
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage,SystemMessage
from . import deps
from .schemas import AskRequest,AskResponse
from src.prompts import QA_PROMPT
import time,uuid
from .main_helpers import extract_citations,extract_tool_calls
from .streaming import sse_stream

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
  description="劳动法RAG+Agent，回答带法条引用",
  version="0.1.0",
  lifespan=lifespan
)


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

@app.post("/api/ask/stream")
async def ask_stream(req:AskRequest):
  """
  提问，sse流式
  header：
  Cache-Control: no-cache 告诉浏览器别缓存流
  Connection: keep-alive 保持长连接
  X-Accel-Buffering: no  关掉 nginx 的响应缓冲
  """
  return StreamingResponse(
    sse_stream(req.question,req.thread_id or uuid.uuid4().hex),
    media_type="text/event-stream",
    headers={
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    }

  )