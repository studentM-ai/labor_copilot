"""
FastAPI 应用
D13只弄 api/health与ask 检查与非流式提问
D14加SSE流式

"""
from contextlib import asynccontextmanager    #异步上下文管理器(进入执行一段，退出执行清理)
from dotenv import load_dotenv
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse

from . import deps
from .schemas import AskRequest,AskResponse,SessionCreateResponse,SessionMessagesResponse,MessageItem

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
async def ask(req:AskRequest):
  """
  提问（非流式）
  """
  rt=deps.get_runtime()

  thread_id=req.thread_id or uuid.uuid4().hex
  ts=time.perf_counter()

  async with deps.open_agent_async(rt,thread_id) as (agent,pool,config):
    payload=await deps.build_input_async(agent,config,req.question)
    try:
      result=await agent.ainvoke(
        payload,
        config=config
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

@app.post("/api/sessions",response_model=SessionCreateResponse)
def create_session():
  """
  新建一个会话，只发一个thread_id
  这里只是给一个可用的id 给前端用的
  """
  return SessionCreateResponse(thread_id=uuid.uuid4().hex)

@app.get("/api/sessions/{thread_id}/messages",response_model=SessionMessagesResponse)
async def session_messages(thread_id:str):
  """
  取某个会话的历史消息，要从数据库读，所以用异步
  """
  rt=deps.get_runtime()
  history=await deps.read_history(rt,thread_id)

  items=[]
  for h in history:
    items.append(MessageItem(role=h["role"],content=h["content"]))
  return SessionMessagesResponse(thread_id=thread_id,messages=items)

@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id:str):
  """
  删除会话
  返回deleted 表示原来有无这个会话
  如果不存在也返回200
  """
  rt=deps.get_runtime()
  existed=await deps.delete_session(rt,thread_id)
  return {"thread_id":thread_id,"deleted":existed}