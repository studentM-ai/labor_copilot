"""
SSE流式输出

事件（自定义）：
status       阶段变化（thinking）
tool_call    开始调某个工具
tool_result  工具返回（只给前 100 字预览）
token        模型吐出的文字片段
done         结束，带完整的 citations / thread_id / elapsed_ms
error        出错（此时 HTTP 状态码已经发出去了，改不了，只能在流里报）

不直接用astream是因为它拿不到正在调用工具的事件，无法插入心跳
加心跳是因为在合同体验途中会占用时间，在这个时间内可能回断链，心跳就是保持连接的

事件结构（astream里的） v2标准结构
{
    "event": "on_chat_model_stream",   #事件类型
    "name": "ChatOpenAI",              #这个事件是谁发出来的（组件名字）
    "run_id": "uuid-xxxx",
    "parent_ids": [],
    "metadata": {                      #元数据，用来过滤 langgraph_node 的地方
        "langgraph_node": "agent"
    },
    "data": {                          # 真正的数据本体！每个事件不一样
        "chunk": AIMessageChunk(content="你好")
    }
}
"""
import asyncio,json,time

from . import memory
from . import deps
from . import observability
from .main_helpers import extract_citations


#时间间隔（心跳间隔），用于合同体验中间保持连接
SECONDS=5 
def sse(event:str,data:dict)->str:
  """
  拼一条SSE消息
  SSE格式固定：
  event: 事件名\n
  data: 一行 JSON\n
  \n      空行表示这条消息结束
  """
  payload=json.dumps(data,ensure_ascii=False)
  return f"event:{event}\ndata:{payload}\n\n"

def heartbeat()->str:
  """
  SSE注释行：以冒号开头，客户端会忽略，但连接不会被判定为空闲
  """
  return ": keep-alive\n\n"

async def sse_stream(question:str,thread_id:str,user_id:str="anonymous"):
  """
  SSE生成器
  question：用户问题
  thread_id:会话id
  返回SSE格式的字符串片段
  """
  question_len=len(question)   # 只记长度，不记原文
  tokens_in=0
  tokens_out=0
  status="ok"
  # 是否把 done 事件完整发出去了  用来区分「正常完成」和「客户端中途断开」
  finished=False
  answer_parts=[] #模型吐出 的文字，拼成完整答案
  tool_rows=[]  #工具调用记录
  st=time.perf_counter()

  try:
    rt=deps.get_runtime() #拿到共享状态
    config={"configurable":{"thread_id":thread_id},"recursion_limit":rt.limit}

    async with deps.open_agent_async(rt,thread_id) as (agent,pool,config):
      #续聊只传human
      payload=await deps.build_input_async(rt,agent,config,question,user_id)
  
      yield sse("status",{"stage":"thinking"})  #表示正在干活正在思考中

      #用队列桥接【异步事件流】和【心跳】 不用wait_for是因为超时会引发错误
      queue:asyncio.Queue=asyncio.Queue()
      error_box={"err":None}  #存放异常信息

      async def producer():
        """
        把agent的事件放进队列，结束时放None标记
        """
        try:
          #ev是事件对象，是字典形状的
          async for ev in agent.astream_events(payload,config=config,version="v2"):
            await queue.put(ev) #把每个事件都放进去队列
        except Exception as e:
          error_box["err"]=e
        finally:
          await queue.put(None)

      task=asyncio.create_task(producer())  #启动，开始跑Agent，事件不断入队
      try:
        while True:
          try:
            #等下一个事件，最多SECONDS秒
            ev=await asyncio.wait_for(queue.get(),timeout=SECONDS)
          except asyncio.TimeoutError:  #超时没有新事件的时候就发一条心跳防止断开连接
            yield heartbeat()
            continue

          if ev is None:
            break

          #处理需要的三类事件
          kind=ev.get("event")
          metadata=ev.get("metadata") or {}
          #按langgraph_node 过滤，因为astream会把每次嵌套调用都报上来，可能会收到重复的token
          node=metadata.get("langgraph_node")
          data=ev.get("data")or {}

          if kind=="on_chat_model_stream":  #如果是模型吐出字段事件
            if node=="agent":
              chunk=ev.get("data",{}).get("chunk")
              text=getattr(chunk,"content","") or ""  #拿到内容
              if text:
                answer_parts.append(text)
                yield sse("token",{"text":text})

          elif kind=="on_chat_model_end":
            # 只有这个事件里有 usage_metadata，stream 事件的 chunk 上没有
            msg=data.get("output")
            usage=getattr(msg,"usage_metadata",None) or {}
            tokens_in+=int(usage.get("input_tokens") or 0)
            tokens_out+=int(usage.get("output_tokens") or 0)
        
          elif kind=="on_tool_start": #工具开始事件
            name=ev.get("name","")
            args=(ev.get("data")or {}).get("input") or {}
            #只保留字符串/数字这类可json化的值
            safe_args={}
            for k ,v in args.items():
              if isinstance(v,(str,int,float,bool))or v is None:
                safe_args[k]=v
            tool_rows.append({"name":name,"args":safe_args})
            yield sse("tool_call",{"name":name,"args":safe_args})

          elif kind=="on_tool_end": #工具结束事件  
            name=ev.get("name","")
            out=(ev.get("data")or{}).get("output")
            #输出可能是字符串也可能是对象
            preview=""
            if isinstance(out,str):
              preview=out[:100]
            else:
              preview=str(out)[:100]
            yield sse("tool_result",{"name":name,"preview":preview})

      finally:
        #客户端断开时，生成器会被关闭，这里要取消生产者任务 否则白花token
        if not task.done():
          task.cancel()
          try:
            await task
          except asyncio.CancelledError:
            pass
          except Exception:
            pass

    #出了 async with 连接关闭，这里用内存里的数据
    if error_box["err"] is not None:
      status="error"   # 让日志里能看出这次失败了
      yield sse("error",{"message":f"Agent执行失败：{error_box['err']}"})
      return

    answer="".join(answer_parts)

    #记忆：抽取 + 存储丢到后台，不 await
    memory.spawn(memory.remember_turn(rt,user_id,question,answer))

    elapsed_ms=int((time.perf_counter()-st)*1000)
    citations=extract_citations(answer,pool)
    finished=True   # 先置标志再发，客户端收到后立刻断开也不会漏
    yield sse("done",{
      "answer":answer,
      "citations":[c.model_dump() for c in citations],   #转普通dict
      "tool_calls":tool_rows,
      "thread_id":thread_id,
      "elapsed_ms":elapsed_ms,
    })
  except Exception:
    status="error"
    raise
  finally:
    # 三段式：报错 / 正常发完 / 中途断开（断开也要留日志）
    if status=="error":
      final_status="error"
    elif finished:
      final_status="ok"
    else:
      final_status="aborted"
    observability.log_event({
      "kind":"ask_stream",
      "thread_id":thread_id,
      "user_id":user_id,
      "question_len":question_len,
      "elapsed_ms":int((time.perf_counter()-st)*1000),
      "tokens_in":tokens_in,
      "tokens_out":tokens_out,
      "tools":[r["name"] for r in tool_rows],
      "status":final_status,
      "answer_len":len("".join(answer_parts)),
    })

