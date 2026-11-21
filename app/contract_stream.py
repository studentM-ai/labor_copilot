"""
合同体检的 SSE 流式接口

/api/ask/stream   是模型 + 工具边跑边吐 token
这个是一个确定的长任务边跑边推进度，不吐 token
两者都用 SSE，事件类型不同：
  status    开始（thinking）
  progress  每判完一条推一次：{done, total, clause, risk}
  done      完成：{file, total, high, medium, none, unknown, report, report_path}
  error     出错

check_contract_file 内部是同步的：它调 openai 的同步客户端，
如果在 async 函数里直接调用，这十几秒会把整个事件循环占死，
期间其他请求一个都进不来（连 /api/health 都不响应）。
所以用 asyncio.to_thread 把它丢到工作线程，
主协程继续跑「取进度 + 发心跳」的循环。

【为什么回调里用 call_soon_threadsafe】
  on_progress 是在**工作线程**里被调用的，
  而 asyncio.Queue 不是线程安全的，不能直接 await queue.put。

  正确做法：用 loop.call_soon_threadsafe 把「往队列里放东西」这个动作
  调度回事件循环线程去执行。
"""
import asyncio
import time
from pathlib import Path

from src.contract import check_contract_file
from src.tools.check import build_report_lines

from .streaming import heartbeat, sse

# 心跳间隔（秒）
SECONDS = 5

# 报告写的目录（和 017 的产物放一起，方便找）
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

async def contract_check_stream(rt,filename):
  """
  体验一份合同，边跑边推进度
  rt:deps.Runtime（需要它的retriever和top_k）
  filename:合同文件名
  返回 sse格式的字符串片段
  """
  yield sse("status",{"stage":"thinking","file":filename})

  queue:asyncio.Queue=asyncio.Queue()   #这个队列只能在主线程使用
  loop=asyncio.get_running_loop() #拿到当前线程正在跑的事件循环
  st=time.perf_counter()

  def on_progress(done,total,clause,risk):
    """
    done:已完成条数  total：总条数  clause:条款原文 risk：风险值
    每判断完一条就在工作线程里调用一次(进行签名)
    主要原因还是check_contract_file内部是同步阻塞的，协程无法切换，只能扔给线程来切换
    这里只用call_soon_threadsafe，不能直接用queue.put_nowait
    子线程想给async塞东西要用前者，后者操作的队列只在事件循环里安全
    call_soon_threadsafe 线程安全地把某个函数交给事件循环执行（实际上就是把一个东西转交给主线程的事件循环去执行）
    """
    one_line=clause[:40].replace("\n"," ")  #截断限长
    #执行函数将数据放入队列
    loop.call_soon_threadsafe(
      queue.put_nowait,
      (
        "progress",{
          "done":done,  
          "total":total,
          "clause":one_line,
          "risk":risk,
        }
      ),
    )

  def work():
    """
    合同体验，在工作线程里跑
    """
    try:
      result=check_contract_file(
        filename,
        rt.retriever,
        rt.top_k,
        on_progress=on_progress
      )
      loop.call_soon_threadsafe(queue.put_nowait,("done",result)) #放合同体验结果（推最终结果到队列）
    except Exception as e:
      loop.call_soon_threadsafe(queue.put_nowait,("error",str(e)))

  task=asyncio.create_task(asyncio.to_thread(work)) #放线程里运行
  try:
    while True:
      try:
        #等下一个消息，最多SECONDS秒，超过就心跳
        kind,payload=await asyncio.wait_for(queue.get(),timeout=SECONDS)    #接收端
      except asyncio.TimeoutError:
        yield heartbeat()
        continue
      if kind=="progress":
        yield sse("progress",payload)
      elif kind =="error":
        yield sse("error",{"message":payload})

      elif kind=="done":
        #check_contract_file出错时不抛异常，而是返回{"file":..., "error": "..."}
        if "error" in payload:
          yield sse("error",{"message":payload["error"]})
          return
        result =payload
        clauses=result["clauses"]
        #统计各风险等级
        counts={"high":0,"medium":0,"none":0,"unknown":0}
        for c in clauses:
          if c["risk"]in counts:
            counts[c["risk"]]+=1

        #写一份md文档与017一致
        report=build_report_lines(result)
        EVAL_DIR.mkdir(parents=True,exist_ok=True)
        out_path=EVAL_DIR/("contract_report_"+Path(result["file"]).stem+".md")
        out_path.write_text(report,encoding="utf-8")

        yield sse("done",{
          "file":result["file"],
          "total":len(clauses),
          "high":counts["high"],
          "medium":counts["medium"],
          "none":counts["none"],
          "unknown":counts["unknown"],
          "warning":result.get("warning"),
          "report":report,
          "report_path":str(out_path.relative_to(EVAL_DIR.parent.parent)),
          "elapsed_ms":int((time.perf_counter()-st)*1000)
        })
        return 
  finally:
    #客户端断开时生成器会被关闭们这里需要取消任务
    if not task.done():
      task.cancel()
      




