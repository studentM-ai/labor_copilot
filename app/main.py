"""
FastAPI 应用
D13只弄 api/health与ask 检查与非流式提问
D14加SSE流式

"""
from contextlib import asynccontextmanager    #异步上下文管理器(进入执行一段，退出执行清理)
from pathlib import Path
import time,uuid

from dotenv import load_dotenv
from fastapi import FastAPI,File,Header,HTTPException,UploadFile
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles    #把 app/static 目录挂到 /static

from . import deps
from . import memory
from .contract_stream import contract_check_stream
from .main_helpers import extract_citations,extract_tool_calls
from .schemas import (
  AskRequest,AskResponse,
  SessionCreateResponse,SessionMessagesResponse,MessageItem,
  ContractFile,ContractsResponse,UploadResponse,ContractCheckRequest,
)
from .streaming import sse_stream
from src.contract import CONTRACTS,decode_bytes_smart,list_contracts

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

# 静态文件目录（D16 前端页面）
STATIC_DIR = Path(__file__).resolve().parent / "static"

# 开发期：让浏览器每次都重新校验静态文件
#  StaticFiles 默认不设 Cache-Control，浏览器会自己猜一个缓存时间。改了 app.js，刷新页面却还是跑旧代码
@app.middleware("http")
async def no_cache_static(request,call_next):
  response=await call_next(request)
  path=request.url.path #本次访问路径
  # 判断：首页 /  或者 以 /static/ 开头的静态资源
  if path=="/" or path.startswith("/static/"):
    response.headers["Cache-Control"]="no-cache"  #如果更新了就用新的
  return response


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static") #挂载前端静态页面


@app.get("/",include_in_schema=False)
def index():
  """
  首页：直接返回前端单页
  用 FileResponse 而不是模板渲染：
  页面是纯静态的，没有任何服务端变量要注入，读文件发出去最简单。
  include_in_schema=False 让它不出现在 /docs 的接口列表里，它是给人看的页面，不是给程序调的接口。
  """
  return FileResponse(str(STATIC_DIR / "index.html"))


#路由
@app.get("/api/health")
def health():
  """
  健康检查，只检验服务存在，与模型无关
  """
  return {"status":"ok"}

@app.post("/api/ask",response_model=AskResponse)
async def ask(req:AskRequest,x_user_id:str=Header(default="anonymous")):
  """
  提问（非流式）
  """
  rt=deps.get_runtime()

  thread_id=req.thread_id or uuid.uuid4().hex
  ts=time.perf_counter()

  async with deps.open_agent_async(rt,thread_id) as (agent,pool,config):
    payload=await deps.build_input_async(rt,agent,config,req.question,x_user_id)
    try:
      result=await agent.ainvoke(
        payload,
        config=config
      )
    except Exception as e:
      raise HTTPException(status_code=500,detail=f"Agent执行失败:{e}")

  #答案已经出来了，记忆抽取丢后台，不阻塞响应
  memory.spawn(memory.remember_turn(rt,x_user_id,req.question,result["messages"][-1].content or ""))

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
async def ask_stream(req:AskRequest,x_user_id:str=Header(default="anonymous")):
  """
  提问，sse流式
  header：
  Cache-Control: no-cache 告诉浏览器别缓存流
  Connection: keep-alive 保持长连接
  X-Accel-Buffering: no  关掉 nginx 的响应缓冲
  """
  return StreamingResponse(
    sse_stream(req.question,req.thread_id or uuid.uuid4().hex,x_user_id),
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
    items.append(MessageItem(role=h["role"],content=h["content"],citations=h["citations"]))
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



# 合同：列表 / 上传 / 体检（SSE）
# 单个文件上限。512KB
MAX_UPLOAD_BYTES = 512 * 1024

# 只接受纯文本。docx 要额外解析库，暂时不做
ALLOWED_SUFFIX = {".txt"}

@app.get("/api/contracts",response_model=ContractsResponse)
def list_contract_files():
  """
  列出 data/contracts 下可体检的合同

  前端需要这个接口是因为上传之前，用户不知道系统里有哪几份合同可以试；
  有了它，前端可以先展示「示例合同」列表，点一下就体检。
  """
  files=[]
  for path in sorted(CONTRACTS.glob("*.txt")):
    files.append(ContractFile(name=path.name,size=path.stat().st_size))
  return ContractsResponse(files=files)


@app.post("/api/contracts/upload",response_model=UploadResponse)
async def upload_contract(file:UploadFile=File(...)):
  """
  上传一份合同（.txt）
  1.文件名清洗：浏览器可以传 "../../etc/passwd" 这种文件名，所以只取文件名部分，反斜杠也切掉（Windows 客户端可能带 \）
  2.大小：边读边累加，超限立刻拒绝，不等整个文件落盘再判断
  3.类型：只收 .txt；再检查解码后是不是 几乎是正常文本
  4.重名：先比内容，一模一样就复用；不同才加后缀

  """
  #1.文件名清洗
  raw_name=file.filename or ""
  name=raw_name.replace("\\","/").split("/")[-1].strip()  #只要最后的纯文件名

  if name=="" or name.startswith("."):
    raise HTTPException(status_code=400,detail="文件名不合法")

  suffix=Path(name).suffix.lower()  #后缀小写
  if suffix not in ALLOWED_SUFFIX:
    raise HTTPException(status_code=400,detail=f"只支持.txt 文件，收到的是 {suffix or '无扩展名'}")

  #2.分块读，超限就停
  #UploadFile 超过一定大小会落到临时文件，但一次性 read() 仍然会把
  #整个文件读进内存。分块能在超限那一刻立刻停，不给内存压力。
  chunks=[]
  total=0
  while True:
    chunk=await file.read(64*1024) #64kb
    if not chunk:
      break
    total+=len(chunk)
    if total>MAX_UPLOAD_BYTES:
      raise HTTPException(status_code=413,detail=f"文件超过 {MAX_UPLOAD_BYTES//1024} KB")
    chunks.append(chunk)

  data=b"".join(chunks)
  if len(data.strip())==0:
    raise HTTPException(status_code=400,detail="文件是空的")

  #3.内容检查：改名成 .txt 的二进制文件要拦住
  #（1）能不能用 utf-8 或 gbk 干净解码
  #（2）解码后控制字符占比是否过高（二进制里 0x00-0x1f 很多）
  #两个都要做
  probe=decode_bytes_smart(data)
  if probe is None:
    raise HTTPException(status_code=400,detail="这不是文本文件，请上传纯文本的合同")

  ctrl=0  #非法控制字符
  for ch in probe:
    if ord(ch)<32 and ch not in "\r\n\t": #ASCII码小于32的是不可见控制字符
      ctrl+=1
  if len(probe)>0 and ctrl/len(probe)>0.02: #非法字符太多
    raise HTTPException(status_code=400,detail="这不像文本文件，请上传纯文本的合同")

  #4.存盘
  #先比内容：只要有一个文件的字节和这次上传的完全一样，就直接用它，不新建。
  #用户测试时经常反复传同一个文件，否则 data/contracts 会堆满 sample_1_1.txt、sample_1_1_1.txt
  for existing in CONTRACTS.glob("*.txt"):
    if existing.stat().st_size != len(data):
      continue
    if existing.read_bytes() == data:
      return UploadResponse(filename=existing.name,size=len(data))

  #内容不同才加后缀。stem 是从【原始文件名】算的（常量），
  #所以每次生成的是 sample_1_1、sample_1_2，不会滚成 sample_1_1_1
  stem=Path(name).stem   #去掉后缀，比如 sample_1.txt -> sample_1

  target=CONTRACTS/name
  index=1
  while target.exists():
    target=CONTRACTS/(stem+"_"+str(index)+suffix)
    index+=1

  target.write_bytes(data)
  return UploadResponse(filename=target.name,size=len(data))


@app.post("/api/contracts/check/stream")
async def contract_check(req:ContractCheckRequest):
  """
  体检一份合同，SSE 推进度

  为什么用 POST 体传文件名，而不是放在 URL 路径里：
    文件名可能含中文和空格，放路径要来回 encode/decode，容易出错。
  """
  name=Path(req.filename).name
  if name not in list_contracts():
    raise HTTPException(status_code=404,detail=f"没有找到 {name}")

  rt=deps.get_runtime()

  return StreamingResponse(
    contract_check_stream(rt,name),
    media_type="text/event-stream",
    headers={
      "Cache-Control":"no-cache",
      "Connection":"keep-alive",
      "X-Accel-Buffering":"no",
    }
  )
