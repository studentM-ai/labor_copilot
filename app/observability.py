"""
可观测性的 结构化日志与统计
不能影响主流程且不能记用户问题原文
用contextvars上下文变量来获取请求id

"""
import json,os,time,contextvars
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
LOG_DIR=ROOT/"data"/"logs"
LOG_FILE=LOG_DIR/"app.jsonl"
MAX_BYTES=20*1024*1024  #20MB

#request_id 
#用contextvars 而不是全局变量的原因是asyncio是单线程并发，await事件循环会跑别的协程，这个时候全局变量会被覆盖
#不用local是因为它按线程隔离，而所有协程在同一线程，一样会被覆盖
_request_id:contextvars.ContextVar[str]=contextvars.ContextVar("request_id",default="-")

def set_request_id(rid:str)->None:
  """
  给当前请求设置编号id 
  """
  _request_id.set(rid)

def get_request_id()->str:
  """
  读到当前请求的编号
  """
  return _request_id.get()

def new_request_id()->str:
  """
  生成一个短id
  """
  return os.urandom(4).hex()  #随机生成，4字节  8位十六进制

#日志
def _rotate_if_needed()->None:
  """
  超过20MB就把当前文件名改为app.jsonl.1(覆盖旧的) 需要轮转不然日志会无限增大
  这是只是简单的一份轮转
  """
  try:
    if LOG_FILE.exists() and LOG_FILE.stat().st_size>MAX_BYTES:
      backup=LOG_FILE.with_suffix(".jsonl.1") #替换后缀
      if backup.exists():
        backup.unlink() #如果已经存在旧的备份文件，先删除它
      LOG_FILE.rename(backup) #把当前超限的日志文件重命名
  except Exception:
    pass

def log_event(record:dict)->None:
  """
  追加一行Json到日志文件
  """
  try:
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    _rotate_if_needed()
    row={"ts":time.strftime("%Y-%m-%dT%H:%M:%S")}
    row["request_id"]=get_request_id()
    row.update(record)  #record里的字段覆盖同名的

    with open(LOG_FILE,"a",encoding="utf-8")as f:
      f.write(json.dumps(row,ensure_ascii=False)+"\n")
  except Exception as e:
    print("[obs] 写日志失败（不影响主流程）：",e,flush=True)

def read_records(limit:int=0)->list[dict]:
  """
  读日志文件，返回记录列表
  
  limit>0时只读最后limit行
  """
  if not LOG_FILE.exists():
    return []

  records=[]
  with open(LOG_FILE,"r",encoding="utf-8")as f:
    for line in f:
      line=line.strip()
      if not line:
        continue
      try:
        records.append(json.loads(line))
      except Exception:
        continue
  if limit>0:
    return records[-limit:]
  return records

def pct(values:list[float],p:float)->float:
  """
  分位数
  """
  if not values:
    return 0.0
  ordered=sorted(values)
  idx=int(round((p/100)*(len(ordered)-1)))
  return ordered[idx]

def usage_of(messages)->tuple[int,int]:
  """
  从消息列表里累加token用量
  """
  total_in=0
  total_out=0
  for m in messages or []:
    usage=getattr(m,"usage_metadata",None) or {}
    total_in+=int(usage.get("input_tokens")or 0)
    total_out+=int(usage.get("output_tokens")or 0)
  return total_in,total_out

def build_stats(limit:int=500)->dict:
  """
  汇总最近的请求，给/api/stats用
  只统计最近limit条
  """
  records=read_records(limit=limit)

  total=len(records)
  errors=0
  latencies=[]  #延迟
  tokens_in=0
  tokens_out=0
  tool_counter:dict[str,int]={}
  slow=0  #慢请求
  aborted=0  #用户主动断开：不算错误，但单独计数（反映「等不及的人」有多少）

  for r in records:
    if r.get("status") == "error":
      errors += 1
    # aborted 是用户主动断开，不算错误 但单独计数
    elif r.get("status") == "aborted":
      aborted += 1

    lat=r.get("elapsed_ms") #延迟时间
    if isinstance(lat,(int,float)):
      latencies.append(float(lat))
      if lat >10000:    #超过10s算慢请求
        slow+=1
    tokens_in+=int(r.get("tokens_in") or 0)
    tokens_out+=int(r.get("tokens_out")or 0)
    for name in (r.get("tools")or []):
      tool_counter[name]=tool_counter.get(name,0)+1
      
  return {
      "window": total,  # 统计了多少条
      "total": total,
      "error_rate": round(errors / total, 4) if total else 0.0, #错误率
      "latency_p50_ms": round(pct(latencies, 50), 1),
      "latency_p95_ms": round(pct(latencies, 95), 1),
      "latency_max_ms": round(max(latencies), 1) if latencies else 0.0,
      "slow_requests": slow,  #慢请求 >10 秒
      "tokens_in": tokens_in,
      "tokens_out": tokens_out,
      "tokens_avg": round((tokens_in + tokens_out) / total, 1) if total else 0.0, #平均token
      "aborted": aborted,   # 用户中途断开次数
      "tools": dict(sorted(tool_counter.items(), key=lambda kv: -kv[1])),
  }

