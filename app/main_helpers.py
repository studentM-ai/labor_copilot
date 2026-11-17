"""
路由共用的辅助函数
"""
from .schemas import Citation,ToolCallInfo

def extract_citations(answer:str,pool)->list[Citation]:
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
      out.append(Citation(
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