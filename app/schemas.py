"""
请求与响应的数据结构
使用pydantic模型进行校验 
如果没校验question传空字符串也会进到agent，白烧一次LLM
"""

from pydantic import BaseModel,Field
class AskRequest(BaseModel):
  """
  提问请求
  """
  question:str=Field(
    ...,
    min_length=1,
    max_length=1000,
    description="用户的问题，1–1000 字"
  )

  thread_id:str | None=Field(
    None,
    description="会话id。不传则新建一个；D15 接上 checkpointer 后用于多轮"
  )

class citation(BaseModel):
  """
  一条被答案引用到的法条
  """
  idx:int=Field(...,description="答案里的引用编号，对应[idx]")
  law: str = Field(..., description="法律名称")
  no: str = Field(..., description="条号，如「第六十条」")
  text: str = Field(..., description="法条原文")

class ToolCallInfo(BaseModel):
  """
  一次工具调用
  """
  name:str=Field(...,description="工具名")
  args:dict=Field(default_factory=dict,description="调用参数")

class AskResponse(BaseModel):
  """
  提问响应
  """
  answer:str =Field(...,description="答案正文，含 [1] [2] 引用标记")
  citations:list[citation]=Field(default_factory=list,description="答案里实际引用到的法条")
  tool_calls:list[ToolCallInfo]=Field(default_factory=list, description="这轮调用过的工具")
  thread_id: str = Field(..., description="会话 id，下次追问带上它")
  elapsed_ms: int = Field(..., description="本次耗时（毫秒）")