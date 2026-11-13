"""
Agent图，模型+工具 只管模型节点+工具节点+循环判断
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END,START,StateGraph,MessagesState
from langgraph.prebuilt import ToolNode

def build_graph(tools,api_key,model="deepseek-chat", max_tokens=1200):
  """
  给定工具列表，返回编译好的Agent图
  """
  client=ChatOpenAI(
    model_name=model,
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=max_tokens
  )

  #绑定工具
  client_with_tools=client.bind_tools(tools)

  def call_model(state:MessagesState):
    """
    节点1：把当前对话历史交给模型，模型可能回复内容，也可能要求调工具
    """
    response=client_with_tools.invoke(state["messages"])
    return {"messages":[response]}  #追加到历史末尾（键必须是 messages，MessagesState 的合并器挂在这个键上）

  def should_continue(state:MessagesState):
    """
    分支判断：最后一条消息如果有tool_calls，就去工具节点；否则结束
    """
    last=state["messages"][-1]
    if getattr(last,"tool_calls",None):
      return "tools"
    return END

  g=StateGraph(MessagesState)
  g.add_node("agent",call_model)
  g.add_node("tools",ToolNode(tools))
  g.add_edge(START,"agent")
  g.add_conditional_edges("agent",should_continue,["tools",END])
  g.add_edge("tools","agent")
  return g.compile()   


