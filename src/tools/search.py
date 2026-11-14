"""
search_law 工具：检索法条，并把结果放进引用池
RefPool 保证同一部法条跨多次检索拿到同一个编号。

"""
from langchain_core.tools import tool


def make_search_law(retriever, pool, top_k=5):
  """
  造一个 search_law 工具
  retriever: 检索器（BM25 / 向量 / 混合都行，只要实现了 retrieve(query, k)）
  pool:引用池 RefPool，负责编号
  top_k: 默认检索几条
  返回LangChain 的 StructuredTool，可以直接放进 tools 列表
  """
  @tool
  def search_law(query: str, k: int = top_k) -> str:
    """检索劳动法条文。

    注意：检索词要用法律术语，不要直接用用户的口语。
      用户说「加班费」→ 用「延长工作时间 工资 报酬」
      用户说「社保」  → 用「社会保险 缴纳」
      用户说「年假」  → 用「年休假 天数」
    同一个问题可以检索多次，换不同关键词。
    """
    # 检索
    hits = retriever.retrieve(query, k)
    
    # 进引用池，拿回带全局编号的 (编号, 法条) 列表
    numbered = pool.add(hits)

    # 返回给模型的是文本，每条前面带全局编号
    return "\n\n".join(
      f"[{nidx}] 《{r['law_title']}》{r['display_no']}\n{r['text']}"
      for nidx, r in numbered
    )

  return search_law