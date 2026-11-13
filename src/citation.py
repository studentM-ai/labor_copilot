"""
引用池，用于多次检索时编号不串台
RefPool维护一张全局编号表
"""
class RefPool:
  """
  全局引用池，按法条id去重，编号跨轮连续
  """
  def __init__(self):
    self.items=[] #按法条顺序存对象 编号=下标+1
    self.by_id={} #法条id:编号

  def add(self,rows):
    """
    把检索结果加进池子，返回[(编号，法条)]
    """
    out=[]
    for r in rows:
      if r["id"]in self.by_id:
        out.append((self.by_id[r["id"]],r)) #如果已存在就用旧编号，不重复占位
      else:
        #新的追加到末尾，编号就是当前长度
        self.items.append(r)
        nidx=len(self.items)
        self.by_id[r["id"]]=nidx  #用长度为编号即最后一个
        out.append((nidx,r))
    return out
