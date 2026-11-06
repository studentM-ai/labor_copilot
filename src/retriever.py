"""
封装检索器类
有三种检索器
"""

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# 参数默认值 和之前的一样
K1 = 1.5  #词频饱和速度
B = 0.75  #长度归一化力度
TOP_K = 5
RRF_K = 60  #RRF排名里的k
RRF_POOL = 20 #RRF里的池子

# 向量文件位置
VEC_PATH = Path("data/clean/articles_vec.npy")
def tokenize(s):
  """
  切分 字符二元组
  """
  s=re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s) #只要汉字英文数字
  return [s[i:i+2] for i in range(len(s)-1)]  #每两个为一次

class BM25Retriever:
  """
  按字面重合度打分
  长度归一化，方便直接算点积
  """
  def __init__(self, rows):
    self.rows = rows
    self.N = len(rows)

    #每条法条切词、统计词频
    self.docs = []
    for r in rows:
      toks = tokenize(r["text"] + " " + r["law_title"])
      self.docs.append({
        "row": r,
        "tokens": toks,
        "tf": Counter(toks),  #词频：一篇文档里包含几个这个词   词：次数
      })

    #平均文档长度
    total=0
    for d in self.docs:
      total+=len(d["tokens"]) #全部文档token之和
    self.avgdl=total/self.N     #平均文档参数  所有文档总和token/文档数量

    #文档频率：有多少条法条包含该词
    self.df=Counter()
    for d in self.docs:
      self.df.update(set(d["tokens"]))  #利用set去重  词：包含法条数

  def score(self,q_tokens,d):
    """
    计算法条对这次查询的分数
    q_tokens：问题切好的token列表
    d:一条法条的预处理结果
    单term得分是：f(k1+1)*idf/(f+k1*(1-b+b*(dl/avgdl)))
    """
    s=0.0
    for q in q_tokens:
      #该词在本法条里出现几次，可以理解d["tf"]是个counter包着的字典
      f=d["tf"].get(q,0)    #词频
      if f==0:
        continue

      #计算idf逆文档频率 衡量词语稀有度的   公式：ln(1 + (N - df + 0.5) / (df + 0.5))
      num=self.N-self.df.get(q,0)+0.5
      den=self.df.get(q,0)+0.5
      idf=math.log(1 + num / den) 

      #词频饱和与长度归一化
      #f:词频  dl/avgdl：本条词长度/平均长度  k1=饱和速度  b=长度归一化力度
      doc_len=len(d["tokens"])
      denom=f+K1*(1-B+B*doc_len/self.avgdl)

      s+=idf*f*(K1+1)/denom #累加这个分
    return s

  def retrieve(self,query,k=TOP_K):
    """
    返回得分最高的k条法条
    query：问题
    k：TOPK
    """
    q_tokens=list(set(tokenize(query)))  #利用set去重
    #给每一条法条打分  （分数，文档）
    scored=[]
    for d in self.docs:
      scored.append((self.score(q_tokens,d),d))

    #按分数从高到低排序
    scored.sort(key=lambda x:-x[0])

    #只取出原始记录，不要token与tf
    result=[]
    for _,d in scored[:k]:
      result.append(d["row"])
    return result

class VectorRetriever:
  """
  按语义相似度检索 (向量)
  使用D4已经算好的，归一化的向量文件
  """
  def __init__(self,rows, vec_path=VEC_PATH):
    self.rows=rows
    #读向量矩阵
    self.vecs=np.load(vec_path)
    #加载嵌入模型，记得开镜像
    self.model=SentenceTransformer("BAAI/bge-m3")

  def retrieve(self,query,k=TOP_K):
    """
    返回语义最相近的k条
    """
    #问题编向量
    q_vec=self.model.encode([query],normalize_embeddings=True)[0] #要归一化方便算点积

    #算出所有相似度
    scores=self.vecs @ q_vec
    #取分数最高的k个
    top_idx=np.argsort(-scores)[:k]
    
    #用下表取回原始记录
    result=[]
    for i in top_idx:
      result.append(self.rows[i])
    return result

class HybridRetriever:
  """
  混合检索，只看排名不看分数
  使用RRF更方便
  总分=1/(k+r1+1) + 1/(k+r2+1) 
  """
  def __init__(self,bm25, vec, k_rrf=RRF_K, pool=RRF_POOL):
    self.bm25 = bm25
    self.vec = vec
    self.k = k_rrf
    self.pool = pool

    #id->原始记录  融合的时候只操作id，最后需要这个找回完整记录
    self.id_to_row={}
    for r in bm25.rows:
      self.id_to_row[r["id"]]=r

  def retrieve(self, query, k=TOP_K):
    scores={} #id：分数
    #bm25的
    bm25_hits=self.bm25.retrieve(query,self.pool)
    rank=0 #0为第一名
    for r in bm25_hits:
      this_id=r["id"]
      add=1/(self.k+rank+1)

      if this_id in scores:
        scores[this_id]+=add
      else:
        scores[this_id]=add
      rank+=1

    #向量的
    rank = 0
    vec_hits = self.vec.retrieve(query, self.pool)
    for r in vec_hits:
      this_id = r["id"]
      add = 1/(self.k+rank+1)
      if this_id in scores:
        scores[this_id]+=add
      else:
        scores[this_id]=add
      rank+=1

    #总分从高到低
    items=list(scores.items())
    items.sort(key=lambda x:-x[1])

    result=[]
    for this_id,_ in items[:k]:
      result.append(self.id_to_row[this_id])
    return result






    



    



