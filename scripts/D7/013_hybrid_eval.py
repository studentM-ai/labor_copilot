"""
做混合检索对比 测试集固化
使用RRF只看排名来评价
得到对比表来总结
"""

import json,re,math
from collections import Counter
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

CLEAN = Path("data/clean")
EVAL = Path("data/eval")
EVAL.mkdir(parents=True, exist_ok=True)
K1,B = 1.5,0.75      # BM25 参数，与 07 一致
TOP_K=5
RRF_K=60  #使用默认值60
RRF_POOL = 20     # 两路各取 20 条再融合

#BM25检索器 与07一样

#切分一句话为字符二元组
def tokenize(s):
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s)   #只保留中文
    return [s[i:i+2]for i in range(len(s)-1)] #每次取两个字切分

#定义简单的BM25检索器   复制的08
class BM25Retriever:
  def __init__(self,rows):
    self.rows = rows
    self.N = len(rows)
    self.docs = []
    for r in rows:
      toks = tokenize(r["text"] + " " + r["law_title"])
      self.docs.append({
        "row": r,
        "tokens": toks, 
        "tf": Counter(toks)})
    self.avgdl = sum(len(d["tokens"]) for d in self.docs) / self.N

    # 文档频率  多少个法条包含这个词
    self.df = Counter()
    for d in self.docs:
      self.df.update(set(d["tokens"]))

  #bm25计算分数方法
  def score(self, q_tokens, d):
    score = 0.0
    for w in q_tokens:
      f = d["tf"].get(w, 0)
      if not f:
        continue
      idf = math.log(1 + (self.N - self.df.get(w, 0) + 0.5) / (self.df.get(w, 0) + 0.5))
      score += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * len(d["tokens"]) / self.avgdl))
    return score

  #检索分最高的topk条
  def retrieve(self,query,k=TOP_K):
    q=list(set(tokenize(query)))
    scored=[(self.score(q,d),d) for d in self.docs]
    scored.sort(key=lambda x: -x[0])
    return [d["row"] for _, d in scored[:k]]    #只要row对象

#向量检索器  复制010的
class VectorRetriever:
  """向量检索器 接口和 BM25Retriever 完全一样，可以直接换
  只有一个方法：retrieve(query, k) -> [row, ...]
  """
  def __init__(self, rows, vec_path="data/clean/articles_vec.npy"):
    self.rows = rows
    # 读预先算好的向量矩阵
    self.vecs = np.load(vec_path) #把二进制文件读回数组

    self.model = SentenceTransformer("BAAI/bge-m3")

  def retrieve(self, query, k=TOP_K):
    # 问题也编码成向量
    # encode返回二维数组 (1, 1024)，可以批量编码多条；取第一条，得到 (1024,) 的一维向量
    q_vec = self.model.encode([query], normalize_embeddings=True)[0]

    # 算相似度
    #因为都归一化过，这个分数就是余弦相似度
    scores = self.vecs @ q_vec    

    # 取分数最高的 k 个
    #返回的是【下标】，不是分数
    top_idx = np.argsort(-scores)[:k]

    # ④ 用下标取回对应的法条
    return [self.rows[i] for i in top_idx]

#混合检索器  使用RRF
class HybridRetriever:
  def __init__(self,bm25,vec,rrf_k=RRF_K,pool=RRF_POOL):
    self.bm25 = bm25
    self.vec = vec
    self.k = rrf_k
    self.pool = pool

    #id->row 用于融合后按id找原始记录
    self.id_row={r["id"]:r for r in bm25.rows}

  def retrieve(self,query,k=TOP_K):
    scores={} #id:score

    #1.bm25的前20条
    for rank,r in enumerate(self.bm25.retrieve(query,self.pool)):
      scores[r["id"]]= scores.get(r["id"], 0) + 1 / (self.k + rank + 1) #累加分数

    #2.向量的前20条
    for rank, r in enumerate(self.vec.retrieve(query, self.pool)):
      scores[r["id"]] = scores.get(r["id"], 0) + 1 / (self.k + rank + 1)

    top=sorted(scores.items(),key=lambda x:-x[1])[:k]#从高到低进行刨析
    return [self.id_row[i] for i ,_ in top]

#题目 11道原有的+3道新的
#(问题，[（法律名，条文号）,...])
QUESTIONS = [
    #11 道口语题（沿用 D3–D6）
    ("公司不给我交社保怎么办", [
        ("中华人民共和国社会保险法", "第五十八条"),
        ("中华人民共和国社会保险法", "第六十三条"),
    ], "kouyu"),

    ("试用期最长能有多久", [
        ("中华人民共和国劳动法", "第二十一条"),
        ("中华人民共和国劳动合同法", "第十九条"),
    ], "kouyu"),

    ("被裁员了能拿多少钱，我工作了 3 年半，月薪 12000", [
        ("中华人民共和国劳动合同法", "第四十七条"),
    ], "kouyu"),

    ("加班费怎么算", [
        ("中华人民共和国劳动法", "第四十四条"),
    ], "kouyu"),

    ("公司让我签竞业协议合法吗", [
        ("中华人民共和国劳动合同法", "第二十三条"),
        ("中华人民共和国劳动合同法", "第二十四条"),
    ], "kouyu"),

    ("工伤怎么认定", [
        ("工伤保险条例", "第十四条"),
    ], "kouyu"),

    ("年假有几天", [
        ("职工带薪年休假条例", "第三条"),
    ], "kouyu"),

    ("女员工怀孕能被辞退吗", [
        ("女职工劳动保护特别规定", "第五条"),
    ], "kouyu"),

    ("劳动仲裁要多久", [
        ("中华人民共和国劳动争议调解仲裁法", "第四十三条"),
    ], "kouyu"),

    ("拖欠工资找谁", [
        ("劳动保障监察条例", "第九条"),
        ("保障农民工工资支付条例", "第十条"),
    ], "kouyu"),

    ("房东不退我押金怎么办", [], "out_of_scope"),   # 无 gold，应拒答

    # 3 道法条原词密集题
    # 查询词几乎逐字来自法条原文，BM25 应该占优
    ("延长工作时间 百分之一百五十 工资报酬", [
        ("中华人民共和国劳动法", "第四十四条"),
    ], "term"),

    ("裁减人员 二十人以上 提前三十日 说明情况", [
        ("中华人民共和国劳动合同法", "第四十一条"),
    ], "term"),

    ("不能胜任工作 经过培训 调整工作岗位 提前三十日", [
        ("中华人民共和国劳动合同法", "第四十条"),
    ], "term"),
]

def main():
  rows = [json.loads(l) for l in open(CLEAN / "articles.jsonl", encoding="utf-8") if l.strip()]
 
  #建立 （法律名,条文号）：id 用于解析上面问题前部分
  key_id={(r["law_title"], r["display_no"]): r["id"] for r in rows}

  print("初始化三种检索器...")
  bm25=BM25Retriever(rows)
  vec=VectorRetriever(rows)
  hyb=HybridRetriever(bm25,vec)
  retrievers=[("BM25", bm25), ("向量", vec), ("混合", hyb)]

  #统计：每种方法命中几题
  hit_count={name:0 for name,_ in retrievers}

  md=["#混合检索对比（BM25 / 向量 / RRF）", "",
        f"- 语料：{len(rows)} 条",
        f"- 三种检索都取 Top{TOP_K}；混合用 RRF(k={RRF_K})，两路各取 {RRF_POOL} 条",
        "- 判定：gold 法条是否出现在 Top5 里",
        "",
        "| # | 问题 | 类型 | BM25 | 向量 | 混合 |",
        "|---|---|---|---|---|---|"]

  print()
  print(f"{'#':>2} {'类型':<11} {'BM25':>10} {'向量':>10} {'混合':>10} 问题")
  print("="*50)

  #这里用gold表示问题上面的中间部分
  golden=[]  #用于生成问题与title的jsonl文件

  #逐题跑三种检索
  for qidx,(query,gold_pairs,qtype) in enumerate(QUESTIONS,1):
    #把（法律名，条文号）解析成真实的id
    gold_ids=[key_id[k] for k in gold_pairs if k in key_id] #拿到 问题相关的法条id

    missing=[k for k in gold_pairs if k not in key_id]
    if missing:
      print(f" [警告] 题 {qidx} 的 gold 找不到: {missing}")

    #三种检索各跑一次，记录gold排在第几位
    ranks={}   #检索器名：排名
    for name,ret in retrievers: #遍历名与检索器
      hits=ret.retrieve(query,TOP_K)  #拿到检索到的列表

      #在返回的TOPk条里找gold，找到就记下位置，找不到记0
      pos=0 #命中下标
      for idx ,h in enumerate(hits,1):  #
        if h["id"] in gold_ids:  #如果检索到的法条id在问题相关法条id里，说明命中了
          pos=idx
          break
      ranks[name]=pos #记下这个方法在这个题上的排名

      if pos>0: #大于0就算命中了
        hit_count[name]+=1

    #超范围的题没有gold部分，不参与命中统计
    if not gold_ids:
      ranks={k:0 for k in ranks}

    #把排名变成好读的符号
    def fmt(p): #p是排名数字
      #- 表示超范围拒答  x没进top5  √表示排到第p位
      if qtype == "out_of_scope":
        return "—"
      else:
        if p == 0:
          return "×"
        else:
          return f"√{p}"
        
    print(f"{qidx:>2} {qtype:<11} {fmt(ranks['BM25']):>10} "
          f"{fmt(ranks['向量']):>10} {fmt(ranks['混合']):>10}   {query[:34]}")

    #加入gold 测试集记录
    golden.append({
      "id": f"q{qidx:02d}",
      "question": query,
      "gold_law_ids": gold_ids,
      "type": qtype,
      "must_refuse": qtype == "out_of_scope", #拒答
    })

  #汇总
  #可判定的题 其实就是排除超范围题目
  n_scored = sum(1 for _, _, t in QUESTIONS if t != "out_of_scope") #要范围内的
  print("=" * 96)
  print(f"命中合计（共 {n_scored} 道可判定题）：")
  for name, _ in retrievers: 
      print(f"{name} {hit_count[name]} / {n_scored}")

  md += ["", f"**命中合计（共 {n_scored} 道可判定题）：**", "", 
         "| 检索方式 | 命中 |", "|---|---|"]  
  for name, _ in retrievers:  
      md.append(f"| {name} | {hit_count[name]} / {n_scored} |")

  #新增术语 题单独来看
  print()
  print("术语密集题（第 12–14 题）单独看：")
  md += ["", "**术语密集题（第 12–14 题）：**", ""]

  for qindx in (12,13,14):
    query,gold_pairs,_=QUESTIONS[qindx-1] #读后三题
    gold_ids=[key_id[k] for k in gold_pairs if k in key_id] #拿到id

    parts=[]
    for name,ret in retrievers:
      hits=ret.retrieve(query,TOP_K)
      #在返回的TOPk条里找gold，找到就记下位置，找不到记0
      pos=0 #命中下标
      for idx ,h in enumerate(hits,1):  
        if h["id"] in gold_ids:  #如果检索到的法条id在问题相关法条id里，说明命中了
          pos=idx
          break
      parts.append(f"{name}={pos if pos else '×'}")

    line = f"[{qindx}] {query[:30]:<32} " + "  ".join(parts)
    print(line)
    md.append(f"- {line.strip()}")


  #保存
  (EVAL / "hybrid_eval_result.md").write_text("\n".join(md), encoding="utf-8")

  #生成测试集（一行一条）
  with open(EVAL / "golden_set.jsonl", "w", encoding="utf-8") as f:
    for g in golden:
      f.write(json.dumps(g, ensure_ascii=False) + "\n")
  print()

if __name__ == "__main__":
    main()
    