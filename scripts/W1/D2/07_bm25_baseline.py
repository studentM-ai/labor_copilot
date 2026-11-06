"""
验证vs法条  收集改写素材，用于后续的提升检索的对照

初步试着字面匹配，得到粗命中率
这里的一些测试问题是网上查找的常问语句
这里输出的是md格式的报告，便于观看
"""
import json,math,re
from collections import Counter
from pathlib import Path
CLEAN = Path("data/clean")
EVAL = Path("data/eval")
EVAL.mkdir(parents=True, exist_ok=True)

K1 = 1.5      # 词频饱和速度
B = 0.75      # 长度归一化强度
TOP_K = 3     # 只看前 3 条

#（用户可能会提到的，法条实际写的）term
TERM_GAP=[
  ("社保","社会保险"),
  ("五险一金", "社会保险"),
  ("年假","年休假"),
  ("竞业协议", "竞业限制"),
  ("辞退","解除劳动合同"),
  ("开除","解除劳动合同"),
  ("n+1","经济补偿"),
  ("裁员","经济性裁员"),
  ("加班费", "延长工作时间"),
  ("试用期", "试用期"),  
  ("公司","用人单位"),
]

#网上找的常见的10个问题 +参考答案法
QUESTIONS=[
  ("公司不给我交社保怎么办","社会保险法 第58/63条"),
  ("试用期最长能有多久","劳动法 第21条"),
  ("被裁员了能拿多少钱","劳动合同法 第47条"),
  ("加班费怎么算","劳动法 第44条"),
  ("公司让我签竞业协议合法吗", "劳动合同法 第23/24条"),
  ("工伤怎么认定","工伤保险条例 第14条"),
  ("年假有几天","职工带薪年休假条例 第3条"),
  ("女员工怀孕能被辞退吗","女职工劳动保护特别规定 第5条"),
  ("劳动仲裁要多久","劳动争议调解仲裁法 第43条"),
  ("拖欠工资找谁","劳动保障监察条例 第9条"),
]

#切分一句话为字符二元组
def tokenize(s):
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s)   #只保留中文
    return [s[i:i+2]for i in range(len(s)-1)] #每次取两个字切分

def main():
  rows=[json.loads(line) for line in 
        open(CLEAN / "articles.jsonl", encoding="utf-8") if line.strip()]

  N=len(rows)

  #报告写入到md里
  md=[
    "#BM25前期测试","",f"语料{N}条","",
    "## 一、术语差别","",
    "| 用户可能说的 | 条数 | 法条实际写的 | 条数 | ","|...|...|...|...|"
  ]

  #术语差别
  print("术语差别（包含该词的法条数）")
  print(f"{'用户可能说的':<12}{'条数':>6} | {'法条实际写的':<16}{'条数':>6}")

  for user_term,law_term in TERM_GAP:
    #只要用户说的在text或title中就算包含
    c_user=sum(1 for r in rows if user_term in r["text"] or user_term in r["law_title"])
    c_law=sum(1 for r in rows if law_term in r["text"] or law_term in r["law_title"])
    print(f"{user_term:<12}{c_user:>6}   | {law_term:<16}{c_law:>6}")
    #添加到md里
    md.append(f"| {user_term} | {c_user} | {law_term} | {c_law} |")
  print("如果用户词接近0而法条词很多，那么就算字面检索0命中；")
  print("如果两边都是同样条数，这是对照组，说明有落差的才是需要解决的问题")

  #建立BM25初步的简单索引
  #正文+法律名的分词 不用条文号是因为用户一般不会问
  docs=[]
  for r in rows:
    toks=tokenize(r['text']+" "+ r["law_title"])  #得到列表
    docs.append({
      "row":r,
      "tokens":toks,  
      "tf":Counter(toks)  #词频
    })

  avgdl = sum(len(d["tokens"]) for d in docs) / N   # 平均文档长度 （计算切分的那个）

  #看多少条法条包含这个词
  #set（）去重：一条里出现5次也只算1条  转字典特性
  df=Counter()
  for d in docs:
    df.update(set(d["tokens"]))

  #给法条打分 （手写的bm25公式）
  # 公式：  集合 idf * (f * (k1+1)) / (f + k1 * (1 - b + b*dl/avgdl))
  def bm25(q_tokens,d):
    score=0
    for w in q_tokens:
      f = d["tf"].get(w, 0)      # 这个词在本条里出现几次
      if not f:
        continue                 # 没出现就跳过（顺带避免 df 取不到）
      idf = math.log(1 + (N - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
      score += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * len(d["tokens"]) / avgdl))
    return score

  print("\n")
  print(f"10个问题的TOP{TOP_K}")
  md+=["","##二、bm25简单检索结果（自判定）",""]

  #序号，（问题，回答）
  for qidx ,(q,ref) in enumerate(QUESTIONS,1):
    #利用集合性质再次去重
    q_tokens=list(set(tokenize(q)))   #得到问题切分字符列表

    #给所有的法条打分，从高到低 取3个 （bm25分数，docs对象）
    scored = [(bm25(q_tokens, d), d) for d in docs]
    scored.sort(key=lambda x:-x[0])

    print(f"[{qidx} {q}]")
    print(f"参考答案：{ref}")
    md+=[f"###{qidx}.{q}","",f"参考答案：{ref}",""]

    #排名 （分数，d对象）
    for rank,(s,d)in enumerate(scored[:TOP_K],1): #要3条分数最高的
      r=d["row"]
      #拿到片段，太长了不全部展示了
      snippet=r['text'][:45].replace("\n"," ")
      print(f"{rank}：{s:.2f} -[{r['law_title']} {r['display_no']}]  {snippet}...")
      md.append(f"{rank}. {r['law_title']} {r['display_no']} （{s:.2f}）{snippet}...")
    
    md += ["", "判定：[ ] 命中  [ ] 未命中", ""]

    #保存md文件
    (EVAL / "bm25_baseline_result.md").write_text("\n".join(md), encoding="utf-8")
    print("结束")

if __name__=="__main__":
  main()



