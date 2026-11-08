"""
用 Ragas 评估生成质量
主要量faithfulness忠诚度，answer_relevancy(回答相关度)
先看忠诚度能不能行再测回答相关度
超范围题不参与，因为它的回答是拒答，没有可拆的句子
"""
import json,os,sys,time
from pathlib import Path

#到项目根
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from openai import OpenAI
from src.retriever import VectorRetriever
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset,evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness,AnswerRelevancy
#回答相关度要用的
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper



load_dotenv()
CLEAN = ROOT / "data" / "clean"
EVAL = ROOT / "data" / "eval"

TOP_K = 5
MODEL = "deepseek-chat"

#调试条数，先是测前几条
RUN_LIMIT=999

#是否强制重新生成答案  true是重新调llm生成     false是有缓存就用缓存（调指标时不用重跑）
FORCE_REGENERATE = False  

#是否要算answer_relevancy
USE_ANSWER_RELEVANCY = True

#中间文件：问题+答案+检索到的法条
FILE=EVAL/"ragas_samples.jsonl"

SYSTEM_PROMPT="""你是劳动法助手，服务对象是不懂法的普通人。

严格遵守以下规则：
1. 只依据【参考法条】里的内容回答，不要使用你自己的法律知识。
2. 每句话后面用 [1] [2] 这样的编号，标注依据的是哪一条。
3. 检索到的法条不足以回答时，直接说「未找到相关依据，建议咨询专业律师」，不许推测。
4. 用大白话解释。可以引用法条原文，但要再解释一遍它是什么意思。
5. 答案控制在 400 字以内。先给结论，再给依据。不要写「1. 2. 3.」的分点结构，不要展开法条没写的延伸建议。"""

def load_rows():
  """
  读法条语料
  """
  rows = []
  for line in open(CLEAN / "articles.jsonl", encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows

def load_golden_set():
  """
  读40条测试集，返回列表,大致内容如下
  {
    "id": "q01",
    "question": "公司不给我交社保怎么办",
    "gold_law_ids": ["social-insurance-law-0058", "..."],
    "type": "kouyu",
    "must_refuse": false
  }
  """
  records=[]
  for line in open(EVAL/"golden_set.jsonl",encoding="utf-8"):
    if line.strip():
      records.append(json.loads(line))
  return records

#生成答案
def build_messages(question,hits):
  """
  把检索到的法条拼成prompt
  hits:检索到的法条
  """
  refs=[]
  for i,r in enumerate(hits,1):
    refs.append(f"[{i}] 《{r['law_title']}》{r['display_no']}\n{r['text']}")
  user_content="【参考法条】\n"+"\n\n".join(refs)+f"\n\n【问题】\n{question}"

  return [
    {"role":"system","content":SYSTEM_PROMPT},
    {"role":"user","content":user_content}
  ]

def generate_samples(retriever,records):
  """
  调llm生成答案，返回样本列表
    [{
      "id": rec["id"],
      "type": rec["type"],
      "user_input": question,
      "response": answer,
      "retrieved_contexts": contexts,
    }]

  records:测试集
  """
  api_key=os.getenv("DEEPSEEK_API_KEY")
  client=OpenAI(api_key=api_key,base_url="https://api.deepseek.com")

  samples=[]

  for i,rec in enumerate(records,1):
    question=rec["question"]

    #检索
    hits=retriever.retrieve(question,TOP_K)

    #拼prompt
    messages=build_messages(question,hits)

    #调模型
    response=client.chat.completions.create(
      model=MODEL,
      messages=messages,
      temperature=0,
      max_tokens=2500
    )
    answer=response.choices[0].message.content or ""

    #收集contexts 只放正文，不放法律名和条文号  因为 faithfulness 判断的是「答案能不能从这些文字推出来」
    contexts=[]

    for h in hits:
      contexts.append(f"《{h['law_title']}》{h['display_no']}：{h['text']}")

    samples.append({
      "id": rec["id"],
      "type": rec["type"],
      "user_input": question,
      "response": answer,
      "retrieved_contexts": contexts,
    })

    print(f"[{i}/{len(records)}] {rec['id']} 生成 {len(answer)} 字")

    #轻微限速
    time.sleep(0.3)

  return samples

def save_samples(samples):
  """
  把样本写进jsonl，一行一条
  """
  with open(FILE,"w",encoding="utf-8")as f:
    for s in samples:
      f.write(json.dumps(s,ensure_ascii=False)+"\n")
  print(f"已保存-> {FILE}")

def load_cached_samples():
  """
  从jsonl读回样本；文件不存在返回空列表
  """
  if not FILE.exists():
    return []

  samples=[]
  for line in open(FILE,encoding="utf-8"):
    if line.strip():
      samples.append(json.loads(line))
  return samples

def get_samples(retriever,records):
  """
  取样本，优先用缓存，条数不够才重新生成（这个是用于减少时间成本的）
  records:测试集
  因为一开始用的调试次数LIMIT是2，所以缓存也是2，如果后续调了limit到10，此时就得重新调
  """
  if FORCE_REGENERATE or not FILE.exists():   #如果不用缓存(重新生成)或者文件不存在
    print("\n生成答案（调llm可能有点慢）")
    samples=generate_samples(retriever,records) #生成样本
    save_samples(samples) #顺便存样本
    return samples

  #用缓存的话直接就从文件里载入
  cached=load_cached_samples()
  if len(cached)>=len(records):
    print(f"\n使用缓存：{FILE}（{len(cached)} 条）")
    print("想重新生成就把 FORCE_REGENERATE 改成 True")
    return cached[:len(records)]

  #缓存不够用，则重新生成
  print(f"\n缓存只有 {len(cached)} 条，不足 {len(records)} 条，重新生成...")
  samples=generate_samples(retriever,records)
  save_samples(samples)
  return samples

def main():
  rows=load_rows()
  all_records=load_golden_set()   #全部 40 条

  print(f"语料 {len(rows)} 条，测试集 {len(all_records)} 条")

  #只评估有gold的题(不要超范围题)
  records=[]
  for rec in all_records:
    if len(rec["gold_law_ids"])>0:
      records.append(rec)
  print(f"可判定题 {len(records)} 条")

  records=records[:RUN_LIMIT]
  print(f"本次跑 {len(records)} 条（RUN_LIMIT={RUN_LIMIT}）")

  if len(records)==0:
    print("没有要评估的题，退出")
    return

  #建检索器
  retriever = VectorRetriever(rows)

  #取样本
  samples=get_samples(retriever,records)
  if len(samples)==0:
    print("没有样本，退出")
    return

  #准备ragas
  print("\n准备 Ragas ...")

  api_key=os.getenv("DEEPSEEK_API_KEY")

  #包装ragas的llm
  llm=ChatOpenAI(
    model=MODEL,
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
  )
  judge=LangchainLLMWrapper(llm)

  #组装指标
  metrics=[Faithfulness()]
  metric_names=["faithfulness"]
  embeddings=None
  if USE_ANSWER_RELEVANCY:  #如果要算回答相关度的话
    try:
      hf_emb=HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}, #向量归一化
      )
      #封装嵌入模型
      embeddings=LangchainEmbeddingsWrapper(hf_emb)

      metrics.append(AnswerRelevancy(strictness=1)) #生成1一个候选问题，因为deepseek只接受一个
      metric_names.append("answer_relevancy")
      print("已启用 answer_relevancy")

    except ImportError as e:
      print(f"[提示] 算不了 answer_relevancy：{e}")

  #调ragas
  print("\n调用 Ragas（会再调很多次 LLM，几分钟）...")
  dataset=EvaluationDataset.from_list(samples)  #数据集
  result=evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=judge,
    embeddings=embeddings
  )

  print("\nRagas 原始结果：")
  print(result)

  #取分数（to_pandas） 转表格形式的
  df=result.to_pandas()   #这是方法，必须加括号才会返回表格

  averages={}
  for name in metric_names: #指标名
    if name in df.columns:  #如果有在这个表里
      averages[name]=df[name].mean()  #mean是取出这一列有数字求和 ÷ 元素个数 就是平均值

  #写md
  md = []
  md.append("# Ragas 生成质量评估")
  md.append("")
  md.append(f"- 模型：{MODEL}")
  md.append(f"- 检索：向量检索 Top{TOP_K}")
  md.append(f"- 样本：{len(samples)} 条（只含可判定题）")
  md.append("")
  md.append("## 指标说明")
  md.append("")
  md.append("| 指标 | 含义 |")
  md.append("|---|---|")
  md.append("| faithfulness | 答案每句能否由给定法条推出。1.0 = 全部有依据 |")
  if "answer_relevancy" in averages:
    md.append("| answer_relevancy | 由答案反推问题，与原问题的相似度。低分说明答非所问 |")
  md.append("")
  md.append("## 平均分")
  md.append("")
  md.append("| 指标 | 平均分 |")
  md.append("|---|---|")
  for name in metric_names:
    if name in averages:
      md.append(f"| {name} | **{averages[name]:.3f}** |")
  md.append("")
  md.append("## 逐条明细")
  md.append("")

  # 表头按 metric_names 动态拼：启用几个指标就显示几列
  header = "| 题号 | 类型 | 问题 |"
  divider = "|---|---|---|---|"
  for name in metric_names:
    header = header + f" {name} |"
    divider = divider + "---|"
  md.append(header)
  md.append(divider)

  for i, s in enumerate(samples):
    # 每行先写题号、类型、问题
    line = f"| {s['id']} | {s['type']} | {s['user_input'][:24]} |"

    # 再按 metric_names 的顺序，把每个指标的分依次拼上去
    for name in metric_names:
      cell = "-"
      if name in df.columns and i < len(df):   #不超过行数
        v = df[name].iloc[i]                   #取第i行的值
        if v is not None and v == v:           # v == v 用来排除 NaN
          cell = f"{float(v):.2f}"
      line = line + f" {cell} |"

    md.append(line)

  md.append("")
  md.append("## 分数最低的 3 条（便于人工复看）")
  md.append("")

  if "faithfulness" in df.columns:
    pairs=[]  #存（分数，下标）

    #先收集所有有效分数
    for i in range(len(samples)):
      v = df["faithfulness"].iloc[i]  #iloc 按下标取出元素
      if v is not None and v==v:      #v==v 用来排除 NaN
        pairs.append((float(v), i))   #注意：要打包成一个元组

    #按分数从低到高排
    pairs.sort(key=lambda x:x[0])

    shown=0
    for score_value,index in pairs:
      if shown>=3:
        break
      s=samples[index]  #拿到对应样本
      # 标题里把所有指标的分都列出来，方便对照
      parts = []
      for name in metric_names:
        if name in df.columns and index < len(df):
          v2 = df[name].iloc[index]
          if v2 is not None and v2 == v2:
            parts.append(f"{name}={float(v2):.2f}")
      md.append("### " + s["id"] + "  " + "  ".join(parts))
      md.append("")
      md.append(f"**问题**：{s['user_input']}")
      md.append("")
      md.append("**答案**：")
      md.append("")
      md.append(s["response"][:800])
      md.append("")
      shown += 1

  output = EVAL / "ragas_eval.md"
  with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
  print()

if __name__ == "__main__":
  main()



                                                                                