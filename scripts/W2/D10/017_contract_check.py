"""
初次合同体验
输入一份劳动合同文本 切成条款  每条检索相关法条  判断有没有法律风险  输出带法条依据的风险清单。
"""

import json,os,re,sys
from pathlib import Path
#到项目根
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from openai import OpenAI
from src.retriever import VectorRetriever
load_dotenv()
CLEAN = ROOT / "data" / "clean"
EVAL = ROOT / "data" / "eval"
CONTRACTS = ROOT / "data" / "contracts"   #待体检的合同（.txt）

MODEL = "deepseek-chat"
TOP_K = 5  
MIN_PARA_LEN = 15  #短于这个字数的段落并到上一条

SYSTEM_PROMPT="""你是劳动法审查助手。下面会给你一条劳动合同条款，以及检索到的相关法条。

请判断这条条款是否存在法律风险。

严格遵守：
1. 只依据给出的【参考法条】判断，不要使用你自己的法律知识。
2. 如果给出的法条不足以判断，risk 必须填 "unknown"，不要猜。
3. 只输出一个 JSON 对象，不要输出任何其他文字。

输出格式：
{"risk": "high" | "medium" | "none" | "unknown",
 "reason": "一句话说明这条为什么有/没有问题",
 "law_refs": ["《法律名》第N条"]}

risk 的含义：
  high     明显违反法条，或明显对劳动者不利
  medium   可能违法，或表述模糊、有争议
  none     对照法条未发现问题
  unknown  给出的法条不足以判断
"""

def load_rows():
  """
  读法条语料
  """
  rows = []
  for line in open(CLEAN / "articles.jsonl", encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows

def load_contracts():
  """
  读所有待检验的合同，返回[(文件名，全文),...]
  只读 .txt，按文件名排序
  """
  contracts=[]
  for path in sorted(CONTRACTS.glob("*.txt")):
    text=path.read_text(encoding="utf-8")
    contracts.append((path.name,text))
  return contracts

def merge_short(parts):
  """
  把过短的段落并到上一条，去掉空段落
  parts是分割后的字符串列表
  """
  out = []
  for p in parts:
    p=p.strip()
    if p=="":
      continue
    #如果太短并且有上一条，那么就并进去
    if len(p)<MIN_PARA_LEN and len(out)>0:
      out[-1]=out[-1] +"\n"+p #更新最后一条，合并上去
    else:
      out.append(p)

  #应对第一条是标题的情况
  if len(out)>=2 and len(out[0])<MIN_PARA_LEN:
    out[1]=out[0]+"\n"+out[1]
    out.pop(0)  #删除首元素
  return out

NUMBER_START = re.compile(r"^(第[一二三四五六七八九十百零]+条|\d+[.、]|[一二三四五六七八九十]+、)")
def split_by_number(block):
  """
  在按空行切出来的块内部，再按首编号切分
  只有块出现两个以上编号行首才切分
  返回切好的段落列表
  """
  lines = block.split("\n") #切成行
  #先记下所有以编号开头的行号下标
  starts=[]
  for i,line in enumerate(lines):
    if NUMBER_START.match(line.strip()):  #如果匹配了就记下其下标
      starts.append(i)

  #编号少于两个的不切
  if len(starts)<2:
    return [block]

  parts=[]

  #第一个编号之前如果有内容(首段说明)，单独成一段
  if starts[0]>0: #如果这个大于 0说明第一个编号之前是有内容的
    head="\n".join(lines[:starts[0]]).strip() #取到第一个匹配到的编号行 为一个段落
    if head:
      parts.append(head)

  # 按编号位置把块切开
  i=0
  while i<len(starts):
    start_line=starts[i]  #开始
    if i+1 <len(starts):
      end_line=starts[i+1]
    else:
      end_line=len(lines) #到末尾行
    p="\n".join(lines[start_line:end_line])
    parts.append(p)
    i+=1
  return parts

def split_clauses(text):
  """
  分割合同成一条条 条款
  先按空行切成块，如果内部还有多个编号开头再按编号细分，切出来不到3条说明这个合同没用空行；
  那么用编号正则切全篇
  返回条款字符串列表
  """
  text=text.replace("\r\n","\n").strip()

  #1.按空行切块，再内部按编号细分
  blocks=re.split(r"\n\s*\n",text)
  parts=[]
  for b in blocks:
    if b.strip()=="":
      continue
    for s in split_by_number(b):
      parts.append(s)

  #2.兜底 切不出3块说明没用空行分隔，改为按编号切全部
  if len(parts)<3:
    parts=re.split(r"\n(?=第[一二三四五六七八九十百零]+条|\d+[.、]|[一二三四五六七八九十]+、)",text)

  return merge_short(parts)

def build_messages(clause,hits,full_text):
  """
  把条款和检索到的法条拼成prompt
  clause:一条合同条款
  hits:检索到的法条列表
  full_text:整份合同全文（作为背景方便更好判断）
  """
  refs=[]
  for i,r in enumerate(hits,1):
    refs.append(f"[{i}] 《{r['law_title']}》{r['display_no']}：{r['text']}")

  user_content = (
    "【参考法条】\n" + "\n\n".join(refs) +
    "\n\n【整份合同（背景，用于判断条款之间是否冲突）】\n" + full_text +
    "\n\n【需要审查的条款】\n" + clause
  )

  return [
    {"role":"system","content":SYSTEM_PROMPT},
    {"role":"user","content":user_content}
  ]

def parse_verdict(raw):
  """
  把模型返回的文本解析成dict
  模型有时会在json外包一层```json，所有先要去掉这个，如果解析失败就返回unknown
  raw:模型原始输出
  返回 {"risk": ..., "reason": ..., "law_refs": [...]}
  """
  text=raw.strip()

  #去掉 ```josn...```
  if text.startswith("```"):
    text=re.sub(r"^```[a-zA-Z]*\n","",text) #比如这种：```python
    text=re.sub(r"\n```$","",text)  #去除末尾的
    text=text.strip()

  #只取第一个{ 到最后一个} 之间的内容
  start=text.find("{")
  end=text.find("}")
  if start>=0 and end>start:
    text=text[start:end+1]

  try:
    data=json.loads(text) #转字典
  except Exception:
    return {"risk": "unknown", "reason": "模型输出无法解析", "law_refs": []}

  #字段缺失或取值非法时兜底
  risk=data.get("risk","unknown")
  if risk not in ("high","medium","none","unknown"):
    risk="unknown"

  return {
    "risk":risk,
    "reason":str(data.get("reason","")),
    "law_refs":data.get("law_refs",[])or [],
  }

def judge_clause(client,clause,hits,full_text):
  """
  让模型判断一条条款有无风险

  clause：合同条款
  hits:检索到的法条
  返回parse_verdict的结果
  """
  messages=build_messages(clause,hits,full_text)
  response=client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0,
    max_tokens=600
  )
  raw=response.choices[0].message.content or ""
  return parse_verdict(raw)

def check_one_contract(client,retriever,name,text):
  """
  检验一份合同                                                                    
  返回
  {
    "file":文件名，
    "clauses": [{index,text,risk,reason,law_refs,hit_laws}, ...]
  }
  """
  clauses=split_clauses(text) #先切分
  print(f"\n{name}：切成 {len(clauses)} 条")
  results=[]
  for i ,clause in enumerate(clauses,1):
    hits=retriever.retrieve(clause,TOP_K)
    #判断
    verdict=judge_clause(client,clause,hits,text)

    #记住这次检索到的法条，方便人工判断
    hit_laws=[]
    for h in hits:
      hit_laws.append(f"《{h['law_title']}》{h['display_no']}")

    results.append({
      "index": i,
      "text": clause,
      "risk": verdict["risk"],
      "reason": verdict["reason"],
      "law_refs": verdict["law_refs"],
      "hit_laws": hit_laws,
    })

    #条款原文可能换行，打印的时候弄成一行                
    one_line = clause[:30].replace("\n", " ")
    print(f"[{i}/{len(clauses)}] {verdict['risk']:<8} {one_line}")
  return {"file":name,"clauses":results}

def build_report(checks):
  """
  把检验结果拼成md报告
  checks:check_one_contract的返回值列表
  """
  md = []
  md.append("# 合同体检报告")
  md.append("")
  md.append(f"- 模型：{MODEL}")
  md.append(f"- 每条条款检索 Top{TOP_K} 法条")
  md.append(f"- 合同份数：{len(checks)}")
  md.append("")

  md.append("")
  md.append("| 合同 | 条款数 | high | medium | none | unknown |")
  md.append("|---|---|---|---|---|---|")

  for chk in checks:
    counts={"high":0, "medium":0, "none":0, "unknown":0}
    for c in chk["clauses"]:
      counts[c["risk"]]=counts.get(c["risk"],0)+1

    md.append(
      f"| {chk['file']} | {len(chk['clauses'])} | "
      f"{counts['high']} | {counts['medium']} | {counts['none']} | {counts['unknown']} |"
    )
    md.append("")

  #逐份明细
  for chk in checks:
    md.append(f"## {chk['file']}")
    md.append("")

    #有风险的逐条展开
    risky=[]
    for c in chk["clauses"]:
      if c["risk"] in ("high","medium"):
        risky.append(c)
    if len(risky)==0:
      md.append("没有检测出风险条款")
      md.append("")
    else:
      md.append(f"检出 {len(risky)} 条有风险的条款：")
      md.append("")

    for c in risky:
      #high用高风险，medium用有风险
      if c["risk"]=="high":
        mark="高风险"
      else:
        mark="有风险"

      md.append(f"### {mark} 第 {c['index']} 条（{c['risk']}）")
      md.append("")
      md.append("**条款原文**：")
      md.append("")
      md.append("> " + c["text"].replace("\n", " "))
      md.append("")
      md.append(f"**判断理由**：{c['reason']}")
      md.append("")

      if len(c["law_refs"]) > 0:
        md.append(f"**依据**：{'、'.join(c['law_refs'])}")
      else:
        md.append("**依据**：（模型未给出）")
      md.append("")

      md.append(f"**本次检索到的法条**：{'、'.join(c['hit_laws'])}")
      md.append("")

    #没有发现问题的只列编号
    ok=[]
    for c in chk["clauses"]:
      if c["risk"]=="none":
        ok.append(f"第{c['index']}条")
    if len(ok)>0:
      md.append(f"**未发现问题的条款**：{'、'.join(ok)}")
      md.append("")

    #法条不足的单独列出，提醒人工看
    unknown=[]
    for c in chk["clauses"]:
      if c["risk"]=="unknown":
        unknown.append(f"第{c['index']}条")
    if len(unknown)>0:
      md.append(f"**法条不足、无法判断的条款**：{'、'.join(unknown)}")
      md.append("")

  return md
    
def main():
  contracts=load_contracts()
  if len(contracts)==0:
    print(f"没有找到合同文件，在{CONTRACTS}下放txt文件")
    return
  print(f"找到{len(contracts)}份合同")
  api_key=os.getenv("DEEPSEEK_API_KEY")
  client=OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
  )
  #读法条与检索器
  rows=load_rows()
  retriever=VectorRetriever(rows)

  #逐份检验
  checks=[]
  for name,text in contracts:
    chk=check_one_contract(client,retriever,name,text)
    checks.append(chk)
  
  #md
  md=build_report(checks)
  output = EVAL / "contract_check_result.md"
  with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
  print()

  #除了 md 报告，再存一份结构化结果，给 018 的评估脚本比对用
  raw_output = EVAL / "contract_check_raw.json"
  with open(raw_output, "w", encoding="utf-8") as f:
    json.dump(checks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
  main()

