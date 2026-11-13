"""
合同体检（命令行版）
逐条检索法条并判断风险，输出带依据的清单
切分、判断、prompt 都在 src/contract.py —— 因为 Agent 版（019）也要用同一套。
这个脚本只负责三件事：遍历 data/contracts 下的文件、打印进度、写 md 和 json 报告。

md 报告是「命令行版」的输出格式；Agent 版输出的是摘要 + 另一个 md。
两种格式不同，所以拼装逻辑不放进 src/。
"""
import json,os,sys
from pathlib import Path

#到项目根
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.contract import CONTRACTS, TOP_K, check_contract_file, list_contracts, load_rows
from src.retriever import VectorRetriever

load_dotenv()

EVAL = ROOT / "data" / "eval"

MODEL = "deepseek-chat"
TOP_K = TOP_K


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
  names = list_contracts()

  if len(names) == 0:
    print(f"没有找到合同文件，在 {CONTRACTS} 下放 .txt 文件")
    return

  print(f"找到 {len(names)} 份合同")

  #读法条、建检索器
  rows = load_rows()
  retriever = VectorRetriever(rows)

  #逐份体检
  #注意：check_contract_file 内部会自己读文件，所以这里只传文件名
  checks = []
  for name in names:
    chk = check_contract_file(name, retriever, TOP_K, verbose=True)
    checks.append(chk)

  #写md
  md = build_report(checks)
  output = EVAL / "contract_check_result.md"
  with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

  # 写结构化结果，给 018 
  raw_output = EVAL / "contract_check_raw.json"
  with open(raw_output, "w", encoding="utf-8") as f:
    json.dump(checks, f, ensure_ascii=False, indent=2)

  print()
  print(f"-> {output}")
  print(f"-> {raw_output}")

if __name__ == "__main__":
  main()
