"""
check_contract 工具：体检一份合同
传文件名：一个短字符串，和 012 的 save_document(filename) 设计一致。
"""
from pathlib import Path
import sys
from langchain_core.tools import tool
from src.contract import TOP_K, check_contract_file

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "data" / "eval"


def build_report_lines(result):
  """
  把体检结果拼成md （和017差不多）
  """
  md = []
  md.append(f"# 合同体检报告：{result['file']}")
  md.append("")

  risky=[]
  for c in result["clauses"]:
    if c["risk"] in ("high","medium"):
      risky.append(c)

  md.append(f"共 {len(result['clauses'])} 条条款，检出 {len(risky)} 条风险。")
  md.append("")

  for c in risky:
    if c["risk"]=="high":
      mark="高风险"
    else:
      mark="有风险"
    md.append(f"## {mark} 第 {c['index']} 条")
    md.append("")
    md.append(">"+ c["text"].replace("\n", " "))
    md.append("")
    md.append(f"**理由**：{c['reason']}")
    md.append("")    
    if len(c["law_refs"]) > 0:
      md.append(f"**依据**：{'、'.join(c['law_refs'])}")
    else:
      md.append("**依据**：（模型未给出）")
    md.append("")         
  return "\n".join(md)   


def make_check_contract(retriever, top_k=TOP_K):
  """
  造一个 check_contract 工具
  retriever: 检索器（传进来而不是内部新建，避免重复加载 bge-m3）
  top_k: 每条条款检索几条法条
  """

  @tool
  def check_contract(filename:str)->str:
    """
    体检一份劳动合同，逐条检查法律风险
    filename:合同文件名
    文件放在 data/contracts/ 目录下。

    什么时候用：用户要求检查一份合同、或问「我的合同有没有问题」时。
    什么时候不用：用户只是问某个法律问题时，用 search_law 就够了。

    必须让用户提供文件名。不要自己编文件名，也不要要求用户贴合同全文。
    """
    result=check_contract_file(filename,retriever,TOP_K,verbose=False)

    #出错时把原因原样告诉模型，让它转述给用户
    if "error" in result:
      return f"体检失败：{result['error']}"

    clauses=result["clauses"]

    #统计风险
    risky=[]
    for c in clauses:
      if c["risk"] in ("high","medium"):
        risky.append(c)

    #只把最严重的3条摘要给模型，避免返回值太长挤占上下文
    risky.sort(key=lambda c:0 if c["risk"]=="high" else 1)  #把high的放前面
    top3=[]
    for c in risky[:3]:
      top3.append(f"第{c['index']}条（{c['risk']}）：{c['reason'][:50]}")

    #完整报告写进文件
    out_path=EVAL/ f"contract_agent_{Path(result['file']).stem}.md"
    EVAL.mkdir(parents=True,exist_ok=True)
    with open(out_path,"w",encoding="utf-8")as f:
      f.write(build_report_lines(result))

    #统计 high / medium 各自多少条
    high_n=0
    for c in risky:
      if c["risk"]=="high":
        high_n+=1
    medium_n=len(risky)-high_n

    summary=(
      f"合同 {result['file']} 体检完成：共 {len(clauses)} 条条款，"
      f"检出 {len(risky)} 条有风险的条款"
      f"（高风险 {high_n} 条，中风险 {medium_n} 条）。\n"
    )

    if len(top3)>0:
      summary+=f"下面只列出最严重的 {len(top3)} 条（不是全部）：\n"
      summary+="\n".join(top3)+"\n"

    summary+=f"完整报告已写入 {out_path.name}"

    return summary

  return check_contract