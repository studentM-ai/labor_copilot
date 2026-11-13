"""
合同检验的Agent版
017是自己运行脚本，指定目录下的所有合同
019是用户说话agent自己判断要不要体检，体检哪份

工具：
search_law      检索法条（和 011 一样）
calc_severance  算经济补偿（和 011 一样）
check_contract  体检一份合同（新增）

"""
import json,os,sys
from pathlib import Path

# 到项目根
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.tools import tool
from src.agent import build_graph
from src.citation import RefPool
from src.contract import TOP_K,check_contract_file,load_rows
from src.retriever import VectorRetriever

load_dotenv()
EVAL=ROOT / "data" / "eval"
MODEL="deepseek-chat"
TOP_K=TOP_K
LIMIT= 20   #圈循环

SYSTEM_PROMPT = """你是劳动法助手，服务对象是不懂法的普通人与秋招大学生。

工作方式（必须遵守）：
1.回答劳动法问题时，先调用 search_law 检索法条，不要凭自己的知识回答。
2.用户让你看一份合同、问「我的合同有没有问题」时，调用 check_contract。
  这个工具需要合同文件名（放在 data/contracts 下），
  如果用户没给文件名，就问用户要，不要自己猜。
3.需要算经济补偿金额时，调用 calc_severance，不要自己心算。
  缺工作年限或月工资时先问清楚，不要假设数字。
4.引用法条时用 [1] [2] 编号，编号来自 search_law 的返回。
5.检索到的法条不足以回答时，直接说「未找到相关依据，建议咨询专业律师」。
6.问题不属于劳动法范畴时，说明属于哪个领域并建议咨询，不要调工具。
7.如果拒答（没有调用检索工具），不要标注任何引用编号。"""

def make_tools(retriever,pool):
  """
  造三个工具，工具定义在函数里是因为要闭包捕获retriever和pool
  """
  @tool
  def search_law(query:str,k:int =TOP_K):
    """
    检索劳动法条文。
    注意：检索词要用法律术语，不要直接用用户的口语。
    比如：
    用户说[加班费] 用[延长工作时间 工资 报酬]
    用户说[社保]   用[社会保险 缴纳]
    用户说[年假]   用[年休假 天数]
    同一个问题可以检索多次，换不同关键词
    """
    hits=retriever.retrieve(query,k) #检索内容
    numbered=pool.add(hits)  #加入全局引用池，得到编号
    
    # 注意返回给模型的是文本，每条前面需要带全局编号
    return "\n\n".join(
      f"[{nidx}] 《{r['law_title']}》{r['display_no']}\n{r['text']}" for nidx,r in numbered)

  @tool 
  def calc_severance(years: float, monthly_salary: float)->str:
    """
    按《劳动合同法》第47条第一款计算经济补偿金（N）。
       
    规则：每满一年支付一个月工资；六个月以上不满一年的按一年计算；不满六个月的支付半个月。
    只在用户明确问了金额、且已知工作年限和月工资时调用。
    缺参数就反问用户，一定不要猜。
    注意：本工具没有考虑特殊情况，如《劳动合同法》第47条第二款的规定。（高薪封顶和年限）
    """
       
    #年限
    if years<0:
      return "工作年限不能为负数，请检查输入。"
    if years<0.5:
      months=0.5   #不满半年则算半个月
    elif years<1:
      months=1     #六个月以上（含）不满一年  按一年
    else:
      whole_years=int(years) #取整
      fraction=years-whole_years  #得到年份差值
      if fraction>=0.5:
        months=float(whole_years+1)   #满半年加一个月
      elif fraction>0:
        months=whole_years+0.5  #不满半年加半个月
      else:
        months=float(whole_years)   #整年
       
    amount=months*monthly_salary    #计算金额
    return(
      f"计算结果：\n"
      f"工作年限 {years} 年 -> 折算 {months} 个月工资\n"
      f"月工资 {monthly_salary:.0f} 元\n"
      f"经济补偿金 = {months} × {monthly_salary:.0f} = {amount:.0f} 元\n"
      f"依据：《中华人民共和国劳动合同法》第四十七条\n"
      f"限制：未计入第二款的高薪封顶与十二年上限"
    )

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


  return [search_law, calc_severance, check_contract]


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

#三组测试（注重测试能力）
TEST_CASES=[
  ("帮我看看 sample_1.txt 这份合同有没有问题", "check_contract"),
  ("公司不给我交社保怎么办", "search_law"),
  ("我想知道试用期最长能有多久", "search_law"),
]

def tool_calls_of(messages):
  """
  从消息历史里提取调用过哪些工具，按顺序返回工具名列表
  """
  names=[]
  for m in messages:
    if m.type=="ai" and getattr(m,"tool_calls",None):
      for tc in m.tool_calls:
        names.append(tc["name"])
  return names

def main():
  api_key=os.getenv("DEEPSEEK_API_KEY")
  rows=load_rows()
  retriever=VectorRetriever(rows)
  md = ["# 合同体检 Agent 测试结果", "", f"- 模型：{MODEL}，recursion_limit={LIMIT}", ""]
  passed = 0

  for idx,(question,expected)in enumerate(TEST_CASES,1):
    print(f"\n[{idx}]{question}")

    #每组用新的引用池，编号不串台
    pool=RefPool()
    tools=make_tools(retriever,pool)
    app=build_graph(tools,api_key,model=MODEL)

    try:
      result=app.invoke(
        {
          "messages":[
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
          ]
        },
        config={
          "recursion_limit":LIMIT
        }
      )
    except Exception as e:
      print(f"[出错]{e}")
      md+=[f"## {idx}. {question}", "", f"**出错**：{e}", "", "---", ""]
      continue

    names=tool_calls_of(result["messages"])
    answer=result["messages"][-1].content or ""

    #验收 这组该调哪个工具
    if expected == "check_contract":
      ok="check_contract" in names
    else:
      #这组不该调check_contract
      ok = ("search_law" in names) and ("check_contract" not in names)

    if ok:
      passed+=1

    print(f"调用的工具：{names}")
    print(f"期望：{expected}  ->  {'√' if ok else '×'}")
    print(f"答案（前 100 字）：{answer[:100]}")

    md.append(f"## {idx}. {question}")
    md.append("")
    md.append(f"- 调用的工具：{names}")
    md.append(f"- 期望：{expected}")
    md.append(f"- 判定：{'√ 通过' if ok else '× 失败'}")
    md.append("")
    md.append("**答案**：")
    md.append("")
    md.append(answer)
    md.append("")
    md.append("---")
    md.append("")
  md.insert(4, f"**通过 {passed} / {len(TEST_CASES)}**")
  md.insert(5, "")

  output = EVAL / "contract_agent_result.md"
  with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

  print()
  print(f"通过 {passed} / {len(TEST_CASES)}")

if __name__=="__main__":
  main()