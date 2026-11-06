"""
用于生成40条测试集 
并且之前的测试题硬编码在013里，加一道题就得该评估代码
现在独立出“题目定义”
题目格式：（问题，【（法律名，条文号）】，类型）
两种用法：
1.生成测试集 python scripts/D8/014_build_golden_set.py
2.加新题时查候选法条 python scripts/D8/014_build_golden_set.py --find 经济补偿
"""

import json
import sys
from collections import Counter
from pathlib import Path

CLEAN = Path("data/clean")
EVAL = Path("data/eval")
EVAL.mkdir(parents=True, exist_ok=True)

#定义40道题
QUESTIONS = [
  #  口语题（18 道）
  # 用户用大白话问，需要检索来把口语翻译成法条
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

  #  新增 8 道 

  ("产假有多少天", [
    ("女职工劳动保护特别规定", "第七条"),
  ], "kouyu"),

  ("公司能随便给我调岗降薪吗", [
    ("中华人民共和国劳动合同法", "第三十五条"),
  ], "kouyu"),

  ("试用期被辞退有补偿吗", [
    ("中华人民共和国劳动合同法", "第三十九条"),
    ("中华人民共和国劳动合同法", "第四十六条"),
  ], "kouyu"),

  ("年假没休完怎么办", [
    ("职工带薪年休假条例", "第五条"),
  ], "kouyu"),

  ("工伤期间工资怎么算", [
    ("工伤保险条例", "第三十三条"),
  ], "kouyu"),

  ("拖欠工资能要赔偿金吗", [
    ("中华人民共和国劳动合同法", "第八十五条"),
  ], "kouyu"),

  ("什么情况下可以主动辞职还要补偿", [
    ("中华人民共和国劳动合同法", "第三十八条"),
    ("中华人民共和国劳动合同法", "第四十六条"),
  ], "kouyu"),

  ("劳动仲裁的时效是多久", [
    ("中华人民共和国劳动争议调解仲裁法", "第二十七条"),
  ], "kouyu"),


  # 术语题（8 道）
  # 查询词几乎逐字来自法条原文

  ("延长工作时间 百分之一百五十 工资报酬", [
    ("中华人民共和国劳动法", "第四十四条"),
  ], "term"),

  ("裁减人员 二十人以上 提前三十日 说明情况", [
    ("中华人民共和国劳动合同法", "第四十一条"),
  ], "term"),

  ("不能胜任工作 经过培训 调整工作岗位 提前三十日", [
    ("中华人民共和国劳动合同法", "第四十条"),
  ], "term"),

  #  新增 5 道 
  ("用人单位 单方解除劳动合同 事先通知工会", [
    ("中华人民共和国劳动合同法", "第四十三条"),
  ], "term"),

  ("女职工 产假 九十八天 难产 多胞胎", [
    ("女职工劳动保护特别规定", "第七条"),
  ], "term"),

  ("停工留薪期 原工资福利待遇 不变", [
    ("工伤保险条例", "第三十三条"),
  ], "term"),

  ("未休年休假 日工资收入 百分之三百", [
    ("职工带薪年休假条例", "第五条"),
  ], "term"),

  ("用人单位 未依法缴纳社会保险费 劳动者解除劳动合同", [
    ("中华人民共和国劳动合同法", "第三十八条"),
  ], "term"),

  # 多步题（8 道，全新）
  # 需要多次检索，或检索 + 计算才能回答
  ("我工作 3 年半月薪 12000，公司以不能胜任为由辞退我，"
    "还欠我 2 个月工资，一共能主张多少钱", [
    ("中华人民共和国劳动合同法", "第四十七条"),
    ("中华人民共和国劳动法", "第五十条"),
  ], "duobu"),

  ("试用期约定了 6 个月但合同只签 1 年，超期部分能要赔偿吗", [
    ("中华人民共和国劳动合同法", "第十九条"),
    ("中华人民共和国劳动合同法", "第八十三条"),
  ], "duobu"),

  ("公司一直不给我交社保，我主动辞职能要经济补偿吗", [
    ("中华人民共和国劳动合同法", "第三十八条"),
    ("中华人民共和国劳动合同法", "第四十六条"),
  ], "duobu"),

  ("工伤之后公司想把我辞退，合法吗", [
    ("中华人民共和国劳动合同法", "第四十二条"),
    ("工伤保险条例", "第三十三条"),
  ], "duobu"),

  ("竞业协议里没写补偿，这个协议还有效吗", [
    ("中华人民共和国劳动合同法", "第二十三条"),
    ("中华人民共和国劳动合同法", "第二十四条"),
  ], "duobu"),

  ("公司拖欠工资 3 个月，我能直接辞职并要补偿吗", [
    ("中华人民共和国劳动合同法", "第三十八条"),
    ("中华人民共和国劳动合同法", "第四十六条"),
  ], "duobu"),

  ("怀孕期间公司给我调岗降薪，合法吗", [
    ("女职工劳动保护特别规定", "第五条"),
    ("中华人民共和国劳动合同法", "第三十五条"),
  ], "duobu"),

  ("公司说我不能胜任工作要辞退，但没培训也没调岗，合法吗", [
    ("中华人民共和国劳动合同法", "第四十条"),
  ], "duobu"),


  # 超范围题（6 道）
  # gold 为空，正确行为是拒答 + 说明属于哪个领域
  ("房东不退我押金怎么办", [], "out_of_scope"),
  ("离婚财产怎么分割", [], "out_of_scope"),
  ("朋友借钱不还能起诉吗", [], "out_of_scope"),
  ("买房交了定金能退吗", [], "out_of_scope"),
  ("交通事故对方全责怎么索赔", [], "out_of_scope"),
  ("别的公司欠我货款一直不还怎么办", [], "out_of_scope"),
]


def load_rows():
  """
  读法条语料。返回列表，每条是dict
  """
  rows=[]
  for line in open(CLEAN/"articles.jsonl",encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows

def build_key_to_id(rows):
  """
  建立查找表：（法律名，条文号）-》法条id 实际上就是gold-》id
  """
  table={}
  for r in rows:
    key=(r["law_title"],r["display_no"])
    table[key]=r["id"]
  return table

def find_articles(rows,keyword):
  """
  find 打印所有包含该关键词的法条
  用来在加新题时，快速确认某法条
  """
  print(f"包含[{keyword}的法条]")
  count=0
  for r in rows:
    text=r["text"]+" "+r["law_title"]
    if keyword in text:
      count+=1
      snippet=r["text"][:46].replace("\n"," ")  #截取片段
      print(f"《{r["law_title"]}》{r["display_no"]}")
      print(f"id={r["id"]}")
      print(f"{snippet}")
  print(f"共{count}条")


def build():
  """
  主流程：读语料-解析gold-校验-生成jsonl
  """
  rows=load_rows()
  print(f"语料：{len(rows)}条法条")

  key_to_id=build_key_to_id(rows)
  
  records=[]  #jsonl的每一行
  missing_count=0 #有些gold可能找不到
  type_count=Counter()

  for index,item in enumerate(QUESTIONS,1):
    question=item[0]  
    gold_pairs=item[1]
    qtype=item[2]

    #把gold部分转id
    gold_ids=[]
    for key in gold_pairs:
      if key in key_to_id:
        gold_ids.append(key_to_id[key]) #添加id
      else:
        missing_count+=1
        print(f"第{index}题找不到{key}")

    #超范围题gold必须是空的 gold_pairs为空则key空 则gold_ids为空
    if qtype=="out_of_scope" and len(gold_ids)>0:
      print(f"第 {index} 题是超范围，但填了 gold")

    #非超范围题gold不能为空
    if qtype!="out_of_scope" and len(gold_ids)==0:
      print(f"  [警告] 第 {index} 题不是超范围，但没填 gold")

    records.append({
      "id": f"q{index:02d}",
      "question": question,
      "gold_law_ids": gold_ids,
      "type": qtype,
      "must_refuse": (qtype == "out_of_scope"),
    })
  
    type_count[qtype] += 1     # 这道题的类型 +1

  #保存
  output = EVAL / "golden_set.jsonl"
  with open(output, "w", encoding="utf-8") as f:
        for rec in records:
          f.write(json.dumps(rec, ensure_ascii=False) + "\n")

  #汇总
  print()
  print("="*88)
  print(f"生成 {len(records)} 条")
  print()

  print("类型分布：")
  for name in ("kouyu", "term", "duobu", "out_of_scope"):
    print(f"{name:<14} {type_count[name]} 条")
  print()
  if missing_count > 0:
    print(f"有 {missing_count} 个 gold 找不到，请用 --find 查正确的条文号")
  else:
    print("所有 gold 都能在语料里找到")

def main():
  #有find参数就进入查询模式
  if len(sys.argv)>=3 and sys.argv[1]=="find":
    keyword = sys.argv[2]
    rows = load_rows()
    find_articles(rows, keyword)
    return

  build()

if __name__=="__main__":
  main()

# 语料：1251条法条

# ========================================================================================
# 生成 40 条

# 类型分布：
# kouyu          18 条
# term           8 条
# duobu          8 条
# out_of_scope   6 条

# 所有 gold 都能在语料里找到