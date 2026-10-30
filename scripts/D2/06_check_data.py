"""
检查数据，查看产出的data/clean/articles.jsonl，是否符合预期
主要检查 ：
字段完整，id唯一，长度范围，日期格式，溯源（看输出的每一条回到原始语料核对）

"""
import json,random,re
from pathlib import Path
from collections import Counter
RAW = Path("data/raw")
CLEAN = Path("data/clean")
EVAL=Path("data/eval")
EVAL.mkdir(parents=True, exist_ok=True)

#所需字段
# REQUIRED=["id","law_id","law_title","display_no","text","status","effective_date"]

#（第一次结果的改动）
# 必填：缺了下游会崩
REQUIRED = ["id", "law_id", "law_title", "display_no", "text", "status"]

# 可选：缺了不影响功能，但必须在报告里留痕
OPTIONAL = ["effective_date", "chapter"]


#原始的语料
law_files = {p.stem: p for p in RAW.glob("**/laws/*.json")}
#核心法
CORE_TITLES = {
    "国务院关于职工工作时间的规定",
    "保障农民工工资支付条例",
    "中华人民共和国社会保险法",
    "工伤保险条例",
    "中华人民共和国就业促进法",
    "中华人民共和国职业病防治法",
    "女职工劳动保护特别规定",
    "中华人民共和国劳动争议调解仲裁法",
    # 部贯穿性基础法 
    "中华人民共和国劳动法",
    "中华人民共和国劳动合同法"
}

def main():
  #d1得到的最终结果
  rows=[json.loads(line) for line in open(CLEAN/"articles.jsonl",encoding="utf-8") if line.strip()]

  #白名单,用于对账
  selected=json.load(open(RAW/"_selected.json",encoding="utf-8"))
  print("==对账==")
  print(f"法律总条数: {len(rows)}")
  print(f"覆盖法律: {len({r['law_id'] for r in rows})}部")

  ok=True   #开关，有一项NG就会变成False
  report={}   #总体报告，存成json

  #检查字段完整性
  miss=[] #缺失的
  for r in rows:
    for k in REQUIRED:
      if not r.get(k):  #如果缺失了一个字段,就加上这个id
        miss.append(r["id"])
        break

  #记录失去字段的条数
  report["miss_fields"]=len(miss) 
  #如果是有缺失的
  if miss:
    ok=False
    print(f"缺失{len(miss)}条，示例如: {miss[:5]}")
  else:
    print(f"字段完整性检查通过,共{len(rows)}条")

  print("可选字段缺口：")
  for k in OPTIONAL:
    miss_k = [r for r in rows if not r.get(k)]  #如果缺失可选字段
    report["miss_" + k] = len(miss_k) #加到报告中
    if miss_k:
      law_num = len({r["law_id"] for r in miss_k})
      print(f"{k}缺失{len(miss_k)}条，涉及{law_num}部法")
    else:
      print(f"{k}无缺失")

  # status 分布：记录库里有没有非"现行有效"的法规
  status_count = {}
  status_count = dict(Counter(r.get("status") for r in rows) )#
  report["status_distribution"] = status_count  #状态分布，看有效和失效的
  print(f"status分布：{status_count}")
    

  #检查id唯一性
  dup=len(rows)-len({r["id"] for r in rows})    #总数-只计算id的数=重复数 应该是0
  report["duplicate"]=dup #记录重复数
  if dup: #如果有重复id
    ok=False
    print(f"id有重复的，重复{dup}条")
  else:
    print("没有重复的id，所有id唯一")


  #检查长度
  #最长的text：
  # lens=sorted(len({r["text"]}) for r in rows)
  # max_len=lens[-1],min_len=lens[0],mid_len=lens[len(lens)//2]
  # print(f"max_len:{max_len},min_len:{min_len},mid_len:{mid_len}")

  worse_len=[r for r in rows if not (20<=len(r["text"]))]   
  #内容长度不在这个区域的加上
  report["worse_len"]=len(worse_len)
  if worse_len:
    ok=False
    print(f"内容不在范围内的有{len(worse_len)}条")
  else:
    lens = sorted(len(r["text"]) for r in rows)
    print(f"长度正常：最短{lens[0]} / 中位{lens[len(lens)//2]} / 最长{lens[-1]}")
    print("  最长的3条（确认不是整章被塞成一条）：")
    for r in sorted(rows, key=lambda x: -len(x["text"]))[:3]:
      print(f"{len(r['text']):>4} 字  {r['law_title']} {r['display_no']}")


  #检查日期格式（为了后续检查有效法规）
  data_re=re.compile(r"^\d{4}-\d{2}-\d{2}$") #xxxx-xx-xx
  #match是开头，search是任意
  worse_date=[r for r in rows 
              if not data_re.match(r["effective_date"] )and r["effective_date"]] #如果有效时间没有匹配到规定格式则加
  report["worse_date"]=len(worse_date)
  if worse_date:
    ok=False
    print(f"有日期格式错误的{len(worse_date)}条")
    for r in worse_date[:3]:
      print(f"{r['id']} → {r['effective_date']!r}")
  else:
    n_has = sum(1 for r in rows if r["effective_date"])
    print(f"日期格式都正确（{n_has}条有日期，{len(rows)-n_has}条为空，空值已在上面记录）")

  #核心法覆盖
  have={r["law_title"] for r in rows} #所有的titile
  lost=CORE_TITLES-have    #应有title-实际有title=缺少的 如果不缺少就是空集
  report["lost_core_laws"]=sorted(lost) #如果有丢失的那么就加进去
  if lost:
    ok=False
    print(f"核心法缺少{lost}")
  else:
    print(f"核心法都在")
  


  #溯源
  src_index={}    #建立原文索引id :(text,title)
  declared_ids=set()  #白名单的全部id，用来数覆盖了多少法
  for item in selected:
    p=law_files.get(item["id"]) #从白名单拿到id，在原始语料取那一条法律
    if not p:
      continue
    j=json.loads(p.read_text(encoding="utf-8")) #拿到那一条数据
    law_title=j.get("title","") #得到法律title

    for a in j.get("articles",[]):
      text=(a.get("text")or "").strip()
      declared_ids.add(a["id"]) #条id
      src_index[a["id"]]=(text,law_title)

  #逐条对比：
  mismatch=[]
  for r in rows:
    s=src_index.get(r["id"]) 

    if s is None:
      mismatch.append((r["id","原文不存在"]))

    elif s[0]!=r["text"]:
      #正文被修改
      mismatch.append((r["id"],"text与原文不一致"))

    elif s[1]!=r["law_title"]:
      #法律名对不上，说明挂错了来源
      mismatch.append((r["id"],"law_title与原文不一致"))

  report["mismatch"]=len(mismatch)  #不匹配条数

  if mismatch:
    ok=False
    print(f"有{len(mismatch)}条对不上原文")
    for mid,why in mismatch[:5]:
      print(f"{mid}:{why}")
  else:
    print(f"OK,{len(rows)}条全部在原文中找到，正文与法律名一致")

  n_out=len({r["law_id"] for r in rows})  #成品法律数
  n_sel=len(selected) #                 白名单法律数

  report["laws_output"]=n_out
  report["laws_whitelist"]=n_sel

  if n_out!=n_sel:  #如果白名单数不等于成品后的法律数   就是说明整部丢失了
    ok=False
    print(f"输出{n_out}部！=白名单{n_sel}部")
    out_ids={r["law_id"]for r in rows}  #拿最终所得的法律id
    sel_ids={x["id"] for x in selected} #拿白名单的法律id
    missing=sel_ids-out_ids #如果有差值 说明有法律没有展开
    if missing:
      print(f"有法律一条都没展开：{sorted(missing)}")
  else:
    print(f"输出{n_out}部，与白名单一致，没有丢失")


  #抽样人眼确认，看看有没有奇怪的东西
  print("-"*20)
  print("随机抽3条")
  for r in random.sample(rows,3):
    print(f'[{r["law_title"]}-{r["display_no"]}]-({len(r["text"])} 字)')
    print(f'{r["text"][:80]}...')
    print(f'chapter: {r["chapter"] or "(无)"}')


  #保存最终报告
  report["passed"] = ok                 #报告是否通过
  report["total_articles"] = len(rows)  #总法律数
  report["total_laws"] = n_out          #总条数

  (EVAL / "check_report.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2),
      encoding="utf-8"
  )

  print()
  if ok:
      print("结论: 全部通过")
  else:
      print("结论: 有问题，先修再往下走，数据坏了，后面所有实验结论都不可信")

if __name__=="__main__":
  main()


# ==对账==
# 法律总条数: 1251
# 覆盖法律: 30部
# 字段完整性检查通过,共1251条
# 可选字段缺口：
# effective_date缺失326条，涉及14部法
# chapter缺失164条，涉及10部法
# status分布：{'effective': 1251}
# 没有重复的id，所有id唯一
# 长度正常：最短20 / 中位86 / 最长797
#   最长的3条（确认不是整章被塞成一条）：
#  797 字  女职工劳动保护特别规定 第十六条
#  692 字  使用有毒物品作业场所劳动保护条例 第四十一条
#  615 字  中华人民共和国职业病防治法 第七十二条
# 日期格式都正确（925条有日期，326条为空，空值已在上面记录）
# 核心法都在
# OK,1251条全部在原文中找到，正文与法律名一致
# 输出30部，与白名单一致，没有丢失
# --------------------
# 随机抽3条
# [中华人民共和国尘肺病防治条例-第二十条]-(67 字)
# 各企业、事业单位必须贯彻执行职业病报告制度，按期向当地卫生行政部门、劳动部门、工会组织和本单位的主管部门报告职工尘肺病发生和死亡情况。...
# chapter: 第四章健康管理
# [中华人民共和国社会保险法-第十八条]-(49 字)
# 国家建立基本养老金正常调整机制。根据职工平均工资增长、物价上涨情况，适时提高基本养老保险待遇水平。...
# chapter: 第二章基本养老保险
# [中华人民共和国社会保险法-第四十七条]-(43 字)
# 失业保险金的标准，由省、自治区、直辖市人民政府确定，不得低于城市居民最低生活保障标准。...
# chapter: 第五章失业保险

# 结论: 全部通过




#第一次
#==对账==    
# 法律总条数: 1251
# 覆盖法律: 30部
# 缺失326条，示例如: ['xzfg-ec5eb2c922-0001', 'xzfg-ec5eb2c922-0002', 'xzfg-ec5eb2c922-0003', 'xzfg-ec5eb2c922-0004', 'xzfg-ec5eb2c922-0005']
# 没有重复的id，所有id唯一
# 内容都在长度范围内
# 日期格式都正确！
# 核心法都在
# OK1251条全部在原文中找到，正文与法律名一致
# 输出{n_out}部，与白名单一致，没有丢失
# --------------------
# 随机抽3条
# [保障农民工工资支付条例-第二十三条]-(123 字)
# 建设单位应当有满足施工所需要的资金安排。没有满足施工所需要的资金安排的，工程建设项目不得开工建设；依法需要办理施工许可证的，相关行业工程建设主管部门不予颁发施工...
# chapter: 第四章工程建设领域特别规定
# [中华人民共和国劳动合同法实施条例-第三十三条]-(68 字)
# 用人单位违反劳动合同法有关建立职工名册规定的，由劳动行政部门责令限期改正；逾期不改正的，由劳动行政部门处2000元以上2万元以下的罚款。...
# chapter: 第五章法律责任
# [中华人民共和国劳动合同法实施条例-第十六条]-(85 字)
# 劳动合同法第二十二条第二款规定的培训费用，包括用人单位为了对劳动者进行专业技术培训而支付的有凭证的培训费用、培训期间的差旅费用以及因培训产生的用于该劳动者的其他...
# chapter: 第二章劳动合同的订立

# 结论: 有问题，先修再往下走，数据坏了，后面所有实验结论都不可信