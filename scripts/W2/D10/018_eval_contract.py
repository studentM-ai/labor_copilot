"""
合同检验评估所用的，把017的结果与labels.json标注后的文件进行比对，算出检出率、误报率、切分准确率。
检出率：被报high或medium的数/标注的隐患总数
误报率：被报high或medium的正常条款数/正常条款总数
切分准确率：切出的条款合标注一致的合同数/合同总数
"""

import json,os,sys
from pathlib import Path
#到项目根
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "data" / "eval"
CONTRACTS = ROOT / "data" / "contracts"

def load_json(path):
  """
  读一个json文件
  """
  return json.loads(path.read_text(encoding="utf-8"))

def main():
  raw_path=EVAL/"contract_check_raw.json"
  label_path=CONTRACTS/"labels.json"

  if not raw_path.exists():
    print("没有找到 contract_check_raw.json，先跑一次 017_contract_check.py")
    return
  if not label_path.exists():
    print(f"没有找到标注文件 {label_path}")
    return

  raw=load_json(raw_path)
  labels=load_json(label_path)

  #把体验结果按文件名索引，方便按合同查
  raw_by_file={}
  for chk in raw:
    raw_by_file[chk["file"]]=chk

  total_risks = 0   #标注的雷总数
  hit_risks = 0   #被检出的雷数
  total_normal = 0   #标注的正常条款总数
  false_pos = 0    #被误报的正常条款数
  split_ok = 0     #切分正确的合同数

  rows=[] #存每份合同的明细
  for file_name,label in labels.items():
    if file_name not in raw_by_file:
      print(f"[跳过]{file_name}没有体验结果")
      continue

    chk=raw_by_file[file_name]
    clauses=chk["clauses"]

    #条款序号：判断结果
    by_index={}
    for c in clauses:
      by_index[c["index"]]=c["risk"]

    #切分准不准
    split_correct=(len(clauses))==label["expected_clauses"]
    if split_correct:
      split_ok+=1

    #隐患有没有检测出
    hit=0
    missed=[] #没有检测出的条款号
    for r in label["risks"]:
      total_risks+=1
      risk=by_index.get(r["index"],"missing")
      if risk in ("high","medium"):
        hit+=1
        hit_risks+=1
      else: 
        missed.append(f'第{r["index"]}条({risk})')

    #正常条款是否有误报
    fp=0
    false_list=[]
    for index in label["normal"]:
      total_normal+=1
      risk=by_index.get(index,"missing")
      if risk in ("high","medium"):
        fp+=1
        false_pos+=1
        false_list.append(f"第{index}条({risk})")

    rows.append(
      {
        "file":file_name,
        "clauses":len(clauses),
        "expected_clauses":label["expected_clauses"],
        "split_correct":split_correct,
        "risks":len(label["risks"]),
        "hit":hit,
        "missed":missed,
        "normal":len(label["normal"]),
        "false_pos":fp,
        "false_list":false_list,
      }
    )

  #汇总
  recall=hit_risks/total_risks if total_risks else 0
  fp_rate=false_pos/total_normal if total_normal else 0
  split_rate=split_ok/len(rows) if rows else 0

  print()
  print(f"检出率{hit_risks}/{total_risks} = {recall:.1%}")
  print(f"误报率{false_pos}/{total_normal} = {fp_rate:.1%}")
  print(f"切分正确 {split_ok}/{len(rows)} = {split_rate:.0%}")

  #报告
  md = []
  md.append("# 合同体检评估")
  md.append("")
  md.append(f"- 合同份数：{len(rows)}")
  md.append(f"- 标注风险总数：{total_risks}")
  md.append(f"- 标注正常条款总数：{total_normal}")
  md.append("")
  md.append("## 总指标")
  md.append("")
  md.append("| 指标 | 结果 |")
  md.append("|---|---|")
  md.append(f"| 检出率 | **{hit_risks} / {total_risks} = {recall:.1%}** |")
  md.append(f"| 误报率 | {false_pos} / {total_normal} = {fp_rate:.1%} |")
  md.append(f"| 切分正确 | {split_ok} / {len(rows)} = {split_rate:.0%} |")
  md.append("")
  md.append("## 逐份明细")
  md.append("")
  md.append("| 合同 | 条款数 | 切分 | 雷数 | 检出 | 误报 |")
  md.append("|---|---|---|---|---|---|")

  for r in rows:
    if r["split_correct"]:
      split_text="√"
    else:
      split_text=f"X{r["clauses"]}/{r["expected_clauses"]}"

    md.append(
      f'| {r["file"]} | {r["clauses"]} | {split_text} | '
      f'{r["risks"]} | {r["hit"]}/{r["risks"]} | {r["false_pos"]} |'
    )
  md.append("")

  #标注对照检查
  md.append("")
  md.append("## 标注对齐检查")
  md.append("")
  md.append("逐条对照【标注的风险 index】和【实际切出来的条款原文】。")
  md.append("如果这里出现明显不匹配，")
  md.append("说明标注错位或切分错了，要先修正再来看上面的指标。")
  md.append("")
  for file_name,label in labels.items():
    if file_name not in raw_by_file:
      continue

    clauses=raw_by_file[file_name]["clauses"]
    md.append(f"### {file_name}")
    md.append("")
    md.append("| 标注 | 标注写的问题 | 实际这一条的原文 |")
    md.append("|---|---|---|")

    for r in label["risks"]:
      idx=r["index"]
      if idx<=len(clauses):
        #只要钱36个字够了
        actual=clauses[idx-1]["text"][:36].replace("\n"," ")
        md.append(f'| 第 {idx} 条 | {r["issue"][:22]} | {actual} |')
      else:
        md.append(f'| 第 {idx} 条 | {r["issue"][:22]} | **切分只有 {len(clauses)} 条，该条不存在** |')
  md.append("")

  
  #漏检和误报明细
  md.append("## 漏检与误报明细")
  md.append("")
  for r in rows:
    if len(r["missed"])==0 and len(r["false_list"])==0:
      continue
    md.append(f'### {r["file"]}')
    md.append("")
    if len(r["missed"])>0:
      md.append(f'**漏检**：{"、".join(r["missed"])}')
    if len(r["false_list"])>0:
      md.append(f'**误报**：{"、".join(r["false_list"])}')
    md.append("")

  output=EVAL/"contract_eval.md"
  with open(output,"w",encoding="utf-8")as f:
    f.write("\n".join(md))
  print()

if __name__=="__main__":
  main()



    
