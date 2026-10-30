"""
内容密度扫描：从正文找劳动相关法律

用于那些标题不相关劳动但是正文大量涉及劳动权益的那种
例如《尘肺病防治条例》《禁止使用童工规定》。

遍历全部 1513 部法律，逐条统计正文命中劳动相关词的条数和密度，
把"命中 >= 阈值"的全部列出来（不预先排除任何东西，判断交给 04_select.py）
"""
import json
import re
from pathlib import Path

RAW = Path("data/raw")

CONTENT_KW = re.compile(
    r"劳动|工资|工伤|社会保险|社保|工时|加班|辞退|解雇|竞业|"
    r"试用期|病假|产假|年休假|经济补偿|劳动者"
)

# 命中条数阈值：调低候选变多噪声也多，调高可能漏
THRESHOLD = 8


def main():
    law_files = {p.stem: p for p in RAW.glob("**/laws/*.json")}
    idx = json.load(open(RAW / "_index.json", encoding="utf-8"))    #拿id用的

    print("=" * 50)
    print(f"法律文件: {len(law_files)} 个")
    print(f"阈值:正文命中需要 >= {THRESHOLD} 条")

    rows = []
    for meta in idx:    #遍历_index索引文件里的信息
        p = law_files.get(meta["id"])   #通过id找到对应的法律文件路径
        if not p:
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))   #读取法律内容
        except Exception:
            continue
        arts = j.get("articles") or []  #拿到articles列表
        if not arts:
            continue

        #从正文里匹配关键词，如果匹配到就计数
        hits = 0    #命中法条数
        for a in arts:
            text = a.get("text") or ""
            if CONTENT_KW.search(text):
                hits += 1

        rows.append({
            "id": meta["id"], "title": j.get("title", ""),
            "articleCount": len(arts), "hits": hits,
            "density": round(hits / len(arts), 3),  #密度=命中条数/总条数 round（...，3）是保留3小数
        })

    cand = [r for r in rows if r["hits"] >= THRESHOLD]  #如果命中数大于8的就放入候选列表
    def get_hits(r):
        return r["hits"]

    cand.sort(key=get_hits,reverse=True)  #命中条数倒叙

    json.dump(cand, open(RAW / "_content_candidates.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"扫描:{len(rows)} 部 / {sum(r['articleCount'] for r in rows)} 条法条")
    print(f"正文匹配得到的候选:{len(cand)} 部（正文命中 >= {THRESHOLD}）")
    print("\n")
    for r in cand[:15]:
        print(f"{r['hits']:>4}/{r['articleCount']:<5} {r['density']*100:>3.0f}%  {r['title']}")
    print("\n" )

if __name__ == "__main__":
    main()


# 法律文件: 1514 个
# 阈值:正文命中需要 >= 8 条
# 扫描:1513 部 / 56182 条法条
# 正文匹配得到的候选:58 部（正文命中 >= 8）


#  101/107    94%  中华人民共和国劳动法
#   96/98     98%  中华人民共和国劳动合同法
#   71/98     72%  中华人民共和国社会保险法
#   67/67    100%  工伤保险条例
#   60/64     94%  保障农民工工资支付条例
#   57/63     90%  社会保险经办条例
#   49/54     91%  最高人民法院关于审理劳动争议案件适用法律问题的解释（一）
#   46/54     85%  中华人民共和国劳动争议调解仲裁法
#   42/70     60%  使用有毒物品作业场所劳动保护条例
#   41/69     59%  中华人民共和国就业促进法
#   39/88     44%  中华人民共和国职业病防治法
#   37/38     97%  中华人民共和国劳动合同法实施条例
#   35/36     97%  劳动保障监察条例
#   30/31     97%  社会保险费征缴暂行条例
#   28/30     93%  劳动就业服务企业管理规定