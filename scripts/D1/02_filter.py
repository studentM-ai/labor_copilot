"""
关键词粗筛：主题里找劳动相关词

只筛标题，一定会漏。所以正文高度相关、但标题不含关键词的法律（如《尘肺病防治条例》），交给 03补充。
"""
import json
import re

KW = re.compile(r"劳动|工伤|社会保险|工资|就业|女职工|工会|职业病|年休假|工作时间")

idx = json.load(open("data/raw/_index.json", encoding="utf-8"))
hits = [x for x in idx if KW.search(x["title"])]    #标题关键词匹配

out = [
    {"id": x["id"], "title": x["title"],
     "articleCount": x["articleCount"], "source": "keyword"}
    for x in hits
]
json.dump(out, open("data/raw/_keyword_candidates.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("关键词候选:", len(out), "部")
print("总法条:", sum(x["articleCount"] for x in hits))

#使用条数排序
def get_article_num(law_item):
    return law_item["articleCount"]

#取前15条
for x in sorted(hits, key=get_article_num)[-1:-16:-1]:
    print(f"  {x['articleCount']:>4}条  {x['title']}")
print("-" * 50)

# 关键词候选: 31 部
# 总法条: 1208
#    107条  中华人民共和国劳动法
#     98条  中华人民共和国劳动合同法
#     98条  中华人民共和国社会保险法
#     88条  中华人民共和国职业病防治法
#     70条  使用有毒物品作业场所劳动保护条例
#     69条  中华人民共和国就业促进法
#     67条  工伤保险条例
#     64条  保障农民工工资支付条例
#     63条  社会保险经办条例
#     58条  中华人民共和国工会法
#     54条  中华人民共和国劳动争议调解仲裁法
#     54条  最高人民法院关于审理劳动争议案件适用法律问题的解释（一）
#     48条  退役军人就业创业促进条例
#     38条  中华人民共和国劳动合同法实施条例
#     36条  劳动保障监察条例
# --------------------------------------------------