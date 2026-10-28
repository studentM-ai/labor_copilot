"""
测试查看与大致主题相关的法律看看够不够用
"""
import json
import re

index_file = open("data/raw/_index.json", encoding="utf-8")
all_law_list = json.load(index_file)
index_file.close()

#设置关键词
keyword_rule = re.compile(r"劳动|工伤|社会保险|工资|就业|女职工|工会|职业病|年休假|工作时间")

# 新建一个列表，用来存放筛选出来的法律
selected_law = []
for one_law in all_law_list:
    law_title = one_law["title"]
    # 检查标题里有没有匹配到关键词
    if keyword_rule.search(law_title):
        selected_law.append(one_law)    #匹配到就加入列表

print("相关法律:", len(selected_law), "部")

#按法条数量排
def get_article_num(law_item):
    return law_item["articleCount"]

selected_law_sorted = sorted(selected_law, key=get_article_num)

# 只拿法条数量最多的15条
top15_law = selected_law_sorted[-1:-16:-1]

# 循环打印

for law in top15_law:
    article_total = law["articleCount"] #条数
    title = law["title"]
    # :>4 让数字靠右对齐，排版整齐
    print(f"  {article_total:>4}条  {title}")

# 
print("总法条数:", sum(law["articleCount"] for law in selected_law))

# 相关法律: 31 部
#    107条  中华人民共和国劳动法
#     98条  中华人民共和国社会保险法
#     98条  中华人民共和国劳动合同法
#     88条  中华人民共和国职业病防治法
#     70条  使用有毒物品作业场所劳动保护条例
#     69条  中华人民共和国就业促进法
#     67条  工伤保险条例
#     64条  保障农民工工资支付条例
#     63条  社会保险经办条例
#     58条  中华人民共和国工会法
#     54条  最高人民法院关于审理劳动争议案件适用法律问题的解释（一）
#     54条  中华人民共和国劳动争议调解仲裁法
#     48条  退役军人就业创业促进条例
#     38条  中华人民共和国劳动合同法实施条例
#     36条  劳动保障监察条例
# 总法条数: 1208