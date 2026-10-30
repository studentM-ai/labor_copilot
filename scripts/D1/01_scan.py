"""
扫描全部，给出一些测试数据并整理法律索引
"""
import json, glob
from collections import Counter

files = glob.glob("data/raw/chinese-law-corpus-master/laws/*.json")
print("JSON 文件数:", len(files))

#可解析法律
laws = []
for f in files:
    try:
        j = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if isinstance(j, dict) and j.get("articles"): #有某个文件顶层是列表的，所以要判断
        laws.append(j)  #将可解析的法律字典加入列表

print("可解析的法律:", len(laws))
print("总法条数:", sum(len(l["articles"]) for l in laws))   #l["articles"] = [条文1, 条文2, 条文3]

print("\n分类分布:", dict(Counter(l.get("category") for l in laws)))  #Counter可以统计元素的个数
print("效力分布:", dict(Counter(l.get("status") for l in laws)))

lens = sorted(len(a["text"]) for l in laws for a in l["articles"])  #统计每条法条的字数，并排序，小到大
#最小 中间值 90分位值 最大
print("\n法条字数  min/median/p90/max:",
      lens[0], lens[len(lens)//2], lens[int(len(lens)*0.9)], lens[-1])

# 存一份索引，后面不用重复解析
idx = [{k: l.get(k) for k in
        ("id","title","category","articleCount","status","effectiveDate")}
       for l in laws]
json.dump(idx, open("data/raw/_index.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已保存 data/raw/_index.json")


# JSON 文件数: 1514
# 可解析的法律: 1513
# 总法条数: 56182

# 分类分布: {'法律': 303, '宪法': 1, '法律解释': 25, '监察法规': 2, '有关法律问题和重大问题的决定（部分）': 2, '司法解释': 563, '修正案': 12, '行政法规': 605}
# 效力分布: {'effective': 1510, 'pending': 3}

# 法条字数  min/median/p90/max: 4 93 228 16537

# 已保存 data/raw/_index.json