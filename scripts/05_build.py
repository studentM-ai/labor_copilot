"""
按白名单展开成法条级 JSONL

通过04产出：最终白名单，将法律法条转成能检索的单元  并进行清洗

两个清洗规则：
正文 < 20 字，检索时没有信息量；条文号不以"第"开头：如整篇批复被解析成 [全文]，不是条文结构
"""

import json
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/clean")
OUT.mkdir(parents=True, exist_ok=True)

# 白名单：由 04_select.py 生成
selected = json.load(open(RAW / "_selected.json", encoding="utf-8"))

# id与文件路径映射
law_files = {p.stem: p for p in RAW.glob("**/laws/*.json")}

#最终合格法条，文本太短而被过滤的条文数量，条文号错误被过滤的数量，记录在白名单但是本地找不到的法律id
rows, short, weird, missing = [], 0, 0, []

for item in selected:
    law_id = item["id"]
    p = law_files.get(law_id)
    if not p:
        missing.append(f"{law_id} — {item['title']}")
        continue
    j = json.loads(p.read_text(encoding="utf-8"))   #读取法律内容
    for a in j.get("articles", []):
        text = a.get("text", "").strip()
        disp = a.get("displayNumber", "").strip()   # 第xxx条

        if len(text) < 20:              # 长度太短直接过滤
            short += 1
            continue
        if not disp.startswith("第"):    # 没有"第X条"结构也过滤
            weird += 1
            continue

        rows.append({
            "id": a["id"],                              # 条文唯一 id
            "law_id": law_id,                           # 法律 id
            "law_title": j["title"],                    # 法律名
            "display_no": disp,                         # 第X条
            "chapter": " / ".join(a.get("path", [])),   # 所属章节
            "text": text,
            "status": j.get("status", ""),
            "effective_date": j.get("effectiveDate", ""),
        })

(OUT / "articles.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
    encoding="utf-8"
)

print("=" * 45)
print("白名单:", len(selected), "部")
print("实际入库:", len(rows), "条")
print("过滤掉: 短条款", short, "/ 非条文号", weird)
if missing:
    print()
    print("[警告] 名单里有", len(missing), "部找不到文件：")
    for m in missing:
        print("   ", m)




# 白名单: 30 部
# 实际入库: 1251 条
# 过滤掉: 短条款 27 / 非条文号 0