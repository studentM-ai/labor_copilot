"""
得到法条级别的Jsonl
按白名单展开成法条级 JSONL
把法律文件变成能检索的单元 不同与测试2，这里是按主题更全面一点的
清洗过滤一些无信息量的短条款
"""
import json
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/clean")
OUT.mkdir(parents=True, exist_ok=True)

#白名单
LABOR_LAWS = {
    "labor-law":                               "劳动法 — 劳动关系基础法",
    "labor-contract-law":                      "劳动合同法 — 订立/解除/补偿核心",
    "social-insurance-law":                    "社会保险法 — 社保缴纳与待遇",
    "trade-union-law":                         "工会法 — 工会与集体协商",
    "labor-dispute-mediation-arbitration-law": "劳动争议调解仲裁法 — 仲裁程序",
    "law-703831d503":  "职业病防治法 — 职业病认定",
    "law-a7458586db":  "就业促进法 — 就业歧视",
    "xzfg-caf9f56937": "使用有毒物品作业场所劳动保护条例 — 职业危害",
    "xzfg-8e4ae5638e": "工伤保险条例 — 工伤认定与待遇",
    "xzfg-f764e4b761": "保障农民工工资支付条例 — 欠薪维权",
    "xzfg-caa02b145f": "社会保险经办条例 — 社保登记",
    "xzfg-ec5eb2c922": "劳动合同法实施条例 — 细化规则",
    "xzfg-2d114901d5": "劳动保障监察条例 — 劳动监察投诉",
    "xzfg-6a4e3d70d4": "失业保险条例 — 失业金领取",
    "xzfg-715a4a92c6": "工人考核条例 — 技能考核",
    "xzfg-a3c577d934": "社会保险费征缴暂行条例 — 社保欠缴",
    "xzfg-165f33a3ba": "残疾人就业条例 — 残疾人就业",
    "xzfg-ca36f929fb": "劳动就业服务企业管理规定 — 就业服务机构",
    "xzfg-15774d5bea": "关于工资总额组成的规定 — 工资基数口径",
    "xzfg-4eb3304079": "女职工劳动保护特别规定 — 三期保护",
    "xzfg-df951803c0": "职工带薪年休假条例 — 年休假",
    "xzfg-bc9354a6a4": "国务院关于职工工作时间的规定 — 标准工时",
    "sfjs-laodong-1":  "最高法劳动争议解释（一）",
    "sfjs-laodong-2":  "最高法劳动争议解释（二）",
    "sfjs-2af3b157e7": "最高法工伤保险行政案件规定",
    "sfjs-9a7fc04491": "最高法拒不支付劳动报酬解释",
    "sfjs-22c40a3d9d": "最高法工会法适用解释",
}

# 构建id->路径映射  p.stem 是文件名去掉后缀的部分 p是文件对象
law_files = {p.stem: p for p in RAW.glob("**/laws/*.json")}

#最终合格法条，文本太短而被过滤的条文数量，条文号错误被过滤的数量，记录在白名单但是本地找不到的法律id
rows, short, weird, missing = [], 0, 0, []
for law_id, desc in LABOR_LAWS.items():
    p = law_files.get(law_id)   #拿到文件路径
    if not p:   #如果没有找到文件，就记录下来
        missing.append(f"{law_id} — {desc}")
        continue
    j = json.loads(p.read_text(encoding="utf-8"))
    for a in j.get("articles", []):
        text = a.get("text", "").strip()
        disp = a.get("displayNumber", "").strip()   #第xxx条
        if len(text) < 20:          #长度太短直接过滤
            short += 1
            continue
        if not disp.startswith("第"):   #没有第xx条这种也过滤
            weird += 1
            continue
        rows.append({
            "id": a["id"],        #具体条数id
            "law_id": law_id,     #法律id
            "law_title": j["title"],    #法律名
            "display_no": disp,         #第xx条
            "chapter": " / ".join(a.get("path", [])),   #所属章节   xxx/xxx
            "text": text,
            "status": j.get("status", ""),
            "effective_date": j.get("effectiveDate", ""),
        })

# 写出jsonl  在data/clean/articles.jsonl
(OUT / "articles.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
    encoding="utf-8"
)

print("=" * 45)
print("名单:", len(LABOR_LAWS), "部")
print("实际入库:", len(rows), "条")
print("过滤掉: 短条款", short, "/ 非条文号", weird)
if missing:
    print("\n⚠️ 名单里有", len(missing), "部找不到文件：")
    for m in missing:
        print("   ", m)


# 名单: 27 部
# 实际入库: 1193 条
# 过滤掉: 短条款 25 / 非条文号 0