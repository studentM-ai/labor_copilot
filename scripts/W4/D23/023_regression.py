"""
评测回归

之前的评估（D8 那次的 HitRate@5=0.735）是一次性快照。
后来改了工具、改了 SystemMessage、改了历史截断 —— 但没回头验证过有没有变差。

这个脚本就是干这件事：
  改代码之前先 --save 存一份，改完再 --compare 比一次。

用法：
  python scripts/W4/D23/023_regression.py --save       # 存基线
  python scripts/W4/D23/023_regression.py --compare    # 和基线比
  python scripts/W4/D23/023_regression.py              # 只跑，不存不比
"""
import json
import os
import sys
import time
from pathlib import Path

#到项目根
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
  ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

#⚠️ 必须切到项目根：retriever.py 里 VEC_PATH="data/clean/articles_vec.npy" 是相对路径
os.chdir(ROOT)

from src.contract import load_rows
from src.retriever import VectorRetriever

EVAL = ROOT / "data" / "eval"
GOLDEN = EVAL / "golden_set.jsonl"
BASELINE = EVAL / "regression_baseline.json"

TOP_K = 5


def load_golden():
  """读 40 条测试集"""
  records = []
  with open(GOLDEN, encoding="utf-8") as f:
    for line in f:
      if line.strip():
        records.append(json.loads(line))
  return records


def eval_retrieval(retriever, records, top_k=TOP_K):
  """
  跑40条golden_set 算HitRate@1/3/5+MRR+Recall@5
  要排除掉超范围题（gold 为空的那 6 道），否则会把 HitRate 稀释掉
  """
  per_item = []
  hit1 = hit3 = hit5 = 0
  rr_sum = 0.0
  recall_sum = 0.0
  graded_total = 0    # 有 gold 的题数（分母）
  latencies = []

  for rec in records:
    gold = rec.get("gold_law_ids") or []
    if not gold:      #超范围题跳过
      continue

    graded_total += 1

    t0 = time.perf_counter()
    hits = retriever.retrieve(rec["question"], top_k)
    latencies.append((time.perf_counter() - t0) * 1000)
    top_ids = [h["id"] for h in hits]

    #命中排名 从1开始 没命中为0
    rank = 0
    for i, hid in enumerate(top_ids, 1):
      if hid in gold:
        rank = i
        break

    if rank == 1:
      hit1 += 1
    if rank and rank <= 3:
      hit3 += 1
    if rank:
      hit5 += 1
      rr_sum += 1.0 / rank

    #Recall@5：命中了几个gold
    got = 0
    for g in gold:
      if g in top_ids:
        got += 1
    recall_sum += got / len(gold)

    per_item.append({
      "id": rec["id"],
      "rank": rank,
      "hit5": 1 if rank else 0,
      "mrr": (1.0 / rank) if rank else 0.0,
      "recall": got / len(gold),
    })

  #没有可评的题就直接返回，别让它崩（除零）
  if graded_total == 0:
    return {"total": 0, "hit1": 0.0, "hit3": 0.0, "hit5": 0.0,
            "mrr": 0.0, "recall5": 0.0, "avg_ms": 0.0}, []

  return {
    "total": graded_total,
    "hit1": hit1 / graded_total,
    "hit3": hit3 / graded_total,
    "hit5": hit5 / graded_total,
    "mrr": rr_sum / graded_total,
    "recall5": recall_sum / graded_total,
    "avg_ms": sum(latencies) / len(latencies),
  }, per_item


def check_determinism(retriever, records, n=10):
  """
  同一批问题跑两遍，看检索结果是否一致
  先确认它是确定的，后面的严格对比才有意义
  """
  same = 0
  for rec in records[:n]:
    a = [h["id"] for h in retriever.retrieve(rec["question"], 5)]
    b = [h["id"] for h in retriever.retrieve(rec["question"], 5)]
    if a == b:
      same += 1
  return same, min(n, len(records))


def save_baseline(payload):
  """把这次结果存成基线"""
  EVAL.mkdir(parents=True, exist_ok=True)
  BASELINE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compare(baseline, current):
  """打印差异表 + 翻车的题"""
  print(f"  {'指标':<10}{'基线':>9}{'现在':>9}{'变化':>9}   判定")
  print("  " + "-" * 50)

  for k in ["hit1", "hit3", "hit5", "mrr", "recall5"]:
    b = baseline["retrieval"][k]
    c = current["retrieval"][k]
    flag = "√" if c >= b - 1e-9 else " 退化"
    print(f"  {k:<10}{b:>9.3f}{c:>9.3f}{c - b:>+9.3f}   {flag}")

  #逐题对比：哪些翻了
  b_map = {x["id"]: x for x in baseline["per_item"]}
  c_map = {x["id"]: x for x in current["per_item"]}

  lost = []
  gained = []
  for cid in b_map:
    b_hit = b_map[cid]["hit5"]
    c_hit = c_map[cid]["hit5"]
    if b_hit == 1 and c_hit == 0:
      lost.append(cid)
    elif b_hit == 0 and c_hit == 1:
      gained.append(cid)

  print()
  if lost:
    print(f"新丢失（原来命中现在没命中）：{lost}")
  if gained:
    print(f"新命中：{gained}")
  if not lost and not gained:
    print("逐题结果完全一致")


def main():
  #Windows 控制台不是 UTF-8，不打这行中文可能报错
  try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  except Exception:
    pass

  do_save = "--save" in sys.argv
  do_compare = "--compare" in sys.argv

  print("评测回归")

  records = load_golden()
  rows = load_rows()
  retriever = VectorRetriever(rows)

  graded = sum(1 for r in records if r.get("gold_law_ids"))
  print(f"\n测试集 {len(records)} 条，其中有标准答案 {graded} 条")

  #确定性自检
  same, n = check_determinism(retriever, records)
  print(f"\n确定性自检：{same} / {n} 一致", end="")
  if same == n:
    print("√")
  else:
    print(" 检索不稳定，对比结果要打折看")

  #检索评测
  print("\n检索评测：")
  metrics, per_item = eval_retrieval(retriever, records)
  print(f"  total    = {metrics['total']}")
  print(f"  hit@1    = {metrics['hit1']:.3f}")
  print(f"  hit@3    = {metrics['hit3']:.3f}")
  print(f"  hit@5    = {metrics['hit5']:.3f}")
  print(f"  mrr@5    = {metrics['mrr']:.3f}")
  print(f"  recall@5 = {metrics['recall5']:.3f}")
  print(f"  avg_ms   = {metrics['avg_ms']:.2f}   （含 bge-m3 编码）")

  current = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "retrieval": metrics,
    "per_item": per_item,
  }

  if do_save:
    save_baseline(current)
    print(f"基线已保存：{BASELINE.relative_to(ROOT)}")

  if do_compare:
    if not BASELINE.exists():
      print(f"\n没有基线，先跑一次 --save")
      return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"\n和基线对比（{baseline.get('ts', '?')}）：")
    compare(baseline, current)

  if not do_save and not do_compare:
    print("\n（只跑了评测，没存也没比）")
    print("  --save     存为基线")
    print("  --compare  和基线对比")

  return 0


if __name__ == "__main__":
  sys.exit(main())
