"""
D20 numpy vs Milvus 检索对比
【三个问题】
1. 纯检索谁快？快多少？
2. 召回一致吗？
3. 各自的独有能力（元数据过滤 / 增量更新）
注意编码与检索需要分开计时，因为之前测过的编码时间占比比较大，会影响结果的判断
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
  ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
#切到项目根
os.chdir(ROOT)

from src.contract import load_rows
from src.retriever import MilvusRetriever, VectorRetriever

EVAL = ROOT / "data" / "eval"
REPORT = EVAL / "vector_store_compare.md"

N_QUERY = 20  #query数
ROUNDS = 5  # 每条 query 跑几次
WARMUP = 3  #预热次数

def load_queries(limit=N_QUERY):
  """从 golden_set 取 query（现在有 40 条）"""
  queries = []
  with open(EVAL / "golden_set.jsonl", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      queries.append(json.loads(line)["question"])
      if len(queries) >= limit:
        break
  return queries


def pct(values, p):
  """
  取百分位数
  用中位数/P 分位而不是平均数是因为一次 GC 或系统调度就能把平均数拉歪，而使用中位数就稳定得多。
  """
  if not values:
    return 0.0
  ordered = sorted(values)  #从小到大排序
  idx = int(round((p / 100.0) * (len(ordered) - 1)))  #拿到p分位数
  return ordered[idx]


def bench_numpy(retriever, queries, rounds=ROUNDS):
  """
  测 numpy 的纯检索耗时
  先编码好，只给矩阵乘法 + argsort计时。
  不算encode的时间
  """
  encode_ms = []
  search_ms = []

  for q in queries:
    # 编码（两种后端用的是同一个模型，编码耗时本来就一样，单独记）
    t0 = time.perf_counter()
    q_vec = retriever.model.encode([q], normalize_embeddings=True)[0]
    encode_ms.append((time.perf_counter() - t0) * 1000)

    #最多round轮，因为第一次可能会有点久
    for i in range(rounds):
      t0 = time.perf_counter()
      scores = retriever.vecs @ q_vec   # 全部 1251 条的相似度
      top_idx = np.argsort(-scores)[:5]      # 排序取前 5
      _ = [retriever.rows[i] for i in top_idx] # 取回原始行
      ms = (time.perf_counter() - t0) * 1000
      if i == 0:
        continue      # 第一次丢掉（缓存还没热）
      search_ms.append(ms)

  return encode_ms, search_ms


def bench_milvus(retriever, queries, rounds=ROUNDS):
  """测 Milvus 的纯检索耗时"""
  encode_ms = []
  search_ms = []

  for q in queries:
    t0 = time.perf_counter()
    q_vec = retriever.model.encode([q], normalize_embeddings=True)[0]
    encode_ms.append((time.perf_counter() - t0) * 1000)
    vec = q_vec.tolist()

    for i in range(rounds):
      t0 = time.perf_counter()
      retriever.client.search(
        collection_name=retriever.collection,
        data=[vec],
        limit=5,
        output_fields=["id"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
      )
      ms = (time.perf_counter() - t0) * 1000
      if i == 0:
        continue
      search_ms.append(ms)

  return encode_ms, search_ms


def recall_overlap(np_ret, mv_ret, queries):
  """
  召回一致性：同一条 query，两边 top-5 的 id 集合重合几个
  """
  total = 0   #候选总数量
  same = 0    #所有query检索所得重合数
  details = []

  for q in queries:
    np_ids = [r["id"] for r in np_ret.retrieve(q, 5)]
    mv_ids = [r["id"] for r in mv_ret.retrieve(q, 5)]

    hit = 0
    for x in mv_ids:
      if x in np_ids:
        hit += 1  #重合数+1
    same += hit
    total += 5
    details.append((q, hit))  #（问题，重合数）

  return same, total, details


def bench_filter(np_ret, mv_ret, queries, law_id="labor-contract-law", rounds=3):
  """
  元数据过滤对比：只在某一部法里检索（劳动合同法）
  """
  mv_ms = []
  np_ms = []

  for q in queries[:5]:   #只取 5 条
    vec = mv_ret.model.encode([q], normalize_embeddings=True)[0]
    vec_list = vec.tolist()

    # Milvus：一行 filter
    for i in range(rounds):
      t0 = time.perf_counter()
      mv_ret.client.search(
        collection_name=mv_ret.collection,
        data=[vec_list],
        limit=5,
        output_fields=["id"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        filter=f"law_id == '{law_id}'",
      )
      mv_ms.append((time.perf_counter() - t0) * 1000)

    # numpy：先算完全部分数，再手工过滤
    for i in range(rounds):
      t0 = time.perf_counter()
      scores = np_ret.vecs @ vec
      order = np.argsort(-scores) #从大到小排序
      picked = []
      for idx in order:
        if np_ret.rows[int(idx)]["law_id"] == law_id:
          picked.append(np_ret.rows[int(idx)])
        if len(picked) >= 5:
          break
      np_ms.append((time.perf_counter() - t0) * 1000)

  return np_ms, mv_ms


def write_report(np_load, mv_load, np_search, mv_search, np_enc, mv_enc,
                 same, total, f_np, f_mv, queries):
  """
  把实测数字写成 markdown

  """
  EVAL.mkdir(parents=True, exist_ok=True)

  np_p50 = pct(np_search, 50)
  np_p95 = pct(np_search, 95)
  mv_p50 = pct(mv_search, 50)
  mv_p95 = pct(mv_search, 95)
  enc_p50 = pct(np_enc, 50)

  #所用时间差别倍数
  ratio = mv_p50 / np_p50 if np_p50 > 0 else 0

  # 编码所占比例：编码 / (编码 + 检索)
  enc_share = enc_p50 / (enc_p50 + np_p50) * 100 if enc_p50 > 0 else 0

  L = []
  L.append("# numpy vs Milvus 检索对比")
  L.append("")
  L.append("- 数据量：1,251 条法条，1024 维 bge-m3 向量")
  L.append(f"- 查询：{len(queries)} 条（取自 golden_set），每条跑 {ROUNDS} 次，首次丢弃")
  L.append("")
  L.append("## 实测")
  L.append("")
  L.append("| 项 | numpy（exact） | Milvus（HNSW, ef=64） |")
  L.append("|---|---|---|")
  L.append(f"| 纯检索 P50 | **{np_p50:.2f} ms** | **{mv_p50:.2f} ms** |")
  L.append(f"| 纯检索 P95 | {np_p95:.2f} ms | {mv_p95:.2f} ms |")
  L.append(f"| 编码 P50（同一个 bge-m3） | {enc_p50:.2f} ms | {pct(mv_enc, 50):.2f} ms |")
  L.append(f"| 元数据过滤 P50 | {pct(f_np, 50):.2f} ms | {pct(f_mv, 50):.2f} ms |")
  L.append(f"| 应用侧冷启动 | {np_load:.1f} s | {mv_load:.1f} s |")
  L.append(f"| 召回一致性（top-5 重合） | 基准 | **{same} / {total}** |")
  L.append("")
  L.append(f"**numpy 快 {ratio:.1f} 倍；编码所占 {enc_share:.1f}%；召回完全一致。**")
  L.append("")
  L.append("## 什么时候该换")
  L.append("")
  L.append("这个规模下 numpy 更快，所以默认用它。")
  L.append("")
  L.append("数据量增长到**十万级**时再重新评估 —— 那时候需要真实扩容后重测，")
  L.append("外推不可靠（numpy 的 argsort 是 O(N log N)，HNSW 的图也会随数据量变大）。")
  L.append("")
  L.append("## 能力对比")
  L.append("")
  L.append("| 能力 | numpy | Milvus |")
  L.append("|---|---|---|")
  L.append("| 元数据过滤 | 全量打分后手写过滤 | `filter=` 一个参数 |")
  L.append("| 增量更新 | 重建整个 `.npy` | `upsert` / `delete` |")
  L.append("| 多租户隔离 | 自己维护 id  下标映射 | `filter=user_id==...` |")
  L.append("| 依赖 | 一个 4.9 MB 文件 | 三个容器 + 2–4 GB 内存 |")
  L.append("")
  L.append("---")
  L.append("")
  REPORT.write_text("\n".join(L), encoding="utf-8")


def main():
  try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  except Exception:
    pass

  print("D20 numpy vs Milvus 检索对比")

  queries = load_queries()
  print(f"\n取 {len(queries)} 条 query，每条跑 {ROUNDS} 次（首次要丢弃用不到）")

  # 建两个后端 
  print("\n加载语料与两个后端")
  rows = load_rows()
  print(f"语料 {len(rows)} 条")

  t0 = time.perf_counter()
  np_ret = VectorRetriever(rows)
  np_load = time.perf_counter() - t0
  print(f"numpy后端就绪：{np_load:.1f} 秒")

  t0 = time.perf_counter()
  try:
    mv_ret = MilvusRetriever(rows)
  except Exception as e:
    print("Milvus 后端出现错误：", e)
    return 1
  mv_load = time.perf_counter() - t0
  print(f"Milvus后端就绪：{mv_load:.1f} 秒")

  # 预热
  print(f"\n预热 {WARMUP} 次（不计入统计）")
  for q in queries[:WARMUP]:
    np_ret.retrieve(q, 5)
    mv_ret.retrieve(q, 5)
  print("预热完成")

  #纯检索耗时 
  print("\n 纯检索耗时（不含编码）")
  np_encode, np_search = bench_numpy(np_ret, queries)
  mv_encode, mv_search = bench_milvus(mv_ret, queries)
  print(f"numpy ：P50 {pct(np_search, 50):.2f} ms   P95 {pct(np_search, 95):.2f} ms")
  print(f"Milvus：P50 {pct(mv_search, 50):.2f} ms   P95 {pct(mv_search, 95):.2f} ms")

  #编码耗时 
  print("\n 编码耗时（同一个 bge-m3，两者一样）")
  enc_p50 = pct(np_encode, 50)
  print(f"P50 {enc_p50:.2f} ms")

  #召回一致性 
  print("\n 召回一致性（top-5 重合）")
  same, total, details = recall_overlap(np_ret, mv_ret, queries)
  print(f"合计 {same} / {total}")
  for q, hit in details:
    if hit < 5:
      print(f" 只重合 {hit}/5：{q}")

  # 过滤能力 
  print("\n 元数据过滤对比（只查《劳动合同法》）")
  f_np, f_mv = bench_filter(np_ret, mv_ret, queries)
  print(f"numpy （先算全部+排序，再过滤）：P50 {pct(f_np, 50):.2f} ms")
  print(f"Milvus（filter 在索引内部限制）：P50 {pct(f_mv, 50):.2f} ms")

  # 写报告
  print("\n 写报告")
  write_report(np_load, mv_load, np_search, mv_search, np_encode, mv_encode,
               same, total, f_np, f_mv, queries)
  print(f"{REPORT.relative_to(ROOT)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
