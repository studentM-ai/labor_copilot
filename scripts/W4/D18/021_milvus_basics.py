"""
Milvus基础操作
建库 灌数据 建立索引 load 查询 过滤 删除
旨在了解概念,产出报告

search的结果大致是[[{id,distance,entity...},{}]]
"""
import sys,time
from pathlib import Path
import numpy as np
from pymilvus import DataType,MilvusClient

ROOT=Path(__file__).resolve().parent
while not (ROOT/"src").exists():
  ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.contract import load_rows
from src.retriever import VEC_PATH,VectorRetriever

COLL="law_articles"
DIM=1024
MILVUS_URL="http://localhost:19530"
BATCH=500
EVAL=ROOT/"data"/"eval"
REPORT=EVAL/"milvus_basics.md"

def build_schema(client):
  """
  定义schema 就是定义这个collection有哪些列
  law_id单开一列是用来过滤的，想只查某一部法的时候用这个属性
  """
  schema=client.create_schema(auto_id=False,enable_dynamic_field=True)
  #主键:直接用语料里的id，不自己生
  schema.add_field("id",DataType.VARCHAR,is_primary=True,max_length=64)
  #标量字段：过滤和展示用
  schema.add_field("law_id",DataType.VARCHAR,max_length=64)
  schema.add_field("law_title",DataType.VARCHAR,max_length=128)
  schema.add_field("display_no",DataType.VARCHAR,max_length=64)
  #法条原文
  schema.add_field("text",DataType.VARCHAR,max_length=8192) #要给够
  #向量字段
  schema.add_field("vector",DataType.FLOAT_VECTOR,dim=DIM)
  return schema

def build_index(client):
  """
  建立HNSW索引
  M：每个节点所连的边，默认16，调大召回高但是内存涨
  efConstruction:建索引时的候选队列长度 200常用
  由于归一化过，所以使用COSINE
  """
  params=client.prepare_index_params()
  params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M":16,"efConstruction":200}
  )
  return params

def create(client):
  """
  创建collection
  """
  #先删后建
  if client.has_collection(COLL):
    print("已存在，先删再重建")
    client.drop_collection(COLL)

  #create_collection 里传了 index_params，Milvus 会建索引 + 自动 load 否则要自己手动load
  client.create_collection(
    collection_name=COLL,
    schema=build_schema(client),
    index_params=build_index(client)
  )

def load_data():
  """
  读语料+读向量，按顺序配对
  向量的顺序必须和 rows 的顺序严格一致
  """
  rows=load_rows()
  vecs=np.load(ROOT/VEC_PATH)
  print("法条%d条，向量%s"%(len(rows),vecs.shape))
  if len(rows)!=len(vecs):
    raise RuntimeError("条数与向量行数不一致，检查数据")
  return rows,vecs

def insert(client,rows,vecs,batch=500):
  """
  分批插入，分批是因为一次性传的数据太多，避免请求失败
  返回插入后的数量
  """
  total=0
  for start in range(0,len(rows),batch):
    end=min(start+batch,len(rows))

    data=[]
    for i in range(start,end):
      r=rows[i]
      data.append({
        "id":str(r["id"]),
        "law_id":str(r.get("law_id")or ""),
        "law_title":str(r.get("law_title") or ""),
        "display_no":str(r.get("display_no")or ""),
        "text":str(r.get("text")or ""),
        "vector":vecs[i].tolist()
      })

    res=client.insert(collection_name=COLL,data=data)
    total+=res["insert_count"]
    print(f"已插入{total}/{len(rows)}")
  return total

def flush_and_check(client,expected):
  """
  落盘并看真实条数
  """
  print("flush 前 stats：", client.get_collection_stats(COLL))
  ts=time.perf_counter()
  client.flush(COLL)
  print("flush 耗时：%.2f 秒" % (time.perf_counter() - ts))

  time.sleep(1)
  rc=client.get_collection_stats(COLL).get("row_count")
  print(f"期望{expected}条，stats报{rc}条")
  return expected



def encode_query(retriever,text):
  """
  把中文问题编成1024维向量
  """
  vec=retriever.model.encode([text],normalize_embeddings=True)[0]
  return vec.tolist()

def search(client,query_vec,k=5,filter_expr=""):
  """
  向量检索
  output_fields:要Milvus把哪些标量字段一起返回，如果不加只返回id和距离，拿不到原文
  search_params：ef是查询时参数
  """
  res=client.search(
    collection_name=COLL,
    data=[query_vec],
    limit=k,
    output_fields=["law_title","display_no","text","law_id"],
    search_params={"metric_type":"COSINE","params":{"ef":64}},
    filter=filter_expr  #空字符串就是不过滤
  )
  return res[0]

#普通查询+标量过滤
def verify_plain(client,q_vec,out):
  """
  看看能不能查出来 返回查询结果（普通查询，不带过滤）
  """
  print("\n普通查询（取前5）")
  hits=search(client,q_vec)
  rows=[]
  for h in hits:
    line = "%.4f %s %s" % (
      h["distance"], h["entity"]["law_title"], h["entity"]["display_no"]
    ) #相似度  h["entity"]是标量数据
    print(line)
    rows.append(line.strip())

  out.append("### 普通查询（top5）")
  out.append("")
  out.append("```")
  for r in rows:
    out.append(r)
  out.append("```")
  out.append("")
  return hits

def verify_filter(client,q_vec, plain_hits,out):
  """
  标量过滤
  只在某一步法律里检索。比如问：劳动合同法怎么规定试用期  可以先圈定法律范围
  """
  print("\n带标量过滤（只在《劳动合同法》里搜）")
  hits_f=search(client,q_vec,filter_expr='law_id=="labor-contract-law"')  #过滤条件是只要劳动合同法

  print(f"不带过滤{len(plain_hits)}条，带过滤{(len(hits_f))}条")
  rows=[]
  for h in hits_f:
    line=" %.4f %s %s" % (
      h["distance"], h["entity"]["law_title"], h["entity"]["display_no"]
    )
    print(line)
    rows.append(line.strip())
  out.append("### 标量过滤（只查劳动合同法）")
  out.append("")
  out.append(f"不带过滤{len(plain_hits)}条，带过滤{(len(hits_f))}条")
  out.append("")
  out.append("```")
  for r in rows:
    out.append(r)
  out.append("```")
  out.append("")

#ef对比
def verify_ef(client,q_vec,out):
  """
  对比ef（查询时的，控制召回率与检索速度）  
  ef=512几乎等于暴力搜，所以它的top5可当标准答案
  对不同ef值跑 看top5和基准重合几条，同时对每个ef跑50次，取平均耗时，降低单词噪声
  召回重合率 能说明 ef 小到什么程度才开始丢结果。
  
  """
  print("\nef 参数对比")
  #基准
  def top5(ef): 
    res=client.search(
      collection_name=COLL,
      data=[q_vec],
      limit=5,
      output_fields=["id"],
      search_params={"metric_type":"COSINE","params":{"ef":ef}},
    )[0]
    return [h["id"]for h in res]  #返回id
  
  base=top5(512)  #基准的id

  lines=[]
  lines.append("| ef | 平均耗时 | 与 ef=512 的 top-5 重合 |")
  lines.append("|---|---|---|")

  for ef in [8,16,64,256]:
    #耗时，跑50次取平均
    ts=time.perf_counter()
    for _ in range(50):
      client.search(
        collection_name=COLL,
        data=[q_vec],
        limit=5,
        output_fields=["law_title"],
        search_params={"metric_type":"COSINE","params":{"ef":ef}}
      )
    avg_ms=(time.perf_counter()-ts)/50*1000

    got=top5(ef)
    same=0  #重合数
    for x in got:
      if x in base:
        same+=1
    line=f"| {ef} | {avg_ms:.2f}ms | {same}/5 |"
    print(" "+line)
    lines.append(line)

  out.append("### ef 参数对比")
  out.append("")
  out.extend(lines)
  out.append("")



def verify_delete(client,retriever,out):
  """
  删除测试
  删除最后一条，查询确认没了，再插回去保持库数据完整

  Milvus 的 delete 是 逻辑删除
  这是和 MySQL 那种物理删除不一样的地方
  """

  print("删除测试")
  last=retriever.rows[-1] #拿最后一条
  target_id=str(last["id"])

  client.delete(collection_name=COLL,filter=f"id=='{target_id}'") #删除最后一个
  time.sleep(2)

  #查询还有没有这个数据
  left=client.query(  
    collection_name=COLL,
    filter=f"id=='{target_id}'",
    output_fields=["id"],
    limit=5,
  )
  rc_del=client.get_collection_stats(COLL).get("row_count")
  print(f" 删除 {target_id} , 按 id 查 {len(left)} 条，row_count 仍为 {rc_del}")

  #插回去
  client.insert(collection_name=COLL,data=[{
    "id": target_id,
    "law_id": str(last.get("law_id") or ""),
    "law_title": str(last.get("law_title") or ""),
    "display_no": str(last.get("display_no") or ""),
    "text": str(last.get("text") or ""),
    "vector": retriever.vecs[-1].tolist(),
  }])
  client.flush(COLL)

  out.append("### 删除")
  out.append("")
  out.append(f"- 删除 `{target_id}` 后立即查询：{len(left)} 条（已删掉）" )
  out.append("")
  out.append(f"> 另外删除后 `row_count` 是 {rc_del}：它是物理行数，")
  return 

#写参数表
def write_report(client,row_count,out):
  """
  把此次跑出的数据写出md
  """
  EVAL.mkdir(parents=True,exist_ok=True)

  lines=[]
  lines.append("# Milvus 基础操作测试 D18 021")
  lines.append("")
  lines.append(f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append(f"- 服务端版本：{client.get_server_version()}")
  lines.append(f"- 维度：{ DIM}（bge-m3）")
  lines.append(f"- 度量：COSINE")
  lines.append(f"- 索引：HNSW，M=16，efConstruction=200")
  lines.append(f"- 数据量：{row_count}条")
  lines.append("")
  lines.extend(out)
  lines.append("---")
  lines.append("注意")
  lines.append("")
  lines.append("1.insert后row_count是0，milvus是先写内存后落盘的")
  lines.append("2.numpy的float32必须.tolist()")
  lines.append("")

  REPORT.write_text("\n".join(lines),encoding="utf-8")
  print(f"\n 测试数据已写入：{REPORT.relative_to(ROOT)}")


def main():
  try:
    # Windows 控制台默认可能是 GBK，输出中文会炸，先切 UTF-8
    sys.stdout.reconfigure(encoding="utf-8",errors="replace")
  except Exception:
    pass
  print("D18 Milvus基础操作")

  #连服务
  print("\n连接Milvus")
  client=MilvusClient(uri=MILVUS_URL)
  print(f"服务器版本：{client.get_server_version()}")

  #建库
  print("\n 建collection")
  create(client)  

  #加载模型与向量
  print("\n加载模型与向量")
  ts=time.perf_counter()
  rows,vecs=load_data() #拿到语料与向量
  retriever=VectorRetriever(rows)
  print(f"加载耗时{time.perf_counter()-ts}")

  #灌数据
  print("灌入数据")
  data=insert(client,rows,vecs)
  row_count=flush_and_check(client,data)

  #查询向量
  print("\n编码查询")
  q_vec=encode_query(retriever,"试用期最长能有多久")

  out=[]

  #拿结果
  plain=verify_plain(client,q_vec,out)
  verify_filter(client,q_vec,plain,out)
  verify_ef(client,q_vec,out)
  verify_delete(client,retriever,out)

  #写数据表
  print("\n写数据表")
  write_report(client,row_count,out)

  print("D18完成")

if __name__=="__main__":
  main()