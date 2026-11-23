"""
长期记忆（跨会话）
按 user_id 存，跨会话有效
抽取  每轮回答结束后，让模型从这轮对话里抽 0~3 条「关于用户的事实」
存储  编码成向量，upsert 进 Milvus（主键 = hash(user_id + kind)）
召回  新会话第一句时，拿问题去检索这个用户的记忆，取 top-3 拼进 SystemMessage

用 Milvus 而不是 SQLite
记忆条数很少（几十条），SQLite 其实更快。
用 Milvus 是因为它需要「按 user_id 过滤」和「upsert 覆盖」这两个能力：
filter='user_id == "..."'  保证不串号
upsert + 主键设计            用户改口时覆盖而不是累加

"""
import asyncio
import hashlib
import json
import threading

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import DataType, MilvusClient

from src.prompts import MEMORY_EXTRACT_PROMPT

MEMORY_COLL = "user_memory"
MEMORY_DIM = 1024          # 必须和法条库一致（同一个 bge-m3 模型）
RECALL_TOP_K = 3           # 每次最多召回 3 条
MAX_FACTS_PER_TURN = 3     # 一轮最多存 3 条，防止模型抽太多

# kind 的合法值（必须和 src/prompts.py 里的枚举表一致）
VALID_KINDS = {
  "work_years", "salary", "city", "end_reason",
  "contract", "social_insurance", "dispute_type", "other",
}

# 兜底别名表：万一模型没听指令、输出了中文 kind，这里把它映射回枚举值
KIND_ALIASES = {
  "工作年限": "work_years", "工龄": "work_years", "工作年份": "work_years",
  "月薪": "salary", "工资": "salary", "收入": "salary", "薪资": "salary",
  "城市": "city", "所在城市": "city", "工作城市": "city",
  "离职原因": "end_reason", "被辞退": "end_reason", "辞退原因": "end_reason",
  "合同": "contract", "合同情况": "contract",
  "社保": "social_insurance", "社保情况": "social_insurance", "公积金": "social_insurance",
  "争议诉求": "dispute_type", "诉求": "dispute_type", "赔偿诉求": "dispute_type",
  "其他": "other",
}

# 三个反引号（markdown 代码块围栏），用 chr 拼出来避免源码里出现反引号
FENCE = chr(96) * 3


def normalize_kind(raw):
  """
  把模型给出的 kind 规范化

  顺序：去空格小写  查枚举表  查别名表  都不中返回 None（丢弃）

  other 只有一个主键，不同的未知类别会互相覆盖，反而更糟。
  丢弃并打日志，能让你发现 prompt 需要调整，而不是静默污染数据。
  """
  k = (raw or "").strip().lower()
  if not k:
    return None
  if k in VALID_KINDS:
    return k
  if k in KIND_ALIASES:
    return KIND_ALIASES[k]
  print("[memory] 未知 kind，已丢弃：", raw, flush=True)
  return None


#Milvus 客户端（单例）
_client = None
_lock = threading.Lock()

# 后台任务集合：持有引用，防止任务被垃圾回收
_TASKS = set()


def spawn(coro):
  """
  起一个后台任务，并持有引用防止被 GC

  不用 await：记忆抽取要调一次模型（1~2 秒），如果 await 它，用户会在看到答案之后再多等 1~2 秒。
  """
  task = asyncio.create_task(coro)  #放到事件循环里执行
  _TASKS.add(task)  #只要task在集合里就不会被gc清除
  task.add_done_callback(_TASKS.discard)  #只要结束就触发回调（discard删除掉）
  return task


def _build_schema(client):
  """记忆库的 schema"""
  schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
  schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
  schema.add_field("user_id", DataType.VARCHAR, max_length=64)
  schema.add_field("kind", DataType.VARCHAR, max_length=32)
  schema.add_field("text", DataType.VARCHAR, max_length=512)
  schema.add_field("vector", DataType.FLOAT_VECTOR, dim=MEMORY_DIM)
  return schema


def _build_index(client):
  """HNSW 索引，参数和法条库保持一致，方便对比"""
  params = client.prepare_index_params()
  params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200},
  )
  return params


def get_client(uri):
  """
  拿 Milvus 客户端（全局单例）
  """
  global _client
  with _lock:
    if _client is None:
      client = MilvusClient(uri=uri)
      if not client.has_collection(MEMORY_COLL):
        client.create_collection(
          collection_name=MEMORY_COLL,
          schema=_build_schema(client),
          index_params=_build_index(client),
        )
      _client = client
    return _client


def memory_id(user_id, kind):
  """
  主键 = hash(user_id + kind)

  为什么不用随机 id
    如果用随机 id，用户说三次「月薪」就会存三条，检索时互相挤占 top-3。
    用 (用户, 类别) 做哈希，同一个人的同一类事实永远只有一条，
    upsert 时自然覆盖

  kind需要枚举是因为
    kind 一旦漂移（「月薪」vs「工资」），哈希就不同，覆盖失效。
    所以 normalize_kind 会先把它归一化成固定枚举值。
  """
  raw = f"{user_id}|{kind}" 
  return hashlib.md5(raw.encode("utf-8")).hexdigest() #十六进制


# 抽取 
_llm_cache = {}

def _get_llm(rt):
  """抽取用的小模型客户端（缓存，避免每次新建）"""
  key = (rt.api_key, rt.model)
  if key not in _llm_cache:
    _llm_cache[key] = ChatOpenAI(
      model_name=rt.model,
      api_key=rt.api_key,
      base_url="https://api.deepseek.com",
      temperature=0.0,      #不要创造力
      max_tokens=300,       #输出就是个小 JSON 数组，不用给太多
    )
  return _llm_cache[key]


def parse_json_array(text):
  """
  从模型输出里抠出 JSON 数组

  处理顺序：
  1. 去掉 markdown 代码块围栏
  2. 截取第一个 [ 到最后一个 ]
  3. 解析，失败就返回空列表
  4. 用 normalize_kind 归一化 kind，字段不合法就跳过
  """
  if not text:
    return []

  s = text.strip()

  # 去掉代码块围栏
  if s.startswith(FENCE): #```
    parts = s.split("\n")
    if len(parts) > 1:
      s = "\n".join(parts[1:])  #去掉第一行，拼回后面的
    s = s.rstrip()
    if s.endswith(FENCE):
      s = s[:-3]  #取出后面的```

  # 截取 [ ... ]
  i = s.find("[")
  j = s.rfind("]")
  if i < 0 or j < i:
    return []

  try:
    data = json.loads(s[i:j + 1])
  except Exception:
    return []

  if not isinstance(data, list):
    return []

  out = []
  for item in data:
    if not isinstance(item, dict):
      continue
    kind = normalize_kind(item.get("kind"))
    txt = str(item.get("text") or "").strip()
    if not kind or not txt:
      continue
    out.append({"kind": kind, "text": txt[:200]})
    if len(out) >= MAX_FACTS_PER_TURN:
      break
  return out


async def extract_facts(rt, question, answer, known_text=""):
  """
  从这一轮对话里抽 0~3 条用户事实

  known_text:这个用户已有的记忆摘要，喂给模型让它别重复抽。
              传空字符串也行，只是可能重复抽到同一件事（无害，会被 upsert 覆盖）。

  """
  llm = _get_llm(rt)

  prompt = (MEMORY_EXTRACT_PROMPT
              .replace("{known}", known_text or "（暂无）")
              .replace("{question}", question[:800])
              .replace("{answer}", answer[:1200]))

  try:
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    return parse_json_array(resp.content or "")
  except Exception as e:
    print("[memory] 抽取失败（不影响主流程）：", e, flush=True)
    return []


# 存储 
async def save_memories(rt, user_id, facts):
  """
  把抽取事实编码后 upsert 进 Milvus

  注意：
  1.encode 是同步阻塞的（bge-m3），必须丢到线程里，否则阻塞事件循环
  2.用 upsert 不是 insert —— 同主键覆盖，避免同类事实累加
  """
  if not facts:
    return 0

  model = rt.retriever.model   #复用已经加载好的 bge-m3

  rows = []
  for f in facts:
    vec = await asyncio.to_thread(
      model.encode, [f["text"]], normalize_embeddings=True
    )   #转线程
    rows.append({
      "id": memory_id(user_id, f["kind"]),
      "user_id": user_id,
      "kind": f["kind"],
      "text": f["text"],
      "vector": vec[0].tolist(),
    })

  client = get_client(rt.milvus_uri)
  await asyncio.to_thread(client.upsert, collection_name=MEMORY_COLL, data=rows)
  return len(rows)


#召回
async def recall_memories(rt, user_id, question, top_k=RECALL_TOP_K):
  """
  召回这个用户的相关记忆
  """
  model = rt.retriever.model
  vec = await asyncio.to_thread(
    model.encode, [question], normalize_embeddings=True
  ) #问题向量化

  client = get_client(rt.milvus_uri)
  res = await asyncio.to_thread(
    client.search,
    collection_name=MEMORY_COLL,
    data=[vec[0].tolist()],
    limit=top_k,
    output_fields=["kind", "text"],
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    filter='user_id == "%s"' % user_id,
  )
  return res[0] if res else []


def format_memory(hits):
  """
  把召回结果拼成一段文字，塞进 SystemMessage

  输出例：
    月薪 12000 元；工作 3 年半；在杭州
  """
  if not hits:
    return ""
  parts = []
  for h in hits:
    txt = (h.get("entity") or {}).get("text") or ""
    if txt:
      parts.append(txt)
  return "；".join(parts)


async def memory_context(rt, user_id, question):
  """
  召回 + 格式化，供 build_input_async 调用

  整个过程失败都返回空字符串 记忆没了可以，答案必须给。
  """
  try:
    hits = await recall_memories(rt, user_id, question)
    return format_memory(hits)
  except Exception as e:
    print("[memory] 召回失败（不影响回答）：", e, flush=True)
    return ""



#一轮结束后的入口 
async def remember_turn(rt, user_id, question, answer):
  """
  一轮对话结束后：抽取 + 存储

  由 spawn() 丢到后台跑，所以这里必须自己兜住所有异常
  （后台任务抛异常没人接，只会打一堆 traceback）。
  """
  if not user_id or user_id == "anonymous":
    # 匿名用户没有稳定身份，存了也没法跨会话召回
    return
  try:
    #先把这个用户已有的记忆取出来，喂给抽取器，避免每轮重复抽同一条
    known_text = ""
    try:
      hits = await recall_memories(rt, user_id, question)
      known_text = format_memory(hits)
    except Exception:
      known_text = ""

    #抽取
    facts = await extract_facts(rt, question, answer, known_text)
    if not facts:
      return
    n = await save_memories(rt, user_id, facts)  #存入
    print("[memory] 记住 %d 条：%s" % (n, [f["kind"] for f in facts]), flush=True)
  except Exception as e:
    print("[memory] 保存失败（不影响主流程）：", e, flush=True)


async def clear_user_memory(rt, user_id):
  """清空某个用户的记忆（测试和【忘记我】功能用）"""
  try:
    client = get_client(rt.milvus_uri)
    await asyncio.to_thread(
      client.delete,
      collection_name=MEMORY_COLL,
      filter='user_id == "%s"' % user_id,
    )
    return True
  except Exception as e:
    print("[memory] 清空失败：", e, flush=True)
    return False

