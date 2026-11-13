"""
合同体检的核心逻辑

【为什么单独抽出来】
  命令行版（017_contract_check.py）和 Agent 版（019_contract_agent.py）都要
  「切分 → 逐条检索 → 判断」这一套。如果两边各写一份，改 prompt 或改切分规则时
  必然有一边忘改，导致 Agent 的实际表现和评估报告对不上。

  所以这里放唯一实现，两边都 import。报告格式不在这里——那属于各自的输出，
  命令行版输出带表格的 md，Agent 版输出摘要。

【注意路径】
  ROOT 用 __file__ 往上两层（src/ 的父目录就是项目根），
  这样不管从哪个目录运行脚本都能找到 data/。
"""
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
CONTRACTS = ROOT / "data" / "contracts"

# ---------- 参数 ----------
MODEL = "deepseek-chat"
TOP_K = 5
MIN_PARA_LEN = 15    # 短于这个字数的段落并到相邻条款

# 行首是编号的样子：第X条 / 1. / 1、 / 一、
NUMBER_START = re.compile(r"^(第[一二三四五六七八九十百零]+条|\d+[.、]|[一二三四五六七八九十]+、)")


SYSTEM_PROMPT = """你是劳动法审查助手。下面会给你一条劳动合同条款，以及检索到的相关法条。

请判断这条条款是否存在法律风险。

严格遵守：
1. 只依据给出的【参考法条】判断，不要使用你自己的法律知识。
2. 如果给出的法条不足以判断，risk 必须填 "unknown"，不要猜。
3. 只输出一个 JSON 对象，不要输出任何其他文字。

输出格式：
{"risk": "high" | "medium" | "none" | "unknown",
 "reason": "一句话说明这条为什么有/没有问题",
 "law_refs": ["《法律名》第N条"]}

risk 的含义：
  high     明显违反法条，或明显对劳动者不利
  medium   可能违法，或表述模糊、有争议
  none     对照法条未发现问题
  unknown  给出的法条不足以判断
"""


def load_rows():
  """
  读法条语料，返回列表
  """
  rows = []
  for line in open(CLEAN / "articles.jsonl", encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows


def list_contracts():
  """
  列出 data/contracts 下所有待体检的合同文件名（按名字排序）
  """
  names = []
  for p in sorted(CONTRACTS.glob("*.txt")):
    names.append(p.name)
  return names


def merge_short(parts):
  """
  把过短的段落并到上一条，去掉空段落
  parts:分割后的字符串列表
  """
  out = []
  for p in parts:
    p = p.strip()
    if p == "":
      continue
    # 太短且已经有上一条，就并进去
    if len(p) < MIN_PARA_LEN and len(out) > 0:
      out[-1] = out[-1] + "\n" + p
    else:
      out.append(p)

  # 首行往往是合同标题（如「劳动合同（固定期限）」），
  # 它在上面循环里没有「上一条」可并，会单独成一段。
  # 这里再检查一次，过短就并到第二条去。
  if len(out) >= 2 and len(out[0]) < MIN_PARA_LEN:
    out[1] = out[0] + "\n" + out[1]
    out.pop(0)

  return out


def split_by_number(block):
  """
  在一个「按空行切出来的块」内部，再按行首编号细分

  只有块里出现【两个以上】编号行首时才切——
  因为条款内部的列举（「甲方义务：1. … 2. …」）通常不是独立条款，
  是同一段话的组成部分。

  参数
      block: 一个由空行切出来的块
  返回
      切好的段落列表；不需要切就原样返回 [block]
  """
  lines = block.split("\n")

  # 先记下所有以编号开头的行号
  starts = []
  for i, line in enumerate(lines):
    if NUMBER_START.match(line.strip()):
      starts.append(i)

  # 编号少于 2 个：这段里最多只有一条条款，不切
  if len(starts) < 2:
    return [block]

  parts = []

  # 第一个编号之前如果还有内容（段首说明文字），单独成一段
  if starts[0] > 0:
    head = "\n".join(lines[:starts[0]]).strip()
    if head:
      parts.append(head)

  # 按编号位置把块切开
  for k, s in enumerate(starts):
    if k + 1 < len(starts):
      end = starts[k + 1]
    else:
      end = len(lines)
    parts.append("\n".join(lines[s:end]))

  return parts


def split_clauses(text):
  """
  分割合同成一条条条款

  主策略是「按空行切」而不是「按编号切」：
    编号形式五花八门（第X条 / 1. / 一、），还可能在同一份合同里混用；
    但合同天然用空行分隔条款。以空行为主，就不受编号形式变化影响。

  三级处理：
    1 先按空行切成块
    2 每块内部如果还有多个编号开头，再按编号细分（处理没留空行的情况）
    3 切出来不到 3 条 → 说明这份合同根本没用空行，改用编号正则切全篇

  参数
      text: 合同全文
  返回
      条款字符串列表
  """
  text = text.replace("\r\n", "\n").strip()

  # ---------- 1 + 2：按空行切块，块内再按编号细分 ----------
  blocks = re.split(r"\n\s*\n", text)

  parts = []
  for b in blocks:
    if b.strip() == "":
      continue
    for s in split_by_number(b):
      parts.append(s)

  # ---------- 3：兜底 ----------
  if len(parts) < 3:
    parts = re.split(
      r"\n(?=第[一二三四五六七八九十百零]+条|\d+[.、]|[一二三四五六七八九十]+、)",
      text,
    )

  return merge_short(parts)


def build_messages(clause, hits, full_text):
  """
  把条款、检索到的法条、整份合同拼成 prompt

  clause:一条合同条款
  hits:检索到的法条列表
  full_text:整份合同全文（作为背景）

  为什么要把整份合同也传进去：
    有些风险只在条款之间才成立。比如「试用期六个月」单独看是合法的
    （劳动法第二十一条：试用期最长不得超过六个月），但配上第一条的
    「合同期限一年」，就违反了劳动合同法第十九条（一年期最多两个月）。
    只把这一条条款交给模型，它没法做这个交叉判断，只能报「法条不足」。
  """
  refs = []
  for i, r in enumerate(hits, 1):
    refs.append(f"[{i}] 《{r['law_title']}》{r['display_no']}：{r['text']}")

  user_content = (
    "【参考法条】\n" + "\n\n".join(refs) +
    "\n\n【整份合同（背景，用于判断条款之间是否冲突）】\n" + full_text +
    "\n\n【需要审查的条款】\n" + clause
  )

  return [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_content},
  ]


def parse_verdict(raw):
  """
  把模型返回的文本解析成 dict

  模型有时候会在 JSON 外面包一层 ```json，所以先剥掉；
  解析失败就返回 unknown，不让整个流程崩掉。

  参数
      raw: 模型的原始输出
  返回
      {"risk": ..., "reason": ..., "law_refs": [...]}
  """
  text = raw.strip()

  # 剥掉 ```json ... ``` 包裹
  if text.startswith("```"):
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    text = text.strip()

  # 只取第一个 { 到最后一个 } 之间的内容
  start = text.find("{")
  end = text.rfind("}")
  if start >= 0 and end > start:
    text = text[start:end + 1]

  try:
    data = json.loads(text)
  except Exception:
    return {"risk": "unknown", "reason": "模型输出无法解析", "law_refs": []}

  # 字段缺失或取值非法时兜底
  risk = data.get("risk", "unknown")
  if risk not in ("high", "medium", "none", "unknown"):
    risk = "unknown"

  return {
    "risk": risk,
    "reason": str(data.get("reason", "")),
    "law_refs": data.get("law_refs", []) or [],
  }


def judge_clause(client, clause, hits, full_text):
  """
  让模型判断一条条款有没有法律风险

  client:OpenAI 客户端
  clause:一条合同条款
  hits:检索到的法条
  full_text:整份合同全文
  返回
      parse_verdict 的结果
  """
  messages = build_messages(clause, hits, full_text)

  response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0,
    max_tokens=600,
  )
  raw = response.choices[0].message.content or ""

  return parse_verdict(raw)


def check_contract_file(filename, retriever, top_k=TOP_K, verbose=False):
  """
  体检一份合同，返回结构化结果（不写文件）

  参数
      filename:  合同文件名，如 "sample_1.txt"
                 （内部只取文件名部分，防止传 "../../xxx" 这种路径）
      retriever: 已建好的检索器（由调用方传进来，避免重复加载 bge-m3）
      top_k:     每条条款检索几条法条
      verbose:   是否打印进度（命令行版打开，Agent 版关掉）
  返回
      正常：{"file": 文件名, "clauses": [{index,text,risk,reason,law_refs,hit_laws}, ...]}
      出错：{"file": 文件名, "error": "原因"}
  """
  # 只取文件名部分，防止路径穿越
  name = Path(filename).name

  if not name.endswith(".txt"):
    return {"file": name, "error": f"只支持 .txt 文件，收到的是：{filename}"}

  path = CONTRACTS / name
  if not path.exists():
    return {"file": name, "error": f"没有找到 {name}，可用的是：{list_contracts()}"}

  api_key = os.getenv("DEEPSEEK_API_KEY")
  if not api_key:
    return {"file": name, "error": "没读到 DEEPSEEK_API_KEY"}

  client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

  text = path.read_text(encoding="utf-8")
  clauses = split_clauses(text)

  if verbose:
    print(f"\n{name}：切成 {len(clauses)} 条")

  results = []
  for i, clause in enumerate(clauses, 1):
    # 检索这条条款相关的法条
    hits = retriever.retrieve(clause, top_k)

    # 判断
    verdict = judge_clause(client, clause, hits, text)

    # 记下这次检索到哪些法条，方便人工核对依据对不对
    hit_laws = []
    for h in hits:
      hit_laws.append(f"《{h['law_title']}》{h['display_no']}")

    results.append({
      "index": i,
      "text": clause,
      "risk": verdict["risk"],
      "reason": verdict["reason"],
      "law_refs": verdict["law_refs"],
      "hit_laws": hit_laws,
    })

    if verbose:
      # 条款原文可能有换行，打印时压成一行
      one_line = clause[:30].replace("\n", " ")
      print(f"[{i}/{len(clauses)}] {verdict['risk']:<8} {one_line}")

    # 轻微限速，避免连续调用被判定为高频
    time.sleep(0.3)

  return {"file": name, "clauses": results}
