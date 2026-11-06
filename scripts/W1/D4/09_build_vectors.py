"""
把所有法条算成向量，存到文件里，后续不用重算

"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CLEAN = Path("data/clean")

rows = [json.loads(line) for line in
        open(CLEAN / "articles.jsonl", encoding="utf-8") if line.strip()]
print(f"法条数: {len(rows)}")

texts = [r["text"] + " " + r["law_title"] for r in rows]  #依旧拼接内容与标题

model = SentenceTransformer("BAAI/bge-m3")    #首次加载要下载，记得设置镜像在下载

#编码  得到一个二维数组ndarray
vecs = model.encode(
  texts,
  batch_size=32,    #一次处理32条
  normalize_embeddings=True,    #L2归一化，向量模长为1，这样就可以用点积算了
  show_progress_bar=True,     #进度条
)

#存入文件里  把数组存进.npy二进制文件
np.save(CLEAN / "articles_vec.npy", vecs)

print("完成")
print("形状：",vecs.shape)    
print("类型:",vecs.dtype)   
print("大小:",vecs.nbytes / 1024 / 1024, "MB")

# 形状： (1251, 1024)
# 类型: float32
# 大小: 4.88671875 MB