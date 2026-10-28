# 解释一些东西并记录每天做了什么
>Day1： 下载一些包，试着收数据+定格式

pip install openai python-dotenv chromadb pandas tqdm

python-dotenv   读.env 里的 API Key
tqdm 灌数据的时候看进度条的

从github下载 法律数据 https://github.com/lttxzmj/chinese-law-corpus.git

##
从 chinese-law-corpus 下载了法律的原始 JSON
写了 01_scan.py，扫全量生成索引，确认有 56182 条法条
写了 02_filter.py，用关键词粗筛，发现有 31 部
写了 03_content_scan.py 使用正文密度扫描 发现58部法律有命中且达到阈值的
04，进行合并去重并 人工剔除与规则剔除
05，过滤掉 25 条字数少且条文号不是以第开头的
最终得到的产物是 data/clean/articles.jsonl，683KB，已提交
数据初步处理后进行第一次上传  原本法律数据raw就不上传了

>Day2: 

