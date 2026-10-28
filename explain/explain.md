# 解释一些东西并记录每天做了什么
>Day1： 下载一些包，试着收数据+定格式

pip install openai python-dotenv chromadb pandas tqdm

python-dotenv   读.env 里的 API Key
tqdm 灌数据的时候看进度条的

从github下载 法律数据 https://github.com/lttxzmj/chinese-law-corpus.git

##
#在测试里先整理总法律数，法条数，类别分布，法条字数（判断后期文字切割） 整理成全文索引
#利用全文索引文件，以关键词判断与主题相关的法律发条数，判断后续检索是否能达到要求
#根据白名单进行整理法条json数据，一条法条就是一个对象
数据初步处理后进行第一次上传  原本法律数据raw就不上传了

>Day2: 

