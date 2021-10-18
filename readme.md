# 概述：一个基于阅读理解的自动问答模型。

模型的整体流程图如下：

![avatar](images/overview.png)

模型主要分为两部分：Document Retriever和Document Reader。其中，Retriever部分主要是根据输入的问题从文档数据库中获取与之最相关的文档（有可能是多篇）；Reader
部分主要是依据预训练模型从文档中进行答案抽取。

# 使用方法
## 1、下载
模型文件及数据文件暂不纳入git管理，需要另外从百度网盘下载。

>链接: https://pan.baidu.com/s/1BxEz2HBUZzwzpS2hDATTfQ 
>
>提取码: 0elm

下载完成后，将`data`文件夹和`model`文件夹放至项目根目录即可。

完成后，项目具有如下结构：
```text
.
├── app.py
├── baidu_stopwords.txt
├── data
│   ├── dev.json
│   ├── doc.db
│   ├── test.json
│   └── train.json
├── images
│   └── overview.png
├── model
│   ├── doc-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
│   └── model_8545
│       ├── model_config.json
│       ├── model_state.pdparams
│       ├── tokenizer_config.json
│       └── vocab.txt
├── predict.py
├── readme.md
├── requirements.txt
├── retriever
│   ├── __init__.py
│   ├── tfidf_doc_ranker.py
│   └── utils.py
├── scripts
│   ├── __init__.py
│   ├── build_db.py
│   ├── build_tfidf.py
│   └── interactive.py
└── tokenizers
    ├── __init__.py
    ├── simple_tokenizer.py
    └── tokenizer.py

```

## 2、启动
利用flask部署，启动`app.py`后，以`POST`的方式请求路由`http://127.0.0.1:5000/qa` ，参数格式为`application/json`， 参数内容为：
```json
{
  "question": "什么时候的龙井最好"
}
```

# 参考内容：
- [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051).

- [ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding](https://arxiv.org/abs/2010.12148).