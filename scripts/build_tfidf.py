# -*- coding: utf-8 -*-
"""对数据库中的所有文档进行分词，并计算tf-idf，然后序列化到磁盘上"""
import argparse
import math
import logging
import os

import numpy as np
import scipy.sparse as sp

from collections import Counter
from functools import partial

from retriever import DocDB, utils
from tokenizers import get_class

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

DOC2IDX = {}


def count(args, doc_id):
    """根据doc_id获取该doc的内容，然后对其进行分词、ngram处理，最后统计词频"""
    global DOC2IDX
    row, col, data = [], [], []
    # 获取一个tokenizer
    tokenizer = get_class(args.tokenizer)()
    # 获取doc的内容
    with DocDB(args.db_path) as doc_db:
        text = doc_db.get_doc_text(doc_id)
    # 对doc的内容进行tokenize
    tokens = tokenizer.tokenize(text)
    # 对tokens进行ngram操作
    ngrams = tokens.ngrams(n=args.ngram)
    # 通过对每个词进行哈希映射来更快地计算词频
    # 由于哈希映射时对``args.hash_size``取了模，因此，counts的长度最长为``args.hash_size``
    counts = Counter([utils.hash_(gram, args.hash_size) for gram in ngrams])
    # 每个词对应的映射后的数字
    row.extend(counts.keys())
    # 每个doc的编号
    col.extend([DOC2IDX.get(doc_id)] * len(counts))
    # 每个词的词频
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args):
    """首先获取数据库中全部文档的id，然后遍历id获取文档内容，再逐文档
    进行分词，生成计数矩阵。"""
    global DOC2IDX
    with DocDB(args.db_path) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    row, col, data = [], [], []
    _count = partial(count, args)
    for i in doc_ids:
        b_row, b_col, b_data = _count(i)
        row.extend(b_row)
        col.extend(b_col)
        data.extend(b_data)

    # 创建稀疏矩阵，这里用的是按行压缩的方法（Compressed Sparse Row, csr）
    # 关于什么是csr_matrix，参考：
    #   https://www.pianshen.com/article/7967656077/
    #   https://zhuanlan.zhihu.com/p/342942385
    count_matrix = sp.csr_matrix((data, (row, col)), shape=(args.hash_size, len(doc_ids)))
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


def get_tfidf_matrix(matrix):
    """根据单词计数矩阵，计算td-idf

    这里用的计算公式为：
    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    其中：
    * tf = 某个词在某篇文档中的词频
    * N = 文档的总数
    * Nt = 出现该词的文档的数量
    """
    Nt = get_doc_freqs(matrix)
    idfs = np.log((matrix.shape[1] - Nt + 0.5) / (Nt + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = matrix.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(matrix):
    """对每个词，统计其在多少篇文章中出现，返回文章的数量"""
    binary = (matrix > 0).astype(int)
    return np.array(binary.sum(1)).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    logging.info('统计词频...')
    count_matrix, doc_dict = get_count_matrix(args)

    logger.info('计算tfidf...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('统计每个词所出现的文章的数量...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('保存至: "%s.npz"' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    utils.save_sparse_csr(filename, tfidf, metadata)

