# -*- coding: utf-8 -*-
"""一些实用函数"""
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32


def save_sparse_csr(filename, matrix, metadata=None):
    """将稀疏矩阵存储至指定位置"""
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    """从指定位置加载稀疏矩阵"""
    loader = np.load(filename, allow_pickle=True)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


def hash_(token, num_buckets):
    """对token利用32位murmurhash进行哈希映射。

    由于对num_buckets求了模，因此，返回值最大为``num_buckets-1``。
    """
    return murmurhash3_32(token, positive=True) % num_buckets
