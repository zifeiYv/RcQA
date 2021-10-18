# -*- coding: utf-8 -*-
from .tokenizer import Tokens, Tokenizer
import jieba
import logging
logger = logging.getLogger('app')


class SimpleTokenizer(Tokenizer):
    def __init__(self):
        self.annotators = set()

    def tokenize(self, text):
        try:
            # 加入停用词表可能会提升效果
            with open('../baidu_stopwords.txt') as f:
                stopwords = f.readlines()
        except FileNotFoundError:
            stopwords = []
        data = [i for i in jieba.cut(text) if i not in stopwords]
        return Tokens(data, self.annotators)
