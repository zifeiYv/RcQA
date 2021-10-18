# -*- coding: utf-8 -*-
from .simple_tokenizer import SimpleTokenizer


def get_class(name):
    if name == 'simple':
        return SimpleTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)
