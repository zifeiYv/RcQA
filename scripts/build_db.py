# -*- coding: utf-8 -*-
import argparse
import sqlite3
import json
import os
from uuid import uuid1


def get_contents(filename):
    """从json文件中读取内容"""
    with open(filename) as f:
        doc = json.load(f)
        paragraphs = doc['data'][0]['paragraphs']
        documents = [(uuid1().hex, i['context']) for i in paragraphs]
    return documents


def store_contents(data_path, save_path):
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")
    documents = get_contents(data_path)
    c.executemany("INSERT INTO documents VALUES (?,?)", documents)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    args = parser.parse_args()
    store_contents(args.data_path, args.save_path)
