# -*- coding: utf-8 -*-
import argparse
import code
import prettytable
import logging
import sys
sys.path.append('/Volumes/工作/算法/RcQA')
import os
os.chdir('/Volumes/工作/算法/RcQA')


from retriever import get_class

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()


logger.info('Initializing ranker...')
ranker = get_class('tfidf')(tfidf_path=args.model)

# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)


code.interact(local=locals())
