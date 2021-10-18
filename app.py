# -*- coding: utf-8 -*-
from flask import Flask, request
from predict import answer
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('./logs/' + 'app.log', maxBytes=10 * 1024 * 1024,
                              backupCount=5)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(fmt)
logger.addHandler(handler)

app = Flask(__name__)


@app.route('/qa', methods=['POST'])
def qa():
    paras = request.json
    # article = paras['article']
    question = paras['question']
    logger.info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    logger.info('Q:' + question)
    asr = answer(question)
    return {'answer': asr}


if __name__ == '__main__':
    app.run()
