# -*- coding: utf-8 -*-
import paddle
from paddlenlp.transformers import ErnieGramForQuestionAnswering, ErnieGramTokenizer
from paddlenlp.metrics.squad import compute_prediction
from retriever import get_class, DocDB
import logging

logger = logging.getLogger('app')
MODEL_CLASSES = (ErnieGramForQuestionAnswering, ErnieGramTokenizer)
# >>>>
model_path = './model/model_8545'
doc_stride = 128
max_seq_length = 128
n_best_size = 20
max_answer_length = 30
db_path = './data/doc.db'
ranker = get_class('tfidf')(tfidf_path='./model/doc-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')

# <<<<
paddle.set_device('cpu')
model_class, tokenizer_class = MODEL_CLASSES
tokenizer = tokenizer_class.from_pretrained(model_path)
model = model_class.from_pretrained(model_path)


def prepare_validation_features(contexts, questions):
    data = [{
        'id': '1',
        'title': '',
        'context': contexts,
        'question': questions,
        'answers': [],
        'answer_starts': [-1]
    }]

    contexts = [contexts]
    questions = [questions]

    tokenized_examples = tokenizer(questions, contexts, stride=doc_stride, max_seq_len=max_seq_length)
    sequence_ids = tokenized_examples[0]['token_type_ids']
    tokenized_examples[0]["example_id"] = '1'
    tokenized_examples[0]["offset_mapping"] = [(o if sequence_ids[k] == 1 else None) for k, o in enumerate(
        tokenized_examples[0]['offset_mapping'])]

    input_ids = tokenized_examples[0]['input_ids']
    token_type_ids = tokenized_examples[0]['token_type_ids']
    offset_mapping = tokenized_examples[0]["offset_mapping"]
    new_data = [{
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'offset_mapping': offset_mapping,
        'overflow_to_sample': 0,
        'example_id': '1'
    }]

    input_ids = paddle.to_tensor(input_ids)
    input_ids = input_ids.reshape(shape=(1, len(input_ids)))
    token_type_ids = paddle.to_tensor(token_type_ids)
    token_type_ids = token_type_ids.reshape(shape=(1, len(token_type_ids)))

    start_logits_tensor, end_logits_tensor = model(input_ids, token_type_ids)
    all_start_logits = [start_logits_tensor.numpy()[0]]
    all_end_logits = [end_logits_tensor.numpy()[0]]
    all_predictions, _, _ = compute_prediction(data, new_data, (all_start_logits, all_end_logits),
                                               False, n_best_size, max_answer_length)

    return all_predictions['1']


def get_closest_docs(query, k=5):
    doc_ids, doc_scores = ranker.closest_docs(query, k)
    docs = []
    with DocDB(db_path) as doc_db:
        for i in range(len(doc_ids)):
            doc_id = doc_ids[i]
            score = doc_scores[i]
            doc = doc_db.get_doc_text(doc_id)
            docs.append(doc)
            logger.info(f'{i+1}, doc id: {doc_id}; score: {score}')
            logger.info(f'   doc: {doc}')
    return docs


def answer(query):
    asrs = []
    docs = get_closest_docs(query)
    asrs.append(prepare_validation_features(docs[0], query))
    return asrs
