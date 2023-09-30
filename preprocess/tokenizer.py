import multiprocessing
import pickle
# import pytorch_pretrained_bert
import transformers
from preprocess.rams_utils import *
import json
import os
import time
from tqdm import tqdm

BERT_BASE_CASED = 'bert-base-cased'
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_LARGE_CASED = 'bert-large-case'
BERT_LARGE_UNCASED = 'bert-large-uncased'

SEP = 102
CLS = 101
PAD = 0

def crop_item(item, ml):
    """
    :param item:
    :param ml: int, max length
    :return:
    """
    l = len(item['token'])
    t = item['trigger'][0]
    a = None
    if len(item['argument']) == 0:
        a = t
    else:
        positions = [arg[0] for arg in item['argument']]
        abs_distances = [abs(t - x) for x in positions]
        min_distance = min(abs_distances)
        for p, d in zip(positions, abs_distances):
            if d == min_distance:
                a = p
                break
        assert a is not None, 'a is None'
    if l <= ml:
        return 0, l
    else:
        middle = (t + a) // 2
        start = max(0, middle - 40)
        end = min(start + ml, l)
        assert end <= l
        return start, end


def tokenize(items, bert_pretrain):
    if bert_pretrain == BERT_BASE_CASED:
        tokenizer = transformers.BertTokenizer.from_pretrained(bert_pretrain, do_lower_case=False)
    else:
        tokenizer = transformers.BertTokenizer.from_pretrained(bert_pretrain, do_lower_case=True)
    with open('../datasets/vocab.pkl', 'rb') as f:
        word_map, _ = pickle.load(f)

    tokenized_items = []

    for item in items:
        glove_indices = []
        bert_indices = []

        for token in item['token']:
            if token in word_map:
                glove_indices.append(word_map[token])
            elif token.lower() in word_map:
                glove_indices.append(word_map[token.lower()])
            else:
                glove_indices.append(1)
            bert_tokens = tokenizer.tokenize(token)
            token_indices = tokenizer.convert_tokens_to_ids(bert_tokens)
            bert_indices.append(token_indices)

        item = {
            'id': item['id'],
            'bert_indices': bert_indices,
            'glove_indices': glove_indices
        }
        tokenized_items.append(item)
    return tokenized_items


def tokenize_bert_uncased(items):
    return tokenize(items, BERT_BASE_UNCASED)


def tokenize_bert_cased(items):
    return tokenize(items, BERT_BASE_CASED)


def split(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def unsplit(lists):
    a = []
    for x in lists:
        a += x
    return a


def tokenize_from_json_file(datasets, settings, corpus=('train', 'dev', 'test')):
    n_process = 40
    pool = multiprocessing.Pool(n_process)
    for d in datasets:
        for s in settings:
            for c in corpus:
                data = load_json('datasets/{}/{}/{}.prune.json'.format(d, s, c))
                splits = split(data, n_process)
                parsed_data = pool.map(tokenize_bert_cased, splits)
                parsed_data = unsplit(parsed_data)
                save_json(parsed_data, 'datasets/{}/{}/{}.{}.json'.format(d, s, c, BERT_BASE_CASED))
    pool.close()


def gen_all():
    n_process = 20
    pool = multiprocessing.Pool(n_process)
    for setting in ['supervised']:
        for corpus in ['train', 'dev', 'test']:
            data = load_json('datasets/{}/{}.json'.format(setting, corpus))
            splits = split(data, n_process)
            parsed_data = pool.map(tokenize_bert_cased, splits)
            parsed_data = unsplit(parsed_data)
            with open('datasets/{}/{}.{}.json'.format(setting, corpus, BERT_BASE_CASED), 'w') as f:
                json.dump(parsed_data, f)
    pool.close()


def do_tokenize(prefix):
    n_process=32
    pool = multiprocessing.Pool(n_process)
    data = load_json('{}.prune.json'.format(prefix))
    splits = split(data, n_process)
    parsed_data = pool.map(tokenize_bert_cased, splits)
    parsed_data = unsplit(parsed_data)
    save_json(parsed_data, '{}.{}.json'.format(prefix, BERT_BASE_CASED))
    pool.close()


if __name__ == '__main__':
    do_tokenize(f'../datasets/ace/fsl/train')
    do_tokenize(f'../datasets/ace/fsl/dev')
    do_tokenize(f'../datasets/ace/fsl/test')

    do_tokenize(f'../datasets/ace/fsl/train.negative')
    do_tokenize(f'../datasets/ace/fsl/dev.negative')
    do_tokenize(f'../datasets/ace/fsl/test.negative')

    # do_tokenize('datasets/rams/fsl/train')
    # do_tokenize('datasets/rams/fsl/dev')
    # do_tokenize('datasets/rams/fsl/test')

    # do_tokenize('datasets/rams/fsl/train.negative')
    # do_tokenize('datasets/rams/fsl/dev.negative')
    # do_tokenize('datasets/rams/fsl/test.negative')

    # do_tokenize(f'../datasets/lowkbp/lowkbp0/fsl/dev')
    # do_tokenize(f'../datasets/lowkbp/lowkbp1/fsl/dev')
    # do_tokenize(f'../datasets/lowkbp/lowkbp2/fsl/dev')
    # do_tokenize(f'../datasets/lowkbp/lowkbp3/fsl/dev')
    # do_tokenize(f'../datasets/lowkbp/lowkbp4/fsl/dev')
    # do_tokenize(f'../datasets/lowkbp/lowkbp0/fsl/test')
    # do_tokenize(f'../datasets/lowkbp/lowkbp1/fsl/test')
    # do_tokenize(f'../datasets/lowkbp/lowkbp2/fsl/test')
    # do_tokenize(f'../datasets/lowkbp/lowkbp3/fsl/test')
    # do_tokenize(f'../datasets/lowkbp/lowkbp4/fsl/test')
    # do_tokenize(f'../datasets/lowkbp/lowkbp0/fsl/train')
    # do_tokenize(f'../datasets/lowkbp/lowkbp1/fsl/train')
    # do_tokenize(f'../datasets/lowkbp/lowkbp2/fsl/train')
    # do_tokenize(f'../datasets/lowkbp/lowkbp3/fsl/train')
    # do_tokenize(f'../datasets/lowkbp/lowkbp4/fsl/train')
    #
    # do_tokenize(f'../datasets/lowkbp/lowkbp0/fsl/dev.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp1/fsl/dev.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp2/fsl/dev.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp3/fsl/dev.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp4/fsl/dev.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp0/fsl/test.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp1/fsl/test.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp2/fsl/test.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp3/fsl/test.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp4/fsl/test.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp0/fsl/train.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp1/fsl/train.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp2/fsl/train.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp3/fsl/train.negative')
    # do_tokenize(f'../datasets/lowkbp/lowkbp4/fsl/train.negative')
