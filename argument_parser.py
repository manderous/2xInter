import random
import datetime
import argparse
from constant import *


def str_list(text):
    return tuple(text.split(','))


def int_list(text):
    return [int(x) for x in text.split(',')]


def parse_int_list(input_):
    if input_ == None:
        return []
    return list(map(int, input_.split(',')))


def parse_float_list(input_):
    if input_ == None:
        return []
    return list(map(float, input_.split(',')))


def one_or_list(parser):
    def parse_one_or_list(input_):
        output = parser(input_)
        if len(output) == 1:
            return output[0]
        else:
            return output

    return parse_one_or_list


def argument_parser():
    parser = argparse.ArgumentParser()
    # Training setting
    parser.add_argument('-m', '--model', default='proto', choices=fsl_class.keys())
    parser.add_argument('-e', '--encoder', default='bertmlp', choices=encoder_class.keys())
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-o', '--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_step_size', default=1000, type=int)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--ex', default='base')

    # Few-shot settings
    parser.add_argument('-d', '--dataset', default='rams',
                        choices=dataset_constant.keys())
    parser.add_argument('--save', default='checkpoints', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--ams', default=0, type=float, help='Additive Margin')

    # Supervised setting
    parser.add_argument('--epoch', default=30, type=int)

    parser.add_argument('-n', '--way', default=5, type=int)
    parser.add_argument('-k', '--shot', default=5, type=int)
    parser.add_argument('-q', '--query', default=4, type=int)
    # Embedding
    parser.add_argument('--embedding', default='glove', type=str_list)
    parser.add_argument('--tune_embedding', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--progress', default=False, action='store_true')
    parser.add_argument('--hidden_size', default=512, type=int)

    # Tree settings
    parser.add_argument('--tree', default='full', type=str, choices=['full', 'prune'])

    # BERT
    parser.add_argument('--bert_pretrained', default='bert-base-cased', type=str)
    parser.add_argument('--bert_layer', default=4, type=int)
    parser.add_argument('--bert_update', default=False, action='store_true')

    # CNN params
    parser.add_argument('--window', default=2, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)

    # CNN, NCNN parameters
    parser.add_argument('--cnn_kernel_sizes', default=[2, 3, 4, 5], type=parse_int_list)
    parser.add_argument('--cnn_kernel_number', default=150, type=int)

    # RNN parameters (i.e, GRU, LSTM)
    parser.add_argument('--rnn_num_hidden_units', default=300, type=int)
    parser.add_argument('--rnn_pooling', default='pool_anchor',
                        choices=['pool_anchor', 'pool_max', 'pool_dynamic', 'pool_entity'])

    # GCNN params
    parser.add_argument('--num_rel_dep', default=50, type=int)
    parser.add_argument('--gcnn_kernel_numbers', default=[300, 300], type=parse_int_list)
    parser.add_argument('--gcnn_edge_patterns', default=[1, 0, 1], type=parse_int_list)
    parser.add_argument('--gcnn_pooling', default='pool_anchor',
                        choices=['pool_anchor', 'pool_max', 'pool_dynamic', 'pool_entity'])

    # Transformer model
    parser.add_argument('--wsd_model', default='none', type=str, choices=['none', 'gcn', 'bert'])

    # Auxilarity params
    parser.add_argument('--alpha', default=0, type=float)
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--gamma', default=0.0, type=float)
    parser.add_argument('--omega', default=0.0, type=float)
    parser.add_argument('--xi', default=0.0, type=float)

    return parser
