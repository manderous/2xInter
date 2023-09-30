import json
from torch.utils.data import Dataset, DataLoader
from preprocess.tokenizer import *
from preprocess.utils import *
import random
import torch
import numpy as np
import os


def merge(r, g, b):
    item = {}
    for k, v in r.items():
        item[k] = v
    for k, v in g.items():
        item[k] = v
    for k, v in b.items():
        item[k] = v
    return item


class RAMSDataset_bart(Dataset):

    def __init__(self,
                 features=(),
                 prefix='datasets/ace/fsl/train',
                 bert_pretrain=BERT_BASE_CASED,
                 device='cpu',
                 load_negative=True):
        super(RAMSDataset_bart, self).__init__()

        self.device = device
        self.features = features
        prefix_parts = prefix.split('/')
        label_map_path = 'datasets/{}/{}/label_map.json'.format(
            prefix_parts[1], prefix_parts[2]
        )
        assert len(prefix_parts) == 4

        self.sen_max_len = 84
        self.prompt_max_len = 19

        # Load label, ner, pos, dep, argument map
        m = load_json(label_map_path)
        self.label_map = m['label']
        self.n_class = len(self.label_map)
        self.argument_map = m['argument']

        # Load data
        self.raw = load_json('{}.template.json'.format(prefix))  # tjy
        self.indices = load_json('{}.{}.json'.format(prefix, bert_pretrain))

        # load_negative:
        self.raw += load_json('{}.negative.template.json'.format(prefix))
        self.indices += load_json('{}.negative.{}.json'.format(prefix, bert_pretrain))

        print('Load')

        self.cache = {}
        print('Check id matching', end=' ')
        assert len(self.raw) == len(self.indices), 'Raw: {}, Indices: {}'.format(len(self.raw), len(self.indices))
        for r, b in zip(self.raw, self.indices):
            assert r['id'] == b['id'], 'Raw and  BERT mismatch'
        print('#Instance: ', len(self.raw), len(self.indices))

    def preprocess(self, i):
        raw = self.raw[i]
        indices = self.indices[i]
        preprocesed = {'i': i, 'target': self.label_map[str(raw['label'])]}
        # ml = self.max_len
        sl = self.sen_max_len
        pl = self.prompt_max_len
        sen_input_ids = indices['sentence_input_ids']
        sen_mask_ids = indices['sentence_mask_ids']
        preprocesed['enc_input_ids'] = sen_input_ids + [0 for _ in range(sl - len(sen_input_ids))]
        preprocesed['enc_mask_ids'] = sen_mask_ids + [0 for _ in range(sl - len(sen_input_ids))]
        for tem_id in range(3):
            prompt_input_ids = indices['tem_{}_prompt_input_ids'.format(str(tem_id))]
            prompt_mask_ids = indices['tem_{}_prompt_mask_ids'.format(str(tem_id))]
            preprocesed['dec_prompt_ids_{}'.format(str(tem_id))] = prompt_input_ids + [0 for _ in range(pl - len(prompt_input_ids))]
            preprocesed['dec_prompt_mask_ids_{}'.format(str(tem_id))] = prompt_mask_ids + [0 for _ in range(pl - len(prompt_input_ids))]
            preprocesed['mask_index_{}'.format(str(tem_id))] = raw['tem_{}_mask_id'.format(str(tem_id))]
        return preprocesed

    def get_one(self, i):
        if i in self.cache:
            return self.cache[i]
        else:
            item = self.preprocess(i)
            self.cache[i] = item
            return item

    def __getitem__(self, item):
        return self.get_one(item)

    def __len__(self):
        return len(self.raw)


def sup_pack(items):
    batches = {}
    for fea in items[0].keys():
        data = [x[fea] for x in items]
        batches[fea] = FeatureTensor[fea](data)
    return batches


def fsl_pack(items):
    batches = {}
    for fea in items[0].keys():
        data = [x[fea] for x in items]
        batches[fea] = torch.cat(data, dim=0)
    return batches


def keep(x):
    return x
