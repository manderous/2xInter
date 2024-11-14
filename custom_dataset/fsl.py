from preprocess.utils import *
from .supervised import RAMSDataset
from constant import *


def merge(r, g, b):
    item = {}
    for k, v in r.items():
        item[k] = v
    for k, v in g.items():
        item[k] = v
    for k, v in b.items():
        item[k] = v
    return item


class FSLDataset(RAMSDataset):
    def __init__(self, N, K, Q, O, length, **kwargs):
        super(FSLDataset, self).__init__(**kwargs)
        self.N = N
        self.K = K
        self.Q = Q
        self.O = O
        self.length = length

        labels = [x['label'] for x in self.raw]
        label_set = sorted(set(labels))
        self.fsl_label_map = {l: i for i, l in enumerate(label_set)}
        print(self.fsl_label_map)
        self.positive_targets = [i for i in range(self.O, len(label_set))]

        print(self.positive_targets)

        self.label_indices_map = [[] for _ in range(len(label_set))]

        for i, label in enumerate(labels):
            fsl_target = self.fsl_label_map[label]
            self.label_indices_map[fsl_target].append(i)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # random.seed(item)
        if self.O == 1:
            selected_fsl_target = [0] + random.sample(self.positive_targets, k=self.N)
        else:
            # print(f'Trying to sample: N={self.N} Positive={len(self.positive_targets)}')
            selected_fsl_target = random.sample(self.positive_targets, k=self.N)

        all_class_indices = []
        targets = []
        sample_per_class = self.K + self.Q

        for i, target in enumerate(selected_fsl_target):
            possible_indices = self.label_indices_map[target]
            if len(possible_indices) < sample_per_class:
                possible_indices = possible_indices + possible_indices + possible_indices
            sampled_indices = random.sample(possible_indices, k=sample_per_class)
            all_class_indices += sampled_indices
            targets.append([i for _ in range(self.Q)])
        all_class_data = [self.get_one(x) for x in all_class_indices]
        batch = dict()
        for fea in self.features:
            data = [x[fea] for x in all_class_data]
            try:
                batch[fea] = FeatureTensor[fea](data)
            except:
                print(fea)
                for x in data:
                    print(len(x))
                exit(0)

        batch['target'] = torch.LongTensor(targets)
        return batch

    @staticmethod
    def fsl_pack(items):
        batches = {}
        for fea in items[0].keys():
            data = [x[fea] for x in items]
            # batches[fea] = FeatureTensor[fea](data)
            batches[fea] = torch.cat(data, dim=0)
            # print(fea, batches[fea].shape)
        # print('------')
        return batches


def keep(x):
    return x


FeatureTensor = {
    'i': torch.LongTensor,  # Index of the sample
    # ANN
    'entity_indices': torch.LongTensor,
    'entity_attention': torch.FloatTensor,
    # CNN, LSTM, GRU
    'indices': torch.LongTensor,
    'prune_indices': torch.LongTensor,
    'length': torch.LongTensor,
    'prune_length': torch.LongTensor,
    'dist': torch.LongTensor,
    'anchor_index': torch.LongTensor,
    'prune_anchor_index': torch.LongTensor,
    'mask': torch.FloatTensor,
    'prune_mask': torch.FloatTensor,
    'prune_footprint': torch.FloatTensor,
    # BERT
    'cls_text_sep_indices': torch.LongTensor,
    'cls_text_sep_length': torch.LongTensor,
    'cls_text_sep_segment_ids': torch.LongTensor,
    'transform': torch.FloatTensor,
    # GCN
    'dep': torch.FloatTensor,
    'prune_dep': torch.FloatTensor,
    'target': torch.LongTensor,
    # Semcor
    'label_mask': torch.FloatTensor,
    # Precompute BERT
    'emb': torch.FloatTensor
}