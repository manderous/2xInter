from preprocess.utils import *
from .supervised_bart import RAMSDataset_bart
from constant import *
import json


def merge(r, g, b):
    item = {}
    for k, v in r.items():
        item[k] = v
    for k, v in g.items():
        item[k] = v
    for k, v in b.items():
        item[k] = v
    return item


class FSLDataset_bart(RAMSDataset_bart):
    def __init__(self, N, K, Q, O, length, **kwargs):
        super(FSLDataset_bart, self).__init__(**kwargs)
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
        if self.O == 1:
            selected_fsl_target = [0] + random.sample(self.positive_targets, k=self.N)
        else:
            selected_fsl_target = random.sample(self.positive_targets, k=self.N)

        all_class_indices = []
        all_class_indices_each_class = {}
        fsl_label_map_reverse = {v: k for k, v in self.fsl_label_map.items()}
        targets = []
        sample_per_class = self.K + self.Q

        for i, target in enumerate(selected_fsl_target):
            possible_indices = self.label_indices_map[target]
            if len(possible_indices) < sample_per_class:
                possible_indices = possible_indices + possible_indices + possible_indices
            sampled_indices = random.sample(possible_indices, k=sample_per_class)
            all_class_indices += sampled_indices
            all_class_indices_each_class[fsl_label_map_reverse[target]] = sampled_indices
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
            batches[fea] = torch.cat(data, dim=0)
        return batches


def keep(x):
    return x


FeatureTensor = {
    'i': torch.LongTensor,  # Index of the sample
    'target': torch.LongTensor,
    'enc_input_ids': torch.LongTensor,
    'enc_mask_ids': torch.LongTensor,
    'dec_prompt_ids_0': torch.LongTensor,
    'dec_prompt_mask_ids_0': torch.LongTensor,
    'mask_index_0': torch.LongTensor,
    'dec_prompt_ids_1': torch.LongTensor,
    'dec_prompt_mask_ids_1': torch.LongTensor,
    'mask_index_1': torch.LongTensor,
    'dec_prompt_ids_2': torch.LongTensor,
    'dec_prompt_mask_ids_2': torch.LongTensor,
    'mask_index_2': torch.LongTensor,
}
