import transformers
from sentence_encoder import *
from fewshot_bart import *
import tqdm

dataset_constant = {
    'ace': {
        'max_length': 40,
        'n_class': 34,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 18,
    },
    'rams': {
        'max_length': 40,
        'n_class': 140,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'lowkbp': {
        'max_length': 40,
        'n_class': 92,
        'other': 0,  # Means no other, doing event classification
        'train_way': 18,
    }
}

dataset_constant['lowkbp0'] = dataset_constant['lowkbp']
dataset_constant['lowkbp1'] = dataset_constant['lowkbp']
dataset_constant['lowkbp2'] = dataset_constant['lowkbp']
dataset_constant['lowkbp3'] = dataset_constant['lowkbp']
dataset_constant['lowkbp4'] = dataset_constant['lowkbp']

feature_map = {'bart_contra_multipos_neg': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0',
                                            'dec_prompt_mask_ids_0', 'mask_index_0', 'dec_prompt_ids_1',
                                            'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                                            'dec_prompt_mask_ids_2', 'mask_index_2']}

encoder_class = {'bart_contra_multipos_neg': Bart_contra_multipos_neg}

fsl_class = {'proto_bart_contra': PrototypicalNetwork_bart_contra}
