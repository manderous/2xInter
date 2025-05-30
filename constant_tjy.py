import transformers
from preprocess.tokenizer import *
from sentence_encoder import *
from fewshot import *
from fewshot_template import *
from fewshot_intervention import *
from fewshot_paradox import *
from fewshot_bart import *
# from fewshot_bart_verb import *
# from fewshot_gpt35 import *
from supervise import *
import tqdm

dataset_constant = {
    'ace': {
        'max_length': 40,
        'n_class': 34,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 18,
    },
    'fed': {
        'max_length': 40,
        'n_class': 450,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'rams': {
        'max_length': 40,
        'n_class': 140,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'rams2': {
        'max_length': 40,
        'n_class': 39,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'semcor': {
        'max_length': 40,
        'n_class': 7435,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'cyber': {
        'max_length': 40,
        'n_class': 32,
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

base = ['i', 'indices', 'length', 'mask', 'anchor_index', 'dist']

feature_map = {
    'ann': base + ['entity_indices', 'entity_attention'],
    'cnn': base + ['prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'lstm': base + ['prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'gru': base + ['prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'gcn': base + ['dep', 'prune_dep'],
    'bertgcn': base + ['cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segment_ids',
                       'transform',
                       'dep',
                       'prune_dep'],
    'bertcnn': base + ['cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segment_ids',
                       'transform',
                       'dep', 'prune_footprint',
                       'prune_dep'],
    'bart_base': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0', 'dec_prompt_mask_ids_0',
                  'mask_index_0', 'dec_prompt_ids_1', 'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                  'dec_prompt_mask_ids_2', 'mask_index_2'],  # 5个样本
    'bart_contra': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0', 'dec_prompt_mask_ids_0',
                  'mask_index_0', 'dec_prompt_ids_1', 'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                  'dec_prompt_mask_ids_2', 'mask_index_2'],  # 5个样本
    'bart_contra_multipos': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0', 'dec_prompt_mask_ids_0',
                  'mask_index_0', 'dec_prompt_ids_1', 'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                  'dec_prompt_mask_ids_2', 'mask_index_2'],  # 3个样本
    'bart_contra_multipos_neg': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0', 'dec_prompt_mask_ids_0',
                  'mask_index_0', 'dec_prompt_ids_1', 'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                  'dec_prompt_mask_ids_2', 'mask_index_2'],  # 3个样本
    'bart_contra_multipos_neg_woAGT': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0', 'dec_prompt_mask_ids_0',
                  'mask_index_0', 'dec_prompt_ids_1', 'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                  'dec_prompt_mask_ids_2', 'mask_index_2'],  # 3个样本
    'bart_contra_multipos_neg_woIC': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0', 'dec_prompt_mask_ids_0',
                  'mask_index_0', 'dec_prompt_ids_1', 'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                  'dec_prompt_mask_ids_2', 'mask_index_2'],  # 3个样本
    'bart_contra_multipos_neg_woALL': ['i', 'target', 'enc_input_ids', 'enc_mask_ids', 'dec_prompt_ids_0', 'dec_prompt_mask_ids_0',
                  'mask_index_0', 'dec_prompt_ids_1', 'dec_prompt_mask_ids_1', 'mask_index_1', 'dec_prompt_ids_2',
                  'dec_prompt_mask_ids_2', 'mask_index_2'],  # 3个样本
    'bertmlp': base + ['cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segment_ids',
                       'transform',
                       'prune_footprint'],
    'bertmlp_template': base + ['cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segment_ids',
                       'transform',
                       'prune_footprint'],
    'berted': base + ['cls_text_sep_indices',
                      'cls_text_sep_length',
                      'cls_text_sep_segment_ids',
                      'transform'],
    'bertlinear': base + ['cls_text_sep_indices',
                          'cls_text_sep_length',
                          'cls_text_sep_segment_ids',
                          'transform',
                          'prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'bertdm': base + ['cls_text_sep_indices',
                      'cls_text_sep_length',
                      'cls_text_sep_segment_ids',
                      'transform'],
    'mlp': base + ['emb']
}

encoder_class = {
    'ann': ANN,
    'cnn': CNN,
    'lstm': LSTM,
    'gru': GRU,
    'gcn': GCN,
    'bertgcn': BertGCN,
    'bertcnn': BertCNN,
    'bart_base': Bart_base,
    # 'bart_verb': Bart_verb,
    'bart_contra': Bart_contra,
    'bart_contra_multipos': Bart_contra_multipos,
    'bart_contra_multipos_neg': Bart_contra_multipos_neg,
    'bart_contra_multipos_neg_woAGT': Bart_contra_multipos_neg_woAGT,
    'bart_contra_multipos_neg_woIC': Bart_contra_multipos_neg_woIC,
    'bart_contra_multipos_neg_woALL': Bart_contra_multipos_neg_woALL,
    'bertmlp': BertMLP,
    'bertmlp_template': BertMLP_template,
    'bertlinear': BertLinear,
    'mlp': MLP,

}

classificaion_class = {
    'cnn': CNNClassifier,
    'gcn': GCNClassifier,
    'berted': BertEDClassifier,
    'bertgcn': BertGCNClassifier,
    'bertdm': BertDMClassifier,
}

fsl_class = {
    'proto': PrototypicalNetwork,
    'proto_bart': PrototypicalNetwork_bart,
    # 'proto_bart_verb': PrototypicalNetwork_bart_verb,
    'proto_bart_contra': PrototypicalNetwork_bart_contra,
    # 'proto_gpt35': PrototypicalNetwork_gpt35,
    'proto_template': PrototypicalNetwork_template,
    'proto_template_multi': PrototypicalNetwork_template_multi,
    'dproto_interv': DProto_Intervention,
    'dproto_interv_svd': DProto_Interv_SVD,
    'dproto_interv_tde': DProto_Interv_TDE,
    'dproto_interv_tde_1': DProto_Interv_TDE_1,
    'dproto_interv_ebm': DProto_Interv_EBM,
    'dproto_interv_ebm_1': DProto_Interv_EBM_1,
    'proto_interv': PrototypicalNetwork_Intervention,
    'proto_interv_svd': PrototypicalNetwork_Interv_SVD,
    'proto_interv_svd_1': PrototypicalNetwork_Interv_SVD_1,
    'proto_interv_svd_cf': PrototypicalNetwork_Interv_SVD_CF,
    'proto_interv_tde': PrototypicalNetwork_Interv_TDE,
    'proto_interv_tde_1': PrototypicalNetwork_Interv_TDE_1,
    'proto_interv_ebm': PrototypicalNetwork_Interv_EBM,
    'proto_interv_ebm_1': PrototypicalNetwork_Interv_EBM_1,
    'proto_interv_ebm_2': PrototypicalNetwork_Interv_EBM_2,
    'proto_interv_ebm_old_2': PrototypicalNetwork_Interv_EBM_OLD_2,
    'proto_interv_ebm_old_3': PrototypicalNetwork_Interv_EBM_OLD_3,
    'proto_interv_ebm_3': PrototypicalNetwork_Interv_EBM_3,
    'proto_interv_ebm_kl': PrototypicalNetwork_Interv_EBM_KL,
    'proto_para': PrototypicalNetwork_Paradox,
    'attproto': AttPrototypicalNetwork,
    'relation': RelationNetwork,
    'matching': MatchingNetwork,
    'induction': InductionNetwork,
    'dmn': DProto,
    'melr': MELRNetwork,
    'melr_interv': MELRNetwork_Intervention,
    'melr_interv_svd': MELRNetwork_Interv_SVD,
    'melr_interv_svd_1': MELRNetwork_Interv_SVD_1,
    'melr_interv_tde': MELRNetwork_Interv_TDE,
    'melr_interv_tde_1': MELRNetwork_Interv_TDE_1,
    'melr_interv_pca': MELRNetwork_Interv_PCA,
    'melr_interv_inter': MELRNetwork_Interv_INTER,
    'melr_interv_inter_cf': MELRNetwork_Interv_INTER_CF,
    'melr_interv_ebm': MELRNetwork_Interv_EBM,
    'melr_interv_ebm_old': MELRNetwork_Interv_EBM_OLD,
    'melr_interv_ebm_1': MELRNetwork_Interv_EBM_1,
    'melr_interv_ebm_split': MELRNetwork_Interv_EBM_SPLIT,
    'melr_interv_ebm_split_1': MELRNetwork_Interv_EBM_SPLIT_1,
    'melr_interv_ebm_cf': MELRNetwork_Interv_EBM_CF,
    'melr_interv_ebm_kl': MELRNetwork_Interv_EBM_KL,
    'melr_interv_ebm_sum': MELRNetwork_Interv_EBM_SUM,
    'melrplus': MELRPlus
}

meta_opt_net_class = {
    'svmcs': SVM_CS,
    'proto': Proto,
    'svmhe': SVM_He,
    'svmww': SVM_WW,
    'ridge': Ridge,
    'r2d2': R2D2
}
