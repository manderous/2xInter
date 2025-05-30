from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
import torch.nn as nn
import torch
from .utils import init_weights
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Bart_contra(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Sequential(nn.Tanh(), nn.Linear(768, 512))
        self.bce_loss = nn.BCEWithLogitsLoss()


    def select_anchor(self, emb, anchor_index):
        B, L, D = emb.shape
        u = torch.tensor([x for x in range(L)]).unsqueeze(0).to(self.device)
        v = anchor_index.view(B, 1)
        mask = (u == v).unsqueeze(dim=2).to(self.device)
        x = torch.masked_select(emb, mask).view(-1, D)
        return x

    def init_weight(self):
        self.fc1.apply(init_weights)

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        # return torch.matmul(norm_x, norm_y.transpose(1, 0))
        return torch.matmul(norm_x, norm_y)

    def similarity(self, x1, x2):
        # # Gaussian Kernel
        # M = euclidean_dist(x1, x2)
        # s = torch.exp(-M/self.tau)

        # dot product
        tau = 0.1
        results = torch.matmul(x1, x2.t())/tau
        # s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return results

    def forward(self, B, N, K, Q, batch):
        enc_input_ids = batch['enc_input_ids']
        enc_mask_ids = batch['enc_mask_ids']

        context_outputs = self.model(
            enc_input_ids,
            attention_mask=enc_mask_ids,
            return_dict=True,
        )
        decoder_context = context_outputs.encoder_last_hidden_state

        all_logits = []
        for tem_id in range(3):  # 3,4,5,6：遍历所有Prompt，1：仅考虑一个Prompt，即去除因果干预
            dec_prompt_ids = batch['dec_prompt_ids_{}'.format(str(tem_id))]
            dec_prompt_mask_ids = batch['dec_prompt_mask_ids_{}'.format(str(tem_id))]
            mask_index = batch['mask_index_{}'.format(str(tem_id))]
            decoder_prompt_outputs = self.model.decoder(
                    input_ids=dec_prompt_ids,
                    attention_mask=dec_prompt_mask_ids,
                    encoder_hidden_states=decoder_context,
                    encoder_attention_mask=enc_mask_ids,
            )
            decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state   #[bs, prompt_len, H]
            rep = self.select_anchor(self.dropout(decoder_prompt_outputs), mask_index)
            embeddings = self.fc1(rep)
            D = embeddings.shape[-1]
            embeddings = embeddings.view(B, N, K + Q, D)
            # print('| Proto > embedding', tuple(embeddings.shape))
            support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
            query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

            prototypes = support.mean(dim=2)  # ->  B x N x D
            unsqueeze_prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D

            # 获取第0次循环的原型和索引
            if tem_id == 0:
                support_0 = support.squeeze(dim=0)  # B x N x K x D -> N x K x D [18, 5, 512]
                query_0 = query.squeeze(dim=0)  # B x N x Q x D -> N x Q x D [18, 4, 512]
                prototypes_0 = prototypes.squeeze(dim=0)  # B x N x D -> N x D [18, 512]

            query = query.view(B, N * Q, D)  # ->  B x NQ x D
            query = query.unsqueeze(dim=2)  # ->  B x NQ x 1 x D

            error = query - unsqueeze_prototypes  # B x NQ x N x D
            logits = -torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N
            all_logits.append(torch.squeeze(logits, 0))

        loss_contra = 0.0
        return all_logits
