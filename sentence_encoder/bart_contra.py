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
        # # Contrastive loss 1
        # # 参考《6_UAI2021_Contrastive prototype learning with augmented embeddings for few-shot learning》
        # # 原型和索引集作对比
        # num = query_0.shape[1]  # 4
        # num_classes = query_0.shape[0]  # 19
        # loss_contra = 0.
        # for i in range(num_classes):
        #     sim_instance = torch.exp(self.sim(query_0, prototypes_0[i]))  # [19,4,768]*[768]=[19,4]
        #     pos_ins = sim_instance[i]  # [4]
        #     neg_ins = (sim_instance.sum(0) - pos_ins)  # [4] -> [1]
        #     loss_contra += - torch.log(pos_ins.sum(0) / (pos_ins.sum(0) + neg_ins.sum(0)))  # [1]
        # loss_contra = loss_contra / (num * num_classes)

        # # Contrastive loss 2
        # # 参考《8_AAAI2022_Contrastnet_A contrastive learning framework for few-shot text classification》
        # # 相同类别的向量表示和不同类别的向量表述作对比
        # support_inds = torch.arange(0, N).view(N, 1, 1).expand(N, K, 1).long()
        # support_inds = Variable(support_inds, requires_grad=False).to('cuda')
        # query_inds = torch.arange(0, N).view(N, 1, 1).expand(N, Q, 1).long()
        # query_inds = Variable(query_inds, requires_grad=False).to('cuda')
        # support_re = support_0.reshape(N*K, D)
        # query_re = query_0.reshape(N*Q, D)
        # x = [support_re, query_re]
        # X = torch.cat(x, 0)
        # batch_labels = torch.cat([support_inds.reshape(-1), query_inds.reshape(-1)], 0)
        # len_ = batch_labels.size()[0]
        #
        # # computing similarities for each positive and negative pair
        # s = self.similarity(X, X)
        #
        # # computing masks for contrastive loss
        # mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device)  # 所有的正负样本（分母）
        # label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        # mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float() * mask_i  # 正样本（但不考虑样本本身）（分子）
        # pos_num = torch.sum(mask_j, 1)
        #
        # # weighted NLL loss
        # s_i = torch.clamp(torch.sum(s * mask_i, 1), min=1e-10)
        # s_j = torch.clamp(s * mask_j, min=1e-10)
        # log_p = torch.sum(-torch.log(s_j / s_i) * mask_j, 1) / pos_num
        # loss_contra = torch.mean(log_p)

        # # Contrastive loss 3
        # # 参考《7_EMNLP2021_Exploring task difficulty for few-shot relation extraction》
        # # 原型和原型作对比
        # # select positive prototypes
        # pos_proto_hyb = prototypes_0.unsqueeze(1)  # [18, 1, 768]
        # # select negative prototypes
        # neg_index = torch.zeros(N, N - 1)  # [18, 17]
        # for i in range(N):
        #     index_ori = [i for i in range(0, N)]
        #     index_ori.pop(i)
        #     neg_index[i] = torch.tensor(index_ori)
        # neg_index = neg_index.long().view(-1).cuda()  # [18*17]
        # neg_proto_hyb = torch.index_select(prototypes_0, dim=0, index=neg_index).view(N, N - 1, -1)  # [18, 17, 768]
        # proto_selected = torch.cat((pos_proto_hyb, neg_proto_hyb), dim=1)  # (18, 18, 768)
        # # 但是这样做没有anchor vector，无法计算logits_proto，参考文献使用label的名称和描述来获得anchor vector。
        # labels_proto = torch.cat((torch.ones(N, 1), torch.zeros(N, N - 1)), dim=-1).cuda()  # (18, 18)
        # loss_contra = self.bce_loss(logits_proto, labels_proto)

        # return all_logits, loss_contra
        return all_logits
