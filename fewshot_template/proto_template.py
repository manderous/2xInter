import torch
import torch.nn as nn
from transformers import BertTokenizer
# from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F


class PrototypicalNetwork_template(nn.Module):

    def __init__(self, encoder, args):
        super(PrototypicalNetwork_template, self).__init__()
        self.args = args
        self.device = args.device
        self.encoder = encoder
        self.head = torch.nn.Linear(3072, args.hidden_size, bias=False)  # 线性函数，将隐藏层映射至低维空间


    def process_hiddens(self, embeddings, batch_length, settings):
        _, N, K, Q = settings
        B = batch_length
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        prototypes = support.mean(dim=2)  # ->  B x N x D
        unsqueeze_prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D

        query = query.view(B, N * Q, D)  # ->  B x NQ x D
        query = query.unsqueeze(dim=2)  # ->  B x NQ x 1 x D

        error = query - unsqueeze_prototypes  # B x NQ x N x D
        proto_logits = -torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N

        return proto_logits

    def forward(self, batch, settings, fsl_label_map):
        _, N, K, Q = settings
        batch_length = int(batch['length'].shape[0] / N / (K + Q))
        encoded = self.encoder(batch)
        last_hidden_state = encoded['last_hidden_state_at_mask']
        proto_logits = self.process_hiddens(self.head(last_hidden_state), batch_length, settings)  # 先映射至低维空间再计算原型
        return proto_logits.squeeze(0)
