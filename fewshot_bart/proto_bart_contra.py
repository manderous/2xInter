import torch

from fewshot.base import *


class PrototypicalNetwork_bart_contra(FSLBaseModel):
    def __init__(self, encoder, args):
        super(PrototypicalNetwork_bart_contra, self).__init__(encoder, args)

    def forward(self, batch, setting, template_ll_list, train_flag):
    # def forward(self, batch, setting, template_ll_list):
        _, N, K, Q = setting
        B = int(batch['i'].shape[0] / N / (K + Q))
        # all_logits, pos_logits, loss_contra = self.encoder(B, N, K, Q, batch, train_flag)
        all_logits, pos_logits, loss_contra = self.encoder(B, N, K, Q, batch)

        proto_logits_all = torch.zeros_like(all_logits[0])
        for idx in range(len(all_logits)):  # 现假设有3,4,5,6个模板
            proto_logits_all += all_logits[idx] * template_ll_list[idx]  # for all and w/o IC
            # proto_logits_all += all_logits[idx]  # for w/o all and w/o AGT,如果仅考虑一个Prompt，就不需要再乘以似然了

        return proto_logits_all, pos_logits, loss_contra
