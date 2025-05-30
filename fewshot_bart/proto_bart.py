from fewshot.base import *


class PrototypicalNetwork_bart(FSLBaseModel):
    def __init__(self, encoder, args):
        super(PrototypicalNetwork_bart, self).__init__(encoder, args)

    def forward(self, batch, setting, template_ll_list):
        _, N, K, Q = setting
        B = int(batch['i'].shape[0] / N / (K + Q))
        all_logits = self.encoder(B, N, K, Q, batch)

        proto_logits_all = torch.zeros_like(all_logits[0])
        for idx in range(len(all_logits)):  # 现假设有3,4,5,6个模板
            # proto_logits_all += all_logits[idx] * template_ll_list[idx]
            proto_logits_all += all_logits[idx]  # 如果仅考虑一个Prompt，就不需要再乘以似然了

        return proto_logits_all
