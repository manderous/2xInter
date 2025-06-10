from fewshot.base import *


class PrototypicalNetwork(FSLBaseModel):
    def __init__(self, encoder, args):
        super(PrototypicalNetwork, self).__init__(encoder, args)

    def euclidean(self, s1, s2):
        return torch.sum(torch.pow(s1 - s2, 2))

    def cosine(self, s1, s2):
        return torch.sum(torch.nn.functional.cosine_similarity(s1, s2, -1))

    def intra_loss(self, support):
        B, N, K, D = support.shape

        support = support.view(-1, K, D)  # BN x K x D
        s1 = support.unsqueeze(dim=1).expand(-1, K, -1, -1)  # BN x K x K x D
        s2 = support.unsqueeze(dim=2).expand(-1, -1, K, -1)  # BN x K x K x D
        return self.euclidean(s1, s2)

    def inter_loss(self, prototypes):
        B, N, D = prototypes.shape
        s1 = prototypes.unsqueeze(dim=1).expand(-1, N, -1, -1)  # B x N x N x D
        s2 = prototypes.unsqueeze(dim=2).expand(-1, -1, N, -1)  # B x N x N x D
        return self.cosine(s1, s2)

    def forward(self, batch, setting):
        _, N, K, Q = setting
        B = int(batch['length'].shape[0] / N / (K + Q))

        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        # print('| Proto > embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        # print('| Proto > support', tuple(support.shape))
        # print('| Proto > query', tuple(query.shape))

        prototypes = support.mean(dim=2)  # ->  B x N x D
        unsqueeze_prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D
        # print('| Proto > prototypes', tuple(prototypes.shape))

        query = query.view(B, N * Q, D)  # ->  B x NQ x D
        # print('| Proto > query', tuple(query.shape))

        query = query.unsqueeze(dim=2)  # ->  B x NQ x 1 x D
        # print('| Proto > query', tuple(query.shape))

        error = query - unsqueeze_prototypes  # B x NQ x N x D
        logits = -torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N

        # print(torch.mean(logits))

        return_item = {
            'logit': logits,
            # 'global_mutual_loss': self.global_mutual_loss(encoded['pool1'], encoded['pool2'], setting),
            'openset_loss': self.openset_loss(embeddings, setting),
            'intra_loss': self.intra_loss(support),
            'inter_loss': self.inter_loss(prototypes)
        }

        for k, v in encoded.items():
            if k != 'embedding':
                return_item[k] = v
        return return_item
