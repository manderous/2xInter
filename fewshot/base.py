import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class FSLBaseModel(nn.Module):

    def __init__(self, encoder, args):
        super(FSLBaseModel, self).__init__()
        self.args = args
        self.device = args.device
        self.encoder = encoder
        self.global_mutual_ce = nn.CrossEntropyLoss()

    def init_weight(self):
        # self.global_mutual.apply(init_weights)
        pass

    def global_mutual_loss_v1(self, embedding, setting):
        B, N, K, Q = setting

        embedding = embedding.view(B, N, K + Q, -1)  # (B,N,K+Q,D)
        e1 = embedding.view(B, N, 1, K + Q, -1).expand(-1, -1, N, -1, -1)  # (B,N, N,K+Q,D)
        e2 = embedding.view(B, 1, N, K + Q, -1).expand(-1, N, -1, -1, -1)  # (B,N, N,K+Q,D)

        z = torch.cat([e1, e2], dim=-1)  # (B,N, N, K+Q, 2D)

        eye = torch.eye(N, N, dtype=torch.int64).view(1, N, N, 1, 1).expand(B, N, N, K + Q, 1).contiguous().to(
            self.device)

        z_scores = self.global_mutual(z)

        mi_loss = self.global_mutual_ce(z_scores.view(-1, 2), eye.view(-1))
        return mi_loss

    def global_mutual_loss_v2(self, e1, e2, setting):
        B, N, K, Q = setting

        e1 = e1.view(B, N, K + Q, -1)
        e2 = e2.view(B, N, K + Q, -1)

        slices = [(i + 1) % N for i in range(N)]

        e3 = e1[:, slices, :, :]

        z_z = torch.cat([e1, e2], dim=-1)  # (B,N, N, K+Q, 2D)
        z_t = torch.cat([e1, e3], dim=-1)  # (B,N, N, K+Q, 2D)

        z_z_scores = self.global_mutual(z_z).view(-1, 2)
        z_t_scores = self.global_mutual(z_t).view(-1, 2)

        logits = torch.cat([z_z_scores, z_t_scores], dim=0)
        zeros = torch.zeros(z_z_scores.shape[0], dtype=torch.int64).to(self.device)
        ones = torch.ones(z_z_scores.shape[0], dtype=torch.int64).to(self.device)
        targets = torch.cat([ones, zeros], dim=0)

        mi_loss = self.global_mutual_ce(logits, targets)
        return mi_loss

    def global_mutual_loss_v3(self, e1, e2, setting):
        B, N, K, Q = setting

        e1 = e1.view(B, N, K + Q, -1)
        e2 = e2.view(B, N, K + Q, -1)

        e1 = e1[:, :, :K, :]
        e2 = e2[:, :, :K, :]

        slices = [(i + 1) % N for i in range(N)]

        e3 = e1[:, slices, :, :]

        z_z = torch.cat([e1, e2], dim=-1)  # (B,N, N, K+Q, 2D)
        z_t = torch.cat([e1, e3], dim=-1)  # (B,N, N, K+Q, 2D)

        z_z_scores = self.global_mutual(z_z).view(-1, 2)
        z_t_scores = self.global_mutual(z_t).view(-1, 2)

        logits = torch.cat([z_z_scores, z_t_scores], dim=0)
        zeros = torch.zeros(z_z_scores.shape[0], dtype=torch.int64).to(self.device)
        ones = torch.ones(z_z_scores.shape[0], dtype=torch.int64).to(self.device)
        targets = torch.cat([ones, zeros], dim=0)

        mi_loss = self.global_mutual_ce(logits, targets)
        return mi_loss

    def global_mutual_loss(self, e1, e2, setting):
        B, N, K, Q = setting

        slices = [(i + 1) % N for i in range(N)]
        e3 = e2.view(B, N, K + Q, -1)[:, slices, :, :]

        m1 = self.global_mutual(e1)
        m2 = self.global_mutual(e2)
        m3 = self.global_mutual(e3).view(-1, self.encoder.hidden_size)

        # p1 = F.softmax(m1, dim=1)
        p2 = F.softmax(m2, dim=1)
        p3 = F.softmax(m3, dim=1)

        mutual_loss = -torch.sum(p2 * ((p2 / p3).log()))
        return mutual_loss

    def openset_loss(self, embedding, setting):
        """

        :param embedding: B, N, K+Q, D
        :param setting: tupple(B, N, K, Q)
        :return:
        """
        B, N, K, Q = setting

        # print('| Proto > embedding', tuple(embedding.shape))

        support = embedding[:, 1:, :K, :]  # B, N, K, D
        negative = embedding[:, 0, :, :]  # B,K+Q,D

        # print('| Proto > support', tuple(support.shape))
        # print('| Proto > negative', tuple(negative.shape))

        prototypes = support.mean(dim=2).unsqueeze(dim=1)  # B,1, N,D

        # print('| Proto > prototypes', tuple(prototypes.shape))

        negative = negative.unsqueeze(dim=2)  # B,KQ,1,D
        # print('| Proto > negative', tuple(negative.shape))

        error = negative - prototypes   # B,K+Q,N,D
        distances = torch.sum(torch.pow(error, 2), dim=3)  # B, K+Q, N
        logits = torch.softmax(distances, dim=2) + 1e-10   # B, K+Q, N
        return torch.mean(logits * torch.log(logits))
