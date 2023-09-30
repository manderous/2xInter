from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
import torch.nn as nn
import torch
from .utils import init_weights
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Bart_contra_multipos_neg(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Sequential(nn.Tanh(), nn.Linear(768, config.hidden_size))
        self.kl_loss = nn.KLDivLoss()

        self.layer1_prediction = nn.Sequential(
            nn.Linear(768, config.hidden_size),  # 768
            nn.LayerNorm(config.hidden_size),  # LayerNormalization
            # nn.BatchNorm1d(config.hidden_size),  # BatchNormalization
            # nn.GELU(),
            nn.ReLU(inplace=True),
        )
        self.layer2_prediction = nn.Linear(config.hidden_size, config.hidden_size)

        self.layer1_projection = nn.Sequential(
            nn.Linear(768, config.hidden_size),
            nn.LayerNorm(config.hidden_size),  # LayerNormalization
            # nn.BatchNorm1d(config.hidden_size),  # BatchNormalization
            # nn.GELU(),
            nn.ReLU(inplace=True),
        )
        self.layer2_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),  # LayerNormalization
            # nn.BatchNorm1d(config.hidden_size),  # BatchNormalization
            # nn.GELU(),
            nn.ReLU(inplace=True),
        )
        self.layer3_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),  # LayerNormalization
            # nn.BatchNorm1d(config.hidden_size),  # BatchNormalization
        )

    def select_anchor(self, emb, anchor_index):
        B, L, D = emb.shape
        u = torch.tensor([x for x in range(L)]).unsqueeze(0).to(self.device)
        v = anchor_index.view(B, 1)
        mask = (u == v).unsqueeze(dim=2).to(self.device)
        x = torch.masked_select(emb, mask).view(-1, D)
        return x

    def init_weight(self):
        self.fc1.apply(init_weights)

    def distance(self, p, z):
        z = z.detach()  # stop gradient
        softmax_Z = F.softmax(z, dim=1)
        log_softmax_P = F.log_softmax(p, dim=1)
        ce_loss = -torch.sum(softmax_Z * log_softmax_P, dim=1).mean()
        return ce_loss

    def forward(self, B, N, K, Q, batch, train_flag):
        enc_input_ids = batch['enc_input_ids']
        enc_mask_ids = batch['enc_mask_ids']

        # Prompt intervention
        pos_decoder_context = []
        for pos_id in range(3):
            context_outputs = self.model(
                enc_input_ids,
                attention_mask=enc_mask_ids,
                return_dict=True,
            )
            if pos_id == 0:
                decoder_context = context_outputs.encoder_last_hidden_state
            else:
                pos_decoder_context.append(context_outputs.encoder_last_hidden_state)

        pos_logits = [0, 0]
        loss_contra = 0.0
        if train_flag == 'train':
            preject_logits = []
            predict_logits = []
            for pos_id in range(len(pos_decoder_context)):
                dec_prompt_ids = batch['dec_prompt_ids_0']
                dec_prompt_mask_ids = batch['dec_prompt_mask_ids_0']
                mask_index = batch['mask_index_0']

                decoder_prompt_outputs = self.model.decoder(
                    input_ids=dec_prompt_ids,
                    attention_mask=dec_prompt_mask_ids,
                    encoder_hidden_states=decoder_context,
                    encoder_attention_mask=enc_mask_ids,
                )
                decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state
                rep = self.select_anchor(self.dropout(decoder_prompt_outputs), mask_index)

                # projection_MLP
                rep_project = self.layer1_projection(rep)
                rep_project = self.layer2_projection(rep_project)
                rep_project = self.layer3_projection(rep_project)
                preject_logits.append(rep_project)

                # prediction_MLP
                rep_predict = self.layer1_prediction(rep_project)
                rep_predict = self.layer2_prediction(rep_predict)
                predict_logits.append(rep_predict)

            # Contrastive loss
            loss_contra = self.distance(predict_logits[0], preject_logits[1]) / 2 + \
                          self.distance(predict_logits[1], preject_logits[0]) / 2

        # Instance intervention
        all_logits = []
        for tem_id in range(3):
            dec_prompt_ids = batch['dec_prompt_ids_{}'.format(str(tem_id))]
            dec_prompt_mask_ids = batch['dec_prompt_mask_ids_{}'.format(str(tem_id))]
            mask_index = batch['mask_index_{}'.format(str(tem_id))]

            decoder_prompt_outputs = self.model.decoder(
                    input_ids=dec_prompt_ids,
                    attention_mask=dec_prompt_mask_ids,
                    encoder_hidden_states=decoder_context,
                    encoder_attention_mask=enc_mask_ids,
            )
            decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state
            rep = self.select_anchor(self.dropout(decoder_prompt_outputs), mask_index)
            embeddings = self.fc1(rep)
            D = embeddings.shape[-1]
            embeddings = embeddings.view(B, N, K + Q, D)
            support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
            query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

            prototypes = support.mean(dim=2)  # ->  B x N x D
            unsqueeze_prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D

            query = query.view(B, N * Q, D)  # ->  B x NQ x D
            query = query.unsqueeze(dim=2)  # ->  B x NQ x 1 x D

            error = query - unsqueeze_prototypes  # B x NQ x N x D
            logits = -torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N
            all_logits.append(torch.squeeze(logits, 0))

        return all_logits, pos_logits, loss_contra
