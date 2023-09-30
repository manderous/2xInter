import torch
import numpy
import numpy as np
import torch.nn as nn
import json
import random
from sklearn.metrics import *
import time
from preprocess.utils import save_json


class FSLTrainer_bart_contra(torch.nn.Module):

    def __init__(self, fsl_model, train_dl, dev_dl, test_dl, template_ll_list, args, wsd_model=None, wsd_dl=None):
        super(FSLTrainer_bart_contra, self).__init__()
        self.fsl_model = fsl_model
        self.wsd_model = wsd_model
        self.wsd_dl = wsd_dl

        self.TN = args.train_way
        self.N = args.way
        self.K = args.shot
        self.Q = args.query
        self.B = args.batch_size
        self.O = args.other

        self.train_setting = (self.B, self.TN + self.O, self.K, self.Q)
        self.eval_setting = (self.B, self.N + self.O, self.K, self.Q)

        self.train_dl = train_dl
        self.dev_dl = dev_dl
        self.test_dl = test_dl

        self.ignore_cuda_feature = ['token', 'label', 'target']

        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.template_ll_list = template_ll_list

    def do_train(self):
        wsd_dl = None
        if self.wsd_dl:
            wsd_dl = iter(self.wsd_dl)

        B, N, K, Q = self.train_setting

        params = [x for x in self.parameters() if x.requires_grad]

        trainable = 0
        for p in params:
            trainable += p.numel()

        print('Trainable params: ', trainable)

        print('Optimizer: SGD')
        # optimizer = torch.optim.SGD(params, lr=self.args.lr)
        optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=0.0005)

        self.fsl_model.train()
        for i, batch in enumerate(self.train_dl):
            for k, v in batch.items():
                if k not in self.ignore_cuda_feature:
                    batch[k] = batch[k].cuda()

            if i == 0:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(k, tuple(v.shape))
                    elif isinstance(v, list):
                        print(k, 'list of ', len(v), type(v[0]))
            optimizer.zero_grad()

            return_item, pos_logits, loss_contra = self.fsl_model(batch, self.train_setting, self.template_ll_list, 'train')
            target = batch['target'].view(-1).cuda()
            logits = return_item.view(-1, N)
            total_loss = 0.0
            loss = self.ce(logits, target)
            total_loss = total_loss + loss + loss_contra

            mutual_loss = None
            wsd_mutual_loss = None
            global_mi_loss = None

            if self.args.alpha > 0.0:
                if 'mutual_loss' in return_item:
                    mutual_loss = self.args.alpha * return_item['mutual_loss']
                    total_loss = total_loss + mutual_loss

            if self.args.beta > 0.0:
                if self.wsd_model is not None:
                    batch = next(wsd_dl, None)
                    if batch is None:
                        wsd_dl = iter(self.wsd_dl)
                        batch = next(wsd_dl, None)
                    for k, v in batch.items():
                        if k not in self.ignore_cuda_feature:
                            batch[k] = batch[k].cuda()
                    wsd_mutual_loss = self.args.beta * self.wsd_mutual(batch)
                    total_loss = total_loss + wsd_mutual_loss

            inter_loss = None
            intra_loss = None
            if self.args.xi > 0.0:
                if 'inter_loss' in return_item:
                    _inter = return_item['inter_loss']
                    scale = loss.detach() / _inter.detach()
                    inter_loss = self.args.xi * _inter * scale
                    total_loss += inter_loss
                if 'intra_loss' in return_item:
                    _intra = return_item['intra_loss']
                    scale = loss.detach() / _intra.detach()
                    intra_loss = self.args.xi * _intra * scale
                    total_loss += intra_loss

            total_loss.backward()
            optimizer.step()

            if self.args.debug:
                train_log_iter = 5
                train_iter = 10
            else:
                train_log_iter = 400
                train_iter = 400

            if i % train_log_iter == 0 and i > 0:
                l = loss.detach().cpu().numpy()
                m_loss = self.get_detach_loss(mutual_loss)
                w_loss = self.get_detach_loss(wsd_mutual_loss)
                ia_loss = self.get_detach_loss(intra_loss)
                ir_loss = self.get_detach_loss(inter_loss)

                output_format = '| @ {} Loss={:.4f} Tree={:.4f} WSD={:.4f} II={:.4f}/{:.4f}'
                print(output_format.format(i, l,
                                           m_loss,
                                           w_loss,
                                           ia_loss,
                                           ir_loss))

            if i % train_iter == 0 and i > 0:
                self.do_eval('dev', i)
                self.do_eval('test', i)
                self.fsl_model.train()
            if i > 6000:
                return

    def get_detach_loss(self, loss):
        if loss:
            return loss.detach().cpu().numpy()
        else:
            return 0.0

    def do_eval(self, part='test', iteration=0):
        if part == 'dev':
            dl = self.dev_dl
        else:
            dl = self.test_dl
        B, N, K, Q = self.eval_setting
        self.fsl_model.eval()

        with torch.no_grad():
            predictions = []
            targets = []
            sample_indices = []
            for i, batch in enumerate(dl):
                for k, v in batch.items():
                    if k not in self.ignore_cuda_feature:
                        batch[k] = batch[k].cuda()
                return_item, pos_logits_all, loss_contra = self.fsl_model(batch, self.eval_setting, self.template_ll_list, 'dev')
                logits = return_item.view(-1, N)
                _pred = torch.argmax(logits, dim=1).view(-1).cpu().numpy()
                _target = batch['target'].view(-1).numpy()

                predictions.append(_pred)
                targets.append(_target)
                sample_indices.append(batch['i'].tolist())

            p, r, f = self.metrics(targets, predictions)
            print('-> {:4s}: {:6.2f} {:6.2f} {:6.2f}'.format(part, p, r, f))

        # Save for further analysize
        predictions = np.stack(predictions).tolist()
        targets = np.stack(targets).tolist()
        item = {'iteration': iteration,
                'sample': sample_indices,
                'target': targets,
                'prediction': predictions}
        path = 'checkpoints/rams/proto_5_5/{}.{}.{}.json'.format(self.args.ex, part, iteration)  # rams
        # path = 'checkpoints/lowkbp/lowkbp0/proto_5_5/{}.{}.{}.json'.format(self.args.ex, part, iteration)  # lowkbp
        # path = 'checkpoints/ace/InterIntra_5_5/{}.{}.{}.json'.format(self.args.ex, part, iteration)  # ace
        save_json(item, path)

    def wsd_mutual(self, batch):
        p_gcn = self.wsd_model.forward_with_external_encoder(batch, self.fsl_model.encoder)
        p_wsd = self.wsd_model(batch)
        p = torch.softmax(p_gcn['logit'], dim=-1)
        q = torch.softmax(p_wsd['logit'], dim=-1)
        mutual_loss = torch.sum((p - q) ** 2)
        return mutual_loss

    def metrics(self, targets, predictions):

        def avg(x):
            return sum(x) / len(x)

        precisions, recalls, fscores = [], [], []
        labels = [x for x in range(self.O, self.N + self.O)]  # works for either self.O = O or 1
        # print('Evaluation label set: ', labels)
        for _t, _p in zip(targets, predictions):
            p = 100 * precision_score(_t, _p, labels=labels, average='micro', zero_division=0)
            r = 100 * recall_score(_t, _p, labels=labels, average='micro', zero_division=0)
            f = 100 * f1_score(_t, _p, labels=labels, average='micro', zero_division=0)
            precisions.append(p)
            recalls.append(r)
            fscores.append(f)
        return avg(precisions), avg(recalls), avg(fscores)

    def fscore(self, targets, predictions):
        targets = numpy.concatenate(targets)
        predictions = numpy.concatenate(predictions)
        zeros = np.zeros(predictions.shape, dtype='int')
        numPred = np.sum(np.not_equal(predictions, zeros))
        numKey = np.sum(np.not_equal(targets, zeros))
        predictedIds = np.nonzero(predictions)
        preds_eval = predictions[predictedIds]
        keys_eval = targets[predictedIds]
        correct = np.sum(np.equal(preds_eval, keys_eval))
        # print('correct : {}, numPred : {}, numKey : {}'.format(correct, numPred, numKey))
        precision = 100.0 * correct / numPred if numPred > 0 else 0.0
        recall = 100.0 * correct / numKey
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0. else 0.0
        return precision, recall, f1
