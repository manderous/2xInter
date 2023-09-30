import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import random
import datetime

from custom_dataset import *
from argument_parser import *
from fewshot_bart.trainer_bart_contra import FSLTrainer_bart_contra
from transformers import BartConfig


def main(args):
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda'
    args.max_length = dataset_constant[args.dataset]['max_length']
    args.n_class = dataset_constant[args.dataset]['n_class']
    args.train_way = dataset_constant[args.dataset]['train_way']
    args.other = dataset_constant[args.dataset]['other']

    B = args.batch_size  # default = 4
    TN = args.train_way  # default = 20
    N = args.way
    K = args.shot  # default = 5
    Q = args.query  # default = 5
    O = args.other

    current_time = str(datetime.datetime.now().time())
    args.log_dir = 'logs/{}-{}-way-{}-shot-{}'.format(args.model, args.way, args.shot, current_time)

    print('Before load')
    feature_set = feature_map[args.encoder]
    train_dl = FSLDataset_bart(TN, K, Q, O,
                               features=feature_set,
                               length=1000000,
                               prefix='datasets/{}/template_argument_multi_bart/train'.format(args.dataset))
    dev_dl = FSLDataset_bart(N, K, Q, O,
                             features=feature_set,
                             length=500,
                             prefix='datasets/{}/template_argument_multi_bart/dev'.format(args.dataset))
    test_dl = FSLDataset_bart(N, K, Q, O,
                              features=feature_set,
                              length=500,
                              prefix='datasets/{}/template_argument_multi_bart/test'.format(args.dataset))

    train_dl = DataLoader(train_dl, batch_size=B, num_workers=0, collate_fn=FSLDataset.fsl_pack)
    dev_dl = DataLoader(dev_dl, batch_size=B, num_workers=0, collate_fn=FSLDataset.fsl_pack, shuffle=False)
    test_dl = DataLoader(test_dl, batch_size=B, num_workers=0, collate_fn=FSLDataset.fsl_pack, shuffle=False)

    print('-' * 80)
    for k, v in args.__dict__.items():
        print('{}\t{}'.format(k, v))
    print('-' * 80)

    config = BartConfig.from_pretrained(args.bert_pretrained)
    encoder = encoder_class[args.encoder].from_pretrained(args.bert_pretrained, config=config)
    encoder.init_weight()

    fsl_model = fsl_class[args.model](encoder, args)
    fsl_model.cuda()

    wsd_model = None
    wsd_train_dl = None
    # ace, lowkbp0, lowkbp1, lowkbp2, lowkbp3, lowkbp4, rams
    with open('datasets/lowkbp0/template_argument_multi_bart/template_log_likelihood.txt') as f_template_ll:
        template_ll_str = f_template_ll.read()
    template_ll_list = [float(ll) for ll in template_ll_str.split('\t')]

    fsl_trainer = FSLTrainer_bart_contra(fsl_model, train_dl, dev_dl, test_dl, template_ll_list, args, wsd_model, wsd_train_dl)
    fsl_trainer.do_train()


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
