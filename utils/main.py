# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy  # needed (don't change it)
import importlib
import os
import sys
import socket

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime


def top5_examples(model, dataset, path):
    # Load model + initialize 20 list of tuples( one for each superclass)
    model.load_state_dict(torch.load(path))
    model.to(model.device)
    tuple_list = [[] for i in range(20)]
    tmp = (0, 0)
    for s in tuple_list:
        for i in range(5):
            s.append(tmp)

    with torch.no_grad():
        for t in range(dataset.N_TASKS):
            _, test_loader = dataset.get_data_loaders()
            for i, data in enumerate(tqdm(test_loader)):
                inputs, _ = data
                inputs = inputs.to(model.device)
                outputs = model(inputs)
                class_outputs = class_logits_from_subclass_logits(outputs, 20)
                for n, logit_tensor in enumerate(class_outputs):
                    val, idx = torch.max(logit_tensor, 0)
                    val, idx = val.item(), idx.item()
                    if tuple_list[idx][-1][1] < val:
                        tuple_list[idx][-1] = (inputs[n], val)
                        tuple_list[idx].sort(key=lambda tup: tup[1], reverse=True)

    plt.figure(figsize=(50, 50))
    for list_idx in range(20):
        for img_idx in range(5):
            img = tuple_list[list_idx][img_idx][0].cpu().numpy()
            plt.subplot(20, 5, 5 * list_idx + img_idx + 1)
            plt.axis('off')
            plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.tight_layout()
    plt.savefig('activation_superclasses.png')
    plt.show()


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'        
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

        # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    backbone = dataset.get_backbone(args.subclass)
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    # set job name
    setproctitle.setproctitle(
        '{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)

    # model_path = f"data/saved_model/{args.dataset[4:]}/model.pth.tar"
    # top5_examples(model, dataset, model_path)


if __name__ == '__main__':
    main()
