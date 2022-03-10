# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK

    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs, n):

        logits = self.net(inputs)
        logits = logits[:,self.task*self.cpt:(self.task+1)*self.cpt]
        labels_mapped = torch.remainder(labels, self.cpt)

        self.opt.zero_grad()

        loss = self.loss(logits, labels_mapped)
        loss_re = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            buf_logits = self.net(buf_inputs)
            buf_logits = buf_logits[:,:(self.task+1)*self.cpt]
            loss_re = self.loss(buf_logits, buf_labels)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()
