# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.distill import class_logits_from_subclass_logits, aux_loss
from datasets import get_dataset


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class DerppACE(ContinualModel):
    NAME = 'derpp_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(DerppACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK

    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs,n):

        logits = self.net(inputs)
        logits_mapped = logits[:, self.task * self.cpt:(self.task + 1) * self.cpt]
        labels_mapped = torch.remainder(labels, self.cpt)

        self.opt.zero_grad()
        loss = self.loss(logits_mapped, labels_mapped)

        # if self.task > 0:
        #     masked_logits = class_logits.masked_fill(mask == 0, torch.finfo(class_logits.dtype).min)
        #     loss = self.loss(masked_logits, labels)
        # else:
        #     loss = self.loss(class_logits, labels)

        # if self.args.aux:
        #     auxiliary = aux_loss(logits,self.device)
        #     loss += (self.args.aux_weight * auxiliary.squeeze())

        loss_re = torch.tensor(0.)
        loss_mse = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buf_outputs = buf_outputs[:, :(self.task + 1) * self.cpt]
            loss_re = self.args.beta * F.cross_entropy(buf_outputs,buf_labels)

            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss = loss + loss_re + loss_mse
        loss.backward()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=logits.data)

        return loss.item()
