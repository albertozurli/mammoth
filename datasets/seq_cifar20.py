# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import numpy as np


class CIFAR100Super(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Super, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.targets = coarse_labels[self.targets].tolist()

        # update classes
        self.classes = ['aquatic mammals',  # ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        'fish',  # ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        'flowers',  # ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        'food containers',  # ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        'fruit and veg',  # ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        'household electrical devices',  # ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        'household furniture',  # ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        'insects',  # ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        'large carnivores',  # ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        'large man-made outdoor things',  # ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        'large natural outdoor scenes',  # ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        'large omnivores and herbivores',  # ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        'medium - sized mammals',  # ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        'non-insect invertebrates',  # ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        'people',  # ['baby', 'boy', 'girl', 'man', 'woman'],
                        'reptiles',  # ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        'small mammals',  # ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        'trees',  # ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        'vehicles 1',  # ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        'vehicles 2'  # ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
                        ]


class MyCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.targets = coarse_labels[self.targets].tolist()

        # update classes
        self.classes = ['aquatic mammals',  # ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        'fish',  # ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        'flowers',  # ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        'food containers',  # ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        'fruit and veg',  # ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        'household electrical devices',  # ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        'household furniture',  # ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        'insects',  # ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        'large carnivores',  # ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        'large man-made outdoor things',  # ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        'large natural outdoor scenes',  # ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        'large omnivores and herbivores',  # ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        'medium - sized mammals',  # ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        'non-insect invertebrates',  # ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        'people',  # ['baby', 'boy', 'girl', 'man', 'woman'],
                        'reptiles',  # ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        'small mammals',  # ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        'trees',  # ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        'vehicles 1',  # ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        'vehicles 2'  # ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
                        ]

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):
    NAME = 'seq-cifar100-super'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_SUBCLASSES_PER_CLASS = 5
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2615))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100SUPER', train=True,
                                   download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = CIFAR100Super(base_path() + 'CIFAR100SUPER', train=False,
                                          download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100SUPER', train=True,
                                   download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS*SequentialCIFAR100.N_SUBCLASSES_PER_CLASS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform
