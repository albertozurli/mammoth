# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch

Official repository of [Class-Incremental Continual Learning into the eXtended DER-verse](https://arxiv.org/abs/2201.00766) and [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

## WARNING
Ignore `setup.sh` file.
## Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

## Models

+ Experience Replay (ER)
+ ER-ACE
+ Dark Experience Replay (DER)
+ Dark Experience Replay++ (DER++)
+ DER++-ACE

## Datasets

**Class-Il / Task-IL settings**

+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential CIFAR-20 (CIFAR-100 with superclasses)

