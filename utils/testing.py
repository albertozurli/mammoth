from collections import Counter

import torch
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
import numpy as np
import itertools
import time


def load_model(model, args):
    dataset_name = getattr(args, 'dataset')[4:]
    model_name = getattr(args, 'model')
    if args.subclass:
        if args.aux:
            model.load_state_dict(torch.load(f"data/saved_model/{dataset_name}/{model_name}/model_sub_aux.pth.tar"))
        else:
            model.load_state_dict(torch.load(f"data/saved_model/{dataset_name}/{model_name}/model_sub.pth.tar"))
    else:
        model.load_state_dict(torch.load(f"data/saved_model/{dataset_name}/{model_name}/model.pth.tar"))

    return model


def eval100(model: ContinualModel, args, last=False):
    labels_lists = [[] for i in range(100)]
    model.net.to(model.device)
    model.net.eval()

    args.dataset = "seq-cifar100"
    dataset = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset.get_data_loaders()

    # Save for each C100 class the logit head activated
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                for idx, val in enumerate(labels):
                    labels_lists[val.item()].append(pred[idx].item())
        print(f"Task {k} test completed")

    # Create a mapping for each (sub)class the head activated most
    mapping_dict = {}
    for index, l in enumerate(labels_lists):
        c = Counter(l)
        arg_max = max(c, key=c.get)
        mapping_dict[index] = arg_max

    # Made another pass on test set with new mapping to compute accuracy
    correct, total = 0, 0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                labels = [mapping_dict[l] for l in labels.tolist()]
                pred = pred.tolist()
                for x, y in zip(pred, labels):
                    if x == y:
                        correct += 1

                total += len(labels)

    print(f"Accuracy on CIFAR100: {correct / total * 100}")
    print("Test Completed")



    # TODO Impossible to generate and eval all possible permutations, also on HPC cluster, too much memory required for 100! permutations
    # TODO Maybe we can eval pemutations in each superclass (5!*20) or on each task (2 superclasses --> 10!*10)

    # start = time.time()
    # first_list = list(np.arange(100))
    # permutations = list(itertools.permutations(first_list))
    #
    # max_acc = 0
    # best_perm = []
    # for n, p in enumerate(permutations):
    #     correct, total = 0, 0
    #     for k, test_loader in enumerate(dataset.test_loaders):
    #         if last and k < len(dataset.test_loaders) - 1:
    #             continue
    #         for data in test_loader:
    #             with torch.no_grad():
    #                 inputs, labels = data
    #                 inputs, labels = inputs.to(model.device), labels.to(model.device)
    #                 outputs = model(inputs)
    #
    #                 _, pred = torch.max(outputs.data, 1)
    #                 labels = [p[l] for l in labels.tolist()]
    #                 pred = pred.tolist()
    #                 for x, y in zip(pred, labels):
    #                     if x == y:
    #                         correct += 1
    #
    #                 total += len(labels)
    #     acc = correct / total * 100
    #     if acc > max_acc:
    #         best_perm = p
    #         max_acc = acc
    #     print(f"Permutation {n} completed with accuracy {acc}%")
    #
    # print(f"\n\nBest accuracy: {max_acc}%")
    # end = time.time()
    # print(f"Elapsed time: {start - end}s")
    # print("Eval completed")
