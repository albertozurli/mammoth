from collections import Counter

import numpy.lib.recfunctions
import torch
import pickle
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import matplotlib.pyplot as plt
import numpy as np


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


def save_lists(filename, listname):
    with open(filename, 'wb') as handle:
        pickle.dump(listname, handle)


def load_list(filename):
    with open(filename, 'rb') as handle:
        lst = pickle.load(handle)
    return lst


def eval100(model: ContinualModel, args, last=False):
    example_list, label_list = [], []
    model.net.to(model.device)
    model.net.eval()

    args.dataset = "seq-cifar100"
    dataset = get_dataset(args)

    # Generate training set couple examples/labels
    for t in range(dataset.N_TASKS):
        train_loader, _ = dataset.get_data_loaders()
        for data in train_loader:
            with torch.no_grad():
                inputs, labels, _ = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                for x, y in zip(outputs, labels):
                    example_list.append(x)
                    label_list.append(y.item())
        print(f"Task {t} train completed")
    save_lists('train_examples.pickle', example_list)
    save_lists('train_labels.pickle', label_list)

    # Generate training set couple examples/labels
    example_list, label_list = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                for x, y in zip(outputs, labels):
                    example_list.append(x)
                    label_list.append(y.item())
        print(f"Task {k} test completed")

    save_lists('test_examples.pickle', example_list)
    save_lists('test_labels.pickle', label_list)

    x_train = load_list('train_examples.pickle')
    x_train = [i.cpu().numpy() for i in x_train]
    x_train = np.vstack(x_train)
    x_test = load_list('test_examples.pickle')
    x_test = [i.cpu().numpy() for i in x_test]
    x_test = np.vstack(x_test)

    feature_list = [[] for i in range(x_train.shape[1])]
    for f in range(x_train.shape[1]):
        feature_list[f] = x_train[:,f]
    mean_list = [np.mean(feature) for feature in feature_list]
    std_list = [np.std(feature) for feature in feature_list]
    for idx,l in enumerate(feature_list):
        feature_list[idx] = (l-mean_list[idx])/std_list[idx]
    x_train = np.transpose(np.vstack(feature_list))

    feature_list = [[] for i in range(x_test.shape[1])]
    for f in range(x_test.shape[1]):
        feature_list[f] = x_test[:, f]
    for idx, l in enumerate(feature_list):
        feature_list[idx] = (l - mean_list[idx]) / std_list[idx]
    x_test = np.transpose(np.vstack(feature_list))

    y_train = load_list('train_labels.pickle')
    y_test = load_list('test_labels.pickle')

    knn = KNeighborsClassifier(n_neighbors=64)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

