#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from collections import defaultdict
import os
import torch
import numpy as np
import pickle
from sklearn.linear_model import Ridge, Lasso
from biplnn.log import getLogger
from biplnn.utils import load_data_new, load_model, PathManager, load_data_new
from biplnn.mdl.lmt import LinearModelTree
from biplnn import config

logger = getLogger(__name__)

IMG_SIZE = 28
MIN_NODE_SIZE = 15
MIN_SPLIT_IMPROVEMENT = 10


def fit_linear_model(x, y):
    logger.info("Using Lasso")
    lr = Lasso(alpha=0.01)
    lr.fit(x, y)
    return SharedScalerModel(lr)


class SharedScalerModel:
    def __init__(self, lm):
        self.lm = lm
        self.coef_ = torch.unsqueeze(torch.tensor(lm.coef_, dtype=torch.float64), dim=0)
        self.intercept_ = lm.intercept_

    def predict(self, X):
        return torch.tensor(self.lm.predict(X), dtype=torch.float64)


def train(c_1, c_2, id2pn, dataset):
    if not os.path.exists("../model"):
        os.mkdir("../model")
    mdl_name = "{}_{}".format(c_1, c_2)
    logger.info("Train the model: {} {}".format(mdl_name, id2pn))
    train_data, train_labels = load_data_new(c_1, c_2, train=True, dataset=dataset)
    train_data = train_data.view(-1, IMG_SIZE * IMG_SIZE).cpu()
    train_labels = torch.tensor([id2pn[i.item()] for i in train_labels], dtype=torch.float64)

    test_data, test_labels = load_data_new(c_1, c_2, train=False, dataset=dataset)
    test_data = test_data.view(-1, IMG_SIZE * IMG_SIZE)
    test_labels = np.array([id2pn[i.item()] for i in test_labels])

    counter = defaultdict(int)
    for i in train_labels:
        counter[i.item()] += 1
    train_pos, train_neg = counter[1], counter[0]
    counter = defaultdict(int)
    for i in test_labels:
        counter[i.item()] += 1
    test_pos, test_neg = counter[1], counter[0]

    logger.info("Train_labels {} ".format(train_labels))
    logger.info("Test_labels {} ".format(test_labels))

    logger.info("""
    ======================================================================
    Data Information
    & \# Positive & \# Negative & \# Positive & \# Negative \\
    \hline
    {} {} & {} & {} & {} & {} \\
    \hline
    ======================================================================
    """.format(c_1, c_2, train_pos, train_neg, test_pos, test_neg))

    logger.info("train_labels {}".format(train_labels))

    lmt = LinearModelTree(MIN_NODE_SIZE, fit_linear_model, min_split_improvement=MIN_SPLIT_IMPROVEMENT)
    lmt.build_tree(train_data, train_labels)
    logger.info("Finish building trees")
    lmt.merge_lrs(lmt.root)
    logger.info("Finish merging trees")

    path_manager = PathManager(mdl_name="LMT", c_1=c_1, c_2=c_2, dataset=dataset, if_train_set=None)
    model_path = path_manager.mdl_path()
    with open(model_path, "wb") as f:
        pickle.dump(lmt, f)

    lmt = load_model(c_1, c_2, dataset=dataset, model_name="LMT")
    train_data = train_data.to(config.DEVICE)
    _test("LMT", lmt, train_data, train_labels, "Trainset")
    _test("LMT", lmt, test_data, test_labels, "Testset")


def _test(mdl_name, lmt, test_data, test_labels, if_train):
    y_pred = lmt.predict_positive(test_data)

    correct = 0
    for i in range(len(test_labels)):
        p_label = 1 if y_pred[i] > 0.5 else 0
        logger.debug("p_label: {} Prob: {} train_label: {}".format(p_label, y_pred[i], test_labels[i]))
        if p_label == test_labels[i]:
            correct += 1
    precision = correct * 1.0 / len(test_labels)
    logger.info("[{} dataset] Model: {} Accuracy: {}/{}={}".format(if_train, mdl_name, correct, len(test_labels), precision))

def load(model_path):
    lmt = pickle.load(open(model_path, "rb"))
    return lmt


def test_1():
    mdl = load_model("Pullover", "Coat", "FMNIST", "LMT")
    images, labels = load_data_new("Pullover", "Coat", train=False, dataset="FMNIST")
    images = images.view(-1, 784)
    forward = mdl.forward(images)
    logger.info("forward.size() => {}".format(forward.size()))
    prob = mdl.predict_positive(images)
    logger.info("prob.size() => {}".format(prob.size()))

if __name__ == '__main__':
    # main()
    # train_main("Pullover", "Coat")
    # test("Pullover", "Coat", FMNIST.id2pn_label(FMNIST.str2id("Pullover"), FMNIST.str2id("Coat")))
    test_1()
