#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch import optim
from tqdm import trange
from openapi.utils import PathManager, gen_batch, FMNIST
from openapi.log import getLogger
from openapi import config

logger = getLogger(__name__)
BATCH_SIZE = 1000
EPOCH = 10
NAME = "LINEAR"


class Linear(nn.Module):
    def __init__(self, cls_num, input_dim):
        # input_dim -> 500 -> 500 -> 256 -> cls_num
        super(Linear, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(input_dim, cls_num)
        self.fc1.to(torch.double)

    def logit(self, x):
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.softmax(self.logit(x))

    def compute_gradient_auto(self, x, clss):
        # Get the gradient
        x_variable = Variable(x, requires_grad=True)
        prob = self.forward(x_variable)
        prob = prob[0, clss]
        prob.backward()
        W_grad = x_variable.grad
        return W_grad.detach()

    def compute_gradient_manual(self, x):
        return self.fc1.weight.detach(), None


def train_model(train_data, train_labels, cls_names, dataset, verbose=False):
    path_manager = PathManager("LINEAR", cls_names, dataset, if_train_set=None)
    logger.info("Train the model: {}".format("-".join(cls_names)))

    train_loader = list(gen_batch(train_data, train_labels, batch_size=BATCH_SIZE))

    model = Linear(len(cls_names), train_data[0, :].size()[0]).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()

    for epoch in trange(1, EPOCH + 1):
        total_loss = 0
        for batch_idx in trange(0, len(train_loader)):
            x_tensor, target_tensor = train_loader[batch_idx]
            optimizer.zero_grad()

            loss = loss_func(model.logit(x_tensor), target_tensor)
            total_loss = total_loss + loss.data.item()

            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 5 == 0 and verbose:
                logger.info(
                    '=> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx + 1, total_loss))

        logger.info("Save model path => {}".format(path_manager.mdl_path()))
        torch.save(model.state_dict(), path_manager.mdl_path())

    logger.info("Finish training")
    return model


def load_model(model_path, cls_num, input_dim):
    if not os.path.isfile(model_path):
        raise Exception("The PLNN model of {} does not exist.".format(model_path))

    mdl = Linear(cls_num, input_dim)
    mdl.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        mdl = mdl.cuda(device=config.DEVICE)  # Send the model to GPU
    return mdl


def evaluate(model, test_data, test_labels, dataset):
    # On the test data set
    total_acc, total = 0, test_data.size()[0]
    test_loader = list(gen_batch(test_data, test_labels, batch_size=BATCH_SIZE))
    for batch_idx, (x_tensor, target) in enumerate(test_loader):
        prob = model(x_tensor)
        p_labels = prob.argmax(dim=1)
        total_acc += (p_labels == target).sum().item()

    precision = total_acc / (1.0 * total)
    logger.info("[{} dataset] Model: LINEAR Accuracy: {}/{}={}".format(dataset, total_acc, total, precision))
    return precision, total, total_acc


def test_cases():
    def train_fmnist():
        IMG_SIZE = 28
        cls_names = ["T-shirt_top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                     "Ankle_boot"]
        fmnist = FMNIST(cls_names)
        path_manager = PathManager(NAME, "FMNIST")
        train_imgs, train_labels, test_imgs, test_labels = fmnist.load_mnist()
        fmnist.display(train_labels, test_labels)
        train_imgs = train_imgs.view(train_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        train_model(train_imgs, train_labels, cls_names, "FMNIST", verbose=True)
        mdl = load_model(path_manager.mdl_path(), len(cls_names), IMG_SIZE * IMG_SIZE)
        evaluate(mdl, train_imgs, train_labels, "train")

    def evaluate_fmnist():
        IMG_SIZE = 28
        cls_names = ["T-shirt_top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                     "Ankle_boot"]
        path_manager = PathManager(NAME, "FMNIST")
        fmnist = FMNIST(cls_names)
        train_imgs, train_labels, test_imgs, test_labels = fmnist.load_mnist()
        fmnist.display(train_labels, test_labels)
        test_imgs = test_imgs.view(test_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        mdl = load_model(path_manager.mdl_path(), len(cls_names), IMG_SIZE * IMG_SIZE)
        evaluate(mdl, test_imgs, test_labels, "test")

    train_fmnist()
    evaluate_fmnist()


if __name__ == '__main__':
    test_cases()
