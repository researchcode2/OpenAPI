#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import torch.nn as nn
import torch
import torch.nn.functional as F
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torch.autograd import Variable
from torch import optim
from tqdm import trange
from openapi.utils import PathManager, gen_batch
from openapi.log import getLogger
from openapi import config

logger = getLogger(__name__)
BATCH_SIZE = 1000
EPOCH = 10


class MLPNet(nn.Module):
    NAME = "MLP"

    def __init__(self, cls_num, input_dim):
        # input_dim -> 500 -> 500 -> 256 -> cls_num
        super(MLPNet, self).__init__()
        self.cls_num = cls_num
        self.input_dim = input_dim
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc1.to(torch.double)
        self.fc2 = nn.Linear(256, 128)
        self.fc2.to(torch.double)
        self.fc3 = nn.Linear(128, 100)
        self.fc3.to(torch.double)
        self.fc4 = nn.Linear(100, cls_num)
        self.fc4.to(torch.double)

    def logit(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def forward(self, x):
        return self.softmax(self.logit(x))

    def compute_decision_feature(self, x, cls):
        W = self.compute_corep_and_polytope(x)[0]
        decision = W - W[cls, :]
        decision = torch.sum(decision, dim=0).unsqueeze(0)
        return -decision / (self.cls_num - 1)

    def compute_gradient_auto(self, x, clss):
        # Get the gradient
        x_variable = Variable(x, requires_grad=True)
        prob = self.forward(x_variable)
        prob = prob[0, clss]
        prob.backward()
        W_grad = x_variable.grad
        return W_grad

    def compute_corep_and_polytope(self, x):
        W_1 = self.fc1.weight.data
        z_2 = self.fc1(x)
        r_2 = (F.relu(z_2) > 0).type(torch.float64)
        a_2 = F.relu(z_2)

        W_2 = self.fc2.weight.data
        z_3 = self.fc2(a_2)
        r_3 = (F.relu(z_3) > 0).type(torch.float64)
        a_3 = F.relu(z_3)

        W_3 = self.fc3.weight.data
        z_4 = self.fc3(a_3)
        r_4 = (F.relu(z_4) > 0).type(torch.float64)

        W_4 = self.fc4.weight.data

        W_2_h = W_2 * r_2
        W_3_h = W_3 * r_3
        W_4_h = W_4 * r_4

        W_comp = W_4_h

        for W in [W_3_h, W_2_h, W_1]:
            W_comp = torch.mm(W_comp, W)
        # Feature weights & Polytope
        return W_comp, torch.cat((r_2, r_3, r_4), dim=1)

    def integrated_gradient_random(self, x_tensor, clss, steps=50, init_num=10):
        integrated_grads = []
        for _ in range(init_num):
            baseline = torch.zeros(x_tensor.size(), device=config.DEVICE, dtype=torch.float64).uniform_(0, 1)
            scaled_inputs = [baseline + (float(i) / steps) * (x_tensor - baseline) for i in range(0, steps + 1)]
            grads = [self.compute_gradient_auto(i, clss) for i in scaled_inputs]
            avg_grads = torch.mean(torch.cat(grads, dim=0), dim=0).unsqueeze()
            assert avg_grads.size() == x_tensor.size()
            integrated_grads.append((x_tensor - baseline) * avg_grads)  # shape: <inp.shape>
        integrated_grads = torch.mean(torch.cat(integrated_grads, dim=0), dim=0).unsqueeze()
        assert integrated_grads.size() == x_tensor.size()
        return integrated_grads

    def integrated_gradient(self, x_tensor, clss, steps=5):
        baseline = 0 * torch.zeros(x_tensor.size(), device=config.DEVICE, dtype=torch.float64)
        scaled_inputs = [baseline + (float(i) / steps) * (x_tensor - baseline) for i in range(0, steps + 1)]
        grads = [self.compute_gradient_auto(i, clss) for i in scaled_inputs]
        avg_grads = torch.mean(torch.cat(grads, dim=0), dim=0).unsqueeze(0)
        assert avg_grads.size() == x_tensor.size()
        integrated_gradients = (x_tensor - baseline) * avg_grads  # shape: <inp.shape>
        return integrated_gradients

    def gradient_input(self, x_tensor, clss):
        grad = self.compute_gradient_auto(x_tensor, clss)
        return grad * x_tensor

    def saliency_map(self, x_tensor, clss):
        grad = abs(self.compute_gradient_auto(x_tensor, clss))
        return grad * x_tensor

    def lime_interpret(self, x_tensor, clss):
        def predict_fun(x):
            # ====================================================================================
            # The original gray image is change to RGB image by LIME by default.
            # To use the original classifier, we need to remove the channels added by LIME
            x = torch.tensor(x[:, :, :, 0], device=config.DEVICE, dtype=torch.float64).view(-1, self.input_dim)
            # ====================================================================================
            rst = self.forward(x).detach().cpu().numpy()  # Output 1 * 2 array
            return rst

        # ====================================================================================
        # Each pixel is separated as a single segmentation
        segmenter = SegmentationAlgorithm("quickshift", kernel_size=1, max_dist=0.0001, ratio=0.2)
        # ====================================================================================
        var_num = x_tensor.size()[1]
        x_tensor = x_tensor.view(28, 28)
        explainer = lime_image.LimeImageExplainer(feature_selection='none')
        explanation = explainer.explain_instance(x_tensor.cpu().numpy(), predict_fun, top_labels=None, hide_color=0,
                                                 num_samples=var_num + 2, num_features=var_num,
                                                 segmentation_fn=segmenter, labels=(clss,))

        w_lime = sorted(explanation.local_exp[clss], key=lambda i: i[0])
        w_lime = torch.tensor([v for _, v in w_lime], dtype=torch.float64, device=config.DEVICE).unsqueeze(0)
        return w_lime


def train_model(train_data, train_labels, cls_names, dataset, verbose=False):
    path_manager = PathManager(MLPNet.NAME, dataset)
    logger.info("Train the model: {}".format("-".join(cls_names)))

    train_loader = list(gen_batch(train_data, train_labels, batch_size=BATCH_SIZE))

    model = MLPNet(len(cls_names), train_data[0, :].size()[0]).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
    logger.info("Load model {}".format(model_path))
    mdl = MLPNet(cls_num, input_dim)
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
    logger.info("[{} dataset] Model: MLP Accuracy: {}/{}={}".format(dataset, total_acc, total, precision))
    return precision, total, total_acc


def test():
    def train_fmnist():
        from openapi.utils import FMNIST
        IMG_SIZE = 28
        # cls_names = ["Shirt", "Pullover", "Coat"]
        cls_names = FMNIST._ID2STR
        fmnist = FMNIST(cls_names)
        path_manager = PathManager(MLPNet.NAME, FMNIST.NAME)
        train_imgs, train_labels, test_imgs, test_labels = fmnist.load_mnist()
        fmnist.display(train_labels, test_labels)
        train_imgs = train_imgs.view(train_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        train_model(train_imgs, train_labels, cls_names, "FMNIST", verbose=True)
        mdl = load_model(path_manager.mdl_path(), len(cls_names), IMG_SIZE * IMG_SIZE)
        evaluate(mdl, train_imgs, train_labels, "train")

    def evaluate_fmnist():
        from openapi.utils import FMNIST
        IMG_SIZE = 28
        # cls_names = ["Shirt", "Pullover", "Coat"]
        cls_names = FMNIST._ID2STR
        path_manager = PathManager(MLPNet.NAME, FMNIST.NAME)
        fmnist = FMNIST(cls_names)
        _, _, test_imgs, test_labels = fmnist.load_mnist()
        test_imgs = test_imgs.view(test_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        mdl = load_model(path_manager.mdl_path(), len(cls_names), IMG_SIZE * IMG_SIZE)
        evaluate(mdl, test_imgs, test_labels, "test")

    def train_mnist():
        from openapi.utils import MNIST
        IMG_SIZE = 28
        cls_names = MNIST._ID2STR
        mnist = MNIST(cls_names)
        path_manager = PathManager(MLPNet.NAME, MNIST.NAME)
        train_imgs, train_labels, test_imgs, test_labels = mnist.load_mnist()
        mnist.display(train_labels, test_labels)
        train_imgs = train_imgs.view(train_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        train_model(train_imgs, train_labels, cls_names, MNIST.NAME, verbose=True)
        mdl = load_model(path_manager.mdl_path(), len(cls_names), IMG_SIZE * IMG_SIZE)
        evaluate(mdl, train_imgs, train_labels, "train")

    def evaluate_mnist():
        from openapi.utils import MNIST
        IMG_SIZE = 28
        cls_names = MNIST._ID2STR
        path_manager = PathManager(MLPNet.NAME, MNIST.NAME)
        mnist = MNIST(cls_names)
        _, _, test_imgs, test_labels = mnist.load_mnist()
        test_imgs = test_imgs.view(test_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        mdl = load_model(path_manager.mdl_path(), len(cls_names), IMG_SIZE * IMG_SIZE)
        evaluate(mdl, test_imgs, test_labels, "test")

    train_fmnist()
    evaluate_fmnist()

    train_mnist()
    evaluate_mnist()


if __name__ == '__main__':
    test()
