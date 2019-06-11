#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Code adopted and modified from https://medium.com/convoy-tech/the-best-of-both-worlds-linear-model-trees-7c9ce139767d

from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
import random
import time

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from openapi.utils import PathManager
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from openapi import config
from openapi.log import getLogger

_MACHINE_EPSILON = torch.tensor([np.finfo(np.float64).eps], dtype=torch.float64, device=config.DEVICE)

logger = getLogger(__name__)


def weighted_mse(input, target, weights):
    out = (input - target) ** 2
    out = out * weights
    # expand_as because weights are prob not defined for mini-batch
    loss = torch.sum(out)
    return loss


class LocalLasso(nn.Module):
    def __init__(self, var_num, clss_num, alpha=0.01):
        # y -> batch * clss_num
        super(LocalLasso, self).__init__()
        logger.info("Using Lasso")
        self.linear = nn.Linear(var_num, clss_num)
        self.linear.to(torch.float64)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = alpha
        self.clss_num = clss_num

    def fit(self, X, y, weight, parent_mdl):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=0.03)

        prev_loss = 0

        for i in range(100):
            optimizer.zero_grad()
            outputs = self.linear(X)
            all_linear_params = torch.cat([x.view(-1) for x in self.linear.parameters()])
            l1_regularization = self.alpha * torch.norm(all_linear_params, 1)
            loss = weighted_mse(outputs, y, weight) + l1_regularization
            if i % 50 == 49:
                print("Loss", loss)
            loss.backward()
            optimizer.step()
            # Early stop
            if abs(prev_loss - loss) < 1e-8:
                break
            prev_loss = loss

        self.linear.weight.data.detach()
        self.linear.bias.data.detach()
        weight = self.linear.weight.data
        bias = self.linear.bias.data
        norm_weight = (self.clss_num - 1) / self.clss_num * (weight - torch.sum(weight, dim=0) / self.clss_num)
        norm_bias = (self.clss_num - 1) / self.clss_num * (bias - torch.sum(bias, dim=0) / self.clss_num)

        if parent_mdl is not None:
            norm_weight += parent_mdl.linear.weight.data
            norm_bias += parent_mdl.linear.bias.data

        self.linear.weight.data = norm_weight
        self.linear.bias.data = norm_bias

        total = X.size()[0]
        prob = self.predict(X).detach()
        p_labels = prob.argmax(dim=1)
        y = torch.argmax(y, dim=1)
        total_acc = (p_labels == y).sum().item()
        precision = total_acc / (1.0 * total)
        logger.info("Accuracy: {}/{}={}".format(total_acc, total, precision))
        return precision

    def predict(self, X):
        return self.softmax(self.linear(X))


def _update_weights_and_response(y, prob, z_max=1.):
    """Compute the working weights and response for a boosting iteration."""
    # Samples with very certain probabilities (close to 0 or 1) are weighted
    # less than samples with probabilities closer to 1/2. This is to encourage
    # the higher-uncertainty samples to contribute more during model training
    sample_weight = prob * (1. - prob)

    # Don't allow sample weights to be too close to zero for numerical stability
    # (cf. p. 353 in Friedman, Hastie, & Tibshirani (2000)).
    sample_weight = torch.max(sample_weight, 2. * _MACHINE_EPSILON)

    # Compute the regression response z = (y - p) / (p * (1 - p))
    z = torch.where(y > 0, 1. / prob, -1. / (1. - prob))

    # Very negative and very positive values of z are clipped for numerical
    # stability (cf. p. 352 in Friedman, Hastie, & Tibshirani (2000)).
    z = z.clamp(-z_max, z_max)

    return sample_weight, z


class LinearModelTree:
    NAME = "LMT"

    def __init__(self, min_node_size, clss_num, var_num):
        self.min_node_size = min_node_size
        self.path2classifiers = {}
        self.path_len_max = -1
        self.clss_num = clss_num
        self.var_num = var_num
        self.root = None
        self.node = 0

    def flat_tree(self):
        pass

    def build_tree(self, X, y):
        y = nn.functional.one_hot(y)
        logger.info("One-hot y {}".format(y.size()))
        row_num = X.size()[0]
        # Initialize with uniform class probabilities
        prob = torch.full(size=(row_num, self.clss_num), fill_value=1. / self.clss_num,
                          dtype=torch.float64, device=config.DEVICE)

        # Used for debug purpose only. The index of the train examples
        _ids = torch.tensor(range(row_num), dtype=torch.int64)

        self.root = NodeFunction.build_node_recursive(self, X, y, prob, parent_mdl=None, depth=0, ids=_ids)

    def gpu_forward(self, X):
        row_num = X.shape[0]
        ids = torch.tensor(list(range(row_num)), device=config.DEVICE)
        prob = torch.zeros((row_num, self.clss_num), device=config.DEVICE, dtype=torch.float64)
        queue = [(self.root, X, ids)]
        while queue:
            node, X, ids = queue.pop(0)
            if node.feature_idx is not None:
                left_ids = ids[X[:, node.feature_idx] <= node.pivot_value]
                left_X = X[left_ids]
                left_node = node.left
                queue.append((left_node, left_X, left_ids))
                right_ids = ids[X[:, node.feature_idx] > node.pivot_value]
                right_X = X[right_ids]
                right_node = node.right
                queue.append((right_node, right_X, right_ids))
            else:
                local_prob = node.mdl.predict(X).detach()
                prob[ids] = local_prob
        return prob

    def forward(self, X):
        row_num = X.shape[0]
        predicts = []
        for i in range(row_num):
            x_i = X[i, :].view(1, -1)
            path, mdl = Node.get_leaf_mdl(self.root, x_i, i)
            predicts.append(mdl.predict(x_i).detach())
        predicts = torch.cat(predicts, dim=0)
        return predicts

    def compute_gradient_auto(self, x, clss):
        path, mdl = Node.get_leaf_mdl(self.root, x, 0)
        x_variable = Variable(x, requires_grad=True)
        prob = mdl.predict(x_variable)
        prob = prob[0, clss]
        prob.backward()
        W_grad = x_variable.grad
        return W_grad

    def gpu_compute_core_p_and_polytope(self, X):
        row_num = X.shape[0]
        ids = torch.tensor(list(range(row_num)), device=config.DEVICE)
        corep = torch.zeros((row_num, self.var_num), device=config.DEVICE, dtype=torch.float64)
        polytope = torch.zeros((row_num, 51), device=config.DEVICE)

        queue = [(self.root, X, ids, "")]
        while queue:
            node, X, ids, depth = queue.pop(0)
            if node.feature_idx is not None:
                left_ids = ids[X[:, node.feature_idx] <= node.pivot_value]
                left_X = X[left_ids]
                left_node = node.left
                polytope[left_ids, depth] = 0
                queue.append((left_node, left_X, left_ids, depth + 1))
                right_ids = ids[X[:, node.feature_idx] > node.pivot_value]
                right_X = X[right_ids]
                right_node = node.right
                polytope[left_ids, depth] = 1
                queue.append((right_node, right_X, right_ids, depth + 1))
            else:
                corep[ids] = node.mld.linear.weight.data
        return corep, polytope


    def compute_decision_feature(self, x, cls):
        mdl = Node.get_leaf_mdl(self.root, x, 0)[1]
        decision = mdl.linear.weight.data - mdl.linear.weight.data[cls, :]
        decision = torch.sum(decision, dim=0).unsqueeze(0)
        return -decision / (self.clss_num - 1)

    def compute_corep_and_polytope(self, x):
        path, mdl = Node.get_leaf_mdl(self.root, x, 0)
        # Append the paths such that all instances have the same length of path
        # Here we use 51 because we restrict the depth of the tree to 50
        assert len(path) < 51
        while len(path) < 51:
            path.append(0)
        return mdl.linear.weight.data, torch.tensor(path).view(1, -1)

    def node_count(self):
        return self.root.node_count()

    def to(self, device):
        queue = [self.root, ]
        while queue:
            node = queue.pop(0)
            node.mdl = node.mdl.cuda(device)
            if node.feature_idx is not None:
                queue.append(node.left)
                queue.append(node.right)
                node.mdl = None  # Ignore models in the intermediate nodes
                node.pivot_value = node.pivot_value.to(config.DEVICE)

    def integrated_gradient_random(self, x_tensor, clss, steps=50, init_num=10):
        integrated_grads = []
        for _ in range(init_num):
            baseline = torch.zeros(x_tensor.size(), device=config.DEVICE, dtype=torch.float64).uniform_(0, 1)
            scaled_inputs = [baseline + (float(i) / steps) * (x_tensor - baseline) for i in range(0, steps + 1)]
            grads = [self.compute_gradient_auto(i, clss) for i in scaled_inputs]
            avg_grads = torch.mean(torch.cat(grads, dim=0), dim=0).unsqueeze()
            assert avg_grads.size() == x_tensor.size()
            integrated_grads.append( (x_tensor - baseline) * avg_grads )# shape: <inp.shape>
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
            x = torch.tensor(x[:, :, :, 0], device=config.DEVICE, dtype=torch.float64).view(-1, self.var_num)
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


def cal_entropy(clss_dist, count):
    ############################################################
    # DEBUG
    # assert torch.sum(clss_dist) == count, (torch.sum(clss_dist), count)
    ############################################################
    clss_dist = clss_dist / count
    log_dist = torch.log(clss_dist + 1e-12)
    entropy = -torch.sum(clss_dist * log_dist)
    return entropy


def find_best_split(X, Y, feature_idx, tree, _right_dist, _ylabels, entropy):
    best_ig, best_feature, best_feature_pivot = 0, None, None
    row_count = X.size()[0]
    uniq_value = torch.unique(X, sorted=True)
    left_dist = torch.zeros(Y.size()[1], dtype=torch.float64, device=config.DEVICE)
    left_count = 0
    for i, value in enumerate(uniq_value):
        idxs = (X == value)
        _Y = torch.sum(Y[idxs], dim=0).double()
        count = torch.sum(_Y)
        left_count += count
        if left_count < tree.min_node_size:
            continue
        if row_count - left_count < tree.min_node_size:
            break
        left_dist += _Y
        left_entropy = cal_entropy(left_dist, left_count)
        right_entropy = cal_entropy(_right_dist - left_dist, row_count - left_count)
        right_count = row_count - left_count
        ig = entropy - (left_count / row_count) * left_entropy - (right_count / row_count) * right_entropy

        if ig > best_ig:
            best_ig = ig
            best_feature = feature_idx
            best_feature_pivot = (uniq_value[i] + uniq_value[i + 1]) / 2.0

    return best_feature, best_feature_pivot, best_ig


class NodeFunction:
    @staticmethod
    def find_best_split(tree, X, Y):

        logger.debug("Y => {}".format(Y.size()))
        best_ig, best_feature, best_feature_pivot = 0, None, None
        row_count, var_num = X.size()

        if row_count < tree.min_node_size * 2:
            return best_feature, best_feature_pivot

        _right_dist = torch.sum(Y, dim=0).double()
        entropy = cal_entropy(_right_dist, row_count)
        _ylabels = torch.argmax(Y, dim=1)

        st = time.time()
        for i in random.sample(range(var_num), 100):
            _X = X[:, i]
            feature, feature_pivot, ig = find_best_split(_X, Y, i, tree, _right_dist, _ylabels, entropy)
            if best_ig < ig:
                best_ig = ig
                best_feature = feature
                best_feature_pivot = feature_pivot

        ed = time.time()
        logger.info("Time used {} ig {} feature {} value {}".format(ed - st, best_ig, best_feature, best_feature_pivot))

        return best_feature, best_feature_pivot

    @staticmethod
    def split_on_pivot(X, y, feature_idx, pivot_value, probs, ids):
        logger.debug("pivot_value: {}".format(pivot_value))
        left_idxs = X[:, feature_idx] <= pivot_value.item()
        left_probs = probs[left_idxs]
        left_y = y[left_idxs]
        left_X = X[left_idxs]
        left_ids = ids[left_idxs]

        ############################################################
        # DEBUG
        logger.debug("{} {}".format(left_probs.size(), probs.size()))
        assert left_probs.size() != probs.size()
        ############################################################

        right_idxs = X[:, feature_idx] > pivot_value.item()
        right_probs = probs[right_idxs]
        right_y = y[right_idxs]
        right_X = X[right_idxs]
        right_ids = ids[right_idxs]

        return (
            left_probs, left_y, left_X, left_ids,
            right_probs, right_y, right_X, right_ids
        )

    @staticmethod
    def build_node_recursive(tree, X, y, prob, parent_mdl, depth, ids):
        ############################################################
        # For debug purpose only
        # assert torch.sum(X_cpu.clone().to(config.DEVICE) - X) == 0
        # assert torch.sum(y_cpu.clone().to(config.DEVICE) - y) == 0

        logger.info("Build a node. Total: {} Distribution {}".format(y.size()[0], torch.sum(y, dim=0)))
        tree.node += 1
        sample_weight, z = _update_weights_and_response(y, prob)

        # Train model on z (response)
        mdl = LocalLasso(tree.var_num, tree.clss_num)
        mdl.to(config.DEVICE)
        accuracy = mdl.fit(X, z, sample_weight, parent_mdl)

        pred_prob = mdl.predict(X).detach()

        ############################################################
        # Stop criteria used to avoid overfitting
        # 1. If the local model is very accurate.
        # 2. If the depth is larger than 50
        ############################################################
        if accuracy < 0.99 and depth < 50:
            (feature_idx, pivot_value) = NodeFunction.find_best_split(tree, X, y)
        else:
            feature_idx, pivot_value = None, None

        node = Node(feature_idx, pivot_value, mdl, ids)
        logger.info("Node: {} Depth: {}".format(tree.node, depth))

        if feature_idx is not None:
            (
                left_prob, left_y, left_X, left_ids,
                right_prob, right_y, right_X, right_ids
            ) = NodeFunction.split_on_pivot(X, y, feature_idx, pivot_value, pred_prob, ids)

            logger.info("left {}".format(left_X.size()))
            node.left = NodeFunction.build_node_recursive(tree, left_X, left_y, left_prob, mdl, depth + 1,
                                                          left_ids)
            logger.info("right {}".format(right_X.size()))
            node.right = NodeFunction.build_node_recursive(tree, right_X, right_y, right_prob, mdl, depth + 1,
                                                           right_ids)

        return node


class Node:
    def __init__(self, feature_idx, pivot_value, mdl, _ids):
        self.feature_idx = feature_idx
        self.pivot_value = pivot_value
        self.mdl = mdl
        self.left = None
        self.right = None
        self._train_ids = _ids

    def node_count(self):
        if self.feature_idx is not None:
            return 1 + self.left.node_count() + self.right.node_count()
        else:
            return 1

    @staticmethod
    def get_leaf_mdl(root, x, _id):
        node = root
        path = []
        while node.feature_idx is not None:
            if x[0, node.feature_idx].item() <= node.pivot_value:
                node = node.left
                path.append(0)
            else:
                node = node.right
                path.append(1)
        return path, node.mdl


def train_model(train_data, train_labels, cls_names, dataset, verbose=False):
    logger.info("Train the model: {}".format("-".join(cls_names)))
    path_manager = PathManager(LinearModelTree.NAME, dataset)
    var_num = train_data[0, :].size()[0]
    lmt = LinearModelTree(min_node_size=100, clss_num=len(cls_names), var_num=var_num)
    lmt.build_tree(train_data, train_labels)
    logger.info("Finish training Node {}".format(lmt.node))
    with open(path_manager.mdl_path(), "wb") as f:
        pickle.dump(lmt, f)
    return lmt


def load_model(model_path):
    lmt = pickle.load(open(model_path, "rb"))
    lmt.to(config.DEVICE)
    return lmt


def evaluate(model, test_data, test_labels, dataset):
    # On the test data set
    total_acc, total = 0, test_data.size()[0]
    prob = model.forward(test_data)
    p_labels = prob.argmax(dim=1)
    total_acc += (p_labels == test_labels).sum().item()
    precision = total_acc / (1.0 * total)
    logger.info("[{} dataset] Model: MLP Accuracy: {}/{}={}".format(dataset, total_acc, total, precision))
    return precision, total, total_acc


def test():
    def train_fmnist():
        from openapi.utils import FMNIST
        IMG_SIZE = 28
        cls_names = FMNIST._ID2STR
        fmnist = FMNIST(cls_names)
        train_imgs, train_labels, test_imgs, test_labels = fmnist.load_mnist()
        fmnist.display(train_labels, test_labels)
        train_imgs = train_imgs.view(train_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        mdl = train_model(train_imgs, train_labels, cls_names, FMNIST.NAME, verbose=True)
        evaluate(mdl, train_imgs, train_labels, "train")

    def evaluate_fmnist():
        from openapi.utils import FMNIST
        IMG_SIZE = 28
        cls_names = FMNIST._ID2STR
        path_manager = PathManager(LinearModelTree.NAME, FMNIST.NAME)
        fmnist = FMNIST(cls_names)
        _, _, test_imgs, test_labels = fmnist.load_mnist()
        test_imgs = test_imgs.view(test_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        mdl = load_model(path_manager.mdl_path())
        evaluate(mdl, test_imgs, test_labels, "test")

    def train_mnist():
        from openapi.utils import MNIST
        IMG_SIZE = 28
        cls_names = MNIST._ID2STR
        fmnist = MNIST(cls_names)
        train_imgs, train_labels, test_imgs, test_labels = fmnist.load_mnist()
        fmnist.display(train_labels, test_labels)
        train_imgs = train_imgs.view(train_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        mdl = train_model(train_imgs, train_labels, cls_names, MNIST.NAME, verbose=True)
        evaluate(mdl, train_imgs, train_labels, "train")

    def evaluate_mnist():
        from openapi.utils import MNIST
        IMG_SIZE = 28
        cls_names = MNIST._ID2STR
        path_manager = PathManager(LinearModelTree.NAME, MNIST.NAME)
        fmnist = MNIST(cls_names)
        _, _, test_imgs, test_labels = fmnist.load_mnist()
        test_imgs = test_imgs.view(test_imgs.size()[0], IMG_SIZE * IMG_SIZE)
        mdl = load_model(path_manager.mdl_path())
        evaluate(mdl, test_imgs, test_labels, "test")

    train_fmnist()
    evaluate_fmnist()
    # train_mnist()
    # evaluate_mnist()


if __name__ == '__main__':
    test()
