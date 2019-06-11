#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from openapi.abs_explainer import AbsExplainer
from openapi.utils import set_random_seed, compute_gradient
from openapi.log import getLogger
from openapi import config

logger = getLogger(__name__)


def solve_lin(X, Y):
    from numpy.linalg import solve as lin_solve
    w = lin_solve(X, Y)
    return w


def unif_sample(x_tensor, sample_size, dist):
    """
    Uniformly samples points around the given point within a specific distance
    :param x_tensor: tensor of the target point
    :param sample_size: number of samples to be drawn
    :param dist: the distance threshold between the sampled points and the target point
    :return: tensor of samples points. (sample_num, features)
    """
    feature_len = x_tensor.size()[1]
    samples_tensor = torch.zeros((sample_size, feature_len),
                                 dtype=torch.float64, device=config.DEVICE).uniform_(-1.0 * dist, dist)
    samples_tensor += x_tensor
    return samples_tensor


def reverse(x_tensor, model, X_tensor, clss):
    Y = model.forward(X_tensor)
    y = model.forward(x_tensor)
    Dccs = {}
    for i in range(Y.size()[1]):
        if i != clss:
            _Y = torch.log(Y[:, clss] / Y[:, i])
            _y = torch.log(y[:, clss] / y[:, i])
            Dccs[i] = _reverse(x_tensor, X_tensor, _y, _Y)
        else:
            Dccs[clss] = torch.zeros((1, x_tensor.size()[1] + 1), dtype=torch.float64, device=config.DEVICE)
    return Dccs


def _reverse(x_tensor, X_tensor, y, Y):
    num = X_tensor.size()[0]  # Sample * var_num
    X_aug_tensor = torch.cat((X_tensor, torch.ones(num, 1, dtype=torch.float64, device=config.DEVICE)), dim=1)

    var_num = X_aug_tensor.size()[1]
    # st = time.time()
    # wb_est_np = solve_lin(X_aug_tensor.cpu().detach(), Y.cpu().detach())
    # print("CPU Solve linear", time.time() - st)
    # wb_est_tensor = torch.tensor(wb_est_np, dtype=torch.float64, device=config.DEVICE).view(1, var_num)
    # wb1 = wb_est_tensor.view(var_num, 1)
    # assert torch.sum(wb1 - wb2) == 0, "Difference {}".format(torch.sum(abs(wb1 - wb2)))
    # st = time.time()
    # print("GPU Solve linear", time.time() - st)
    ##################################################################################
    wb2 = torch.solve(Y.view(-1, 1), X_aug_tensor)[0]
    wb_est_tensor = wb2.view(1, var_num)
    ##################################################################################

    x_agu_tensor = torch.cat((x_tensor, torch.tensor([[1]], dtype=torch.float64, device=config.DEVICE)), dim=1)

    # logger.debug("WB: {} Agu Tensor : {}".format(wb_est_tensor, x_agu_tensor))
    x_y_est_tensor = torch.mm(wb_est_tensor, torch.t(x_agu_tensor))

    logger.debug("x_logit_tensor: {} x_y_est_tensor: {}".format(y, x_y_est_tensor))


    assert abs(y - x_y_est_tensor) <= 1e-10, \
        "Cannot find solution. x's logit diff is {}".format(
            (y - x_y_est_tensor).item()
        )
    logger.debug("x_logit_diff: {}".format(abs(y - x_y_est_tensor)))
    return wb_est_tensor.detach()


class OpenAPI(AbsExplainer):
    def __init__(self, sample_num, threshold, var_num):
        AbsExplainer.__init__(self, sample_num, var_num)
        self.threshold = threshold

    def weight_diff(self, x_tensor, model, clss, clss_num):
        set_random_seed()
        dist = 1
        w_diff_est_tensor = None
        samples_tensor = None
        counter = 0
        error = None
        while dist > self.threshold:
            counter += 1
            try:
                samples_tensor = unif_sample(x_tensor, self.sample_num, dist)
                w_diff_est_tensor = reverse(x_tensor, model, samples_tensor, clss)
                break
            except Exception as e:
                error = e
                dist /= 2.0
            logger.debug("Counter: {}".format(counter))
        if w_diff_est_tensor is None:
            logger.exception(error)
            raise Exception(
                "Due to the limitations of python's computing error, we cannot estimate the gradient of x given the distance threshold.")
        return w_diff_est_tensor, dist, samples_tensor

    def decision_f(self, x_tensor, model, clss, clss_num):
        w_diff_tensor, dist, samples_tensor = self.weight_diff(x_tensor, model, clss, clss_num)
        assert type(w_diff_tensor) is dict
        decision_f = self._decision_f(w_diff_tensor, clss_num)
        return decision_f, None, dist, samples_tensor, w_diff_tensor
