#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np
from openapi.abs_explainer import AbsExplainer

from openapi.log import getLogger
from openapi import config

logger = getLogger(__name__)


class Zeroth(AbsExplainer):
    def __init__(self, var_num, dist):
        sample_num = var_num * 2
        AbsExplainer.__init__(self, sample_num, var_num)
        self.dist = dist

    def weight_diff(self, x_tensor, model, clss, clss_num):
        img_perturb = x_tensor.repeat(self.sample_num, 1)
        for i, var_idx in enumerate(range(self.var_num)):
            var_idx = np.array([var_idx])
            img_perturb[i * 2, var_idx] += self.dist
            img_perturb[i * 2 + 1, var_idx] -= self.dist

        probs = model.forward(img_perturb)
        weight_diffs = {}
        for diff_j in range(clss_num):
            weight_diff = torch.zeros((1, self.var_num), dtype=torch.float64, device=config.DEVICE)
            if diff_j != clss:
                for i, var_idx in enumerate(range(self.var_num)):
                    prob_a = torch.log(probs[i * 2][clss] / probs[i * 2][diff_j])
                    prob_b = torch.log(probs[i * 2 + 1][clss] / probs[i * 2 + 1][diff_j])
                    weight_diff[0, var_idx] = (prob_a - prob_b) / (2 * self.dist)
            weight_diffs[diff_j] = weight_diff.detach()
        return weight_diffs, self.dist, img_perturb

    def decision_f(self, x_tensor, model, clss, clss_num):
        w_diff_tensor, dist, samples_tensor = self.weight_diff(x_tensor, model, clss, clss_num)
        assert type(w_diff_tensor) is dict
        decision_f = self._decision_f(w_diff_tensor, clss_num)
        return decision_f, None, dist, samples_tensor, w_diff_tensor
