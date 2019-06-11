#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from openapi.utils import compute_gradient
import openapi.config as config


class AbsExplainer:
    def __init__(self, sample_num, var_num):
        self.sample_num = sample_num
        self.var_num = var_num

    def _decision_f(self, w_diff_tensor, clss_num):
        dc = torch.zeros((1, self.var_num), dtype=torch.float64, device=config.DEVICE)
        for dcc in w_diff_tensor.values():
            dc += dcc[:, :self.var_num]
        return dc / (clss_num - 1)

