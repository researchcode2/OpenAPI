#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import torch
from openapi.log import getLogger
from openapi import config

logger = getLogger(__name__)


class LimeOrigin:
    def __init__(self, var_num, img_size):
        # Make sure that each pixel is a segmentation
        self.segmenter = SegmentationAlgorithm("quickshift", kernel_size=1, max_dist=0.0001, ratio=0.2)
        self.var_num = var_num
        self.py = None
        self.img_size = img_size

    def estimate_gradient(self, x_tensor, target_model):
        """
        Estimate the gradient of x_tensor on the target_model
        :param x_tensor:
        :param target_model:
        :return: Estimated W, Sample Radius, Samples
        """
        def predict_fun(x):
            # The original gray image is change to RGB image by LIME by default.
            # To use the original classifier, we need to remove the channels added by LIME
            x = torch.tensor(x[:, :, :, 0], device=config.DEVICE, dtype=torch.float64).view(-1, self.var_num)
            rst = target_model.predicts(x).cpu().numpy()  # Output 1 * 2 array
            return rst

        x_tensor = x_tensor.view(self.img_size, self.img_size)
        explainer = lime_image.LimeImageExplainer(feature_selection='none')
        explanation = explainer.explain_instance(x_tensor.cpu().numpy(), predict_fun, top_labels=None, hide_color=0,
                                                 num_samples=self.var_num + 2, num_features=self.var_num,
                                                 segmentation_fn=self.segmenter, labels=(0, 1))

        self.py = explanation.local_pred
        # We only consider the weights related to positive labels
        w_lime = sorted(explanation.local_exp[1], key=lambda i: i[0])
        w_lime = torch.tensor([v for _, v in w_lime], dtype=torch.float64, device=config.DEVICE).unsqueeze(0)
        return w_lime
