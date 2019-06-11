#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from openapi.exp.abs_exp import AbsExp
from openapi.utils import set_random_seed
from openapi.log import getLogger
from openapi import config
######################################################################
# LMT related module must be imported to make the pickle load successfully
from openapi.mdl.lmt import LinearModelTree, Node, LocalLasso

######################################################################
# Debug images

set_random_seed()

logger = getLogger(__name__)


class Exp5(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize):
        AbsExp.__init__(self, model_name, dataset, "5", expln_name, datasize)

    def run(self):
        label_images = defaultdict(list)
        for i in range(self.images.size()[0]):
            x_tensor = self.images[i].view(-1, self.var_num)
            clss = self.labels[i].item()
            label_images[clss].append((x_tensor, i))

        plt.axis('off')
        for clss, data in label_images.items():
            clss_name = self.cls_names[clss]
            pdf_name = self.path_manager.image_path(self.expln_name, self.data_size, clss_name)
            pp = PdfPages(pdf_name)
            counter = 0
            fig, axes = plt.subplots(1, 3)
            for x_tensor, idx in data:
                prob = self.mdl.forward(x_tensor)
                pred_clss = torch.argmax(prob, dim=1).item()

                if pred_clss == clss:
                    continue

                f_grad = self.mdl.compute_decision_feature(x_tensor, pred_clss).view(self.img_size, self.img_size)

                img = x_tensor.view(self.img_size, self.img_size).cpu()

                scaler = torch.max(abs(f_grad))
                f_grad = f_grad / scaler
                img = img.cpu()
                f_grad = f_grad.cpu()

                axes[0].imshow(img, cmap="gray")
                axes[0].set_axis_off()
                axes[1].imshow(f_grad, cmap="RdBu", vmin=-1, vmax=1)
                axes[1].set_axis_off()
                f_grad = f_grad * img
                axes[2].imshow(f_grad, cmap="RdBu", vmin=-1, vmax=1)
                axes[2].set_axis_off()

                fig.suptitle("Predict {} True {} Idx {}".format(pred_clss, clss, idx))

                plt.savefig(pp, format="pdf")
                plt.close(fig)
                fig, axes = plt.subplots(1, 3)

            if counter % 2 != 0:
                plt.savefig(pp, format="pdf")

            pp.close()

        logger.info("Finish Dataset: {} Datasize: {} Explainer: {} Model: {} Experiment: {}".format(
            self.dataset, self.data_size, self.expln_name, self.model_name, self.exp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl", help="Model Name", choices=["LMT", "MLP"], required=True)
    parser.add_argument("--dataset", help="data. E.g. FMNIST", choices=["FMNIST", "MNIST"], required=True)
    parser.add_argument("--explainer", help="name of explainer", choices=AbsExp.get_method_names())
    parser.add_argument("--datasize", help="number of test images per class", required=True)
    parser.add_argument("--gpu", help="GPU used", choices=["cuda:1", "cuda:0", "cuda:2", "cuda:3"])
    parsedArgs = parser.parse_args(sys.argv[1:])
    model_name = parsedArgs.mdl
    dataset = parsedArgs.dataset
    expln_name = parsedArgs.explainer
    datasize = int(parsedArgs.datasize)
    config.DEVICE = parsedArgs.gpu

    exp = Exp5(model_name, dataset, expln_name, datasize)
    exp.run()

    logger.info("Finish")


if __name__ == '__main__':
    main()

