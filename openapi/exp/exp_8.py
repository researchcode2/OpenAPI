#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import torch
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
# Averaged image

set_random_seed()

logger = getLogger(__name__)


class Exp8(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize):
        AbsExp.__init__(self, model_name, dataset, "8", expln_name, datasize)

    def run(self):
        label_images = defaultdict(list)
        for i in range(self.images.size()[0]):
            x_tensor = self.images[i].view(-1, self.var_num)
            clss = self.labels[i].item()
            label_images[clss].append((x_tensor, i))

        image_average = defaultdict(list)
        feature_average = defaultdict(list)
        plt.axis('off')
        for clss, data in label_images.items():
            for x_tensor, idx in data:
                prob = self.mdl.forward(x_tensor)
                pred_clss = torch.argmax(prob, dim=1).item()

                if pred_clss != clss:
                    continue

                f_grad = self.mdl.compute_decision_feature(x_tensor, pred_clss)

                scaler = torch.max(abs(f_grad))
                f_grad = f_grad / scaler

                feature_average[pred_clss].append(f_grad)
                image_average[pred_clss].append(x_tensor)

        pdf_name = self.path_manager.image_path(self.expln_name, self.data_size, self.dataset)
        pp = PdfPages(pdf_name)
        for clss, average in image_average.items():
            clss_name = self.cls_names[clss]
            fig, axes = plt.subplots(1)

            img = torch.mean(torch.cat(average, dim=0), dim=0)
            print("image size", img.size())
            f_grad = torch.mean(torch.cat(feature_average[clss], dim=0), dim=0)
            img = img.cpu()
            f_grad = f_grad.cpu()
            img = img.view(self.img_size, self.img_size)
            f_grad = f_grad.view(self.img_size, self.img_size)

            axes.imshow(img, cmap="gray")
            axes.set_axis_off()
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            axes.set_frame_on(False)
            plt.savefig(pp, format='pdf', bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            fig, axes = plt.subplots(1)
            scaler = torch.max(abs(f_grad))
            axes.imshow(-f_grad / scaler, cmap="RdBu", vmin=-1, vmax=1)
            axes.set_axis_off()
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            axes.set_frame_on(False)
            plt.savefig(pp, format='pdf', bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            print("{}".format(clss_name))

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

    exp = Exp8(model_name, dataset, expln_name, datasize)
    exp.run()

    logger.info("Finish")


if __name__ == '__main__':
    main()
