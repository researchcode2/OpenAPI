#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('agg')
from matplotlib.backends.backend_pdf import PdfPages

from openapi.exp.abs_exp import AbsExp
from openapi.utils import set_random_seed, PathManager, Plot
from openapi.log import getLogger
from openapi import config

######################################################################
# LMT related module must be imported to make the pickle load successfully
from openapi.mdl.lmt import LinearModelTree, Node, LocalLasso

######################################################################

set_random_seed()

logger = getLogger(__name__)


def similarity_matrix(mat):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2 * r
    return D.sqrt()


class Exp7(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize, feature_type):
        AbsExp.__init__(self, model_name, dataset, "7", expln_name, datasize)
        self.feature_type = feature_type
        self.result_json = self.path_manager.result_json_path("7", feature_type, size=self.data_size)

    @staticmethod
    def plot(model_name, dataset, datasize):
        exp = "7"
        path_manager = PathManager(model_name, dataset)
        cs = defaultdict(list)
        for feature_type in AbsExp.get_attribute_method_names():
            result_file = path_manager.result_json_path(exp, feature_type, datasize)
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    results = json.loads(f.readlines()[0])
                    for value in results.values():
                        cs[feature_type].append(value["Cosine"])

        pdf_name = path_manager.figure_path(datasize, exp)
        pp = PdfPages(pdf_name)
        values, names = [], []
        for i, j in sorted(cs.items(), key=Plot.sort_key, reverse=True):
            values.append(sorted(j, reverse=True))
            names.append(i)
        Plot.plot_line(values, names, pp, model_name, "CS", "Index of Instance", [-0.05, 1.1], 0.5, (18, 6), (100, 300))
        pp.close()

    def run(self):
        result = {}
        logger.info("Compute similarity matrix")
        start_ts = time.time()
        self.images = self.images.view(-1, self.var_num)
        sim_matrix = similarity_matrix(self.images)
        end_ts = time.time()
        logger.info("Finish computing similarity matrix {}".format(end_ts - start_ts))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        for i in range(self.images.size()[0]):
            start_ts = time.time()
            distance = sim_matrix[i, :]
            j = torch.topk(-distance, k=2, dim=0)[1][1]
            neighbor_idx = j.item()
            image_i = self.images[i, :].unsqueeze(dim=0)
            image_j = self.images[neighbor_idx, :].unsqueeze(dim=0)
            i_pred_clss = torch.argmax(self.mdl.forward(image_i), dim=1).item()
            j_pred_clss = torch.argmax(self.mdl.forward(image_j), dim=1).item()
            if self.feature_type == AbsExp.INTEGRATED_GRADIENT:
                f_i = self.mdl.integrated_gradient(image_i, i_pred_clss)
                f_j = self.mdl.integrated_gradient(image_j, j_pred_clss)
            elif self.feature_type == AbsExp.SALIENCY:
                f_i = self.mdl.saliency_map(image_i, i_pred_clss)
                f_j = self.mdl.saliency_map(image_j, j_pred_clss)
            elif self.feature_type == AbsExp.GRADTIMEINPUT:
                f_i = self.mdl.gradient_input(image_i, i_pred_clss)
                f_j = self.mdl.gradient_input(image_j, j_pred_clss)
            elif self.feature_type == AbsExp.OpenAPI:
                f_i = self.mdl.compute_decision_feature(image_i, i_pred_clss)
                f_j = self.mdl.compute_decision_feature(image_j, j_pred_clss)
            elif self.feature_type == AbsExp.LIMEOrigin:
                f_i = self.mdl.lime_interpret(image_i, i_pred_clss)
                f_j = self.mdl.lime_interpret(image_j, j_pred_clss)
            else:
                raise Exception("The attribution method does not support")
            output = cos(f_i, f_j)

            result[i] = {
                "Index_i": i,
                "Index_j": neighbor_idx,
                "Cosine": output.item()
            }
            print(result[i])
            rst_str = json.dumps(result)
            with open(self.result_json, "w") as w:
                w.write("{}\n".format(rst_str))

            end_ts = time.time()
            logger.info("Time Elapse: {}".format(end_ts - start_ts))
            logger.info("=" * 40)
        logger.info("Finish Dataset: {} Datasize: {} Explainer: {} Model: {} Experiment: {}".format(
            self.dataset, self.data_size, self.expln_name, self.model_name, self.exp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl", help="Model Name", choices=["LMT", "MLP"], required=True)
    parser.add_argument("--dataset", help="data. E.g. FMNIST", choices=["FMNIST", "MNIST"], required=True)
    parser.add_argument("--datasize", help="number of test images per class", required=True)
    parser.add_argument("--explainer", help="name of explainer", choices=AbsExp.get_method_names())
    parser.add_argument("--gpu", help="GPU used", choices=["cuda:1", "cuda:0", "cuda:2", "cuda:3"])
    parser.add_argument("--task", help="GPU used", choices=["plot", "compute"], required=True)
    parser.add_argument("--feature", help="Feature type", choices=Exp7.get_attribute_method_names())
    parsedArgs = parser.parse_args(sys.argv[1:])
    model_name = parsedArgs.mdl
    dataset = parsedArgs.dataset
    datasize = int(parsedArgs.datasize)
    task = parsedArgs.task

    if task == "compute":
        expln_name = parsedArgs.explainer
        feature_type = parsedArgs.feature
        config.DEVICE = parsedArgs.gpu
        exp = Exp7(model_name, dataset, expln_name, datasize, feature_type)
        exp.run()
    else:
        Exp7.plot(model_name, dataset, datasize)
    logger.info("Finish")


if __name__ == '__main__':
    main()
