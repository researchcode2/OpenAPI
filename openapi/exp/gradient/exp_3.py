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


class Exp3(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize):
        AbsExp.__init__(self, model_name, dataset, "3", expln_name, datasize)

    @staticmethod
    def plot(model_name, dataset, datasize):
        exp = "3"
        path_manager = PathManager(model_name, dataset)
        cs = defaultdict(list)
        for expl_name in AbsExp.get_method_names():
            result_file = path_manager.result_json_path(exp, expl_name, datasize)
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    results = json.loads(f.readlines()[0])
                    for value in results.values():
                        cs[expl_name].append(value["Cosine"])

        pdf_name = path_manager.figure_path(datasize, exp)
        pp = PdfPages(pdf_name)
        values, names = [], []
        for i, j in sorted(cs.items(), key=Plot.sort_key, reverse=True):
            if "-08" not in i and "0.0001" not in i and "Ground" not in i:
                continue
            values.append(sorted(j, reverse=True))
            names.append(i)
        Plot.plot_line(values, names, pp, model_name, "CS", "Index of Instance", [-0.05, 1.1], 0.5)
        pp.close()

    def run(self):
        W_EST_NPZ = self.path_manager.w_est_file(self.expln_name, self.data_size)
        est_grads = np.load(W_EST_NPZ)["w"].item()
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
            grad_i = est_grads[i]
            grad_idx = est_grads[neighbor_idx]
            output = cos(grad_i, grad_idx)

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


def testcases():
    def test_1():
        logger.info("Using Device: {}".format(config.DEVICE))
        model_name = "MLP"
        dataset = "FMNIST"
        expln_name = AbsExp.OpenAPI
        datasize = 10
        exp = Exp3(model_name, dataset, expln_name, datasize)
        exp.run()
        logger.info("Finish")

    def test_plot():
        logger.info("Using Device: {}".format(config.DEVICE))
        model_name = "MLP"
        dataset = "FMNIST"
        expln_name = AbsExp.OpenAPI
        datasize = 10
        Exp3.plot(model_name, dataset, datasize)
        logger.info("Finish")

    # test_1()
    test_plot()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl", help="Model Name", choices=["LMT", "MLP"], required=True)
    parser.add_argument("--dataset", help="data. E.g. FMNIST", choices=["FMNIST", "MNIST"], required=True)
    parser.add_argument("--datasize", help="number of test images per class", required=True)
    parser.add_argument("--explainer", help="name of explainer", choices=AbsExp.get_method_names())
    parser.add_argument("--gpu", help="GPU used", choices=["cuda:1", "cuda:0", "cuda:2", "cuda:3"])
    parser.add_argument("--task", help="GPU used", choices=["plot", "compute"], required=True)
    parsedArgs = parser.parse_args(sys.argv[1:])
    model_name = parsedArgs.mdl
    dataset = parsedArgs.dataset
    expln_name = parsedArgs.explainer
    datasize = int(parsedArgs.datasize)
    task = parsedArgs.task
    config.DEVICE = parsedArgs.gpu

    if task == "compute":
        exp = Exp3(model_name, dataset, expln_name, datasize)
        exp.run()
    else:
        Exp3.plot(model_name, dataset, datasize)
    logger.info("Finish")


if __name__ == '__main__':
    # testcases()
    main()

