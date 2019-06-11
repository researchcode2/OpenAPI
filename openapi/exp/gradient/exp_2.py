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


class Exp2(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize):
        AbsExp.__init__(self, model_name, dataset, "2", expln_name, datasize)

    @staticmethod
    def plot(model_name, dataset, datasize):
        exp = "2"
        path_manager = PathManager(model_name, dataset)
        dist = defaultdict(list)
        for expl_name in AbsExp.get_method_names():
            result_file = path_manager.result_json_path(exp, expl_name, datasize)
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    results = json.loads(f.readlines()[0])
                    for value in results.values():
                        dist[expl_name].append(value["L1Distance"])

        pdf_name = path_manager.figure_path(datasize, exp)
        pp = PdfPages(pdf_name)
        values, names = [], []
        for i, j in sorted(dist.items(), key=Plot.sort_key, reverse=True):
            values.append(j)
            names.append(i)
        Plot.plot_box(values, names, pp, model_name, "L1Dist", True, False)
        pp.close()

    def run(self):
        W_EST_NPZ = self.path_manager.w_est_file(self.expln_name, self.data_size)
        True_W_EST_NPZ = self.path_manager.w_est_file("GroundTruth", self.data_size)
        est_grads = np.load(W_EST_NPZ)["w"].item()
        true_grads = {}
        result = {}
        for i in range(self.images.size()[0]):
            try:
                start_ts = time.time()
                x_tensor = self.images[i].view(-1, self.var_num)
                prob = self.mdl.forward(x_tensor)
                clss = torch.argmax(prob, dim=1).item()
                f_grad = est_grads[i]
                true_grad = self.mdl.compute_gradient_auto(x_tensor, clss).detach().cpu()

                true_grads[i] = true_grad.cpu().detach()
                np.savez(True_W_EST_NPZ, w=true_grads)

                distance = torch.sum(abs(f_grad - true_grad)).item()
                result[i] = {
                    "L1Distance": distance,
                    "L1Norm": torch.sum(abs(true_grad)).item()
                }
                print(result[i])
                rst_str = json.dumps(result)
                with open(self.result_json, "w") as w:
                    w.write("{}\n".format(rst_str))

                end_ts = time.time()
                logger.info("Time Elapse: {}".format(end_ts - start_ts))
                logger.info("=" * 40)
            except Exception as e:
                logger.exception(e)
        logger.info("Finish Dataset: {} Datasize: {} Explainer: {} Model: {} Experiment: {}".format(
            self.dataset, self.data_size, self.expln_name, self.model_name, self.exp))


def testcases():
    def test_1():
        logger.info("Using Device: {}".format(config.DEVICE))
        model_name = "MLP"
        dataset = "FMNIST"
        expln_name = AbsExp.OpenAPI
        datasize = 10
        exp = Exp2(model_name, dataset, expln_name, datasize)
        exp.run()
        logger.info("Finish")

    def test_plot():
        logger.info("Using Device: {}".format(config.DEVICE))
        model_name = "MLP"
        dataset = "FMNIST"
        expln_name = AbsExp.OpenAPI
        datasize = 10
        Exp2.plot(model_name, dataset, datasize)
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
        exp = Exp2(model_name, dataset, expln_name, datasize)
        exp.run()
    else:
        Exp2.plot(model_name, dataset, datasize)
    logger.info("Finish")


if __name__ == '__main__':
    # testcases()
    main()

