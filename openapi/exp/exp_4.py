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


def flip_pixels(decision_f, x_tensor, target_model, var_num, clss, f_limit=200):
    origin_prob = target_model.forward(x_tensor)
    origin_prob = origin_prob[0, clss]
    # Feature Index, Feature Gradient, Gradient Sign
    f_rank = sorted([(i, abs(decision_f[0, i]), decision_f[0, i] > 0) for i in range(var_num)],
                    key=lambda i: i[1], reverse=True)[:f_limit]
    hacked = x_tensor.clone()
    hacked_imgs = []
    for idx, _, sign in f_rank:
        if sign > 0:
            hacked[0, idx] = 0
        else:
            hacked[0, idx] = 1
        hacked_imgs.append(hacked)
        hacked = hacked.clone()

    hacked_imgs = torch.cat(hacked_imgs, dim=0)
    predicts = target_model.forward(hacked_imgs)
    print("predicts size", predicts.size())
    cpps = abs(origin_prob - predicts[:, clss])
    nlci = (torch.argmax(predicts, dim=1) != clss)
    return cpps.detach().cpu(), nlci.detach().cpu()


class Exp4(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize):
        AbsExp.__init__(self, model_name, dataset, "4", expln_name, datasize)

    @staticmethod
    def plot(model_name, dataset, datasize):
        exp = "4"
        path_manager = PathManager(model_name, dataset)
        cpp = defaultdict(list)
        nlci = defaultdict(list)
        var_num = 0
        for expl_name in AbsExp.get_method_names():
            result_file = path_manager.result_json_path(exp, expl_name, datasize)
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    results = json.loads(f.readlines()[0])
                    for value in results.values():
                        var_num = len(value["CPP"])
                        cpp[expl_name].append(value["CPP"])
                        nlci[expl_name].append(value["NLCI"])

        pdf_name = path_manager.figure_path(datasize, exp)
        pp = PdfPages(pdf_name)
        values = []
        names = []
        for name, instances in sorted(cpp.items(), key=Plot.sort_key, reverse=True):
            value = [0] * (var_num + 1)
            for instance in instances:
                for idx, v in enumerate(instance):
                    value[idx + 1] += v
            values.append([i / len(instances) for i in value])
            names.append(name)
        Plot.plot_line(values, names, pp, model_name, "CPP", "\#Hacked Features", [0, 1.01], 0.5)

        values = []
        names = []
        for name, instances in sorted(nlci.items(), key=Plot.sort_key, reverse=True):
            value = [0] * (var_num + 1)
            for instance in instances:
                for idx, v in enumerate(instance):
                    value[idx + 1] += v
            values.append(value)
            names.append(name)
        logger.info("Number of instance {}".format(len(instances)))
        Plot.plot_line(values, names, pp, model_name, "NLCI", "\#Hacked Features", [0, len(instances)], 500)
        pp.close()

    def run(self):
        W_EST_NPZ = self.path_manager.w_est_file(self.expln_name, self.data_size)
        est_decisions = np.load(W_EST_NPZ)["w"].item()
        result = {}
        for i in range(self.images.size()[0]):
            start_ts = time.time()
            x_tensor = self.images[i].view(-1, self.var_num)
            clss = self.labels[i].item()
            pred_clss = torch.argmax(self.mdl.forward(x_tensor), dim=1).item()
            if clss == pred_clss:
                f_decision = est_decisions[i]

                cpps, nlci = flip_pixels(f_decision, x_tensor, self.mdl, self.var_num, clss)

                result[i] = {
                    "CPP": cpps.numpy().tolist(),
                    "NLCI": nlci.numpy().tolist()
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
    parsedArgs = parser.parse_args(sys.argv[1:])
    model_name = parsedArgs.mdl
    dataset = parsedArgs.dataset
    expln_name = parsedArgs.explainer
    datasize = int(parsedArgs.datasize)
    task = parsedArgs.task
    config.DEVICE = parsedArgs.gpu

    if task == "compute":
        exp = Exp4(model_name, dataset, expln_name, datasize)
        exp.run()
    else:
        Exp4.plot(model_name, dataset, datasize)
    logger.info("Finish")


if __name__ == '__main__':
    main()

