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


def flip_pixels(grad, x_tensor, target_model, var_num, clss, f_limit=200):
    origin_prob = target_model.forward(x_tensor)
    origin_prob = origin_prob[0, clss]
    # Feature Index, Feature Gradient, Gradient Sign
    f_rank = sorted([(i, abs(grad[0, i]), grad[0, i] > 0) for i in range(var_num)],
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
    cpps = abs(origin_prob - predicts[:, clss])
    nlci = (torch.argmax(predicts, dim=1) != clss)
    return cpps.detach().cpu(), nlci.detach().cpu()


class Exp6(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize, feature_type):
        AbsExp.__init__(self, model_name, dataset, "6", expln_name, datasize)
        self.feature_type = feature_type
        self.result_json = self.path_manager.result_json_path(self.exp, feature_type, size=self.data_size)

    @staticmethod
    def plot(model_name, dataset, datasize):
        exp = "6"
        path_manager = PathManager(model_name, dataset)
        cpp = defaultdict(list)
        nlci = defaultdict(list)
        var_num = 0
        for feature_type in AbsExp.get_attribute_method_names():
            result_file = path_manager.result_json_path(exp, feature_type, size=datasize)
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    results = json.loads(f.readlines()[0])
                    for value in results.values():
                        var_num = len(value["CPP"])
                        cpp[feature_type].append(value["CPP"])
                        nlci[feature_type].append(value["NLCI"])

        pdf_name = path_manager.figure_path(datasize, exp)
        pp = PdfPages(pdf_name)
        values = []
        names = []
        # Filter condition
        for name, instances in sorted(cpp.items(), key=Plot.sort_key, reverse=True):
            value = [0] * (var_num + 1)
            for instance in instances:
                for idx, v in enumerate(instance):
                    value[idx + 1] += v
            values.append([i / len(instances) for i in value])
            names.append(name)
        Plot.plot_line(values, names, pp, model_name, "Avg. CPP", "\#Changed Features", [0, 1.1], 0.5, (18, 6),
                       (20, 50))

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
        Plot.plot_line(values, names, pp, model_name, "Avg. NLCI", "\#Changed Features", [0, 1100], 500, (18, 6),
                       (20, 50))
        pp.close()

    def run(self):
        result = {}
        for i in range(self.images.size()[0]):
            start_ts = time.time()
            x_tensor = self.images[i].view(-1, self.var_num)
            pred_clss = torch.argmax(self.mdl.forward(x_tensor), dim=1).item()
            if self.feature_type == AbsExp.INTEGRATED_GRADIENT:
                f_grad = self.mdl.integrated_gradient(x_tensor, pred_clss)
            elif self.feature_type == AbsExp.SALIENCY:
                f_grad = self.mdl.saliency_map(x_tensor, pred_clss)
            elif self.feature_type == AbsExp.GRADTIMEINPUT:
                f_grad = self.mdl.gradient_input(x_tensor, pred_clss)
            elif self.feature_type == AbsExp.OpenAPI:
                openapi = self.explainer
                f_grad = openapi.decision_f(x_tensor, self.mdl, pred_clss, self.cls_num)
            elif self.feature_type == AbsExp.LIMEOrigin:
                f_grad = self.mdl.lime_interpret(x_tensor, pred_clss)
            else:
                raise Exception("Attribute score is not supported")

            cpps, nlci = flip_pixels(f_grad, x_tensor, self.mdl, self.var_num, pred_clss)

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
    parser.add_argument("--feature", help="Feature type", choices=AbsExp.get_attribute_method_names())
    parsedArgs = parser.parse_args(sys.argv[1:])
    model_name = parsedArgs.mdl
    dataset = parsedArgs.dataset
    datasize = int(parsedArgs.datasize)
    task = parsedArgs.task

    if task == "compute":
        expln_name = parsedArgs.explainer
        config.DEVICE = parsedArgs.gpu
        feature_type = parsedArgs.feature
        exp = Exp6(model_name, dataset, expln_name, datasize, feature_type)
        exp.run()
    else:
        Exp6.plot(model_name, dataset, datasize)
    logger.info("Finish")


if __name__ == '__main__':
    main()
