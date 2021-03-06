#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
import json
import time
import torch
import numpy as np
from collections import defaultdict
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


def core_param_diffs(samples, target_model, x_tensor, clss):
    x_weight, x_polytope = target_model.compute_corep_and_polytope(x_tensor)
    x_corep = x_weight - x_weight[clss, :]

    polytopes = []
    corep_diff = 0
    sample_size = samples.size()[0]
    for idx in range(sample_size):
        # If the perturbed instance is equal to the interpreted point, it will be a bug
        assert torch.sum(abs(samples[idx].view(1, -1) - x_tensor)).item() != 0
        sample_weight, sample_polytope = target_model.compute_corep_and_polytope(samples[idx].view(1, -1))
        sample_corep = sample_weight - sample_weight[clss, :]
        corep_diff += torch.sum(abs(sample_corep - x_corep))
        polytopes.append(sample_polytope)

    polytopes = torch.cat(polytopes, dim=0).detach()
    polytope_diff = torch.sum(abs(polytopes - x_polytope), dim=1)
    logger.info("polytope_diff: {}".format(polytope_diff.shape))
    return corep_diff.item() / sample_size, torch.sum(polytope_diff).item() / sample_size


class Exp1(AbsExp):
    def __init__(self, model_name, dataset, expln_name, datasize):
        AbsExp.__init__(self, model_name, dataset, "1", expln_name, datasize)

    @staticmethod
    def plot_new(model_name, dataset, datasize):
        exp = "1"
        wd = defaultdict(lambda : defaultdict(list))
        rd = defaultdict(lambda : defaultdict(list))
        path_manager = PathManager(model_name, dataset)
        params = set()
        for expl_name in AbsExp.get_method_names():
            result_file = path_manager.result_json_path(exp, expl_name, datasize)
            if os.path.isfile(result_file):
                if ":" in expl_name:
                    name, param = expl_name.split(":")
                    params.add(param)
                else:
                    name, param = expl_name, None

                with open(result_file) as f:
                    results = json.loads(f.readlines()[0])
                    for result in results.values():
                        wd[name][param].append(result["CoreParameter"])
                        rd[name][param].append(0 if result["Polytope"] == 0 else 1)

        # Parameters in ascending order
        params = sorted(params, key=lambda x: float(x) if x is not None else 0, reverse=False)

        wd[AbsExp.OpenAPI] = {i: wd[AbsExp.OpenAPI][None] for i in params}
        wd = {
            name: [(float(param), result[param]) for param in params]
            for name, result in wd.items()
        }

        pdf_name = path_manager.figure_path(datasize, exp)
        pp = PdfPages(pdf_name)
        logger.info("WD {}".format({i+":"+str(k): np.mean(p) for i, j in wd.items() for k, p in j}))
        Plot.plot_errorbar(wd, pp, model_name, "WD", "Perturb Distance", (0, 100000), None, (18, 6), 1)
        pp.close()

    @staticmethod
    def plot(model_name, dataset, datasize):
        exp = "1"
        wd = defaultdict(list)
        rd = defaultdict(list)
        path_manager = PathManager(model_name, dataset)
        for expl_name in AbsExp.get_method_names():
            result_file = path_manager.result_json_path(exp, expl_name, datasize)
            # ##############################################
            # if "Open" not in result_file:
            #     continue
            # ##############################################
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    results = json.loads(f.readlines()[0])
                    for result in results.values():
                        wd[expl_name].append(result["CoreParameter"])
                        rd[expl_name].append(0 if result["Polytope"] == 0 else 1)

        pdf_name = path_manager.figure_path(datasize, exp)
        pp = PdfPages(pdf_name)
        values, names = [], []
        logger.info("WD {}".format({i: np.mean(j) for i, j in wd.items()}))
        for i, j in sorted(wd.items(), key=Plot.sort_key, reverse=False):
            values.append(j)
            names.append(i)
        Plot.plot_box(values, names, pp, model_name, "WD", False, True, pdf_name)

        logger.info("RD {}".format({i: np.mean(j) for i, j in rd.items()}))
        values, names = [], []
        for i, j in sorted(rd.items(), key=Plot.sort_key, reverse=False):
            values.append(j)
            names.append(i)
        Plot.plot_scatter(values, names, pp, model_name, "RD")
        pp.close()

    def run(self):
        W_EST_NPZ = self.path_manager.w_est_file(self.expln_name, self.data_size)
        est_grads = {}
        result = {}
        ###################################################################
        # if os.path.isfile(W_EST_NPZ):
        #     est_grads = np.load(W_EST_NPZ)["w"].item()
        #     result = json.loads(open(self.result_json+".bak").readline())
        ###################################################################
        for i in range(self.images.size()[0]):
            ###################################################################
            # if i in est_grads:
            #     continue
            ###################################################################
            try:
                start_ts = time.time()
                x_tensor = self.images[i].view(-1, self.var_num)
                true_clss = self.labels[i].item()
                prob = self.mdl.forward(x_tensor)
                pred_clss = torch.argmax(prob, dim=1).item()
                prob = prob.detach().cpu().numpy().tolist()

                st = time.time()


                decision_f, _, dist, samples_tensor, weight_diff = self.explainer.decision_f(x_tensor, self.mdl, pred_clss, self.cls_num)

                true_decision = self.mdl.compute_decision_feature(x_tensor, pred_clss)
                distance = torch.sum(abs(decision_f - true_decision)).item()
                print("distance", distance)

                print("Explain time used {}".format(time.time() - st))
                est_grads[i] = decision_f.cpu().detach()
                np.savez(W_EST_NPZ, w=est_grads)

                st = time.time()
                corep_diff, polytope_diff = core_param_diffs(samples_tensor, self.mdl, x_tensor, pred_clss)
                print("Corep time used {}".format(time.time() - st))
                result[i] = {
                    "CoreParameter": corep_diff,
                    "Polytope": polytope_diff,
                    "Prob": prob,
                    "Distance": dist,
                    "TrueClass": true_clss,
                    "PredictClass": pred_clss
                }
                print(result[i])
                logger.info("Dist: {} CoreParameter: {} Polytope: {}".format(dist, corep_diff, polytope_diff))
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

        ###################################################################
        print("Est", len(est_grads), "result", len(result))
        ###################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl", help="Model Name", choices=["LMT", "MLP"], required=True)
    parser.add_argument("--dataset", help="data. E.g. FMNIST", choices=["FMNIST", "MNIST"], required=True)
    parser.add_argument("--datasize", help="number of test images per class", required=True)
    parser.add_argument("--explainer", help="name of explainer", choices=AbsExp.get_method_names())
    parser.add_argument("--gpu", help="GPU used", choices=["cuda:1", "cuda:0", "cuda:2", "cuda:3", "cpu"])
    parser.add_argument("--task", help="Task", choices=["plot", "grad"], required=True)
    parsedArgs = parser.parse_args(sys.argv[1:])
    model_name = parsedArgs.mdl
    dataset = parsedArgs.dataset
    expln_name = parsedArgs.explainer
    datasize = int(parsedArgs.datasize)
    task = parsedArgs.task
    config.DEVICE = parsedArgs.gpu

    if task == "grad":
        exp = Exp1(model_name, dataset, expln_name, datasize)
        exp.run()
    else:
        Exp1.plot(model_name, dataset, datasize)
    logger.info("Finish")


if __name__ == '__main__':
    # testcases()
    main()
