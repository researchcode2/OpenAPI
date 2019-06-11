#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from itertools import combinations

import numpy as np
import os
import json
import random
import sys
import torchvision.transforms as transforms
import torchvision.datasets as dset
from biplnn.adv_cli import CliRunner, AdvCli
from biplnn.utils import set_random_seed, norm, PathManager, FMNIST, load_data_new
from biplnn.exp.abs_exp import AbsExp


def load():
    # point = np.load('bad_point.npz')["x"]
    # samples = np.load('bad_point.npz')["s"]
    # print(point.shape)
    # print(samples.shape)
    total = 0
    counter = 0
    for file in os.listdir("w_est"):
        file_path = "w_est/{}".format(file)
        point = np.load(file_path)
        w_true = point["openbox"]
        total += norm(w_true, 1)
        counter += 1
    print("Total", total)
    print("Mean", total / counter)
    # print(point["bap"].shape)
    # print(point["z1"].shape)
    # print(point["z2"].shape)
    # print(point["z4"].shape)
    # print(point["z8"].shape)
    # print(point["z16"].shape)
    # print(point["lime"].shape)
    # print(point["w_bap"].shape)


def save_data(dataset, c_1=None, c_2=None):
    from biplnn.utils import PathManager
    def _save_data(c_1, c_2):
        set_random_seed()
        path_manager = PathManager(None, c_1, c_2, dataset, None)

        trans = transforms.Compose([
            transforms.ToTensor()  # default : range [0, 255] -> [0.0,1.0]
        ])

        if dataset == "FMNIST":
            c_1_id = FMNIST.str2id(c_1)
            c_2_id = FMNIST.str2id(c_2)

            train_set = [(img, t) for img, t in
                         dset.FashionMNIST(root=path_manager.ROOT_DATA_FOLDER, train=True, transform=trans,
                                           download=True)
                         if t.item() in {c_1_id, c_2_id}]
            test_set = [(img, t) for img, t in
                        dset.FashionMNIST(root=path_manager.ROOT_DATA_FOLDER, train=False, transform=trans,
                                          download=True)
                        if t.item() in {c_1_id, c_2_id}]

        # Process training data
        random.shuffle(train_set)
        train_imgs = np.concatenate([i for i, _ in train_set], axis=0)
        train_labels = np.array([j for _, j in train_set])
        np.savez(path_manager.raw_train_file(), x=train_imgs, l=train_labels)

        # Process testing data
        test_imgs = np.concatenate([i for i, _ in test_set], axis=0)
        test_labels = np.array([j for _, j in test_set])
        np.savez(path_manager.raw_test_file(), x=test_imgs, l=test_labels)

    if dataset == "FMNIST":
        if c_1 == None and c_2 == None:
            for i, j in combinations(FMNIST.get_ids(), 2):
                if i != j:
                    _save_data(FMNIST.id2str(i), FMNIST.id2str(j))
        else:
            _save_data(c_1, c_2)
    else:
        raise Exception("Your dataset {} is not supported".format(dataset))


class Analyze:
    def __init__(self, model_name, c_1, c_2, dataset, train_set):
        path_manager = PathManager(model_name, c_1, c_2, dataset, train_set)
        self.EXP124_RST_JSON = path_manager.result_json_path(exp="1_2_4")
        self.EXP3_RST_JSON = path_manager.result_json_path(exp="3")
        self.c_1 = c_1
        self.c_2 = c_2
        self.model_name = model_name

    def analyze_exp1(self):
        print("Class 1: {} Class 2: {} Model: {}".format(self.c_1, self.c_2, self.model_name))
        rst = {}
        counter = 0
        for ln in open(self.EXP124_RST_JSON):
            counter += 1
            obj = json.loads(ln)
            for m, v in obj.items():
                if m != "PROB" and m != AbsExp.OPENBLACKBOX_W and m != "INDEX" \
                        and m != AbsExp.OPENBOX and m != AbsExp.LIME:
                    if m not in rst:
                        rst[m] = []
                    rst[m].append((v["EXP1"][0], v["EXP1"][1], obj[AbsExp.OPENBLACKBOX_W]))

        print("Total:", counter)
        for m, vs in rst.items():
            if m == AbsExp.OPENBLACKBOX:
                for i, k, j in vs:
                    if i != 0:
                        print(j)
            polytope = [i != 0 for i, k, j in vs if j is not None]
            w_dist = [k for i, k, j in vs if j is not None]
            print(m, "Bad points", sum(polytope), max(w_dist))

    def analyze_exp2(self):
        print("Class 1: {} Class 2: {} Model: {}".format(self.c_1, self.c_2, self.model_name))
        _rst = {}
        for ln in open(self.EXP124_RST_JSON):
            obj = json.loads(ln)
            for m, v in obj.items():
                if m != "PROB" and m != AbsExp.OPENBLACKBOX_W and m != "INDEX" and m != AbsExp.OPENBOX:
                    if m not in _rst:
                        _rst[m] = []
                    _rst[m].append((v["EXP2"], obj["PROB"]))
        rst = {}
        for m, vs in _rst.items():
            _len = len(vs)
            vs = sorted([i for i, j in vs if j < 0.9999], reverse=True)
            rst[m] = (np.mean(vs), np.max(vs), len(vs), _len)

        for m, stats in sorted(rst.items(), key=lambda i: i[1][0]):
            print(m, "Filter:", stats[2], "Original:", stats[3])
            print(m, "Mean:", stats[0], "Max:", stats[1])


class Show:
    @staticmethod
    def analyze_exp3():
        obj = json.loads(open("result/PLNN_Ankle_boot_Bag_exp3_result.json").readline())
        for k, vs in obj.items():
            print(k, max(vs), min(vs))

    @staticmethod
    def analyze_exp1():
        def _analyze(c_1, c_2, model):
            path_tmp = "result/{}_{}_{}_Testset_exp_1_2_4_result.json"
            if os.path.isfile(path_tmp.format(model, c_1, c_2)):
                file_path = path_tmp.format(model, c_1, c_2)
            elif os.path.isfile(path_tmp.format(model, c_2, c_1)):
                file_path = path_tmp.format(model, c_2, c_1)
            else:
                raise Exception("The file does not exist")

            print("=" * 40)
            for ln in open(file_path):
                obj = json.loads(ln)
                print("Class 1: {} Class 2: {} Model: {}".format(c_1, c_2, model))
                print("Probability: {}".format(obj["PROB"]))
                print("Whitebox BAP: {}".format(obj["WHITEBOX_BAP"]))
                for m, v in obj.items():
                    if m != "PROB" and m != "WHITEBOX_BAP" and m != "INDEX":
                        print("{} Polyope: {} Sample_WDiff: {} DIST: {}".format(
                            m,
                            v["EXP1"][1],
                            v["EXP1"][0],
                            v["DIST"]
                        ))
                print('-' * 40)

        _analyze("Ankle_boot", "Bag", "PLNN")
        _analyze("Coat", "Pullover", "PLNN")

    @staticmethod
    def analyze_exp2():
        def _analyze(c_1, c_2, model):
            path_tmp = "result/{}_{}_{}_Testset_exp_1_2_4_result.json"
            if os.path.isfile(path_tmp.format(model, c_1, c_2)):
                file_path = path_tmp.format(model, c_1, c_2)
            elif os.path.isfile(path_tmp.format(model, c_2, c_1)):
                file_path = path_tmp.format(model, c_2, c_1)
            else:
                raise Exception("The file does not exist")

            print("=" * 40)
            for ln in open(file_path):
                obj = json.loads(ln)
                print("Class 1: {} Class 2: {} Model: {}".format(c_1, c_2, model))
                print("Probability: {}".format(obj["PROB"]))
                print("Whitebox BAP: {}".format(obj["WHITEBOX_BAP"]))
                for m, v in obj.items():
                    if m != "PROB" and m != "WHITEBOX_BAP" and m != "INDEX":
                        print("{} WDiff: {} DIST: {}".format(
                            m,
                            v["EXP2"],
                            v["DIST"]
                        ))
                print('-' * 40)

        _analyze("Ankle_boot", "Bag", "PLNN")
        _analyze("Coat", "Pullover", "PLNN")

    @staticmethod
    def analyze_exp4():
        def _analyze(c_1, c_2, model):
            path_tmp = "result/{}_{}_{}_Testset_exp_1_2_4_result.json"
            if os.path.isfile(path_tmp.format(model, c_1, c_2)):
                file_path = path_tmp.format(model, c_1, c_2)
            elif os.path.isfile(path_tmp.format(model, c_2, c_1)):
                file_path = path_tmp.format(model, c_2, c_1)
            else:
                raise Exception("The file does not exist")

            print("=" * 40)
            for ln in open(file_path):
                obj = json.loads(ln)
                print("Class 1: {} Class 2: {} Model: {}".format(c_1, c_2, model))
                print("Probability: {}".format(obj["PROB"]))
                print("Whitebox BAP: {}".format(obj["WHITEBOX_BAP"]))
                for m, v in obj.items():
                    if m != "PROB" and m != "WHITEBOX_BAP" and m != "INDEX":
                        print("{} ProbabilityChange: {} LabelChange: {}".format(
                            m,
                            sum([i for i, j in v["EXP4"]]),
                            sum([j for i, j in v["EXP4"]]),
                            v["DIST"]
                        ))
                print('-' * 40)

        _analyze("Ankle_boot", "Bag", "PLNN")
        # _analyze("Coat", "Pullover", "PLNN")


def stat_dataset(c_1, c_2):
    from collections import defaultdict

    train_data, train_labels = load_data_new(c_1, c_2, train=True)
    counter = defaultdict(int)
    print("Train")
    for i in train_labels:
        counter[i.item()] += 1
    print(counter)

    print("Test")
    train_data, train_labels = load_data_new(c_1, c_2, train=False)
    counter = defaultdict(int)
    for i in train_labels:
        counter[i.item()] += 1
    print(counter)
    print("ID to PositiveNegative Label")
    print(FMNIST.id2pn_label(FMNIST.str2id(c_1), FMNIST.str2id(c_2)))


class Tool(CliRunner):
    def initOptions(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", help="Task", choices=["analyze", "stat"], required=True)
        parser.add_argument("--mdl", help="Model Name", choices=["LMT", "PLNN"], required=True)
        parser.add_argument("--dataset", help="data. E.g. FMNIST", choices=["FMNIST", ], required=True)
        parser.add_argument("--c1", help="First class name", required=True)
        parser.add_argument("--c2", help="Second class name", required=True)
        return parser

    def validateOptions(self, args):
        return True

    def start(self, args):
        if args.task == "analyze":
            analyze = Analyze(args.mdl, args.c1, args.c2, args.dataset, train_set=False)
            analyze.analyze_exp1()
            analyze.analyze_exp2()


if __name__ == '__main__':
    AdvCli.initRunner(Tool(), sys.argv)
