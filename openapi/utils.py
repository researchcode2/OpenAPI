#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from collections import defaultdict
import matplotlib
from decimal import Decimal
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import torchvision.datasets as dset
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from torchvision.transforms import transforms
from openapi import config
from openapi.log import getLogger

logger = getLogger(__name__)

try:
    plt.rc('text', usetex=True)
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rcParams['text.latex.preamble'] = ['\\boldmath', '\\usepackage{siunitx}', ]
except:
    pass


def compute_gradient(w_diff_est_tensor, x_tensor, clss_num, var_num):
    x_tensor = torch.cat([x_tensor, torch.tensor([[1]], dtype=torch.float64, device=config.DEVICE)], dim=1)
    Dccs = [w_diff_est_tensor[i] for i in range(clss_num)]
    Dccs = torch.cat(Dccs, dim=0)
    assert Dccs.size()[0] == clss_num and Dccs.size()[1] == var_num + 1, Dccs.size()
    Dccs_x = torch.mm(Dccs, torch.t(x_tensor))
    assert Dccs_x.size()[0] == clss_num and Dccs_x.size()[1] == 1, Dccs_x.size()
    e_Dccs_x = torch.pow(np.e, -Dccs_x)
    assert e_Dccs_x.size()[0] == clss_num and e_Dccs_x.size()[1] == 1, e_Dccs_x.size()
    # Dot product
    times = e_Dccs_x * Dccs
    assert times.size()[0] == clss_num and times.size()[1] == var_num + 1, times.size()
    norminator = times.sum(dim=0)
    assert norminator.size()[0] == var_num + 1, norminator.size()
    denorminator = (e_Dccs_x.sum()) ** 2
    grad = (norminator / denorminator).unsqueeze(0)
    feature_grad = grad[:, 0:-1]
    b_grad = grad[:, -1]
    assert feature_grad.size()[1] == var_num, feature_grad.size()
    return b_grad, feature_grad


def set_random_seed():
    import random
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)


def save_image(img_tensor, name, img_size):
    img_tensor = img_tensor.view(-1, img_size, img_size).squeeze()
    np.savez(name, x=img_tensor.detach().numpy())


def gen_batch(dataset, labels, batch_size):
    logger.info("Dataset.size() => {}".format(dataset.size()))
    data_size = dataset.size()[0]
    idx = 0
    while idx < data_size:
        batch_data = dataset[idx: min(idx + batch_size, data_size)]
        batch_label = labels[idx: min(idx + batch_size, data_size)]
        idx += batch_size
        yield batch_data, batch_label


param_change = {
    "0.01": "10^{-2}",
    "0.0001": "10^{-4}",
    "1e-08": "10^{-8}",
    "1e-16": "10^{-16}",
}


class Plot:
    @classmethod
    def format_decimal(cls, number):
        # return r"{0:.1}".format(number)
        if number < 0.1:
            _str = '$%.1E' % Decimal(number)
            _str = _str.replace("E", "{*}10^{").replace("+", "").replace("^{0", "^{").replace("^{-0", r"^{-") + "}"
            _str = _str.replace("-", r"\text{\textbf{-}}")
            _str = _str + "$"
            _str = _str.replace("1.0{*}", "")
        else:
            _str = r"${0:.1f}$".format(number)
        return _str

    @staticmethod
    def sort_key(x):
        x = x[0]
        if ":" not in x:
            return (1, x, 0)
        else:
            name, param = x.split(":")
            return (3, name, float(param))

    @classmethod
    def get_name_marker(cls, expl_names):
        markers = {"O": 'o',
                   "Z": 'P',
                   "L": 's',
                   "N": 'X',
                   "G": 'D',
                   "I": 'X',
                   "S": 'P',
                   "R": 'P'}
        namesNmarker = []
        for name in expl_names:
            if ":" in name:
                name, param = name.split(":")
                if "Ridge" in name:
                    name = "Ridge"
                name = (r"\textbf{%s($%s$)}" % (name[0], param_change[param]), markers[name[0]])
            elif "Ground" in name:
                name = (r"\textbf{GT}", markers[name[0]])
            else:
                if "Open" in name:
                    name = (r"\textbf{OpenAPI}", markers[name[0]])
                else:
                    name = (r"\textbf{%s}" % name, markers[name[0]])
            namesNmarker.append(name)
        return namesNmarker

    @classmethod
    def format_name_color(cls, expl_names, short):
        # ['#9467bd', , , '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF']
        colors = {"O": 'r',
                  "Z": 'y',
                  "L": 'c',
                  "N": 'm',
                  "G": 'y',
                  "I": 'b',
                  "S": '#e377c2',
                  "R": 'g'}
        namesNcolor = []
        for name in expl_names:
            if "Open" in name:
                if short == True:
                    name = (r"\textbf{OA}", colors[name[0]])
                else:
                    name = (r"\textbf{OpenAPI}", colors[name[0]])
            elif ":" in name:
                name, param = name.split(":")
                if "Ridge" in name:
                    name = "Ridge"
                name = (r"\textbf{%s($%s$)}" % (name[0], param_change[param]), colors[name[0]])
            elif "Ground" in name:
                name = (r"\textbf{GT}", colors[name[0]])
            else:
                if "Ridge" in name:
                    name = "Ridge"
                if short == True:
                    name = (r"\textbf{%s}" % name[0], colors[name[0]])
                else:
                    name = (r"\textbf{%s}" % name, colors[name[0]])
            namesNcolor.append(name)
        return namesNcolor

    @classmethod
    def plot_errorbar(cls, data, pp, mdl, y_label, x_label, y_limit, y_gap, size, marker_freq):
        expl_names = data.keys()
        namesNmarker = cls.get_name_marker(expl_names)
        namesNcolor = cls.format_name_color(expl_names, True)

        f, ax = plt.subplots(1, figsize=(size[0], size[1]))
        legend = []
        x_tickers = [cls.format_decimal(float(i[0])) for i in  data[expl_names[0]]]
        global_max = 0
        for idx in range(len(expl_names)):
            name, color = namesNcolor[idx]
            marker = namesNmarker[idx][1]
            means = []
            for idx, (param, value) in enumerate(data[expl_names[idx]]):
                y_mean = np.mean(value)
                y_max = np.max(value)
                error = np.array([[y_mean - np.min(value)], [y_max - y_mean]])
                global_max = y_max if y_max > global_max else global_max
                (_, caps, _) = ax.errorbar([idx, ], [y_mean, ], yerr=error, marker=marker, ecolor=color, color=color,
                                           elinewidth=5, capsize=20, markersize=40, markerfacecolor='none', markeredgewidth=5)
                for cap in caps:
                    cap.set_color(color)
                    cap.set_markeredgewidth(10)
                means.append(y_mean)

            ax.plot(range(len(means)),
                    means,
                    linewidth=5,
                    color=color,
                    markeredgecolor=color,
                    marker=marker,
                    markersize=40,
                    markerfacecolor='none',
                    markeredgewidth=5
                    )

            legend.append(name)

        # ymajorLocator = MultipleLocator(y_gap)
        # ax.yaxis.set_major_locator(ymajorLocator)
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda i, j: r"\textbf{%s}" % (i)))
        # xmajorLocator = MultipleLocator(int(v_len / 2))
        # ax.xaxis.set_major_locator(xmajorLocator)

        ax.yaxis.label.set_fontweight('bold')
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.set_tick_params(labelsize=70)
        ax.xaxis.set_tick_params(labelsize=70)
        # ax.set_ylim(y_limit[0], y_limit[1])
        # ax.set_yscale("log")
        x_tickers = ["0", ] + x_tickers
        ax.set_xticklabels(x_tickers)
        # ax.set_xlim(0, v_len)
        ax.set_xlabel(r"\textbf{%s}" % x_label, fontsize=70)
        ax.set_ylabel(r"\textbf{%s}" % y_label, fontsize=70)
        ax.legend(legend, prop={'size': 55, 'weight': 'bold'}, ncol=len(legend), loc=[0.005, 1.01],
                  frameon=False, labelspacing=0, handletextpad=0, columnspacing=0, handlelength=2, mode="expand")
        plt.savefig(pp, format='pdf', bbox_inches="tight")

    @classmethod
    def plot_line(cls, values, expl_names, pp, mdl, y_label, x_label, y_limit, y_gap, size, marker_freq):
        namesNcolor = cls.format_name_color(expl_names, True)
        namesNmarker = cls.get_name_marker(expl_names)
        f, ax = plt.subplots(1, figsize=(size[0], size[1]))
        legend, patches = [], []
        v_len = 0
        for idx, v in enumerate(values):
            v_len = len(v)
            name, color = namesNcolor[idx]
            marker = namesNmarker[idx][1]
            l = name
            if l not in legend:
                line = mlines.Line2D([], [], color=color, marker=marker, linestyle='None',
                                     markersize=55, label=l, markerfacecolor='none', markeredgewidth=10)
                patches.append(line)
                legend.append(l)
            # Original linewidth 5, marker 35
            ax.plot(range(len(v)),
                    v,
                    linewidth=15,
                    color=color,
                    markeredgecolor=color,
                    marker=marker,
                    markersize=55,
                    markevery=marker_freq,
                    markerfacecolor='none',
                    markeredgewidth=10
                    )  # label='handletextpad=0.5',

            # legend.append(name)

        ymajorLocator = MultipleLocator(y_gap)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda i, j: r"\textbf{%s}" % (int(i) if i > 5 else i)))
        xmajorLocator = MultipleLocator(int(v_len / 2))
        ax.xaxis.set_major_locator(xmajorLocator)

        ax.yaxis.label.set_fontweight('bold')
        ax.xaxis.label.set_fontweight('bold')
        # Original 70
        ax.yaxis.set_tick_params(labelsize=90)
        ax.xaxis.set_tick_params(labelsize=90)
        ax.set_ylim(y_limit[0], y_limit[1])
        ax.set_xlim(0, v_len)
        ax.set_xlabel(r"\textbf{%s}" % x_label, fontsize=70)
        ax.set_ylabel(r"\textbf{%s}" % y_label, fontsize=70)
        # Original 55
        ax.legend(legend, prop={'size': 85, 'weight': 'bold'}, ncol=len(legend), loc=[0.005, 1.01],
                  frameon=False, labelspacing=0, handletextpad=0.20, columnspacing=100000., handlelength=1, mode="expand", borderaxespad=0)
        plt.savefig(pp, format='pdf', bbox_inches="tight")

    @classmethod
    def plot_scatter(cls, values, expl_names, pp, mdl, y_label):
        namesNcolor = cls.format_name_color(expl_names, False)
        namesNmarker = cls.get_name_marker(expl_names)
        f, ax = plt.subplots(1, figsize=(18, 6))
        legend, patches = [],  []
        for idx, v in enumerate(values):
            mean = np.mean(v)
            name, marker = namesNmarker[idx]
            name, color = namesNcolor[idx]
            ax.plot([name, ], [mean, ], marker=marker, markerfacecolor="none", markeredgecolor=color, markersize=55,
                    linewidth=0, markeredgewidth=10)
            if "Open" in name:
                l = r"\textbf{OA}"
            elif "L" in name:
                l = r"\textbf{L}"
            elif "R" in name:
                l = r"\textbf{R}"
            elif "N" in name:
                l = r"\textbf{N}"
            elif "Z" in name:
                l = r"\textbf{Z}"
            if l not in legend:
                line = mlines.Line2D([], [], color=color, marker=marker, linestyle='None',
                                     markersize=55, label=l, markerfacecolor='none', markeredgewidth=10)
                patches.append(line)
                legend.append(l)
            ##########################################################################################################
            # Draw values of nodes
            if 0.03 > mean > 0:
                mean_str = cls.format_decimal(mean)
                align = "center"
                if "5.0" in mean_str:
                    mean_str = mean_str.replace("5.0", "5")
                    idx = idx - 0.55
                    align = "left"
                elif "10" in mean_str:
                    idx = idx + 0.2

                if mean > 0.01:
                    height = mean + 0.2
                elif mean < 0.009:
                    height = 0.9
                else:
                    height = 0.5
                ax.text(idx, height, mean_str, fontsize=70, fontweight="bold", rotation=70,
                        horizontalalignment=align)
            elif mean == 0:
                ax.text(idx, mean + 0.20, r"$0$", fontsize=90, fontweight="bold", horizontalalignment="center")
            ##########################################################################################################

        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("right")

        ax.set_xticklabels([i[0] for i in namesNcolor], rotation=50, rotation_mode="anchor")
        ymajorLocator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda i, j: r"\textbf{%s}" % (int(i) if i > 5 else i)))
        # Origin value 80
        ax.yaxis.set_tick_params(labelsize=100)
        ax.xaxis.set_tick_params(labelsize=60)
        ax.yaxis.label.set_fontweight('bold')
        ax.xaxis.label.set_fontweight('bold')
        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel(r"\textbf{%s}" % y_label, fontsize=100)
        ax.legend(handles=patches, prop={'size': 85, 'weight': 'bold'}, ncol=len(legend), loc=[0.005, 1.01],
                  frameon=False, labelspacing=0, handletextpad=0.5, columnspacing=100000, handlelength=0, mode="expand")
        plt.savefig(pp, format='pdf', bbox_inches="tight")

    @classmethod
    def plot_box(cls, values, expl_names, pp, mdl, y_label, log_scale, draw_number, file_name):
        namesNmarker = cls.get_name_marker(expl_names)
        namesNcolor = cls.format_name_color(expl_names, False)

        f, ax = plt.subplots(1, figsize=(18, 6))
        global_max = -1

        legend = []
        patches = []
        for idx, v in enumerate(values):
            mean = np.mean(v)
            min = np.min(v)
            max = np.max(v)
            global_max = max if max > global_max else global_max
            error = np.array([[mean - min], [max - mean]])
            name, marker = namesNmarker[idx]
            name, color = namesNcolor[idx]
            if "Open" in name:
                l = r"\textbf{OA}"
            elif "L" in name:
                l = r"\textbf{L}"
            elif "R" in name:
                l = r"\textbf{R}"
            elif "N" in name:
                l = r"\textbf{N}"
            elif "Z" in name:
                l = r"\textbf{Z}"
            if l not in legend:
                line = mlines.Line2D([], [], color=color, marker=marker, linestyle='None',
                                     markersize=55, label=l, markerfacecolor='none', markeredgewidth=10)
                patches.append(line)
                legend.append(l)

            (_, caps, _) = ax.errorbar([name, ], [mean, ], yerr=error, fmt=marker, ecolor=color, color=color,
                                       elinewidth=8, capsize=20, markersize=50, markerfacecolor='none', markeredgewidth=10)
            for cap in caps:
                cap.set_color(color)
                cap.set_markeredgewidth(10)

            ##########################################################################################################
            # Draw values of nodes
            if draw_number == True:
                if 0 < max < 15:
                    if "LMT" in file_name and "FMNIST" in file_name:
                        height = 200
                    elif "LMT" in file_name and "MNIST" in file_name:
                        height = 500
                    elif "MLP" in file_name and "MNIST" in file_name:
                        height = 35
                    max_str = cls.format_decimal(int(max))
                    if ".0" in max_str:
                        max_str = max_str.replace(".0", "")

                    if max < 10:
                        fontsize = 90
                        idx = idx - 0.2
                    else:
                        fontsize = 80
                        idx = idx - 0.60

                    ax.text(idx, height, max_str, fontsize=fontsize, fontweight="bold", horizontalalignment="left", rotation_mode="anchor")
                # elif 50 > max > 0:
                #     max_str = cls.format_decimal(max)
                #     ax.text(idx, max + 5, max_str, fontsize=80, fontweight="bold",
                #             horizontalalignment="left", rotation=35, rotation_mode="anchor")
                elif max == 0:
                    if "MLP" in file_name:
                        height = 35
                    elif "FMNIST" in file_name and "LMT" in file_name:
                        height = 200
                    elif "FMNIST" in file_name and "MLP" in file_name:
                        height = 80
                    else:
                        height = 500
                    ax.text(idx, max + height, r"$0$", fontsize=90, fontweight="bold", horizontalalignment="center")
            #########################################################################################################

        ax.set_xticklabels([i[0] for i in namesNcolor], rotation=50, rotation_mode="anchor")
        if global_max > 500:
            loc = ticker.MultipleLocator(base=2000)  # this locator puts ticks at regular intervals
            ax.yaxis.set_major_locator(loc)

        if global_max > 500:
            ax.yaxis.set_tick_params(labelsize=70)
        else:
            ax.yaxis.set_tick_params(labelsize=100)
        ax.xaxis.set_tick_params(labelsize=60)
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("right")
        if log_scale == True:
            ax.set_yscale("log")
        if global_max < 500:
            ax.set_ylim(bottom=-30)
        elif global_max < 3000:
            ax.set_ylim(bottom=-220)
        else:
            ax.set_ylim(bottom=-500)

        ax.yaxis.label.set_fontweight('bold')
        ax.xaxis.label.set_fontweight('bold')
        ax.set_ylabel(r"\textbf{%s}" % y_label, fontsize=100)
        ax.legend(handles=patches, prop={'size': 85, 'weight': 'bold'}, ncol=len(legend), loc=[0.005, 1.01],
                  frameon=False, labelspacing=0, handletextpad=0.5, columnspacing=100000, handlelength=0, mode="expand")
        plt.savefig(pp, format='pdf', bbox_inches="tight")


def sample_data(data, labels, size):
    set_random_seed()
    label_data = defaultdict(list)
    for idx, i in enumerate(labels):
        label_data[i.item()].append(data[idx])
    sample_data = []
    sample_label = []
    for i, _data in label_data.items():
        for idx in np.random.choice(range(len(_data)), size=size, replace=False):
            smpl = _data[idx]
            sample_data.append(smpl)
            sample_label.append(i)
    sample_data = torch.stack(sample_data, dim=0)
    sample_label = torch.tensor(sample_label, device=config.DEVICE)
    assert sample_data.size()[1:] == data.size()[1:]
    return sample_data, sample_label


class MNIST:
    _ID2STR = ['0-zero', '1-one', '2-two', '3-three', '4-four',
               '5-five', '6-six', '7-seven', '8-eight', '9-nine']

    _STR2ID = {j: i for i, j in enumerate(_ID2STR)}
    NAME = "MNIST"
    IMG_SIZE = 28
    VAR_NUM = IMG_SIZE * IMG_SIZE

    def __init__(self, cls_names):
        self.cls_ids = [self.str2id(i) for i in cls_names]
        self.cls_names = cls_names

    def load_mnist(self):  # -> return numpy.array()
        print("Use dataset", self.NAME)
        set_random_seed()
        path_manager = PathManager(None, self.NAME)
        trans = transforms.Compose([
            transforms.ToTensor()  # default : range [0, 255] -> [0.0,1.0]
        ])
        train_set, test_set = [], []
        for data_set, if_train in [(train_set, True), (test_set, False)]:
            for img, t in dset.MNIST(root=path_manager.ROOT_DATA_FOLDER.format(self.NAME), train=if_train,
                                     transform=trans,
                                     download=True):
                if t in self.cls_ids:
                    data_set.append((img, t))

        train_imgs = torch.tensor(np.concatenate([i for i, _ in train_set], axis=0), dtype=torch.float64).to(
            config.DEVICE)
        train_labels = torch.tensor(np.array([self.cls_ids.index(j) for _, j in train_set])).to(config.DEVICE)

        test_imgs = torch.tensor(np.concatenate([i for i, _ in test_set], axis=0), dtype=torch.float64).to(
            config.DEVICE)
        test_labels = torch.tensor(np.array([self.cls_ids.index(j) for _, j in test_set])).to(config.DEVICE)

        return train_imgs, train_labels, test_imgs, test_labels

    def display(self, train_labels, test_labels):
        counter = defaultdict(lambda: defaultdict(int))
        for i in train_labels:
            counter["train"][self.id2str(i.item())] += 1
        for i in test_labels:
            counter["test"][self.id2str(i.item())] += 1

        logger.info("Train_labels {} ".format(train_labels))
        logger.info("Test_labels {} ".format(test_labels))
        print("=" * 50)
        print("{}".format("&".join(self.cls_names)))
        print("{}".format("&".join(["{}/{}".format(counter["train"][i], counter["test"][i]) for i in self.cls_names])))
        print("=" * 50)

    @classmethod
    def str2id(cls, cat_str):
        return cls._STR2ID[cat_str]

    def id2str(self, cat_id):
        return MNIST._ID2STR[self.cls_ids[int(cat_id)]]


class FMNIST:
    _ID2STR = ['T-shirt_top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle_boot']

    _STR2ID = {j: i for i, j in enumerate(_ID2STR)}
    NAME = "FMNIST"
    IMG_SIZE = 28
    VAR_NUM = IMG_SIZE * IMG_SIZE

    def __init__(self, cls_names):
        self.cls_ids = [self.str2id(i) for i in cls_names]
        self.cls_names = cls_names

    def load_mnist(self):  # -> return numpy.array()
        print("Use dataset", self.NAME)
        set_random_seed()
        path_manager = PathManager(None, self.NAME)
        trans = transforms.Compose([
            transforms.ToTensor()  # default : range [0, 255] -> [0.0,1.0]
        ])
        train_set, test_set = [], []
        for data_set, if_train in [(train_set, True), (test_set, False)]:
            for img, t in dset.FashionMNIST(root=path_manager.ROOT_DATA_FOLDER.format(self.NAME), train=if_train,
                                            transform=trans,
                                            download=True):
                if t in self.cls_ids:
                    data_set.append((img, t))

        train_imgs = torch.tensor(np.concatenate([i for i, _ in train_set], axis=0), dtype=torch.float64).to(
            config.DEVICE)
        train_labels = torch.tensor(np.array([self.cls_ids.index(j) for _, j in train_set])).to(config.DEVICE)

        test_imgs = torch.tensor(np.concatenate([i for i, _ in test_set], axis=0), dtype=torch.float64).to(
            config.DEVICE)
        test_labels = torch.tensor(np.array([self.cls_ids.index(j) for _, j in test_set])).to(config.DEVICE)

        return train_imgs, train_labels, test_imgs, test_labels

    def display(self, train_labels, test_labels):
        counter = defaultdict(lambda: defaultdict(int))
        for i in train_labels:
            counter["train"][self.id2str(i.item())] += 1
        for i in test_labels:
            counter["test"][self.id2str(i.item())] += 1

        logger.info("Train_labels {} ".format(train_labels))
        logger.info("Test_labels {} ".format(test_labels))
        print("=" * 50)
        print("{}".format("&".join(self.cls_names)))
        print("{}".format("&".join(["{}/{}".format(counter["train"][i], counter["test"][i]) for i in self.cls_names])))
        print("=" * 50)

    @classmethod
    def str2id(cls, cat_str):
        return cls._STR2ID[cat_str]

    def id2str(self, cat_id):
        return FMNIST._ID2STR[self.cls_ids[int(cat_id)]]


class PathManager:
    W_EST_FOLDER = "w_est/{}_{}"
    W_EST_FILE = "{}_{}.npz"
    MODEL_FOLDER = "../model"
    ROOT_DATA_FOLDER = "data/{}"
    MODEL_PATH = "../model/{}_{}.mdl"
    BAD_POINTS = "badpoints"
    RESULT_ROOT = "result"
    JSON_RESULT = ""
    FIGURE_FOLDER = "{}/Figure".format(RESULT_ROOT)

    def __init__(self, mdl_name, dataset):
        self.mdl = mdl_name
        self.dataset = dataset

        for folder in [self.ROOT_DATA_FOLDER, self.MODEL_FOLDER,
                       self.BAD_POINTS, self.RESULT_ROOT, "w_est", self.FIGURE_FOLDER]:
            if not os.path.exists(folder):
                logger.info("Creating folder {}".format(folder))
                os.mkdir(folder)
            elif not os.path.isdir(folder):
                raise Exception("Cannot create the folder {}".format(folder))

    def badpoint_path(self, idx):
        return "{}/exp1_bad_point_{}_{}.npz".format(self.BAD_POINTS, idx, self.dataset)

    def w_est_folder(self):
        # Store the estimated gradients from all methods
        assert self.mdl is not None
        return PathManager.W_EST_FOLDER.format(self.dataset, self.mdl)

    def w_est_file(self, method, size):
        # Store the estimated gradients from all methods
        assert self.mdl is not None
        w_est_folder = self.w_est_folder()
        w_est_file = PathManager.W_EST_FILE.format(method, size)
        if not os.path.exists(w_est_folder):
            os.mkdir(w_est_folder)
        return w_est_folder + "/" + w_est_file

    def mdl_path(self):
        assert self.mdl is not None and self.dataset is not None
        return PathManager.MODEL_PATH.format(self.mdl, self.dataset)

    def result_json_path(self, exp, method, size):
        # Store the result of the experiments
        # Each row is a json which contains the result of one instance
        return "{}/{}_{}_{}_{}_exp_{}.json".format(PathManager.RESULT_ROOT,
                                                   self.dataset,
                                                   self.mdl,
                                                   method,
                                                   size,
                                                   exp)

    def debug_mc_figures(self):
        return 'exp_5_figures_{}_.pdf'.format(self.mdl)

    def image_path(self, method, size, clss):
        return '{}/Figure/{}_{}_{}_{}_{}.pdf'.format(self.RESULT_ROOT, self.mdl, self.dataset, method, size, clss)

    def figure_path(self, size, exp):
        return '{}/Figure/{}_{}_{}_EXP{}.pdf'.format(self.RESULT_ROOT, self.mdl, self.dataset, size, exp)
