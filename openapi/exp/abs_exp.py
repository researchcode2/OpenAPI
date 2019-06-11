#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod
import os
from openapi.mdl.dnn import MLPNet
from openapi.lime_linear_regression import LimeLinearRegression
from openapi.lime_ridge_regression import LimeRidgeRegression
from openapi.mdl.lmt import LinearModelTree
from openapi.zeroth import Zeroth
from openapi.naive_est import Naive
from openapi.openapi_est import OpenAPI
from openapi.utils import set_random_seed, PathManager, FMNIST, sample_data, MNIST
from openapi.log import getLogger

set_random_seed()

logger = getLogger(__name__)


class AbsExp:
    OpenAPI = "OpenAPI"
    LIMELinearRegression = "LIMELinearRegression"
    LIMERidgeRegression = "LIMERidgeRegression"
    Naive = "Naive"
    Zeroth = "Zeroth"
    GroundTrueh = "GroundTruth"

    INTEGRATED_GRADIENT = "IntegratedGradient"
    SALIENCY = "Saliency"
    GRADTIMEINPUT = "GradTimeInput"
    LIMEOrigin = "LIMEOrigin"

    @classmethod
    def get_attribute_method_names(cls):
        return [cls.INTEGRATED_GRADIENT, cls.SALIENCY, cls.GRADTIMEINPUT, cls.OpenAPI, cls.LIMEOrigin]

    @classmethod
    def get_method_names(cls):
        methods = []
        methods.append(cls.OpenAPI)
        methods.append(cls.GroundTrueh)
        for i in [0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.0001, 1e-8, 1e-16]:
            methods.append("{}:{}".format(cls.Naive, i))
            methods.append("{}:{}".format(cls.LIMELinearRegression, i))
            methods.append("{}:{}".format(cls.LIMERidgeRegression, i))
            methods.append("{}:{}".format(cls.Zeroth, i))
        return methods

    def __init__(self, model_name, dataset, exp, expln_name, data_size):
        """
        Initialize the environments used in the experiments.
        """
        if dataset == FMNIST.NAME:
            fmnist = FMNIST(FMNIST._ID2STR)
            test_imgs, test_labels = fmnist.load_mnist()[2:]
            self.cls_names = FMNIST._ID2STR
            self.cls_num = len(self.cls_names)
            self.var_num = FMNIST.VAR_NUM
            self.img_size = FMNIST.IMG_SIZE
        elif dataset == MNIST.NAME:
            mnist = MNIST(MNIST._ID2STR)
            test_imgs, test_labels = mnist.load_mnist()[2:]
            self.cls_names = MNIST._ID2STR
            self.cls_num = len(self.cls_names)
            self.var_num = MNIST.VAR_NUM
            self.img_size = MNIST.IMG_SIZE
        else:
            raise Exception("Dataset is not supported")

        (self.images, self.labels) = sample_data(test_imgs, test_labels, data_size)

        self.path_manager = PathManager(model_name, dataset)

        if model_name == MLPNet.NAME:
            from openapi.mdl.dnn import load_model
            self.mdl = load_model(self.path_manager.mdl_path(), self.cls_num, self.var_num)
        elif model_name == LinearModelTree.NAME:
            # LMT related module must be imported to make the pickle load successfully
            from openapi.mdl.lmt import evaluate, train_model, load_model, Node, LocalLasso
            self.mdl = load_model(self.path_manager.mdl_path())
        else:
            raise Exception("Model is not supported")

        self.dataset = dataset
        self.model_name = model_name
        self.data_size = data_size
        self.exp = exp
        if expln_name != AbsExp.GroundTrueh:
            self.explainer = self._methods(self.var_num)[expln_name]
        else:
            self.explainer = None
        self.expln_name = expln_name
        self.result_json = self.path_manager.result_json_path(self.exp, self.expln_name, self.data_size)
        # Create folders of the output
        if os.path.isfile(self.result_json):
            _RST_JSON = self.result_json + ".bak"
            logger.info("Move {} => {}".format(self.result_json, _RST_JSON))
            os.rename(self.result_json, _RST_JSON)

    def _methods(self, var_num):
        """
        Load the input images and initialize gradient estimate methods.
        :param var_num: dimension of the input instance
        :param img_size: width/height of the image. We assume the image has the same width and height.
        """

        # Methods used in the experiment
        methods = {}
        methods[self.OpenAPI] = OpenAPI(var_num + 1, 1e-8, var_num)
        for i in [0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.0001, 1e-8, 1e-16]:
            methods["{}:{}".format(self.Naive, i)] = Naive(var_num, i, var_num)
            methods["{}:{}".format(self.LIMELinearRegression, i)] = LimeLinearRegression(var_num + 1, i, var_num,
                                                                                         False)
            methods["{}:{}".format(self.LIMERidgeRegression, i)] = LimeRidgeRegression(var_num + 1, i, var_num,
                                                                                         False)
            methods["{}:{}".format(self.Zeroth, i)] = Zeroth(var_num, i)
        return methods

    @abstractmethod
    def run(self):
        pass
