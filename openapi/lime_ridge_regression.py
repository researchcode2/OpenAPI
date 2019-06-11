#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
"""
Functions for explaining classifiers that use Image data.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from sklearn.linear_model import LinearRegression

from openapi.utils import set_random_seed
from openapi.abs_explainer import AbsExplainer
from openapi.log import getLogger
from openapi import config
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
import lime.lime_base as lime_base

logger = getLogger(__name__)


class LimeImageExplainerLinearRegression(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    # Modification: Add the parameter dist
    def __init__(self, dist, num_var, kernel_width=.25, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            dist: sampling distance
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose, random_state=self.random_state)
        self.dist = dist
        self.verbose = verbose
        self.num_var = num_var

    def explain_instance(self, image, classifier_fn, clss, labels, distance_metric='cosine', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            predicts: iterable with labels to be explained.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation function

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        #######################################################################
        # Modification:
        # - We use each pixel a segment
        # if len(image.shape) == 2:
        #     image = gray2rgb(image)
        # if random_seed is None:
        #     random_seed = self.random_state.randint(0, high=1000)
        #
        # if segmentation_fn is None:
        #     segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
        #                                             max_dist=200, ratio=0.2,
        #                                             random_seed=random_seed)
        # try:
        #     segments = segmentation_fn(image)
        # except ValueError as e:
        #     raise e
        # fudged_image = image.copy()
        # separate images to different parts based on the segmentation algorithm
        # if hide_color is None:
        #     print("segment shape", segments.shape)
        #     print(np.unique(segments))
        #     for x in np.unique(segments):
        #         print("x shape", x.shape)
        #         fudged_image[segments == x] = (
        #             np.mean(image[segments == x][:, 0]),
        #             np.mean(image[segments == x][:, 1]),
        #             np.mean(image[segments == x][:, 2]))
        # else:
        #     fudged_image[:] = hide_color
        #######################################################################


        ########################################################################################
        # Modification:
        # - fudged_image, segments are not needed as we use each pixel as a segment
        # fudged_image = image.copy()
        # data, labels = self.data_labels(image, fudged_image, segments,
        #                                 classifier_fn, num_samples,
        #                                 batch_size=batch_size)
        # -----------------------------------------------------------------------
        num_samples = self.num_var
        data, predicts = self.data_labels(image, classifier_fn, num_samples)
        ########################################################################################

        data_cpu = data.cpu()

        distances = sklearn.metrics.pairwise_distances(data_cpu, image.cpu(), metric=distance_metric).ravel()

        ###############################################################################
        # Modification:
        # - We are only interested in the local explanation
        # ret_exp = ImageExplanation(image, segments)
        # if top_labels:
        #     top = np.argsort(labels[0])[-top_labels:]
        #     ret_exp.top_labels = list(top)
        #     ret_exp.top_labels.reverse()
        # for label in top:
        #     (ret_exp.intercept[label],
        #      ret_exp.local_exp[label],
        #      ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
        #         data, labels, distances, label, num_features,
        #         model_regressor=model_regressor,
        #         feature_selection=self.feature_selection)
        # return ret_exp
        # --------------------------------------------------------------------------------
        weight_diff = {}
        for i in labels:
            if i != clss:
                coef = self.explain_instance_with_data(data_cpu, predicts.cpu(), distances, clss, i, model_regressor=model_regressor)[0]

                weight_diff[i] = coef
        ###############################################################################
        # Remove the target sample from the perturbed instances
        data = data[1:]
        return weight_diff, data

    def data_labels(self, image, classifier_fn, num_samples):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        #############################################################################
        # Modification:
        # - Sample perturbed instances within the hypercube
        # - Generate perturbed instances by adding perturbations to the original image
        # import sys
        # Each item is a super-pixel
        # n_features = np.unique(segments).shape[0]
        # binary matrix is generated
        # data = self.random_state.randint(0, 2, num_samples * n_features) \
        #     .reshape((num_samples, n_features))
        # labels = []
        # data[0, :] = 1
        # imgs = []
        # for row in data:
        #     temp = copy.deepcopy(image)
        #     zeros = np.where(row == 0)[0]
        #     mask = np.zeros(segments.shape).astype(bool)
        #     for z in zeros:
        #         mask[segments == z] = True
        #     temp[mask] = fudged_image[mask]
        #     imgs.append(temp)
        #     if len(imgs) == batch_size:
        #         preds = classifier_fn(np.array(imgs))
        #         labels.extend(preds)
        #         imgs = []
        # if len(imgs) > 0:
        #     preds = classifier_fn(np.array(imgs))
        #     labels.extend(preds)
        # return data, np.array(labels)
        # ----------------------------------------------------------------------------
        n_features = image.shape[1]
        data = torch.zeros((num_samples + 1, n_features), dtype=torch.float64, device=config.DEVICE).uniform_(
            -self.dist, self.dist)
        data[0, :] = 0
        assert torch.sum(abs(data)) != 0
        data += image
        logger.debug("data size", data.size())
        labels = classifier_fn(data)
        logger.debug("labels", labels.size())
        return data, labels.detach()
        #############################################################################

    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, distances, label_i, label_j, model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
        """

        from sklearn.linear_model import Ridge
        weights = self.base.kernel_fn(distances)
        labels_column_i = neighborhood_labels[:, label_i]
        labels_column_j = neighborhood_labels[:, label_j]
        # P[i] / P[j]
        P_i_j = torch.log(labels_column_i / labels_column_j)

        num_features = self.num_var
        #################################################################################################
        # Modification
        # - we estimate the important for all features
        # used_features = self.base.feature_selection(neighborhood_data, labels_column, weights, num_features,
        #                                             method=feature_selection)
        # assert used_features.shape[0] == num_features
        #################################################################################################
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        easy_model = model_regressor
        ####################################################################################
        # Modification:
        # - Change pytorch tensor to numpy array otherwise they cannot be used by sklearn
        neighborhood_data = neighborhood_data.cpu().numpy()
        labels_column = P_i_j.cpu().numpy()
        ####################################################################################
        easy_model.fit(neighborhood_data, labels_column, sample_weight=weights)
        prediction_score = easy_model.score(neighborhood_data, labels_column, sample_weight=weights)
        logger.info("Prediction_score {}".format(prediction_score))
        if self.verbose:
            local_pred = easy_model.predict(neighborhood_data[0].reshape(1, -1))
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred, )
        #####################################################################################
        # Modification:
        # - only the learned original coef and intercept are needed, no need to return other information
        # return (easy_model.intercept_,
        #         sorted(zip(used_features, easy_model.coef_),
        #                key=lambda x: np.abs(x[1]), reverse=True),
        #         prediction_score, local_pred)
        # ------------------------------------------------------------------------------------
        coef_ = torch.tensor(easy_model.coef_, dtype=torch.float64, device=config.DEVICE).view(-1, num_features)
        intercept_ = torch.tensor([[easy_model.intercept_]], dtype=torch.float64, device=config.DEVICE)
        return coef_, intercept_


class LimeRidgeRegression(AbsExplainer):
    def __init__(self, sample_num, dist, var_num, verbose):
        ##################################################################################################
        # Modification:
        # - We do not need segmenter as we use each pixel as a segmentation
        # Make sure that each pixel is a segmentation
        # self.segmenter = SegmentationAlgorithm("quickshift", kernel_size=1, max_dist=0.0001, ratio=0.2)
        ##################################################################################################
        AbsExplainer.__init__(self, sample_num, var_num)
        self.var_num = var_num
        self.py = None
        self.sample_num = sample_num
        self.dist = dist
        self.verbose = verbose
        self.explainer = LimeImageExplainerLinearRegression(self.dist, var_num, feature_selection='none')


    def weight_diff(self, x_tensor, model, clss, clss_num):
        set_random_seed()
        def predict_fun(x):
            rst = model.forward(x)
            return rst
        weight_diff, samples_tensor = self.explainer.explain_instance(x_tensor, predict_fun, clss, labels=list(range(clss_num)))
        return weight_diff, self.dist, samples_tensor

    def decision_f(self, x_tensor, model, clss, clss_num):
        w_diff_tensor, dist, samples_tensor = self.weight_diff(x_tensor, model, clss, clss_num)
        assert type(w_diff_tensor) is dict
        decision_f = self._decision_f(w_diff_tensor, clss_num)
        return decision_f, None, dist, samples_tensor, w_diff_tensor
