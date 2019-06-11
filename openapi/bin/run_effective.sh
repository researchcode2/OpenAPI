#!/usr/bin/env bash

###############################
# Experiment 6 Integrated Gradient
###############################
nohup python exp/exp_6.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient --gpu cuda:0 > LMT_EXP6_FMNIST_100_INTG.log
#
nohup python exp/exp_6.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient  --gpu cuda:0 > LMT_EXP6_MNIST_100_INTG.log
#
nohup python exp/exp_6.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient  --gpu cuda:0 > MLP_EXP6_FMNIST_100_INTG.log
#
nohup python exp/exp_6.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient  --gpu cuda:0 > MLP_EXP6_MNIST_100_INTG.log

###############################
# Experiment 6 Decision Feature
###############################

nohup python exp/exp_6.py --mdl LMT --dataset FMNIST --datasize 100 --explainer OpenAPI --task compute --feature OpenAPI --gpu cuda:0 > LMT_EXP6_FMNIST_100_OPENAPI.log
#
nohup python exp/exp_6.py --mdl LMT --dataset MNIST --datasize 100 --explainer OpenAPI --task compute --feature OpenAPI --gpu cuda:0 > LMT_EXP6_MNIST_100_OPENAPI.log
#
nohup python exp/exp_6.py --mdl MLP --dataset FMNIST --datasize 100 --explainer OpenAPI --task compute --feature OpenAPI --gpu cuda:0 > MLP_EXP6_FMNIST_100_OPENAPI.log
#
nohup python exp/exp_6.py --mdl MLP --dataset MNIST --datasize 100 --explainer OpenAPI --task compute --feature OpenAPI --gpu cuda:0 > MLP_EXP6_MNIST_100_OPENAPI.log


###############################
# Experiment 6 Saliency Map
###############################

nohup python exp/exp_6.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > LMT_EXP6_FMNIST_100_SALIENCY.log
#
nohup python exp/exp_6.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > LMT_EXP6_MNIST_100_SALIENCY.log
#
nohup python exp/exp_6.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > MLP_EXP6_FMNIST_100_SALIENCY.log
#
nohup python exp/exp_6.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > MLP_EXP6_MNIST_100_SALIENCY.log


###############################
# Experiment 6 Gradient * Input
###############################

nohup python exp/exp_6.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > LMT_EXP6_FMNIST_100_GRADINPUT.log
#
nohup python exp/exp_6.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > LMT_EXP6_MNIST_100_GRADINPUT.log
#
nohup python exp/exp_6.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > MLP_EXP6_FMNIST_100_GRADINPUT.log
#
nohup python exp/exp_6.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > MLP_EXP6_MNIST_100_GRADINPUT.log
#

###############################
# Experiment 6 LIME
###############################

nohup python exp/exp_6.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin --gpu cuda:0 > LMT_EXP6_FMNIST_100_LIMEOrigin.log
#
nohup python exp/exp_6.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin --gpu cuda:0 > LMT_EXP6_MNIST_100_LIMEOrigin.log
#
nohup python exp/exp_6.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin --gpu cuda:0 > MLP_EXP6_FMNIST_100_LIMEOrigin.log
#
nohup python exp/exp_6.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin --gpu cuda:0 > MLP_EXP6_MNIST_100_LIMEOrigin.log
#
