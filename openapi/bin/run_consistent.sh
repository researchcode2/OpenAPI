#!/usr/bin/env bash

###############################
# Experiment 7 Integrated Gradient
###############################
nohup python exp/exp_7.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient --gpu cuda:0 > LMT_EXP7_FMNIST_100_INTG.log
#
nohup python exp/exp_7.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient  --gpu cuda:0 > LMT_EXP7_MNIST_100_INTG.log
#
nohup python exp/exp_7.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient  --gpu cuda:0 > MLP_EXP7_FMNIST_100_INTG.log
#
nohup python exp/exp_7.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature IntegratedGradient  --gpu cuda:0 > MLP_EXP7_MNIST_100_INTG.log

###############################
# Experiment 7 Decision Feature
###############################

nohup python exp/exp_7.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature OpenAPI --gpu cuda:0 > LMT_EXP7_FMNIST_100_OPENAPI.log
#
nohup python exp/exp_7.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature OpenAPI --gpu cuda:0 > LMT_EXP7_MNIST_100_OPENAPI.log
#
nohup python exp/exp_7.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature OpenAPI --gpu cuda:0 > MLP_EXP7_FMNIST_100_OPENAPI.log
#
nohup python exp/exp_7.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature OpenAPI --gpu cuda:0 > MLP_EXP7_MNIST_100_OPENAPI.log


###############################
# Experiment 7 Saliency Map
###############################

nohup python exp/exp_7.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > LMT_EXP7_FMNIST_100_SALIENCY.log
#
nohup python exp/exp_7.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > LMT_EXP7_MNIST_100_SALIENCY.log
#
nohup python exp/exp_7.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > MLP_EXP7_FMNIST_100_SALIENCY.log
#
nohup python exp/exp_7.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature Saliency --gpu cuda:0 > MLP_EXP7_MNIST_100_SALIENCY.log


###############################
# Experiment 7 Gradient * Input
###############################

nohup python exp/exp_7.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > LMT_EXP7_FMNIST_100_GRADINPUT.log
#
nohup python exp/exp_7.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > LMT_EXP7_MNIST_100_GRADINPUT.log
#
nohup python exp/exp_7.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > MLP_EXP7_FMNIST_100_GRADINPUT.log
#
nohup python exp/exp_7.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature GradTimeInput --gpu cuda:0 > MLP_EXP7_MNIST_100_GRADINPUT.log
#

###############################
# Experiment 7 LIMEOrigin
###############################
nohup python exp/exp_7.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin --gpu cuda:0 > LMT_EXP7_FMNIST_100_LIME.log &

nohup python exp/exp_7.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin  --gpu cuda:1 > LMT_EXP7_MNIST_100_LIME.log &

nohup python exp/exp_7.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin  --gpu cuda:2 > MLP_EXP7_FMNIST_100_LIME.log &

nohup python exp/exp_7.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth --task compute --feature LIMEOrigin  --gpu cuda:3 > MLP_EXP7_MNIST_100_LIME.log &
