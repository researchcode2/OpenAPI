#!/usr/bin/env bash

###############################
# Experiment 2 FMNIST
###############################


nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer OpenAPI --gpu cuda:0 --task compute > MLP_EXP2_FMNIST_100_O.log &
 ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:0 --task compute > MLP_EXP2_FMNIST_100_L2.log

nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task compute > MLP_EXP2_FMNIST_100_L4.log

nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:2 --task compute > MLP_EXP2_FMNIST_100_L8.log


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:0 --task compute > MLP_EXP2_FMNIST_100_N2.log &

nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:1 --task compute > MLP_EXP2_FMNIST_100_N4.log &

nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task compute > MLP_EXP2_FMNIST_100_N8.log &
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:0 --task compute > MLP_EXP2_FMNIST_100_Z2.log &

nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:1 --task compute > MLP_EXP2_FMNIST_100_Z4.log &

nohup python exp/exp_2.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:2 --task compute > MLP_EXP2_FMNIST_100_Z8.log &


###############################
# Experiment 2 MNIST
###############################
nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer OpenAPI --gpu cuda:0 --task compute > MLP_EXP2_MNIST_100_O.log &
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:0 --task compute > MLP_EXP2_MNIST_100_L2.log
#
nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task compute > MLP_EXP2_MNIST_100_L4.log

nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:2 --task compute > MLP_EXP2_MNIST_100_L8.log
#
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:0 --task compute > MLP_EXP2_MNIST_100_N2.log &

nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:1 --task compute > MLP_EXP2_MNIST_100_N4.log &

nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task compute > MLP_EXP2_MNIST_100_N8.log &
#
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:0 --task compute > MLP_EXP2_MNIST_100_Z2.log &

nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:1 --task compute > MLP_EXP2_MNIST_100_Z4.log &

nohup python exp/exp_2.py --mdl MLP --dataset MNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:2 --task compute > MLP_EXP2_MNIST_100_Z8.log &
#


###############################
# Experiment 2 FMNIST
###############################


nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer OpenAPI --gpu cuda:0 --task compute > LMT_EXP2_FMNIST_100_O.log &
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:0 --task compute > LMT_EXP2_FMNIST_100_L2.log

nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task compute > LMT_EXP2_FMNIST_100_L4.log

nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:2 --task compute > LMT_EXP2_FMNIST_100_L8.log
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:0 --task compute > LMT_EXP2_FMNIST_100_N2.log &

nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:1 --task compute > LMT_EXP2_FMNIST_100_N4.log &

nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task compute > LMT_EXP2_FMNIST_100_N8.log &
#
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:0 --task compute > LMT_EXP2_FMNIST_100_Z2.log &

nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:1 --task compute > LMT_EXP2_FMNIST_100_Z4.log &

nohup python exp/exp_2.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:2 --task compute > LMT_EXP2_FMNIST_100_Z8.log &

#
##############################
# Experiment 2 MNIST
###############################
nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer OpenAPI --gpu cuda:0 --task compute > LMT_EXP2_MNIST_100_O.log &
 ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:0 --task compute > LMT_EXP2_MNIST_100_L2.log

nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task compute > LMT_EXP2_MNIST_100_L4.log

nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:2 --task compute > LMT_EXP2_MNIST_100_L8.log
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:0 --task compute > LMT_EXP2_MNIST_100_N2.log &

nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:1 --task compute > LMT_EXP2_MNIST_100_N4.log &

nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task compute > LMT_EXP2_MNIST_100_N8.log &

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:0 --task compute > LMT_EXP2_MNIST_100_Z2.log &

nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:1 --task compute > LMT_EXP2_MNIST_100_Z4.log &

nohup python exp/exp_2.py --mdl LMT --dataset MNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:2 --task compute > LMT_EXP2_MNIST_100_Z8.log &
#
