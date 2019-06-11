#!/usr/bin/env bash

###############################
# Experiment 1 FMNIST MLP
###############################
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer OpenAPI --gpu cuda:0 --task grad > MLP_EXP1_FMNIST_100_O.log &
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:0 --task grad > MLP_EXP1_FMNIST_100_L2.log

nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task grad > MLP_EXP1_FMNIST_100_L4.log

nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:2 --task grad > MLP_EXP1_FMNIST_100_L8.log
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:0 --task grad > MLP_EXP1_FMNIST_100_N2.log &
#
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:1 --task grad > MLP_EXP1_FMNIST_100_N4.log &
#
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task grad > MLP_EXP1_FMNIST_100_N8.log &
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:0 --task grad > MLP_EXP1_FMNIST_100_Z2.log &
#
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:1 --task grad > MLP_EXP1_FMNIST_100_Z4.log &
#
nohup python exp/exp_1.py --mdl MLP --dataset FMNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:2 --task grad > MLP_EXP1_FMNIST_100_Z8.log &
#
#

###############################
# Experiment 2 MNIST MLP
###############################
nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer OpenAPI --gpu cuda:0 --task grad > MLP_EXP1_MNIST_100_O.log &
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:0 --task grad > MLP_EXP1_MNIST_100_L2.log

nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task grad > MLP_EXP1_MNIST_100_L4.log

nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:2 --task grad > MLP_EXP1_MNIST_100_L8.log

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:0 --task grad > MLP_EXP1_MNIST_100_N2.log &

nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:1 --task grad > MLP_EXP1_MNIST_100_N4.log &

nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task grad > MLP_EXP1_MNIST_100_N8.log &

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:0 --task grad > MLP_EXP1_MNIST_100_Z2.log &

nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:1 --task grad > MLP_EXP1_MNIST_100_Z4.log &

nohup python exp/exp_1.py --mdl MLP --dataset MNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:2 --task grad > MLP_EXP1_MNIST_100_Z8.log &



###############################
# Experiment 1 MNIST LMT
###############################
nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer OpenAPI --gpu cuda:0 --task grad > LMT_EXP1_MNIST_100_O.log &
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:1 --task grad > LMT_EXP1_MNIST_100_L2.log

nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task grad > LMT_EXP1_MNIST_100_L4.log

nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:1 --task grad > LMT_EXP1_MNIST_100_L8.log

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:2 --task grad > LMT_EXP1_MNIST_100_N2.log &
#
nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:2 --task grad > LMT_EXP1_MNIST_100_N4.log &
#
nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task grad > LMT_EXP1_MNIST_100_N8.log &

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:3 --task grad > LMT_EXP1_MNIST_100_Z2.log &

nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:3 --task grad > LMT_EXP1_MNIST_100_Z4.log &

nohup python exp/exp_1.py --mdl LMT --dataset MNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:3 --task grad > LMT_EXP1_MNIST_100_Z8.log &



###############################
# Experiment 1 FMNIST LMT
###############################
nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer OpenAPI --gpu cuda:1 --task grad > LMT_EXP1_FMNIST_100_O.log &
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.01 --gpu cuda:1 --task grad > LMT_EXP1_FMNIST_100_L2.log

nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:0.0001 --gpu cuda:1 --task grad > LMT_EXP1_FMNIST_100_L4.log

nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer LIMELinearRegression:1e-08 --gpu cuda:1 --task grad > LMT_EXP1_FMNIST_100_L8.log

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Naive:0.01 --gpu cuda:2 --task grad > LMT_EXP1_FMNIST_100_N2.log &

nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Naive:0.0001 --gpu cuda:2 --task grad > LMT_EXP1_FMNIST_100_N4.log &

nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Naive:1e-08 --gpu cuda:2 --task grad > LMT_EXP1_FMNIST_100_N8.log &
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Zeroth:0.01 --gpu cuda:3 --task grad > LMT_EXP1_FMNIST_100_Z2.log &

nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Zeroth:0.0001 --gpu cuda:3 --task grad > LMT_EXP1_FMNIST_100_Z4.log &

nohup python exp/exp_1.py --mdl LMT --dataset FMNIST --datasize 100 --explainer Zeroth:1e-08 --gpu cuda:3 --task grad > LMT_EXP1_FMNIST_100_Z8.log &
#
