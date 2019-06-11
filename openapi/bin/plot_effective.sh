#!/usr/bin/env bash

python exp/exp_6.py --mdl MLP --dataset FMNIST --datasize 100 --explainer GroundTruth  --gpu cuda:0 --task plot
python exp/exp_6.py --mdl MLP --dataset MNIST --datasize 100 --explainer GroundTruth  --gpu cuda:0 --task plot
python exp/exp_6.py --mdl LMT --dataset FMNIST --datasize 100 --explainer GroundTruth  --gpu cuda:0 --task plot
python exp/exp_6.py --mdl LMT --dataset MNIST --datasize 100 --explainer GroundTruth  --gpu cuda:0 --task plot
