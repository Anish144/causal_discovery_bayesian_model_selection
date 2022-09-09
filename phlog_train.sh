#!/bin/sh
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
python3 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="gauss_pairs" \
    --num_inducing=200 \
    --plot_fit \
    --random_restarts=15 \
    --method="gplvm" \
    --data_start=120 \
    --data_end=140 \
