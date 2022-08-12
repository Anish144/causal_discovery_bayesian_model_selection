#!/bin/sh
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
python3 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="cha_pairs" \
    --num_inducing=200 \
    --plot_fit \
    --random_restarts=2 \
    --method="gplvm-adam" \
    --data_start=280 \
    --data_end=300 \
