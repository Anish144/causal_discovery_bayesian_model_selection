#!/bin/sh
python3 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="gauss_pairs" \
    --num_inducing=200 \
    --plot_fit \
    --random_restarts=1 \
    --method="gplvm" \
    --data_start=0 \
    --data_end=1 \
