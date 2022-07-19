#!/bin/sh
python3 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="cha_pairs" \
    --num_inducing=200 \
    --plot_fit \
    --random_restarts=20 \
    --method="gplvm" \
