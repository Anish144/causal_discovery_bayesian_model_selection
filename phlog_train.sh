#!/bin/sh
python3 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="sim" \
    --num_inducing=500 \
    --plot_fit \
    --random_restarts=2 \
