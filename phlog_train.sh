#!/bin/sh
python3 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="add_a-normal" \
    --num_inducing=10 \
    --plot_fit \
    --random_restarts=2 \
    --method="gplvm" \
