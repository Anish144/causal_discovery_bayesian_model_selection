#!/bin/sh
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
python3 bin/synthetic_real.py \
    --work_dir="./gplvm_causal_discovery" \
    --data="gauss_pairs" \
    --num_inducing=200 \
    --plot_fit \
    --random_restarts=1 \
    --method="gplvm-generalised" \
    --data_start=1 \
    --data_end=2 \
