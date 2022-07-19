#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=ad6013 # required to send email notifcations - please replace <your_username> with your college login name or email address
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
/vol/bitbucket/ad6013/envs/gp-causal-3.8/bin/python3.8 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="cha_pairs" \
    --num_inducing=200 \
    --plot_fit \
    --random_restarts=20 \
    --method='gplvm' \
    --data_start=150 \
    --data_end=300 \