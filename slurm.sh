#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<your_username> # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/ad6013/envs/gp-causal-env/bin/:$PATH
source activate
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
python3 train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/gp-causal" \
    --data="cep" \
    --num_inducing=100 \
    --plot_fit \
    --random_restarts=20 \