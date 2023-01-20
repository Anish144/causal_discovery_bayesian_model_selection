import argparse
import gpflow
import numpy as np
import os
import sys
import tensorflow as tf
from experiments.anm_identifiable import anm_main
from experiments.linear_gaussian_sample_size import (
    linear_gaussian_sample_size_exp,
)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        "-w",
        type=str,
        default="/vol/bitbucket/ad6013/Research/gp-causal",
    )
    parser.add_argument(
        "--num_inducing",
        "-ni",
        type=int,
        default=200,
        help="Number of inducing points.",
    )
    parser.add_argument(
        "--sample_size",
        "-ss",
        type=int,
        default=1000,
        help="Sample size for dataset.",
    )
    parser.add_argument(
        "--num_iterations",
        "-num_it",
        type=int,
        default=100000,
        help="NUmber of maximum iterations.",
    )
    parser.add_argument(
        "--num_minibatch",
        "-mini_size",
        type=int,
        default=500,
        help="Size of a minibatch.",
    )
    parser.add_argument(
        "--adam_lr",
        "-lr",
        type=float,
        default=0.01,
        help="Learning rate for adam.",
    )
    parser.add_argument(
        "--mult_pairs",
        "-mp",
        action="store_true",
        default=False,
        help="Do the experiments on mult pairs.",
    )
    args = parser.parse_args()
    for i in range(40, 100):
        tf.print(f"Seed: {i}")
        np.random.seed(i)
        tf.random.set_seed(i)
        anm_main(args, seed=i)
