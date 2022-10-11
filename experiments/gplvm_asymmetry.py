"""
This experiment will show that the GPLVM has an asymmetry in its causal
and anticausal factorisation
"""
import tensorflow as tf
import argparse
import os
import numpy as np


def main(args):
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(False)
    return None


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir', '-w', type=str, required=True,
    )
    args = parser.parse_args()
    main(args)