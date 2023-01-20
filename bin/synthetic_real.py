"""
Script to run the GPLVM on real and synthetic datasets.

Hyperparameters are set at recommended levels.
The results for the paper were set at:
- num_inducing=200
- random_restarts=20

Note that the GPLVMs return -log marginal likelihood. Hence the lower the score
the better.
"""
from gplvm_causal_discovery.data.get_data import (
    get_tubingen_pairs_dataset,
    get_synthetic_dataset,
)
from gplvm_causal_discovery.data.get_data import (
    get_cha_pairs_dataset,
    get_gauss_pairs_dataset,
    get_multi_pairs_dataset,
)
from gplvm_causal_discovery.data.get_data import get_net_pairs_dataset
from gplvm_causal_discovery.data.get_data import (
    get_gplvm_pairs_dataset,
    get_linear_pairs_dataset,
)
from gplvm_causal_discovery.train_methods.gplvm_method import (
    min_causal_score_gplvm,
)
from gplvm_causal_discovery.train_methods.gplvm_generalised import (
    min_causal_score_gplvm_generalised,
)
import argparse
import numpy as np
import tensorflow as tf


methods = {
    "gplvm": min_causal_score_gplvm,
    "gplvm-generalised": min_causal_score_gplvm_generalised,
}


def main(args: argparse.Namespace):
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(False)

    # Choose the dataset
    if args.data == "cep":
        x, y, weight, target = get_tubingen_pairs_dataset(
            data_path=f"{args.work_dir}/data/pairs/files"
        )
    elif args.data == "cha_pairs":
        x, y, weight, target = get_cha_pairs_dataset(
            data_path=f"{args.work_dir}/data/cha_pairs/files"
        )
    elif args.data == "gauss_pairs":
        x, y, weight, target = get_gauss_pairs_dataset(
            data_path=f"{args.work_dir}/data/gauss_pairs/files"
        )
    elif args.data == "multi_pairs":
        x, y, weight, target = get_multi_pairs_dataset(
            data_path=f"{args.work_dir}/data/multi_pairs/files"
        )
    elif args.data == "net_pairs":
        x, y, weight, target = get_net_pairs_dataset(
            data_path=f"{args.work_dir}/data/net_pairs/files"
        )
    elif args.data == "gplvm_pairs":
        x, y, weight, target = get_gplvm_pairs_dataset(
            data_path=f"{args.work_dir}/data/gplvm_pairs/files"
        )
    elif args.data == "linear_pairs":
        x, y, weight, target = get_linear_pairs_dataset(
            data_path=f"{args.work_dir}/data/linear_pairs/files"
        )
    else:
        func_type, noise = args.data.split("-")
        x, y, weight = get_synthetic_dataset(
            num_datasets=100,
            sample_size=100,
            func_string=func_type,
            noise=noise,
        )

    # Train whichever method is chosen
    train_method = methods[args.method]
    train_method(args=args, x=x, y=y, weight=weight, target=target)


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    # Print GPU to check if it is working fine
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        "-w",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="One of [cep, cha_pairs, ...].",
    )
    parser.add_argument(
        "--num_inducing",
        "-ni",
        type=int,
        default=200,
        help="Number of inducing points.",
    )
    parser.add_argument(
        "--debug",
        "-deb",
        action="store_true",
        default=False,
        help="Will run graph eagerly for debugging.",
    )
    parser.add_argument(
        "--plot_fit",
        "-pf",
        action="store_true",
        default=False,
        help="Plot the fit of the conditional estimators.",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help="Choose the method for finding the causal direction. "
        "Should be one of the keys of methods",
    )
    parser.add_argument(
        "--random_restarts",
        "-rr",
        type=int,
        default=1,
        help="Number of random restarts.",
    )
    parser.add_argument(
        "--data_start",
        "-ds",
        type=int,
        required=True,
        help="Data index to start the runs for.",
    )
    parser.add_argument(
        "--data_end",
        "-de",
        type=int,
        required=True,
        help="Data index to end the runs for.",
    )
    parser.add_argument(
        "--num_iterations",
        "-num_it",
        type=int,
        default=100000,
        help="Number of maximum iterations for Adam.",
    )
    parser.add_argument(
        "--minibatch_size",
        "-mini_size",
        type=int,
        default=500,
        help="Size of a minibatch for Adam and stochastic estimator.",
    )
    args = parser.parse_args()
    main(args)
