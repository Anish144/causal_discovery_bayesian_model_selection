from data.get_data import get_tubingen_pairs_dataset, get_synthetic_dataset, get_simulated_pairs_dataset
from data.get_data import get_cha_pairs_dataset, get_gauss_pairs_dataset, get_multi_pairs_dataset
from data.get_data import get_net_pairs_dataset
from train_methods.gplvm_method import min_causal_score_gplvm
import argparse
import numpy as np
import os
import tensorflow as tf


methods = {
    "gplvm": min_causal_score_gplvm,
}


def main(args: argparse.Namespace):
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(False)

    # Choose the dataset
    if args.data == "cep":
        x, y, weight = get_tubingen_pairs_dataset(
            data_path=f'{args.work_dir}/data/pairs/files'
        )
    elif args.data == "sim":
        x, y, weight = get_simulated_pairs_dataset(
            data_path=f'{args.work_dir}/data/sim_pairs/files'
        )
    elif args.data == "cha_pairs":
        x, y, weight, target = get_cha_pairs_dataset(
            data_path=f'{args.work_dir}/data/cha_pairs/files'
        )
    elif args.data == "cha_pairs":
        x, y, weight, target = get_cha_pairs_dataset(
            data_path=f'{args.work_dir}/data/cha_pairs/files'
        )
    elif args.data == "gauss_pairs":
        x, y, weight, target = get_gauss_pairs_dataset(
            data_path=f'{args.work_dir}/data/gauss_pairs/files'
        )
    elif args.data == "multi_pairs":
        x, y, weight, target = get_multi_pairs_dataset(
            data_path=f'{args.work_dir}/data/multi_pairs/files'
        )
    elif args.data == "net_pairs":
        x, y, weight, target = get_net_pairs_dataset(
            data_path=f'{args.work_dir}/data/net_pairs/files'
        )
    else:
        func_type, noise = args.data.split("-")
        x, y, weight = get_synthetic_dataset(
            num_datasets=100,
            sample_size=100,
            func_string=func_type,
            noise=noise
        )

    # Train whichever method is chosen
    train_method = methods[args.method]
    train_method(
        args=args,
        x=x,
        y=y,
        weight=weight,
        target=target
    )


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir', '-w', type=str, required=True,
    )
    parser.add_argument(
        '--data', '-d', type=str, required=True,
        help="One of [cep, mult_a-normal, ..., add_a-uniform, ..., complex_a-exp]."
    )
    parser.add_argument(
        '--num_inducing', '-ni', type=int, required=True,
        help="Number of inducing points."
    )
    parser.add_argument(
        '--debug', '-deb', action="store_true", default=False,
        help="Will run graph eagerly for debugging."
    )
    parser.add_argument(
        '--plot_fit', '-pf', action="store_true", default=False,
        help="Plot the fit of the conditional estimators."
    )
    parser.add_argument(
        '--method', '-m', type=str,
        help="Choose the method for finding the causal direction. "
        "Should be one of [gplvm,]"
    )
    parser.add_argument(
        '--random_restarts', '-rr', type=int, default=1,
        help="Number of random restarts."
    )
    parser.add_argument(
        '--data_start', '-ds', type=int, default=0,
        help="Data index to start the runs for."
    )
    parser.add_argument(
        '--data_end', '-de', type=int, default=1000,
        help="Data index to end the runs for."
    )
    args = parser.parse_args()
    main(args)