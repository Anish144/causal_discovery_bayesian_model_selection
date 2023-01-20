"""
In this file, we show that a GPLVM model can fit both causal directions of an
ANM model equally well, however the marginal likelihood can identify the
causal direction.
"""
from data.get_data import get_multi_pairs_dataset
from gpflow.config import default_float
from models.GeneralisedGPLVM import GeneralisedGPLVM
from models.GeneralisedUnsupGPLVM import GeneralisedUnsupGPLVM
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import argparse
import dill
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from collections import namedtuple


GPLVM_SCORES = namedtuple("SCORES", "loss_x loss_y_x loss_y loss_x_y")
FIT_SCORES = namedtuple("FIT", "fit_x fit_y_x fit_y fit_x_y")


def run_optimizer(model, train_dataset, iterations, minibatch_size, adam_lr):
    """
    Utility function running the Adam optimizer
    Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(adam_lr)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    iterator = range(iterations)
    for step in iterator:
        optimization_step()
        neg_elbo = training_loss().numpy()
        logf.append(neg_elbo)
    return logf


def generate_anm_data(sample_size):
    """
    Data generated from an ANM dataset.
    """
    # Sample the cause
    cause = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1))
    # Sample the effect
    kernel = gpflow.kernels.RBF(variance=5, lengthscales=0.5).K(cause, cause)
    noise = np.random.normal(loc=0, scale=0.5, size=(sample_size, 1))
    effect = (
        np.random.multivariate_normal(mean=np.zeros(sample_size), cov=kernel)[
            :, None
        ]
        + noise
    )
    # Normalise data
    cause = StandardScaler().fit_transform(cause).astype(np.float64)
    effect = StandardScaler().fit_transform(effect).astype(np.float64)
    return (cause, effect)


def get_mult_pairs(work_dir):
    """
    Get the mult pairs
    """
    x, y, weight, target = get_multi_pairs_dataset(
        data_path=f"{work_dir}/data/multi_pairs/files"
    )
    return x, y, target


def train_marginal_model(
    x: np.ndarray,
    num_inducing: int,
    num_minibatch: int,
    num_iterations: int,
    adam_lr: float,
):
    # Set hyperparams
    kernel_variance = 1.0
    likelihood_variance = 1e-5
    lamda = np.random.uniform(low=10, high=100, size=[3])
    kernel_lengthscale = 1.0 / lamda**2
    (
        kernel_lengthscale_1,
        kernel_lengthscale_2,
        kernel_lengthscale_3,
    ) = kernel_lengthscale[:]

    # Define kernels
    sq_exp = gpflow.kernels.SquaredExponential(
        lengthscales=kernel_lengthscale_1
    )
    sq_exp.variance.assign(kernel_variance)
    matern = gpflow.kernels.Matern32(lengthscales=kernel_lengthscale_2)
    matern.variance.assign(kernel_variance)
    rquadratic = gpflow.kernels.RationalQuadratic(
        lengthscales=kernel_lengthscale_3
    )
    rquadratic.variance.assign(kernel_variance)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel, matern, rquadratic])
    Z = np.random.randn(num_inducing, 1)

    # Define the approx posteroir
    X_mean_init = 0.1 * tf.cast(x, default_float())
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (x.shape[0], 1)), default_float()
    )

    # Define marginal model
    marginal_model = GeneralisedUnsupGPLVM(
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=likelihood_variance),
        num_mc_samples=10,
        inducing_variable=Z,
        batch_size=num_minibatch,
    )
    # Run optimisation
    data_idx = np.arange(x.shape[0])
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x, data_idx))
        .repeat()
        .shuffle(x.shape[0])
    )
    logf = run_optimizer(
        model=marginal_model,
        train_dataset=train_dataset,
        iterations=num_iterations,
        adam_lr=adam_lr,
        minibatch_size=num_minibatch,
    )

    marginal_model.num_mc_samples = 100
    full_elbo = marginal_model.elbo((x, data_idx))
    print(f"Full Loss: {full_elbo}")

    loss = full_elbo
    fit_metric = marginal_model.predictive_score((x, data_idx))
    return loss, fit_metric


def train_conditional_model(
    x: np.ndarray,
    y: np.ndarray,
    num_inducing: int,
    num_minibatch: int,
    num_iterations: int,
    adam_lr: float,
    seed: int,
    causal: bool,
    mult_pairs: bool,
):
    """
    Train a conditional model using a partially observed GPLVM.
    """
    # Set hyperparams
    kernel_variance = 1.0
    likelihood_variance = 1e-5
    lamda = np.random.uniform(low=10, high=100, size=[3])
    kernel_lengthscale = 1.0 / lamda**2
    (
        kernel_lengthscale_1,
        kernel_lengthscale_2,
        kernel_lengthscale_3,
    ) = kernel_lengthscale[:]

    # Define kernels
    sq_exp = gpflow.kernels.SquaredExponential(
        lengthscales=[kernel_lengthscale_1, kernel_lengthscale_1 * 0.3]
    )
    sq_exp.variance.assign(kernel_variance)
    matern = gpflow.kernels.Matern32(
        lengthscales=[kernel_lengthscale_2, kernel_lengthscale_2 * 0.3]
    )
    matern.variance.assign(kernel_variance)
    rquadratic = gpflow.kernels.RationalQuadratic(
        lengthscales=[kernel_lengthscale_3, kernel_lengthscale_3 * 0.3]
    )
    rquadratic.variance.assign(kernel_variance)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel, matern, rquadratic])

    Z = np.concatenate(
        [
            np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
            np.random.randn(num_inducing, 1),
        ],
        axis=1,
    )

    # Define the approx posteroir
    X_mean_init = 0.01 * tf.cast(y, default_float())
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (y.shape[0], 1)), default_float()
    )

    # Define the conditional model
    conditional_model = GeneralisedGPLVM(
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=likelihood_variance),
        num_mc_samples=10,
        inducing_variable=Z,
        batch_size=num_minibatch,
    )

    # Run optimisation
    data_idx = np.arange(y.shape[0])
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x, y, data_idx))
        .repeat()
        .shuffle(y.shape[0])
    )
    logf = run_optimizer(
        model=conditional_model,
        train_dataset=train_dataset,
        iterations=num_iterations,
        adam_lr=adam_lr,
        minibatch_size=num_minibatch,
    )

    conditional_model.num_mc_samples = 20
    full_elbo = conditional_model.elbo((x, y, data_idx))
    print(f"Full ELBO: {full_elbo}")

    loss = full_elbo
    fit_metric = conditional_model.predictive_score((x, y, data_idx))

    # Plot the fit to see if everything is ok
    obs_new = np.linspace(x.min() - 2, x.max() + 1, 1000)[:, None]
    # Sample from the prior
    lower, median, upper, samples = conditional_model.predict_credible_layer(
        Xnew=obs_new, obs_noise=True
    )
    textstr = "like_var=%.2f\nelbo=%.2f\n" % (
        conditional_model.likelihood.variance.numpy(),
        loss,
    )
    plt.text(x.min() - 6, 0, textstr, fontsize=8)
    plt.scatter(x[:, 0], y[:, 0], c="r")
    plt.plot(obs_new, median, c="b", alpha=0.2)
    plt.fill_between(obs_new[:, 0], upper[:, 0], lower[:, 0], alpha=0.5)

    save_dir = Path(f"run_plots/anm")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.25)
    causal_direction = "causal" if causal else "anticausal"
    plt.savefig(
        save_dir
        / f"anm_exp_multpairs_{mult_pairs}_{causal_direction}_seed_{seed}"
    )
    plt.close()

    return loss, fit_metric


def anm_main(args, seed):
    """
    Fit models in both directions, plot and return the results
    """
    # Set hyperparams
    num_inducing = args.num_inducing
    num_minibatch = args.num_minibatch
    num_iterations = args.num_iterations
    adam_lr = args.adam_lr

    # Get data
    if not args.mult_pairs:
        cause, effect = generate_anm_data(args.sample_size)
    else:
        x, y, target = get_mult_pairs(args.work_dir)
        if target[seed] == -1:
            cause, effect = y[seed], x[seed]
        else:
            cause, effect = x[seed], y[seed]

    # Fit X -> Y direction
    x_score, x_fit_metric = train_marginal_model(
        x=cause,
        num_inducing=num_inducing,
        num_minibatch=num_minibatch,
        num_iterations=num_iterations,
        adam_lr=adam_lr,
    )
    y_x_score, y_x_fit_metric = train_conditional_model(
        cause,
        effect,
        num_inducing=num_inducing,
        num_minibatch=num_minibatch,
        num_iterations=num_iterations,
        adam_lr=adam_lr,
        seed=seed,
        mult_pairs=args.mult_pairs,
        causal=True,
    )

    # Fit Y -> X direction
    y_score, y_fit_metric = train_marginal_model(
        x=effect,
        num_inducing=num_inducing,
        num_minibatch=num_minibatch,
        num_iterations=num_iterations,
        adam_lr=adam_lr,
    )
    x_y_score, x_y_fit_metric = train_conditional_model(
        effect,
        cause,
        num_inducing=num_inducing,
        num_minibatch=num_minibatch,
        num_iterations=num_iterations,
        adam_lr=adam_lr,
        seed=seed,
        mult_pairs=args.mult_pairs,
        causal=False,
    )

    x_to_y_score = x_score + y_x_score
    x_to_y_fit_metric = x_fit_metric + y_x_fit_metric

    y_to_x_score = y_score + x_y_score
    y_to_x_fit_metric = y_fit_metric + x_y_fit_metric

    print(f"X -> Y: Score {x_to_y_score}, Fit {x_to_y_fit_metric}")
    print(f"X -> Y Fit: X {x_fit_metric}, Y|X {y_x_fit_metric}")
    print(f"Y -> X: Score {y_to_x_score}, Fit {y_to_x_fit_metric}")
    print(f"Y -> X Fit: Y {y_fit_metric}, X|Y {x_y_fit_metric}")

    # Save the scores
    scores = GPLVM_SCORES(x_score, y_x_score, y_score, x_y_score)
    fits = FIT_SCORES(
        x_fit_metric, y_x_fit_metric, y_fit_metric, x_y_fit_metric
    )
    save_dict = {"scores": scores, "fit": fits}
    save_dir = Path(f"{args.work_dir}/results/anm_exp")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"anm_results_multpairs_{args.mult_pairs}_seed_{seed}"
    with open(save_dir / save_name, "wb") as f:
        dill.dump(save_dict, f)


if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
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
        "--num_inducing",
        "-ni",
        type=int,
        required=True,
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
        "--minibatch_size",
        "-mini_size",
        type=int,
        default=500,
        help="Size of a minibatch.",
    )
    parser.add_argument(
        "--mult_pairs",
        "-mp",
        action="store_true",
        default=False,
        help="Do the experiments on mult pairs.",
    )
    args = parser.parse_args()
