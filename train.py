from data.get_data import get_tubingen_pairs_dataset, get_synthetic_dataset
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import ops
from gpflow.utilities import positive
from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm import trange
import argparse
import gpflow
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
from time import process_time
from multiprocessing import Pool


def ml_estimate(x):
    """
    Find the log likelihood.

    This doesn't make sense after I have normalised the data.
    """
    score = - np.sum(
        np.log(
            norm.pdf(x)
        )
    )
    return score


def train(
    x,
    y,
    num_inducing,
    kernel_variance,
    kernel_lengthscale,
    likelihood_variance,
    no_observed=False,
):
    # if len(x) < num_inducing + 1:
    #     inducing = x
    # else:
    #     # kmeans = KMeans(n_clusters=num_inducing).fit(x)
    #     # inducing = kmeans.cluster_centers_
    #     inducing_idx = np.random.choice(x.shape[0], num_inducing, replace=False)
    #     inducing = np.take(x, inducing_idx, axis=0)

    # sq_exp = gpflow.kernels.SquaredExponential()
    # sq_exp.variance.assign(1.0)
    #  # lambda = 5 in this
    # sq_exp.lengthscales.assign(1. / 10)

    # mat32 = gpflow.kernels.Matern32()
    # mat32.variance.assign(1.0)
    # mat32.lengthscales.assign(1. / 10)

    # mat52 = gpflow.kernels.Matern52()
    # mat52.variance.assign(1.0)
    # mat52.lengthscales.assign(1. / 10)

    # kernel = gpflow.kernels.Sum([sq_exp, mat32, mat52])

    # m = gpflow.models.SGPR((x, y), kernel, inducing)
    # # kappa = 10 in this
    # m.likelihood.variance.assign(1. / (100 ** 2))

    # Find the best lengthscale for the observed bit
    sq_exp = gpflow.kernels.SquaredExponential()
    sigmoid = tfp.bijectors.Sigmoid(
        low=tf.cast(1e-6, tf.float64), high=tf.cast(100, tf.float64)
    )
    sq_exp.lengthscales = Parameter(
        kernel_lengthscale,
        transform=sigmoid, dtype=tf.float64
    )
    sigmoid = tfp.bijectors.Sigmoid(
        low=tf.cast(1e-6, tf.float64), high=tf.cast(50, tf.float64)
    )
    sq_exp.variance = Parameter(kernel_variance, transform=sigmoid, dtype=tf.float64)
    m = gpflow.models.GPR(data=(x, y), kernel=sq_exp, mean_function=None)
    m.likelihood.variance = Parameter(likelihood_variance, transform=positive(lower=1e-6))

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=10000)
    )
    found_lengthscale = m.kernel.lengthscales.numpy()
    found_lik_var = m.likelihood.variance.numpy()

    latent_dim = 1
    # need a lengthscale for the latent dim as well as for the oberved
    # Lengthscale of observed is slightly larger
    if not no_observed:
        kernel = gpflow.kernels.SquaredExponential()
        sigmoid = tfp.bijectors.Sigmoid(
            low=tf.cast(1e-6, tf.float64), high=tf.cast(100, tf.float64)
        )
        kernel.lengthscales = Parameter(
            [found_lengthscale] + [kernel_lengthscale],
            transform=sigmoid, dtype=tf.float64
        )
        # kernel.lengthscales = tf.cast(kern_len_param, dtype=default_float())
    else:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[kernel_lengthscale])
    sigmoid = tfp.bijectors.Sigmoid(
        low=tf.cast(1e-6, tf.float64), high=tf.cast(50, tf.float64)
    )
    kernel.variance = Parameter(kernel_variance, transform=sigmoid, dtype=tf.float64)
     # lambda = 5 in this

    X_mean_init = y - m.predict_f(x)[0]
    # X_mean_init = ops.pca_reduce(y, latent_dim)
    # X_mean_init = tfp.distributions.Normal(loc=0, scale=1).sample([y.shape[0], latent_dim])
    # X_mean_init = tf.cast(X_mean_init, dtype=default_float())
    # X_mean_init = tf.zeros((y.shape[0], latent_dim), dtype=default_float())
    X_var_init = tf.math.square(X_mean_init - tf.math.reduce_mean(X_mean_init, axis=0)) + 1
    # X_var_init = tf.ones((y.shape[0], latent_dim), dtype=default_float())

    if not no_observed:
        m = PartObsBayesianGPLVM(
            data=y,
            in_data=x,
            kernel=kernel,
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            num_inducing_variables=num_inducing,
        )
        m.likelihood.variance = Parameter(found_lik_var, transform=positive(lower=1e-6))
    else:
        m = gpflow.models.BayesianGPLVM(
            data=y,
            kernel=kernel,
            X_data_mean=tf.zeros((y.shape[0], latent_dim), dtype=default_float()),
            X_data_var=X_var_init,
            num_inducing_variables=num_inducing,
        )
        m.likelihood.variance = Parameter(likelihood_variance, transform=positive(lower=1e-6))

 # Train only inducing variables
    gpflow.utilities.set_trainable(m.kernel, False)
    gpflow.utilities.set_trainable(m.likelihood, False)
    gpflow.utilities.set_trainable(m.X_data_mean , False)
    gpflow.utilities.set_trainable(m.X_data_var, False)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=10000),
    )

    # Train only x_var
    gpflow.utilities.set_trainable(m.kernel, False)
    gpflow.utilities.set_trainable(m.likelihood, False)
    gpflow.utilities.set_trainable(m.X_data_mean , False)
    gpflow.utilities.set_trainable(m.X_data_var, True)
    gpflow.utilities.set_trainable(m.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=10000),
    )

    # Train all the hyperparameters
    gpflow.utilities.set_trainable(m.kernel, True)
    gpflow.utilities.set_trainable(m.likelihood, True)
    gpflow.utilities.set_trainable(m.X_data_mean , False)
    gpflow.utilities.set_trainable(m.X_data_var, False)
    gpflow.utilities.set_trainable(m.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=10000),
    )

    # Train everything
    gpflow.utilities.set_trainable(m.kernel, True)
    gpflow.utilities.set_trainable(m.likelihood, True)
    gpflow.utilities.set_trainable(m.X_data_mean , True)
    gpflow.utilities.set_trainable(m.X_data_var, True)
    gpflow.utilities.set_trainable(m.inducing_variable, True)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=10000),
    )
    loss = - m.elbo()
    return loss


def calculate_causal_score(seed, x, y, num_inducing):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Set seed for each run (useful for debugging)
    # Sample random hyperparams, one for each experiment
    # Kernel variance will always be 1
    kernel_variance = 5.0
    # Likelihood variance
    kappa = np.random.uniform(
        low=1.0, high=100, size=[4]
    )
    likelihood_variance = 1.0 / (kappa ** 2)
    # Kernel lengthscale
    lamda = np.random.uniform(
        low=1.0, high=10, size=[4]
    )
    kernel_lengthscale = 1.0 / lamda
    if args.debug:
        print(
            f"Initial values: {likelihood_variance}, {kernel_lengthscale}"
        )

    x_train, y_train = x, y
    # Make sure data is standardised
    x_train = StandardScaler().fit_transform(x_train).astype(np.float64)
    y_train = StandardScaler().fit_transform(y_train).astype(np.float64)
    # x -> y score
    loss_x = train(
        x=np.random.normal(loc=0, scale=1,size=x_train.shape),
        y=x_train,
        no_observed=True,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale=kernel_lengthscale[0],
        likelihood_variance=likelihood_variance[0]
    )
    if args.debug:
        loss_x_ml = ml_estimate(x_train)
        print(f"Log: {loss_x_ml}, {loss_x}")
    print("Y|X")
    loss_y_x = train(
        x=x_train,
        y=y_train,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale=kernel_lengthscale[1],
        likelihood_variance=likelihood_variance[1]
    )
    # x <- y score
    loss_y = train(
        x=np.random.normal(loc=0, scale=1,size=x_train.shape),
        y=y_train,
        no_observed=True,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale=kernel_lengthscale[2],
        likelihood_variance=likelihood_variance[2]
    )
    if args.debug:
        loss_y_ml = ml_estimate(y_train)
        print(f"Log: {loss_y_ml}, {loss_y}")
    print("X|Y")
    loss_x_y = train(
        x=y_train,
        y=x_train,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale=kernel_lengthscale[3],
        likelihood_variance=likelihood_variance[3]
    )
    return (loss_x, loss_y_x, loss_y, loss_x_y)
    # Save time
    # if min(rr_loss_x) + min(rr_loss_y_x) < min(rr_loss_y) + min(rr_loss_x_y):
    #     break


def main(args: argparse.Namespace):
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(False)
    # tf.debugging.enable_check_numerics()

    correct_idx = []
    wrong_idx = []
    num_inducing = args.num_inducing

    # Choose the dataset
    if args.data == "cep":
        x, y, weight = get_tubingen_pairs_dataset(
            data_path='/rds/general/user/ad6013/home/Research/gp-causal/data/pairs/files'
        )
    else:
        func_type, noise = args.data.split("-")
        x, y, weight = get_synthetic_dataset(
            num_datasets=100,
            sample_size=100,
            func_string=func_type,
            noise=noise
        )

    scores = []
    for i in tqdm(range(len(x)), desc="Epochs", leave=True, position=0):
        # Ignore the high dim
        if x[i].shape[-1] > 1:
            continue
        np.random.seed(i)
        tf.random.set_seed(i)
        print(f'\n Run: {i}')
        print(f"Correct: {len(correct_idx)}, Wrong: {len(wrong_idx)}")

        rr_loss_x = []
        rr_loss_y_x = []
        rr_loss_y = []
        rr_loss_x_y = []
        for j in range(args.random_restarts):
            seed = args.random_restarts * i + j
            print(f"\n Random restart: {j}")
            (
                loss_x,
                loss_y_x,
                loss_y,
                loss_x_y
            ) = calculate_causal_score(
                seed=seed,
                x=x[i],
                y=y[i],
                num_inducing=num_inducing
            )
            if loss_x is not None:
                rr_loss_x.append(loss_x)
                rr_loss_y_x.append(loss_y_x)
                rr_loss_y.append(loss_y)
                rr_loss_x_y.append(loss_x_y)
                print(loss_x.numpy(), loss_y_x.numpy(), loss_y.numpy(), loss_x_y.numpy())
        # Need to find the best losses from the list
        # Calculate losses
        if args.debug:
            print(
                f"x: {rr_loss_x} \n y_x: {rr_loss_y_x} \n y: {rr_loss_y} \n x_y: {rr_loss_x_y}"
            )
        score_x_y = min(rr_loss_x) + min(rr_loss_y_x)
        score_y_x = min(rr_loss_y) + min(rr_loss_x_y)
        print(f"Run {i}: {score_x_y} ; {score_y_x}")
        if score_x_y < score_y_x:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
        scores.append((score_x_y.numpy(), score_y_x.numpy()))
    return correct_idx, wrong_idx, weight, scores


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    parser = argparse.ArgumentParser()
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
        '--random_restarts', '-rr', type=int, default=1,
        help="Number of random restarts."
    )
    args = parser.parse_args()
    with tf.device('gpu'):
        tf.print(tf.config.list_physical_devices('GPU'))
        corr, wrong, weight, scores = main(args)
    correct_weight = [weight[i] for i in corr]
    wrong_weight = [weight[i] for i in wrong]
    accuracy = np.sum(correct_weight) / (np.sum(correct_weight) + np.sum(wrong_weight))
    print(f"\n Scores: {scores}")
    print(f"\n Final accuracy: {accuracy}")
    import pickle
    save_name = f"fullscore-{args.data}-gplvm-sqexp-reinit{args.random_restarts}"
    # save_name = "test"
    with open(f' /rds/general/user/ad6013/home/Research/gp-causal/results/{save_name}.p', 'wb') as f:
        pickle.dump((accuracy, scores), f)