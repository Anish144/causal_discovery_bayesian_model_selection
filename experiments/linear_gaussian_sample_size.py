"""
Experiment to show that the Linear Gaussian case can be identifiable upto
an error with increasing sample size.
There will be two modes:
    - Optimising the hyperparams
    - Not optimising the hyperparams
"""
from pathlib import Path
from collections import namedtuple
import math
import gpflow
import dill
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import positive
from gpflow.base import Parameter
from tqdm import trange

tfd = tfp.distributions


SS_EXP = namedtuple("SS_EXP", "x_to_y_list y_to_x_list")
GPFLOW_JITTER = 1e-4


class BayesianNormalDE(tf.Module):
    def __init__(
        self,
        data,
        a_0_initial=1.0,
        b_0_initial=1.0,
        mu_0_initial=1.0,
        lambda_0_initial=1.0,
    ):
        super(BayesianNormalDE, self).__init__()
        self.x = data
        self.a_0 = Parameter(
            [a_0_initial], dtype=tf.float64, transform=positive()
        )
        self.b_0 = Parameter(
            [b_0_initial], dtype=tf.float64, transform=positive()
        )
        self.mu_0 = Parameter(mu_0_initial, dtype=tf.float64)
        self.lambda_0 = Parameter(
            lambda_0_initial, dtype=tf.float64, transform=positive()
        )

    def calculate_marginal_likelihood(
        self,
    ):
        """
        Taken from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        """
        # Calculate intermediate variables
        N = tf.cast(self.x.shape[0], tf.float64)
        mean_x = tf.reduce_mean(self.x)
        lambda_n = self.lambda_0 + N
        a_n = self.a_0 + N / 2
        # b_n term
        diff_mean = 0.5 * tf.reduce_sum((self.x - mean_x) ** 2)
        num_term_2 = self.lambda_0 * N * ((mean_x - self.mu_0) ** 2)
        denom_term_2 = 2 * (self.lambda_0 + N)
        b_n = self.b_0 + diff_mean + (num_term_2 / denom_term_2)
        # Calculate marginal likelihood
        marginal_likelihood = -(N / 2) * tf.math.log(
            2 * tf.cast(math.pi, tf.float64)
        )
        marginal_likelihood += 0.5 * (
            tf.math.log(self.lambda_0 + 1e-20) - tf.math.log(lambda_n + 1e-20)
        )
        marginal_likelihood += self.a_0 * tf.math.log(self.b_0 + 1e-20)
        marginal_likelihood -= a_n * tf.math.log(b_n + 1e-20)
        marginal_likelihood += tf.math.lgamma(a_n + 1e-20) - tf.math.lgamma(
            self.a_0 + 1e-20
        )
        return marginal_likelihood

    def training_loss(self):
        return -self.calculate_marginal_likelihood()

    def return_training_variables(self):
        return (self.a_0, self.b_0, self.mu_0, self.lambda_0)


class BayesianLinearRegression(tf.Module):
    def __init__(
        self,
        data,
        a_0_initial=1.0,
        b_0_initial=1.0,
        mu_0_initial=[1.0, 1.0],
        lambda_0_initial=[1.0, 1.0],
    ) -> None:
        super(BayesianLinearRegression, self).__init__()
        self.x, self.y = data
        self.a_0 = Parameter(
            [a_0_initial], dtype=tf.float64, transform=positive()
        )
        self.b_0 = Parameter(
            [b_0_initial], dtype=tf.float64, transform=positive()
        )
        self.mu_0 = Parameter(
            [[mu_0_initial[0]], [mu_0_initial[1]]], dtype=tf.float64
        )
        self.lambda_0 = Parameter(
            lambda_0_initial, dtype=tf.float64, transform=positive()
        )

    def calculate_marginal_likelihood(
        self,
    ):
        # Calculate intermediate variables
        N = tf.cast(self.x.shape[0], tf.float64)
        # [2 X 2]
        design_matrix = tf.linalg.matmul(self.x, self.x, transpose_a=True)
        lambda_0_matrix = tf.linalg.diag(self.lambda_0)
        lambda_n = design_matrix + lambda_0_matrix
        # design_inv = tf.linalg.inv(design_matrix)
        # [2 X 1]
        beta_hat = tf.linalg.solve(
            design_matrix, tf.linalg.matmul(self.x, self.y, transpose_a=True)
        )
        # Calculate mu_n
        # lambda_n_inv = tf.linalg.inv(lambda_n)
        prior_matmul = tf.linalg.matmul(lambda_0_matrix, self.mu_0)
        design_beta_matmul = tf.linalg.matmul(design_matrix, beta_hat)
        mu_n = tf.linalg.solve(lambda_n, design_beta_matmul + prior_matmul)
        # Calculate a_n
        a_n = self.a_0 + N / 2
        # Calculate b_n
        yT_y = tf.linalg.matmul(self.y, self.y, transpose_a=True)
        muT_lambda_mu = tf.linalg.matmul(
            tf.linalg.matmul(self.mu_0, lambda_0_matrix, transpose_a=True),
            self.mu_0,
        )
        muNT_lambdaN_muN = tf.linalg.matmul(
            tf.linalg.matmul(mu_n, lambda_n, transpose_a=True), mu_n
        )
        b_n = self.b_0 + 0.5 * (yT_y + muT_lambda_mu + muNT_lambdaN_muN)
        # Calculate marginal likelihood
        marginal_likelihood = -(N / 2) * tf.math.log(
            2 * tf.cast(math.pi, tf.float64)
        )
        tf.debugging.check_numerics(marginal_likelihood, message="step 1")
        marginal_likelihood += tf.cast(0.5, tf.float64) * (
            tf.linalg.logdet(lambda_0_matrix + 1e-20)
            - tf.linalg.logdet(lambda_n + 1e-20)
        )
        tf.debugging.check_numerics(marginal_likelihood, message="step 2")
        marginal_likelihood += self.a_0 * tf.math.log(
            self.b_0 + 1e-20
        ) - a_n * tf.math.log(b_n + 1e-20)
        tf.debugging.check_numerics(marginal_likelihood, message="step 3")
        marginal_likelihood += tf.math.lgamma(a_n + 1e-20) - tf.math.lgamma(
            self.a_0 + 1e-20
        )
        tf.debugging.check_numerics(marginal_likelihood, message="step 4")
        return marginal_likelihood

    def training_loss(self):
        return -self.calculate_marginal_likelihood()

    def return_training_variables(self):
        return (self.a_0, self.b_0, self.mu_0, self.lambda_0)


def run_adam(model, iterations, train_dataset, minibatch_size):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()


def calculate_marginal_likelihood(
    x,
):
    """
    The marginal model in this case will be a Gaussian distribution. We can thus
    simply find the log pdf.
    """
    mean = tf.reduce_mean(x, axis=0)
    std = tfp.stats.stddev(x, sample_axis=0)
    normal_dist = tfd.Normal(loc=mean, scale=std)
    log_pdf = tf.reduce_sum(normal_dist.log_prob(x))
    # tf.print(f"Marginal score: {log_pdf}")
    return log_pdf


def train_marginal_model(
    x,
):
    # Need to add an offset to the design matrix
    model = BayesianNormalDE(
        data=x,
        a_0_initial=1.0,
        b_0_initial=1.0,
        mu_0_initial=0.0,
        lambda_0_initial=1e-5,
    )
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        model.training_loss,
        model.trainable_variables,
        options=dict(maxiter=1000),
    )
    loss = model.calculate_marginal_likelihood()
    final_loss = loss.numpy()[0]
    return final_loss


def train_conditional_gp_model(x, y):
    """
    We will train the conditional model by using a linear kernel, and using
    the evidence approximation for other hyperparameters.
    """
    num_inducing = 500 if len(x) > 500 else len(x)
    # We will use a linear kernel with zero mean GP
    kernel = gpflow.kernels.Linear(variance=tf.cast(1.0, tf.float64))
    inducing_variable = np.linspace(np.min(x), np.max(x), num_inducing).reshape(
        -1, 1
    )
    m = gpflow.models.SGPR(
        data=(x, y),
        kernel=kernel,
        mean_function=gpflow.mean_functions.Identity(),
        # likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
    )
    m.likelihood.variance.assign(tf.math.reduce_std(y) ** 2)
    # Train for a short while using adam to reduc
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x, y)).repeat().shuffle(x.shape[0])
    )
    run_adam(m, 2000, train_dataset=train_dataset, minibatch_size=x.shape[0])
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=1000)
    )
    loss = m.elbo()
    # loss = m.log_marginal_likelihood()
    # tf.print(f"Conditional score: {loss}")
    return loss


def train_bayesian_linear_regression(x, y):
    """
    Train Bayesian linear regression with marginal likelihood.

    We will use a normal-inverse gamma prior. We will train the hyperparameters
    by maximising the marginal likelihood.
    """
    # Need to add an offset to the design matrix
    x = tf.concat([x, tf.ones_like(x)], 1)
    model = BayesianLinearRegression(
        data=(x, y),
        a_0_initial=1.0,
        b_0_initial=1.0,
        mu_0_initial=[0.0, 0.0],
        lambda_0_initial=[1e-5, 1e-5],
    )
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        model.training_loss,
        model.trainable_variables,
        options=dict(maxiter=1000),
    )
    loss = model.calculate_marginal_likelihood()
    final_loss = loss.numpy()[0][0]
    return final_loss


def sample_datagen_hyperparams():
    a_0 = tfd.Normal(0, 2.0).sample(1)
    sigma_0 = tfd.InverseGamma(1.0, 2.0).sample(1)
    a_1 = tfd.Normal(4, 2.0).sample(1)
    sigma_1 = tfd.InverseGamma(1.0, 2.0).sample(1)
    w_0 = tfd.Normal(-100.0, 10).sample(1)
    return (a_0, sigma_0, a_1, sigma_1, w_0)


def generate_linear_gaussian_data(sample_size):
    a_0, sigma_0, a_1, sigma_1, w_0 = sample_datagen_hyperparams()
    X = tfd.Normal(a_0, sigma_0).sample(sample_size)
    Y = a_1 * X + w_0 + tfd.Normal(0.0, sigma_1).sample(sample_size)
    # Send mean to zero
    # X -= tf.reduce_mean(X)
    # Y -= tf.reduce_mean(Y)
    # X = StandardScaler().fit_transform(X).astype(np.float64)
    # Y = StandardScaler().fit_transform(Y).astype(np.float64)
    assert X.shape == Y.shape
    return (tf.cast(X, tf.float64), tf.cast(Y, tf.float64))


def linear_gaussian_sample_size_exp(args):
    # Generate the data
    x, y = generate_linear_gaussian_data(sample_size=args.sample_size)
    # X -> Y
    x_loss = train_marginal_model(x)
    y_x_loss = train_bayesian_linear_regression(x, y)
    # Y -> X
    y_loss = train_marginal_model(y)
    x_y_loss = train_bayesian_linear_regression(y, x)
    # Full losses
    x_to_y_loss = x_loss + y_x_loss
    y_to_x_loss = y_loss + x_y_loss
    tf.print(f"X -> Y: {x_to_y_loss}", f"Y -> X: {y_to_x_loss}")
    if x_to_y_loss - y_to_x_loss > 1e-6:
        tf.print(f"Correct! Difference is {x_to_y_loss - y_to_x_loss} \n")
    elif y_to_x_loss - x_to_y_loss > 1e-6:
        tf.print(f"Wrong! Difference is {y_to_x_loss - x_to_y_loss} \n")
    else:
        tf.print("Undecided! \n")
    return x_to_y_loss, y_to_x_loss


def repeat_experiment(args, repeat_times):
    x_to_y_list = []
    y_to_x_list = []
    for i in trange(repeat_times, desc=f"SS {args.sample_size}"):
        np.random.seed(i)
        tf.random.set_seed(i)
        x_to_y, y_to_x = linear_gaussian_sample_size_exp(args)
        x_to_y_list.append(x_to_y)
        y_to_x_list.append(y_to_x)
    return x_to_y_list, y_to_x_list


def vary_sample_sizes(args, sample_size_list):
    ans_dict = {}
    for ss in sample_size_list:
        args.sample_size = ss
        x_to_y_list, y_to_x_list = repeat_experiment(args, args.repeat_times)
        results = SS_EXP(x_to_y_list, y_to_x_list)
        ans_dict[ss] = results
    # Save results
    save_dir = Path(f"{args.work_dir}/results/lin_gauss_exp")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"lingauss_results_repeats_{args.repeat_times}_bayesianlr_optimise_genoffset_offset_wideprior.p"
    with open(save_dir / save_name, "wb") as f:
        dill.dump(ans_dict, f)


if __name__ == "__main__":
    import argparse
    import os

    # tf.config.run_functions_eagerly(True)

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
        "--sample_size",
        "-ss",
        type=int,
        default=1000,
        help="Sample size for dataset.",
    )
    parser.add_argument(
        "--repeat_times",
        "-rt",
        type=int,
        default=50,
        help="Number of times an experiment is repeated.",
    )
    args = parser.parse_args()
    vary_sample_sizes(args, np.linspace(10000, 10000, 1))
