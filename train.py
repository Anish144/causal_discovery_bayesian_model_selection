import gpflow
import tensorflow as tf
import numpy as np
from tqdm import trange
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scipy.stats import norm
from data.get_data import get_tubingen_pairs_dataset, get_synthetic_dataset

from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM
from gpflow.utilities import ops
from gpflow.config import default_float

import tensorflow_probability as tfp



def ml_estimate(x):
    """
    Find the log likelihood.

    This doesn't make sense after I have normalised the data.
    """
    score = - np.log(
        np.sum(
            norm.pdf(x)
        )
    )
    return score


def train(x, y, num_inducing, run_number, direction, no_observed=False):
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
    sq_exp.variance.assign(1.0)
     # lambda = 5 in this
    sq_exp.lengthscales.assign(1. / 5)

    m = gpflow.models.GPR(data=(x, y), kernel=sq_exp, mean_function=None)
    m.likelihood.variance.assign(1. / (1 ** 2))
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
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[found_lengthscale, 1./ 5])
    else:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[1./ 5])
    kernel.variance.assign(1.0)
     # lambda = 5 in this

    X_mean_init = y - m.predict_f(x)[0]
    # X_mean_init = ops.pca_reduce(y, latent_dim)
    # X_mean_init = tfp.distributions.Normal(loc=0, scale=1).sample([y.shape[0], latent_dim])
    # X_mean_init = tf.cast(X_mean_init, dtype=default_float())
    # X_mean_init = tf.zeros((y.shape[0], latent_dim), dtype=default_float())
    X_var_init = tf.ones((y.shape[0], latent_dim), dtype=default_float())

    if not no_observed:
        m = PartObsBayesianGPLVM(
            data=y,
            in_data=x,
            kernel=kernel,
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            num_inducing_variables=num_inducing,
        )
        m.likelihood.variance.assign(found_lik_var)
    else:
        m = gpflow.models.BayesianGPLVM(
            data=y,
            kernel=kernel,
            X_data_mean=tf.zeros((y.shape[0], latent_dim), dtype=default_float()),
            X_data_var=X_var_init,
            num_inducing_variables=num_inducing,
        )
        m.likelihood.variance.assign(1.0)

    # Only train variational parameters first
    gpflow.utilities.set_trainable(m.kernel, False)
    gpflow.utilities.set_trainable(m.likelihood, False)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=10000),
    )

    # Train all the parameters
    gpflow.utilities.set_trainable(m.kernel, True)
    gpflow.utilities.set_trainable(m.likelihood, True)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=10000),
    )
    if not no_observed:
        print(f"\n {m.kernel.variance.numpy()}, {m.likelihood.variance.numpy()}, {m.kernel.lengthscales.numpy()}")

    loss = - m.elbo()
    return loss


def main():
    tf.config.run_functions_eagerly(True)

    rng = np.random.RandomState(0)
    tf.random.set_seed(0)
    np.random.seed(0)

    correct_idx = []
    wrong_idx = []
    num_inducing = 100

    # x, y, weight = get_tubingen_pairs_dataset(
    #     data_path='/vol/bitbucket/ad6013/Research/gp-causal/data/pairs/files'
    # )
    x, y, weight = get_synthetic_dataset(
        num_datasets=100,
        sample_size=100,
        func_string="mult_a",
        noise='uniform'
    )

    scores = []
    for i in tqdm(range(len(x)), desc="Epochs", leave=True, position=0):
        rng = np.random.RandomState(0)
        tf.random.set_seed(0)
        np.random.seed(0)
        # i = 15
        print(f'\n {i}')
        # Ignore the high dim
        if x[i].shape[-1] > 1:
            continue
        else:
            # Get data points
            x_train, y_train, weight_train = x[i], y[i], weight[i]
        # Make sure data is standardised
        x_train = StandardScaler().fit_transform(x_train).astype(np.float64)
        y_train = StandardScaler().fit_transform(y_train).astype(np.float64)
        # x -> y score
        loss_x_ml = ml_estimate(x_train)
        loss_x = train(
            x=np.random.normal(loc=0, scale=1,size=x_train.shape),
            y=x_train,
            no_observed=True,
            num_inducing=num_inducing,
            run_number=i,
            direction=0
        )
        print(f"Log: {loss_x_ml}, {loss_x}")
        loss_x_y = train(x=x_train, y=y_train, num_inducing=num_inducing, run_number=i, direction=0)
        # x <- y score
        loss_y_ml = ml_estimate(y_train)
        loss_y = train(
            x=np.random.normal(loc=0, scale=1,size=x_train.shape),
            y=y_train,
            no_observed=True,
            num_inducing=num_inducing,
            run_number=i,
            direction=1
        )
        print(f"Log: {loss_y_ml}, {loss_y}")
        loss_y_x = train(x=y_train, y=x_train, num_inducing=num_inducing, run_number=i, direction=1)
        # Calculate losses
        score_x_y = loss_x + loss_x_y
        score_y_x = loss_y + loss_y_x
        print(f"Run {i}: {score_x_y} ; {score_y_x}")
        if score_x_y < score_y_x:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
        scores.append((score_x_y.numpy(), score_y_x.numpy()))
    return correct_idx, wrong_idx, weight, scores


if __name__ == "__main__":
    corr, wrong, weight, scores = main()
    correct_weight = [weight[i] for i in corr]
    wrong_weight = [weight[i] for i in wrong]
    accuracy = np.sum(correct_weight) / (np.sum(correct_weight) + np.sum(wrong_weight))
    print(f"\n Scores: {scores}")
    print(f"\n Final accuracy: {accuracy}")
    import pickle
    # save_name = "fullscore-cep-gplvm-sqexp-basicinitialisation"
    save_name = "test"
    with open(f'/vol/bitbucket/ad6013/Research/gp-causal/results/{save_name}.p', 'wb') as f:
        pickle.dump((accuracy, scores), f)