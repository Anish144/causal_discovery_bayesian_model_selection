import gpflow
import tensorflow as tf
import numpy as np
from tqdm import trange
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scipy.stats import norm
from data.get_data import get_tubingen_pairs_dataset, get_synthetic_dataset


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


def train(x, y, num_inducing, run_number, direction):
    if len(x) < num_inducing + 1:
        inducing = x
    else:
        # kmeans = KMeans(n_clusters=num_inducing).fit(x)
        # inducing = kmeans.cluster_centers_
        inducing_idx = np.random.choice(x.shape[0], num_inducing, replace=False)
        inducing = np.take(x, inducing_idx, axis=0)

    sq_exp = gpflow.kernels.SquaredExponential()
    sq_exp.variance.assign(1.0)
     # lambda = 5 in this
    sq_exp.lengthscales.assign(1. / 10)

    mat32 = gpflow.kernels.Matern32()
    mat32.variance.assign(1.0)
    mat32.lengthscales.assign(1. / 10)

    mat52 = gpflow.kernels.Matern52()
    mat52.variance.assign(1.0)
    mat52.lengthscales.assign(1. / 10)

    kernel = gpflow.kernels.Sum([sq_exp, mat32, mat52])

    m = gpflow.models.SGPR((x, y), kernel, inducing)
    # kappa = 10 in this
    m.likelihood.variance.assign(1. / (100 ** 2))

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=10000),
    )

    loss = - m.elbo()
    return loss


def main():
    rng = np.random.RandomState(0)
    tf.random.set_seed(0)

    correct_idx = []
    wrong_idx = []
    num_inducing = 500

    x, y, weight = get_tubingen_pairs_dataset(
        data_path='/vol/bitbucket/ad6013/Research/gp-causal/data/pairs/files'
    )
    x, y, weight = get_synthetic_dataset(
        num_datasets=100,
        sample_size=100,
        func_string="mult_b",
        noise='uniform'
    )

    for i in tqdm(range(len(x)), desc="Epochs", leave=True, position=0):
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
        loss_x = ml_estimate(x=x_train)
        loss_x_y = train(x=x_train, y=y_train, num_inducing=num_inducing, run_number=i, direction=0)
        # x <- y score
        loss_y = ml_estimate(x=y_train)
        loss_y_x = train(x=y_train, y=x_train, num_inducing=num_inducing, run_number=i, direction=1)
        # Calculate losses
        score_x_y = loss_x + loss_x_y
        score_y_x = loss_y + loss_y_x
        print(f"Run {i}: {score_x_y} ; {score_y_x}")
        if score_x_y < score_y_x:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
    return correct_idx, wrong_idx, weight


if __name__ == "__main__":
    corr, wrong, weight = main()
    correct_weight = [weight[i] for i in corr]
    wrong_weight = [weight[i] for i in wrong]
    accuracy = np.sum(correct_weight) / (np.sum(correct_weight) + np.sum(wrong_weight))
    import pickle
    save_name = "fullscore-multb-uniform-sgpr-sumgpsqexpmatern3252-initialisation2"
    with open(f'/vol/bitbucket/ad6013/Research/gp-causal/results/{save_name}.p', 'wb') as f:
        pickle.dump(accuracy, f)