import sys
sys.path.append("/vol/bitbucket/ad6013/Research/gp-causal")

from data.get_data import get_gauss_pairs_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from models.GeneralisedGPLVM import GeneralisedGPLVM
from gpflow.config import default_float
from tqdm import trange
from gpflow.optimizers import NaturalGradient
import gpflow
import matplotlib.pyplot as plt


def run_optimizer(model, train_dataset, iterations, minibatch_size):
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
    # gpflow.set_trainable(model.q_mu, True)
    # gpflow.set_trainable(model.q_sqrt, True)
    # gpflow.set_trainable(model.X_data_mean, False)
    # gpflow.set_trainable(model.X_data_var, False)
    # variational_params = [(model.q_mu, model.q_sqrt)]
    # natgrad_opt = NaturalGradient(gamma=0.5)
    optimizer = tf.optimizers.Adam(0.05)
    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        # natgrad_opt.minimize(training_loss, variational_params)
    iterator = trange(iterations, leave=True)
    for step in iterator:
        optimization_step()
        if step % 4999 == 0:
            neg_elbo = training_loss().numpy()
            iterator.set_description(f"EPOCH: {step}, NEG ELBO: {neg_elbo}")

            # plt.plot(np.arange(len(logf)), logf)
            # plt.show()
            # plt.close()

            # obs_new = np.linspace(-5, 5, 3000)[:, None]

            # lower, median, upper, samples = model.predict_credible_layer(
            #     Xnew=obs_new,
            #     obs_noise=True,
            # )
            # plt.scatter(X[:, 0], Y[:, 0], c='r')
            # plt.plot(obs_new, median, c='b', alpha=0.2)
            # # plt.scatter(obs_new[:, 0], samples[0, 0], alpha=0.5)
            # plt.fill_between(obs_new[:, 0], upper[:, 0], lower[:, 0], alpha=0.5)
            # plt.show()
            # plt.close()

            # obs_new = np.linspace(-5, 5, 1000)[:, None]
            # latent_new = np.random.randn(1000, 1)
            # full_new =  np.concatenate([obs_new, latent_new], axis=1)
            # # full_new =  np.concatenate([obs_new, obs_new], axis=1)
            # # full_new = obs_new
            # pred_y_mean, pred_y_var = model.predict_y(
            #     full_new, full_cov=False, full_output_cov=False
            # )
            # plt.scatter(X[:, 0], Y[:, 0], c='r')
            # plt.plot(obs_new, pred_y_mean, c='b', alpha=0.2)
            # # plt.scatter(inducing_in, np.zeros(100) )
            # plt.fill_between(obs_new[:, 0], (pred_y_mean + 2 * np.sqrt(pred_y_var))[:, 0], (pred_y_mean - 2 * np.sqrt(pred_y_var))[:,0], alpha=0.5)
            # plt.show()
            # plt.close()


            # if np.abs(np.mean(logf[-5000:])) - np.abs(np.mean(logf[-100:])) < 0.25 * np.std(logf[-100:]):
            #     print("\n BREAKING! \n")
            #     break

    return logf


def run_model(X, Y, num_minibatch, num_iterations, num_mc):
    M = 200  # Number of inducing locations

    kernel_1 = gpflow.kernels.SquaredExponential(
        lengthscales=[0.1, 0.1]
    )
    kernel_1.variance.assign(1.0)
    kernel_2 = gpflow.kernels.Linear(variance=1.0)
    kernel = gpflow.kernels.Sum([kernel_1, kernel_2])
    Z = np.concatenate(
            [
                np.linspace(X.min(), X.max(), M).reshape(-1, 1),
                # np.linspace(X.min(), X.max(), M).reshape(-1, 1),
                np.random.randn(M, 1),
            ],
            axis=1
        )
    # Z = np.linspace(X.min(), X.max(), M).reshape(-1, 1)
    X_mean_init = 0.01 * tf.cast(Y, default_float())
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (X.shape[0], 1)), default_float()
    )

    model = GeneralisedGPLVM(
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=1e-5),
        num_mc_samples=num_mc,
        inducing_variable=Z,
        batch_size=num_minibatch,
    )
    data_idx = np.arange(X.shape[0])
    loss = - model.elbo((X, Y, data_idx))
    from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM

    X_mean_init = model.X_data_mean.numpy()
    X_var_init = model.X_data_var.numpy()

    # Z = np.concatenate(
    #         [
    #             model.inducing_variable.Z,
    #             # np.linspace(X.min(), X.max(), M).reshape(-1, 1),
    #             np.random.randn(M, 1),
    #         ],
    #         axis=1
    #     )

    Z = model.inducing_variable
    kernel = model.kernel

    likelihood = model.likelihood

    model_bayes = PartObsBayesianGPLVM(
        data=Y,
        in_data=X,
        kernel=kernel,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        inducing_variable=Z,
        jitter=1e-20,
    )
    model_bayes.likelihood.variance.assign(likelihood.variance)
    bayes_gplvm_loss = (- model_bayes.elbo())
    tf.print(f"Gen loss: {loss.numpy()}, Bayes loss: {bayes_gplvm_loss.numpy()}")


    data_idx = np.arange(X.shape[0])
    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y, data_idx)).repeat().shuffle(N)
    logf = run_optimizer(
        model=model,
        train_dataset=train_dataset,
        iterations=num_iterations,
        minibatch_size=num_minibatch
    )
    # half_N = X.shape[0] // 2
    # loss_first_half = - model.elbo((X[:half_N], Y[:half_N], data_idx[:half_N]))
    # loss_second_half = - model.elbo((X[half_N:], Y[half_N:], data_idx[half_N:]))
    # loss = loss_first_half + loss_second_half
    loss = - model.elbo((X, Y, data_idx))
    print(f"Loss is {loss}")

    obs_new = np.linspace(-5, 5, 1000)[:, None]
    latent_new = np.random.randn(1000, 1)
    full_new =  np.concatenate([obs_new, latent_new], axis=1)
    # full_new =  np.concatenate([obs_new, obs_new], axis=1)
    # full_new = obs_new
    pred_y_mean, pred_y_var = model.predict_y(
        full_new, full_cov=False, full_output_cov=False
    )
    plt.scatter(X[:, 0], Y[:, 0], c='r')
    plt.plot(obs_new, pred_y_mean, c='b', alpha=0.2)
    # plt.scatter(inducing_in, np.zeros(100) )
    plt.fill_between(obs_new[:, 0], (pred_y_mean + 2 * np.sqrt(pred_y_var))[:, 0], (pred_y_mean - 2 * np.sqrt(pred_y_var))[:,0], alpha=0.5)
    textstr = f"NEG_ELBO:{loss:.2f}"
    plt.text(X.min() - 5, 1, textstr, fontsize=8)
    plt.subplots_adjust(left=0.25)
    plt.savefig(f"Test_sampling_minibatch_{num_minibatch}_numit_{num_iterations}_nummc_{num_mc}")
    plt.close()

    from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM

    X_mean_init = model.X_data_mean.numpy()
    X_var_init = model.X_data_var.numpy()

    # Z = np.concatenate(
    #         [
    #             model.inducing_variable.Z,
    #             # np.linspace(X.min(), X.max(), M).reshape(-1, 1),
    #             np.random.randn(M, 1),
    #         ],
    #         axis=1
    #     )

    Z = model.inducing_variable
    kernel = model.kernel

    likelihood = model.likelihood

    model_bayes = PartObsBayesianGPLVM(
        data=Y,
        in_data=X,
        kernel=kernel,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        inducing_variable=Z,
        jitter=1e-6,
    )
    model_bayes.likelihood.variance.assign(likelihood.variance)
    bayes_gplvm_loss = (- model_bayes.elbo())
    tf.print(f"Gen loss: {loss.numpy()}, Bayes loss: {bayes_gplvm_loss.numpy()}")
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    # tf.config.run_functions_eagerly(
    #     True
    # )
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    x, y, _, _ = get_gauss_pairs_dataset("/vol/bitbucket/ad6013/Research/gp-causal/data/gauss_pairs/files")
    X = y[180]
    Y = x[180]
    N = X.shape[0]
    num_minibatch = 100
    num_iterations = 100000
    num_mc = [1]
    print(f"\n Nummin: {num_minibatch}, Numit: {num_iterations} \n ")

    X = StandardScaler().fit_transform(X).astype(np.float64)
    Y = StandardScaler().fit_transform(Y).astype(np.float64)
    for i in num_mc:
        run_model(X, Y, num_minibatch, num_iterations, i)