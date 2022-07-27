import gpflux
import numpy as np
import tensorflow_probability as tfp
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def train_gpflux(
    x,
    y,
    num_inducing,
    save_name,
    causal,
    run_number,
    random_restart_number,
    work_dir,
):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    w_dim = 1
    prior_means = np.zeros(w_dim)
    prior_std = np.ones(w_dim)
    encoder = gpflux.encoders.DirectlyParameterizedNormalDiag(1000, w_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)
    lv = gpflux.layers.LatentVariableLayer(prior, encoder)
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[.05, .2], variance=1.)
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.concatenate(
            [
                np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
                np.random.randn(num_inducing, 1),
            ],
            axis=1
        )
    )
    gp_layer = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data=1000,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Zero(),
    )


    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.random.randn(num_inducing, 1),
    )
    gp_layer2 = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data=1000,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Identity(),
    )
    gp_layer2.q_sqrt.assign(gp_layer.q_sqrt * 1e-5);

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))
    gpflow.set_trainable(likelihood_layer, False)
    dgp = gpflux.models.DeepGP([lv, gp_layer, gp_layer2], likelihood_layer)
    model = dgp.as_training_model()
    model.compile(tf.optimizers.Adam(0.005))

    history = model.fit({"inputs": tf.cast(x, tf.float64), "targets": tf.cast(y, tf.float64)}, epochs=int(20e3), verbose=0, batch_size=1000, shuffle=False)

    loss = history.history['loss'][-1]

    # Plot the fit to see if everything is ok
    obs_new = np.linspace(-10, 10, 1000)[:, None]

    def predict_y_samples(prediction_model, Xs, num_samples=25):
        samples = []
        for i in tqdm(range(num_samples)):
            out = prediction_model(Xs)
            s = out.y_mean + out.y_var ** .5 * tf.random.normal(tf.shape(out.y_mean), dtype=out.y_mean.dtype)
            samples.append(s)
        return tf.concat(samples, axis=1)

    samples = predict_y_samples(dgp.as_prediction_model(), obs_new, 1000).numpy().T

    pred_f_mean = np.mean(samples, 0).flatten()
    pred_f_var = np.var(samples, 0).flatten()

    textstr = 'elbo=%.2f\n'%(
        loss
    )
    plt.text(-17, 0, textstr, fontsize=8)
    plt.scatter(x, y, c='r')
    plt.plot(obs_new, pred_f_mean, c='b', alpha=0.25)
    plt.fill_between(obs_new[:, 0], (pred_f_mean + 2 * np.sqrt(pred_f_var)), (pred_f_mean - 2 * np.sqrt(pred_f_var)), alpha=0.5)
    save_dir = Path(f"{work_dir}/run_plots/{save_name}")
    save_dir.mkdir(
        parents=True, exist_ok=True
    )
    plt.subplots_adjust(left=0.25)
    causal_direction = "causal" if causal else "anticausal"
    plt.savefig(
        save_dir / f"run_{run_number}_rr_{random_restart_number}_{causal_direction}_conditional_gpflux"
    )
    plt.close()

    return loss


if __name__ == "__main__":
    # Simple fitting example
    import sys
    sys.path.append("/vol/bitbucket/ad6013/Research/gp-causal")
    from data.get_data import get_gauss_pairs_dataset
    x, y, weight, target = get_gauss_pairs_dataset(
        data_path='/vol/bitbucket/ad6013/Research/gp-causal/data/gauss_pairs/files'
    )