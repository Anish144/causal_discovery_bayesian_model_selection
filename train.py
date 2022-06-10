from data.get_data import get_tubingen_pairs_dataset, get_synthetic_dataset
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive
from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM
from models.BayesGPLVM import BayesianGPLVM
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
import pickle


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
    work_dir,
    data_name,
    jitter,
    causal: Optional[bool] = None,
    run_number: Optional[int] = None,
    random_restart_number: Optional[int] = None,
    no_observed=False,
):
    tf.print(f"Init: ker_like: {kernel_lengthscale}, ker_var: {kernel_variance}, like_var: {likelihood_variance}")
    latent_dim = 1
    if not no_observed:
        # Find the best lengthscale for the observed bit
        sq_exp = gpflow.kernels.SquaredExponential(lengthscales=[kernel_lengthscale])
        sq_exp.variance.assign(kernel_variance)

        m = gpflow.models.GPR(data=(x, y), kernel=sq_exp, mean_function=None)
        m.likelihood.variance = Parameter(likelihood_variance, transform=positive())

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            m.training_loss, m.trainable_variables, options=dict(maxiter=10000)
        )
        found_lengthscale = float(m.kernel.lengthscales.numpy())
        found_lik_var = m.likelihood.variance.numpy()
        tf.print(f"Found: ker_like: {found_lengthscale}, ker_var: {m.kernel.variance.numpy()}, like_var: {found_lik_var}")

        X_mean_init = y - m.predict_f(x)[0]
        X_var_init = tf.math.square(X_mean_init - tf.math.reduce_mean(X_mean_init, axis=0)) + 1
        # need a lengthscale for the latent dim as well as for the oberved
        # Lengthscale of observed is slightly larger
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[found_lengthscale] + [found_lengthscale * 0.67])
        x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())
    else:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[kernel_lengthscale])
        X_mean_init = tfp.distributions.Normal(loc=0, scale=1).sample([y.shape[0], latent_dim])
        X_mean_init = tf.cast(y, dtype=default_float())

        X_var_init = tf.ones((y.shape[0], latent_dim), dtype=default_float())

        x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())

    kernel.variance.assign(kernel_variance)

    if not no_observed:
        m = PartObsBayesianGPLVM(
            data=y,
            in_data=x,
            kernel=kernel,
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            num_inducing_variables=num_inducing,
            X_prior_var=x_prior_var,
            jitter=jitter
        )
        m.likelihood.variance = Parameter(found_lik_var, transform=positive())
    else:
        m = BayesianGPLVM(
            data=y,
            kernel=kernel,
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            num_inducing_variables=num_inducing,
            X_prior_var=x_prior_var,
            jitter=jitter
        )
        m.likelihood.variance = Parameter(likelihood_variance, transform=positive())

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

    if args.plot_fit and not no_observed:
        # Plot the fit to see if everything is ok
        obs_new = np.linspace(-10, 10, 1000)[:, None]
        # Sample from the prior
        Xnew = tfp.distributions.Normal(loc=0, scale=1).sample([obs_new.shape[0], latent_dim])
        Xnew = tf.cast(Xnew, dtype=default_float())
        Xnew = tf.concat(
            [obs_new, Xnew], axis=1
        )
        pred_f_mean, pred_f_var = m.predict_y(
            Xnew=Xnew,
        )
        textstr = 'kern_len_obs=%.2f\nkern_len_lat=%.2f\nkern_var=%.2f\nlike_var=%.2f\nelbo=%.2f\n'%(
            m.kernel.lengthscales.numpy()[0],m.kernel.lengthscales.numpy()[1],
            m.kernel.variance.numpy(), m.likelihood.variance.numpy(),
            - m.elbo().numpy()
        )
        plt.text(-17, 0, textstr, fontsize=8)
        plt.scatter(x, y, c='r')
        plt.plot(obs_new, pred_f_mean, c='b', alpha=0.25)
        plt.fill_between(obs_new[:, 0], (pred_f_mean + 2 * np.sqrt(pred_f_var))[:, 0], (pred_f_mean - 2 * np.sqrt(pred_f_var))[:,0], alpha=0.5)
        save_dir = Path(f"{work_dir}/run_plots/{data_name}")
        save_dir.mkdir(
            parents=True, exist_ok=True
        )
        plt.subplots_adjust(left=0.25)
        causal_direction = "causal" if causal else "anticausal"
        plt.savefig(
            save_dir / f"run_{run_number}_rr_{random_restart_number}_{causal_direction}_conditional"
        )
        plt.close()
    elif args.plot_fit and no_observed:
        # Plot the fit to see if everything is ok
        obs_new = np.linspace(-5, 5, 1000)[:, None]

        pred_y_mean, pred_y_var = m.predict_y(
            Xnew=obs_new,
        )
        textstr = 'kern_len_lat=%.2f\nkern_var=%.2f\nlike_var=%.2f\nelbo=%.2f\n'%(
            m.kernel.lengthscales.numpy(),
            m.kernel.variance.numpy(), m.likelihood.variance.numpy(),
            - m.elbo().numpy()
        )
        plt.text(-5, 0, textstr, fontsize=8)
        plt.scatter(m.X_data_mean, y, c='r')
        plt.plot(obs_new, pred_y_mean, c='b', alpha=0.25)
        plt.fill_between(obs_new[:, 0], (pred_y_mean + 2 * np.sqrt(pred_y_var))[:, 0], (pred_y_mean - 2 * np.sqrt(pred_y_var))[:,0], alpha=0.5)
        save_dir = Path(f"{work_dir}/run_plots/{data_name}")
        save_dir.mkdir(
            parents=True, exist_ok=True
        )
        plt.subplots_adjust(left=0.25)
        causal_direction = "causal" if causal else "anticausal"
        plt.savefig(
            save_dir / f"run_{run_number}_rr_{random_restart_number}_{causal_direction}_marginal"
        )
        plt.close()
    else:
        pass
    return loss


def calculate_causal_score(args, seed, x, y, run_number, restart_number, causal):
    # Sample random hyperparams, one for each experiment
    x_train, y_train = x, y
    # Make sure data is standardised
    x_train = StandardScaler().fit_transform(x_train).astype(np.float64)
    y_train = StandardScaler().fit_transform(y_train).astype(np.float64)
    num_inducing = args.num_inducing if x.shape[0] > args.num_inducing else x.shape[0]
    # Dynamically reduce the jitter if there is an error
    kernel_variance = 2

    jitter_bug = 1e-6
    finish = 0
    loss_x = None
    while finish == 0:
        try:
            # Likelihood variance
            kappa = np.random.uniform(
                low=10.0, high=75, size=[1]
            )
            likelihood_variance = 1. / (kappa ** 2)
            # Kernel lengthscale
            lamda = np.random.uniform(
                low=1.0, high=100, size=[1]
            )
            kernel_lengthscale = 1.0 / lamda
            # x -> y score
            print("X" if causal else "Y")
            loss_x = train(
                x=np.random.normal(loc=0, scale=1, size=x_train.shape),
                y=x_train,
                no_observed=True,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale[0],
                likelihood_variance=likelihood_variance[0],
                work_dir=args.work_dir,
                data_name=args.data,
                run_number=run_number,
                random_restart_number=restart_number,
                jitter=jitter_bug,
                causal=causal,
            )
            finish = 1
        except Exception as e:
            print(e)
            print(f"Increasing jitter to {jitter_bug * 5}")
            jitter_bug *= 10
            if jitter_bug > 1:
                finish = 1
    jitter_bug = 1e-6
    finish = 0
    while finish == 0:
        try:
            # Likelihood variance
            kappa = np.random.uniform(
                low=10.0, high=75, size=[1]
            )
            likelihood_variance = 1. / (kappa ** 2)
            # Kernel lengthscale
            lamda = np.random.uniform(
                low=1.0, high=100, size=[1]
            )
            kernel_lengthscale = 1.0 / lamda
            print("Y|X" if causal else "X|Y")
            loss_y_x = train(
                x=x_train,
                y=y_train,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale[0],
                likelihood_variance=likelihood_variance[0],
                work_dir=args.work_dir,
                data_name=args.data,
                run_number=run_number,
                random_restart_number=restart_number,
                causal=causal,
                jitter=jitter_bug
            )
            finish = 1
        except Exception as e:
            print(e)
            print(f"Increasing jitter to {jitter_bug * 5}")
            jitter_bug *= 10
            if jitter_bug > 1:
                finish = 1
        if loss_x is None:
            raise ValueError("jitter is more than 1!")
    return (loss_x, loss_y_x)


def main(args: argparse.Namespace):
    save_name = f"fullscore-{args.data}-gplvm-sqexp-reinit{args.random_restarts}"
    save_path = Path(f'{args.work_dir}/results/{save_name}.p')
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(False)
    # tf.debugging.enable_check_numerics()

    # Choose the dataset
    if args.data == "cep":
        x, y, weight = get_tubingen_pairs_dataset(
            data_path=f'{args.work_dir}/data/pairs/files'
        )
    else:
        func_type, noise = args.data.split("-")
        x, y, weight = get_synthetic_dataset(
            num_datasets=100,
            sample_size=100,
            func_string=func_type,
            noise=noise
        )
    if save_path.is_file():
        with open(save_path, "rb") as f:
            checkpoint = pickle.load(f)
        correct_idx = checkpoint['correct_idx']
        wrong_idx = checkpoint['wrong_idx']
        scores = checkpoint["scores"]
        starting_run_number = checkpoint["run_number"]
    else:
        correct_idx = []
        wrong_idx = []
        scores = []
        starting_run_number = 0

    for i in tqdm(range(starting_run_number, len(x)), desc="Epochs", leave=True, position=0):
        # Ignore the high dim
        if x[i].shape[-1] > 1:
            continue
        print(f'\n Run: {i}')

        rr_loss_x = []
        rr_loss_y_x = []
        rr_loss_y = []
        rr_loss_x_y = []
        for j in range(args.random_restarts):
            seed = args.random_restarts * i + j
            np.random.seed(seed)
            tf.random.set_seed(seed)
            print(f"\n Random restart: {j}")
            (
                loss_x,
                loss_y_x,

            ) = calculate_causal_score(
                args=args,
                seed=seed,
                x=x[i],
                y=y[i],
                run_number=i,
                restart_number=j,
                causal=True
            )
            (
                loss_y,
                loss_x_y,

            ) = calculate_causal_score(
                args=args,
                seed=seed,
                x=y[i],
                y=x[i],
                run_number=i,
                restart_number=j,
                causal=False
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
        tf.print(f"Run {i}: {score_x_y} ; {score_y_x}")
        if score_x_y < score_y_x:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
        scores.append(((min(rr_loss_x).numpy(), min(rr_loss_y_x).numpy()), (min(rr_loss_y).numpy(), min(rr_loss_x_y).numpy())))
        print(f"Correct: {len(correct_idx)}, Wrong: {len(wrong_idx)}")
        # Save checkpoint
        with open(save_path, 'wb') as f:
            save_dict = {
                "correct_idx": correct_idx,
                "wrong_idx": wrong_idx,
                "weight": weight,
                "scores": scores,
                "run_number": i + 1
            }
            pickle.dump(save_dict, f)
    return correct_idx, wrong_idx, weight, scores


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
    save_name = f"fullscore-{args.data}-gplvm-sqexp-reinit{args.random_restarts}"
    # save_name = "test"
    with open(f'{args.work_dir}/results/{save_name}.p', 'wb') as f:
        pickle.dump((accuracy, scores), f)