"""
This method will fit a gplvm for both the marginal and conditional models and
choose the causal direction as the one with the minimum
-log marginal likelihood.
"""
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive
from models.BayesGPLVM import BayesianGPLVM
from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Optional
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange


def train_marginal_model(
    y: np.ndarray,
    num_inducing: int,
    kernel_variance: float,
    kernel_lengthscale: float,
    likelihood_variance: float,
    work_dir: str,
    save_name: str,
    jitter: float,
    causal: Optional[bool] = None,
    run_number: Optional[int] = None,
    random_restart_number: Optional[int] = None,
    plot_fit: Optional[bool] = False,
):
    latent_dim = 1
    # Define kernel
    sq_exp = gpflow.kernels.SquaredExponential(lengthscales=[kernel_lengthscale])
    sq_exp.variance.assign(kernel_variance)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
    # Initialise approx posteroir and prior
    X_mean_init = 0.1 * tf.cast(y, default_float())
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (y.shape[0], latent_dim)), default_float()
    )
    x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.random.randn(num_inducing, latent_dim),
    )

    # Define marginal model
    marginal_model = BayesianGPLVM(
        data=y,
        kernel=kernel,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        X_prior_var=x_prior_var,
        jitter=jitter,
        inducing_variable=inducing_variable
    )
    marginal_model.likelihood.variance = Parameter(
        likelihood_variance, transform=positive(1e-6)
    )
    loss_fn = marginal_model.training_loss_closure()

    adam_vars = marginal_model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.1)
    @tf.function
    def optimisation_step():
        adam_opt.minimize(loss_fn, adam_vars)

    epochs = int(20e3)
    log_freq = 100
    with trange(1, epochs + 1, leave=True, position=0) as pbar:
        losses = []
        for epoch in pbar:
            optimisation_step()
            if epoch % log_freq == 0:
                pbar.set_description(f"ELBO {- marginal_model.elbo()}")
            losses.append(- marginal_model.elbo())
            if epoch > 101:
                losses.pop(0)
            if epoch > 1000 and np.abs(np.mean(losses[0:50]) - np.mean(losses[50:100])) < np.std(losses):
                print("BREAKING!")
                break

    tf.print("ELBO:", - marginal_model.elbo())

    loss = - marginal_model.elbo()

    if plot_fit:
        # Plot the fit to see if everything is ok
        obs_new = np.linspace(-5, 5, 1000)[:, None]

        pred_y_mean, pred_y_var = marginal_model.predict_y(
            Xnew=obs_new,
        )
        textstr = 'kern_len_lat=%.2f\nkern_var=%.2f\nlike_var=%.2f\nelbo=%.2f\n'%(
            marginal_model.kernel.kernels[0].lengthscales.numpy(),
            marginal_model.kernel.kernels[0].variance.numpy(),
            marginal_model.likelihood.variance.numpy(),
            - marginal_model.elbo().numpy()
        )
        plt.text(-8, 0, textstr, fontsize=8)
        plt.scatter(marginal_model.X_data_mean, y, c='r')
        plt.plot(obs_new, pred_y_mean, c='b', alpha=0.25)
        plt.fill_between(
            obs_new[:, 0],
            (pred_y_mean + 2 * np.sqrt(pred_y_var))[:, 0],
            (pred_y_mean - 2 * np.sqrt(pred_y_var))[:,0],
            alpha=0.5
        )
        save_dir = Path(f"{work_dir}/run_plots/{save_name}")
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


def train_conditional_model(
    x: np.ndarray,
    y: np.ndarray,
    num_inducing: int,
    kernel_variance: float,
    kernel_lengthscale: float,
    likelihood_variance: float,
    work_dir: str,
    jitter: float,
    save_name: str,
    causal: Optional[bool] = None,
    run_number: Optional[int] = None,
    random_restart_number: Optional[int] = None,
    plot_fit: Optional[bool] = False,
):
    """
    Train a conditional model using a partially observed GPLVM.
    """
    tf.print(f"Init: ker_len: {kernel_lengthscale}, ker_var: {kernel_variance}, like_var: {likelihood_variance}")
    latent_dim = 1

    # if not using a GP, put in initial values for hyperparams
    X_mean_init = 0.01 * tf.cast(y, default_float())
    sq_exp = gpflow.kernels.SquaredExponential(
        lengthscales=[kernel_lengthscale] + [kernel_lengthscale / 3]
    )
    sq_exp.variance.assign(kernel_variance + 1e-20)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance + 1e-20)

    # Define rest of the hyperparams
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (y.shape[0], latent_dim)), default_float()
    )
    x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.concatenate(
            [
                np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
                np.random.randn(num_inducing, 1),
            ],
            axis=1
        )
    )

    # Define conditional model
    conditional_model = PartObsBayesianGPLVM(
        data=y,
        in_data=x,
        kernel=kernel,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        X_prior_var=x_prior_var,
        jitter=jitter,
        inducing_variable=inducing_variable
    )
    conditional_model.likelihood.variance = Parameter(
        likelihood_variance, transform=positive(lower=1e-6)
    )

    loss_fn = conditional_model.training_loss_closure()

    adam_vars = conditional_model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.1)
    @tf.function
    def optimisation_step():
        adam_opt.minimize(loss_fn, adam_vars)

    epochs = int(20e3)
    log_freq = 100
    with trange(1, epochs + 1, leave=True, position=0) as pbar:
        losses = []
        for epoch in pbar:
            optimisation_step()
            if epoch % log_freq == 0:
                pbar.set_description(f"ELBO {- conditional_model.elbo()}")
            losses.append(- conditional_model.elbo())
            if epoch > 101:
                losses.pop(0)
            if epoch > 1000 and np.abs(np.mean(losses[0:50]) - np.mean(losses[50:100])) < np.std(losses):
                print("BREAKING!")
                break

    tf.print("ELBO:", - conditional_model.elbo())

    loss = - conditional_model.elbo()

    if plot_fit:
        # Plot the fit to see if everything is ok
        obs_new = np.linspace(x.min() - 2, x.max() + 1, 1000)[:, None]
        # Sample from the prior
        Xnew = tfp.distributions.Normal(loc=0, scale=1).sample(
            [obs_new.shape[0], latent_dim]
        )
        Xnew = tf.cast(Xnew, dtype=default_float())
        Xnew = tf.concat(
            [obs_new, Xnew], axis=1
        )
        pred_f_mean, pred_f_var = conditional_model.predict_y(
            Xnew=Xnew,
        )
        textstr = 'kern_len_obs=%.2f\nkern_len_lat=%.2f\nkern_var=%.2f\nlike_var=%.2f\nelbo=%.2f\n'%(
            conditional_model.kernel.kernels[0].lengthscales.numpy()[0],
            conditional_model.kernel.kernels[0].lengthscales.numpy()[1],
            conditional_model.kernel.kernels[0].variance.numpy(),
            conditional_model.likelihood.variance.numpy(),
            - conditional_model.elbo().numpy()
        )
        plt.text(x.min() - 6, 0, textstr, fontsize=8)
        plt.scatter(x, y, c='r')
        plt.plot(obs_new, pred_f_mean, c='b', alpha=0.25)
        plt.fill_between(
            obs_new[:, 0],
            (pred_f_mean + 2 * np.sqrt(pred_f_var))[:, 0],
            (pred_f_mean - 2 * np.sqrt(pred_f_var))[:,0],
            alpha=0.5
        )
        save_dir = Path(f"{work_dir}/run_plots/{save_name}")
        save_dir.mkdir(
            parents=True, exist_ok=True
        )
        plt.subplots_adjust(left=0.25)
        causal_direction = "causal" if causal else "anticausal"
        plt.savefig(
            save_dir / f"run_{run_number}_rr_{random_restart_number}_{causal_direction}_conditional"
        )
        plt.close()
    else:
        pass

    return loss


def causal_score_gplvm(args, x, y, run_number, restart_number, causal, save_name):
    num_inducing = args.num_inducing if x.shape[0] > args.num_inducing else x.shape[0]
    # Dynamically reduce the jitter if there is an error
    # Sample hyperparams
    kernel_variance = 1
    jitter_bug = 1e-4
    finish = 0
    loss_x = None
    # Likelihood variance
    likelihood_variance = [0.01, 0.001][restart_number]
    kernel_lengthscale = [0.05, 0.1][restart_number]

    while finish == 0:
        try:
            tf.print("X" if causal else "Y")
            loss_x = train_marginal_model(
                y=x,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale,
                likelihood_variance=likelihood_variance,
                work_dir=args.work_dir,
                run_number=run_number,
                random_restart_number=restart_number,
                jitter=jitter_bug,
                causal=causal,
                save_name=save_name,
                plot_fit=args.plot_fit,
            )
            finish = 1
        except Exception as e:
            tf.print(e)
            tf.print(f"Increasing jitter to {jitter_bug * 10}")
            jitter_bug *= 10
            if jitter_bug > 1:
                finish = 1
    # Sample hyperparams
    jitter_bug = 1e-4
    finish = 0
    kernel_variance = 1

    likelihood_variance = [0.01, 0.001][restart_number]
    kernel_lengthscale = [0.05, 0.1][restart_number]

    while finish == 0:
        try:
            tf.print("Y|X" if causal else "X|Y")
            loss_y_x = train_conditional_model(
                x=x,
                y=y,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale,
                likelihood_variance=likelihood_variance,
                work_dir=args.work_dir,
                run_number=run_number,
                random_restart_number=restart_number,
                causal=causal,
                jitter=jitter_bug,
                save_name=save_name,
                plot_fit=args.plot_fit,
            )
            finish = 1
        except Exception as e:
            tf.print(e)
            tf.print(f"Increasing jitter to {jitter_bug * 10}")
            jitter_bug *= 10
            if jitter_bug > 1:
                finish = 1
        if loss_x is None:
            raise ValueError("jitter is more than 1!")
    return (loss_x, loss_y_x)


def min_causal_score_gplvm_adam(args, x, y, weight, target):
    # TODO: The data loading and saving etc should be separate from the model
    # Find data index to start and end the runs on
    data_start_idx = args.data_start
    data_end_idx = args.data_end if args.data_end < len(x) else len(x)
    save_name = f"fullscore-{args.data}-gplvm_adam-reinit{args.random_restarts}-numind{args.num_inducing}" \
                f"_start:{data_start_idx}_end:{data_end_idx}"
    save_path = Path(f'{args.work_dir}/results/{save_name}.p')

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
        starting_run_number = data_start_idx

    for i in tqdm(range(starting_run_number, data_end_idx), desc="Epochs", leave=True, position=0):
        # Find the target
        run_target = target[i]
        # Ignore the high dim
        if x[i].shape[-1] > 1:
            continue
        tf.print(f'\n Run: {i}')

        # Normalise the data
        x_train = StandardScaler().fit_transform(x[i]).astype(np.float64)
        y_train = StandardScaler().fit_transform(y[i]).astype(np.float64)

        rr_loss_x = []
        rr_loss_y_x = []
        rr_loss_y = []
        rr_loss_x_y = []
        for j in range(args.random_restarts):
            seed = args.random_restarts * i + j
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf.print(f"\n Random restart: {j}")
            (
                loss_x,
                loss_y_x,

            ) = causal_score_gplvm(
                args=args,
                x=x_train,
                y=y_train,
                run_number=i,
                restart_number=j,
                causal=True,
                save_name=save_name
            )
            (
                loss_y,
                loss_x_y,

            ) = causal_score_gplvm(
                args=args,
                x=y_train,
                y=x_train,
                run_number=i,
                restart_number=j,
                causal=False,
                save_name=save_name
            )
            if loss_x is not None:
                rr_loss_x.append(loss_x)
                rr_loss_y_x.append(loss_y_x)
                rr_loss_y.append(loss_y)
                rr_loss_x_y.append(loss_x_y)
                tf.print(loss_x.numpy(), loss_y_x.numpy(), loss_y.numpy(), loss_x_y.numpy())
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
            # If target is -1 this is wrong
            if run_target < 0:
                wrong_idx.append(i)
            else:
                correct_idx.append(i)
        else:
            if run_target < 0:
                correct_idx.append(i)
            else:
                wrong_idx.append(i)
        scores.append(((min(rr_loss_x).numpy(), min(rr_loss_y_x).numpy()), (min(rr_loss_y).numpy(), min(rr_loss_x_y).numpy())))
        tf.print(f"Correct: {len(correct_idx)}, Wrong: {len(wrong_idx)}")
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