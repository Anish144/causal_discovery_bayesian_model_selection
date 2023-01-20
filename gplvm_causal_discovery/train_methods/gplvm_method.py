"""
This method will fit a gplvm for both the marginal and conditional models and
choose the causal direction as the one with the minimum
-log marginal likelihood.

This is the training procedure for closed form GPLVM, following Titsias (2010).

Note that it is only possible to use kernels like RBF or linear, that have
a closed form expectation - see Titsias (2010).

GPLVMs have a tendency to find bad local optima when using BFGS - the entire
dataset is explaied by the likelihood noise.
To alleviate these cases, we train a GPLVM first that finds the score if
the GPLVM explained everything with likelihood noise.
Then, we train a new model with Adam for a number of epochs, till it's ELBO
is hgiher than that of the noise model. Once this happens, we train the
resultant model using BFGS.
We found that this is very effective at reducing the chances of local optima.
Other things that are important:
- Data should be normalised
- Likelihood noise should be initialised low
- Kernel lengthscale should be lower than variance of the data.
- Latent approximate posterior should have low variance. This allows the model
to fit the data before using the latent.
- Latent approximate posterior mean should be initialised with the output data.
This is equivalent to doing PCA in 1 dimension.
"""
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive
from ..models.BayesGPLVM import BayesianGPLVM
from ..models.PartObsBayesianGPLVM import PartObsBayesianGPLVM
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Optional
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import dill
import tensorflow as tf
import tensorflow_probability as tfp
from collections import defaultdict
from collections import namedtuple
from ..utils import return_all_scores, return_best_causal_scores, get_correct


GPLVM_SCORES = namedtuple("GPLVM_SCORES", "loss_x loss_y_x loss_y loss_x_y")


def get_marginal_noise_model_score(y: np.ndarray, num_inducing: int):
    # Need to find the ELBO for a noise model
    linear_kernel = gpflow.kernels.Linear(variance=1e-20)
    kernel = linear_kernel

    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.random.randn(num_inducing, 1),
    )
    X_mean_init = 0.1 * tf.cast(y, default_float())
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (y.shape[0], 1)), default_float()
    )
    x_prior_var = tf.ones((y.shape[0], 1), dtype=default_float())
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.random.randn(num_inducing, 1),
    )
    # Define marginal model
    marginal_model = BayesianGPLVM(
        data=y,
        kernel=kernel,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        X_prior_var=x_prior_var,
        jitter=1e-6,
        inducing_variable=inducing_variable,
    )
    marginal_model.likelihood.variance = Parameter(
        1.0, transform=positive(1e-6)
    )
    # Train everything
    gpflow.utilities.set_trainable(marginal_model.kernel, True)
    gpflow.utilities.set_trainable(marginal_model.likelihood, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_mean, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_var, True)
    gpflow.utilities.set_trainable(marginal_model.inducing_variable, True)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        marginal_model.training_loss,
        marginal_model.trainable_variables,
        options=dict(maxiter=10000),
    )
    loss = -marginal_model.elbo()
    return loss


def get_conditional_noise_model_score(
    x: np.ndarray, y: np.ndarray, num_inducing: int
):
    # Need to find the ELBO for a noise model
    linear_kernel = gpflow.kernels.Linear(variance=1)
    kernel = linear_kernel

    Z = gpflow.inducing_variables.InducingPoints(
        np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
    )
    inducing_variable = Z

    reg_gp_model = gpflow.models.SGPR(
        data=(x, y), kernel=kernel, inducing_variable=inducing_variable
    )
    reg_gp_model.likelihood.variance = Parameter(
        1 + 1e-20, transform=positive(lower=1e-6)
    )
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        reg_gp_model.training_loss,
        reg_gp_model.trainable_variables,
        options=dict(maxiter=200000),
    )
    loss = -reg_gp_model.elbo()
    return loss


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
    sq_exp = gpflow.kernels.SquaredExponential(
        lengthscales=[kernel_lengthscale]
    )
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
        inducing_variable=inducing_variable,
    )
    marginal_model.likelihood.variance = Parameter(
        likelihood_variance, transform=positive(1e-6)
    )

    # We will train the Adam until the elbo gets below the noise model score
    noise_elbo = get_marginal_noise_model_score(y=y, num_inducing=num_inducing)
    loss_fn = marginal_model.training_loss_closure()
    adam_vars = marginal_model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.05)

    @tf.function
    def optimisation_step():
        adam_opt.minimize(loss_fn, adam_vars)

    epochs = int(5e3)
    for epoch in range(epochs):
        optimisation_step()
        if -marginal_model.elbo() < noise_elbo:
            print(
                f"Breaking as {- marginal_model.elbo()} is less than {noise_elbo}"
            )
            break

    # Train everything
    tf.print("Training everything")
    gpflow.utilities.set_trainable(marginal_model.kernel, True)
    gpflow.utilities.set_trainable(marginal_model.likelihood, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_mean, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_var, True)
    gpflow.utilities.set_trainable(marginal_model.inducing_variable, True)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        marginal_model.training_loss,
        marginal_model.trainable_variables,
        options=dict(maxiter=10000),
    )
    tf.print("ELBO:", -marginal_model.elbo())

    loss = -marginal_model.elbo()

    if plot_fit:
        # Plot the fit to see if everything is ok
        obs_new = np.linspace(-5, 5, 1000)[:, None]

        pred_y_mean, pred_y_var = marginal_model.predict_y(
            Xnew=obs_new,
        )
        textstr = (
            "kern_len_lat=%.2f\nkern_var=%.2f\nlike_var=%.2f\nelbo=%.2f\n"
            % (
                marginal_model.kernel.kernels[0].lengthscales.numpy(),
                marginal_model.kernel.kernels[0].variance.numpy(),
                marginal_model.likelihood.variance.numpy(),
                -marginal_model.elbo().numpy(),
            )
        )
        plt.text(-8, 0, textstr, fontsize=8)
        plt.scatter(marginal_model.X_data_mean, y, c="r")
        plt.plot(obs_new, pred_y_mean, c="b", alpha=0.25)
        plt.fill_between(
            obs_new[:, 0],
            (pred_y_mean + 2 * np.sqrt(pred_y_var))[:, 0],
            (pred_y_mean - 2 * np.sqrt(pred_y_var))[:, 0],
            alpha=0.5,
        )
        save_dir = Path(f"{work_dir}/run_plots/{save_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.subplots_adjust(left=0.25)
        causal_direction = "causal" if causal else "anticausal"
        plt.savefig(
            save_dir
            / f"run_{run_number}_rr_{random_restart_number}_{causal_direction}_marginal"
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
    tf.print(
        f"Init: ker_len: {kernel_lengthscale}, ker_var: {kernel_variance}, like_var: {likelihood_variance}"
    )
    latent_dim = 1

    # Flip coin to see if we should initialise with a GP
    use_gp_initialise = np.random.binomial(1, 0.5)
    if use_gp_initialise == 1:
        use_gp = True
    else:
        use_gp = False

    # If use_gp, fit a GP to initialise everything
    if use_gp:
        # Find the best lengthscale for the observed bit
        # Define kernel
        sq_exp = gpflow.kernels.SquaredExponential(
            lengthscales=[kernel_lengthscale]
        )
        linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
        sq_exp.variance.assign(kernel_variance)
        kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
        # Define moedl
        reg_gp_model = gpflow.models.GPR(
            data=(x, y), kernel=kernel, mean_function=None
        )
        reg_gp_model.likelihood.variance = Parameter(
            likelihood_variance, transform=positive(lower=1e-6)
        )
        # Fit model
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            reg_gp_model.training_loss,
            reg_gp_model.trainable_variables,
            options=dict(maxiter=10000),
        )
        found_lengthscale = float(
            reg_gp_model.kernel.kernels[0].lengthscales.numpy()
        )
        found_lik_var = reg_gp_model.likelihood.variance.numpy()
        found_kern_var_0 = reg_gp_model.kernel.kernels[0].variance.numpy()
        found_kern_var_1 = reg_gp_model.kernel.kernels[1].variance.numpy()
        tf.print(
            f"Found: ker_len: {found_lengthscale}, ker_var: {found_kern_var_0}, like_var: {found_lik_var}"
        )

        # Put in new values of hyperparams
        X_mean_init = y - reg_gp_model.predict_y(x)[0]
        sq_exp = gpflow.kernels.SquaredExponential(
            lengthscales=[found_lengthscale + 1e-20]
            + [found_lengthscale / 3 + 1e-20]
        )
        sq_exp.variance.assign(found_kern_var_0 + 1e-20)
        linear_kernel = gpflow.kernels.Linear(variance=found_kern_var_1 + 1e-20)
    else:
        # if not using a GP, put in initial values for hyperparams
        X_mean_init = 0.1 * tf.cast(y, default_float())
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
            axis=1,
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
        inducing_variable=inducing_variable,
    )
    if use_gp:
        conditional_model.likelihood.variance = Parameter(
            found_lik_var + 1e-20, transform=positive(lower=1e-6)
        )
    else:
        conditional_model.likelihood.variance = Parameter(
            likelihood_variance, transform=positive(lower=1e-6)
        )

    # We will train the Adam until the elbo gets below the noise model score
    noise_elbo = get_conditional_noise_model_score(
        x=x, y=y, num_inducing=num_inducing
    )
    loss_fn = conditional_model.training_loss_closure()
    adam_vars = conditional_model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.05)

    @tf.function
    def optimisation_step():
        adam_opt.minimize(loss_fn, adam_vars)

    epochs = int(5e3)
    for epoch in range(epochs):
        optimisation_step()
        if -conditional_model.elbo() < noise_elbo:
            print(
                f"Breaking as {- conditional_model.elbo()} is less than {noise_elbo}"
            )
            break

    # Train everything after Adam
    tf.print("Training everything")
    gpflow.utilities.set_trainable(conditional_model.kernel, True)
    gpflow.utilities.set_trainable(conditional_model.likelihood, True)
    gpflow.utilities.set_trainable(conditional_model.X_data_mean, True)
    gpflow.utilities.set_trainable(conditional_model.X_data_var, True)
    gpflow.utilities.set_trainable(conditional_model.inducing_variable, True)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        conditional_model.training_loss,
        conditional_model.trainable_variables,
        options=dict(maxiter=10000),
    )
    tf.print("ELBO:", -conditional_model.elbo())

    loss = -conditional_model.elbo()

    if plot_fit:
        # Plot the fit to see if everything is ok
        obs_new = np.linspace(x.min() - 2, x.max() + 1, 1000)[:, None]
        # Sample from the prior
        Xnew = tfp.distributions.Normal(loc=0, scale=1).sample(
            [obs_new.shape[0], latent_dim]
        )
        Xnew = tf.cast(Xnew, dtype=default_float())
        Xnew = tf.concat([obs_new, Xnew], axis=1)
        pred_f_mean, pred_f_var = conditional_model.predict_y(
            Xnew=Xnew,
        )
        textstr = (
            "kern_len_obs=%.2f\nkern_len_lat=%.2f\nkern_var=%.2f\nlike_var=%.2f\nelbo=%.2f\n"
            % (
                conditional_model.kernel.kernels[0].lengthscales.numpy()[0],
                conditional_model.kernel.kernels[0].lengthscales.numpy()[1],
                conditional_model.kernel.kernels[0].variance.numpy(),
                conditional_model.likelihood.variance.numpy(),
                -conditional_model.elbo().numpy(),
            )
        )
        plt.text(x.min() - 6, 0, textstr, fontsize=8)
        plt.scatter(x, y, c="r")
        plt.plot(obs_new, pred_f_mean, c="b", alpha=0.25)
        plt.fill_between(
            obs_new[:, 0],
            (pred_f_mean + 2 * np.sqrt(pred_f_var))[:, 0],
            (pred_f_mean - 2 * np.sqrt(pred_f_var))[:, 0],
            alpha=0.5,
        )
        save_dir = Path(f"{work_dir}/run_plots/{save_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.subplots_adjust(left=0.25)
        causal_direction = "causal" if causal else "anticausal"
        plt.savefig(
            save_dir
            / f"run_{run_number}_rr_{random_restart_number}_{causal_direction}_conditional"
        )
        plt.close()
    else:
        pass

    return loss


def causal_score_gplvm(
    args, x, y, run_number, restart_number, causal, save_name
):
    num_inducing = (
        args.num_inducing if x.shape[0] > args.num_inducing else x.shape[0]
    )
    # Dynamically reduce the jitter if there is an error
    # Sample hyperparams
    kernel_variance = 1
    jitter_bug = 1e-6
    finish = 0
    loss_x = None
    # Likelihood variance
    kappa = np.random.uniform(low=10.0, high=100, size=[1])
    likelihood_variance = 1.0 / (kappa**2)
    # Kernel lengthscale
    lamda = np.random.uniform(low=1, high=100, size=[1])
    kernel_lengthscale = 1.0 / lamda
    while finish == 0:
        try:
            tf.print("X" if causal else "Y")
            loss_x = train_marginal_model(
                y=x,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale[0],
                likelihood_variance=likelihood_variance[0],
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
    jitter_bug = 1e-6
    finish = 0
    kernel_variance = 1
    # Likelihood variance
    kappa = np.random.uniform(low=10.0, high=100, size=[1])
    likelihood_variance = 1.0 / (kappa**2)
    # Kernel lengthscale
    lamda = np.random.uniform(low=1.0, high=100, size=[1])
    kernel_lengthscale = 1.0 / lamda
    while finish == 0:
        try:
            tf.print("Y|X" if causal else "X|Y")
            loss_y_x = train_conditional_model(
                x=x,
                y=y,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale[0],
                likelihood_variance=likelihood_variance[0],
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


def min_causal_score_gplvm(args, x, y, weight, target):
    # TODO: The data loading and saving etc should be separate from the model
    # Find data index to start and end the runs on
    data_start_idx = args.data_start
    data_end_idx = args.data_end if args.data_end < len(x) else len(x)
    save_name = (
        f"fullscore-{args.data}-gplvm-reinit{args.random_restarts}-numind{args.num_inducing}"
        f"_start:{data_start_idx}_end:{data_end_idx}"
    )
    save_path = Path(f"{args.work_dir}/results/{save_name}.p")
    starting_run_number = data_start_idx

    if save_path.is_file():
        with open(save_path, "rb") as f:
            checkpoint = dill.load(f)
        correct_idx = checkpoint["correct_idx"]
        wrong_idx = checkpoint["wrong_idx"]
        final_scores = checkpoint["final_scores"]
        best_scores = checkpoint["best_scores"]
    else:
        correct_idx = []
        wrong_idx = []
        best_scores = {}
        # Final scores will be saved in a dictionary with the key being the run
        # number that will point to a dictionary of random restarts
        final_scores = defaultdict(dict)

    for j in tqdm(
        range(args.random_restarts), desc="RR", leave=True, position=0
    ):
        for i in tqdm(
            range(starting_run_number, data_end_idx),
            desc="Runs",
            leave=True,
            position=0,
        ):
            # Check if this random restart already has been done for this run
            # Skip if it has...
            curr_run_rr_idx = list(final_scores[i].keys())
            if j in curr_run_rr_idx:
                continue

            # Find the target
            run_target = target[i]

            # Ignore the high dim
            if x[i].shape[-1] > 1:
                continue
            tf.print(f"\n Run: {i}")

            # Set the seed
            seed = args.random_restarts * i + j
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Subsample data
            if x[i].shape[0] > 4000:
                x_train = x[i][
                    np.random.choice(len(x[i]), size=4000, replace=False)
                ]
                y_train = y[i][
                    np.random.choice(len(y[i]), size=4000, replace=False)
                ]
                tf.print(f"SUBSAMPLING! {x[i].shape[0]}")
            else:
                x_train = x[i]
                y_train = y[i]

            # Normalise the data
            x_train = StandardScaler().fit_transform(x_train).astype(np.float64)
            y_train = StandardScaler().fit_transform(y_train).astype(np.float64)

            tf.print(f"\n Run: {i} \n Random restart: {j}")
            (loss_x, loss_y_x,) = causal_score_gplvm(
                args=args,
                x=x_train,
                y=y_train,
                run_number=i,
                restart_number=j,
                causal=True,
                save_name=save_name,
            )
            (loss_y, loss_x_y,) = causal_score_gplvm(
                args=args,
                x=y_train,
                y=x_train,
                run_number=i,
                restart_number=j,
                causal=False,
                save_name=save_name,
            )
            if loss_x is not None:
                this_run_score = GPLVM_SCORES(
                    loss_x, loss_y_x, loss_y, loss_x_y
                )
                final_scores[i].update({j: this_run_score})
                tf.print(
                    loss_x.numpy(),
                    loss_y_x.numpy(),
                    loss_y.numpy(),
                    loss_x_y.numpy(),
                )
                with open(save_path, "wb") as f:
                    save_dict = {
                        "correct_idx": correct_idx,
                        "wrong_idx": wrong_idx,
                        "weight": weight,
                        "final_scores": final_scores,
                        "best_scores": best_scores,
                    }
                    dill.dump(save_dict, f)
    # Can find the best score once the random restarts are finished
    all_loss_x, all_loss_y_x, all_loss_y, all_loss_x_y = return_all_scores(
        final_scores
    )
    best_scores = return_best_causal_scores(
        all_loss_x, all_loss_y_x, all_loss_y, all_loss_x_y
    )

    correct_idx, wrong_idx = get_correct(best_scores, target)

    tf.print(f"Correct: {len(correct_idx)}, Wrong: {len(wrong_idx)}")
    # Save checkpoint
    with open(save_path, "wb") as f:
        save_dict = {
            "correct_idx": correct_idx,
            "wrong_idx": wrong_idx,
            "weight": weight,
            "final_scores": final_scores,
            "best_scores": best_scores,
        }
        dill.dump(save_dict, f)
