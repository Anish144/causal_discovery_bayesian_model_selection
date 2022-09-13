"""
This method will fit a gplvm for both the marginal and conditional models and
choose the causal direction as the one with the minimum
-log marginal likelihood.
"""
from models.GPDE import GPDE
from models.GPCDE import GPCDE
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm import trange
from typing import Optional
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from gpflow.quadrature import NDiagGHQuadrature
from tqdm import trange
from gpflow.optimizers import NaturalGradient
import gpflow
import matplotlib.pyplot as plt


adam_learning_rates = [0.05, 0.01, 0.005, 0.001]
natgrad_learning_rates = [0.1, 0.05]


def run_optimizer(model, ds_iter, iterations, adam_lr, nat_lr):
    """
    Utility function running the Adam optimizer
    Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    training_loss = model.training_loss_closure(ds_iter, compile=True)
    gpflow.set_trainable(model.q_mu, False)
    gpflow.set_trainable(model.q_sqrt, False)
    variational_params = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = NaturalGradient(gamma=nat_lr)
    optimizer = tf.optimizers.Adam(adam_lr)
    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        natgrad_opt.minimize(training_loss, variational_params)
    iterator = trange(iterations)
    for step in iterator:
        optimization_step()
        if step % 1000 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            iterator.set_description(f"EPOCH: {step}, ELBO: {elbo}")

            if step > 5000:
                if np.abs(np.mean(logf[-5000:])) - np.abs(np.mean(logf[-100:])) < np.std(logf[-100:]):
                    print("Breaking!")
                    break
    return logf


def train_marginal_model(
    y: np.ndarray,
    num_inducing: int,
    kernel_variance: float,
    kernel_lengthscale_1: float,
    kernel_lengthscale_2: float,
    likelihood_variance: float,
    work_dir: str,
    save_name: str,
    adam_lr: float,
    nat_lr: float,
    causal: Optional[bool] = None,
    run_number: Optional[int] = None,
    random_restart_number: Optional[int] = None,
    plot_fit: Optional[bool] = False,
):
    # Define kernels
    sq_exp = gpflow.kernels.SquaredExponential(lengthscales=kernel_lengthscale_1)
    sq_exp.variance.assign(kernel_variance)
    matern = gpflow.kernels.Matern32(lengthscales=kernel_lengthscale_2)
    matern.variance.assign(kernel_variance)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel, matern])
    Z = np.random.randn(num_inducing, 1)

    # Define marginal model
    marginal_model = GPDE(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(
            variance=likelihood_variance
        ),
        inducing_variable=Z,
        num_quadrature=100,
        whiten=True
    )
    ds_iter = marginal_model._init_quadrature(Y=y, num_minibatch=100)
    logf = run_optimizer(
        model=marginal_model,
        ds_iter=ds_iter,
        iterations=50000,
        adam_lr=adam_lr,
        nat_lr=nat_lr
    )

    N = y.shape[0]
    marginal_model.num_minibatch = N
    num_quadrature = 100
    def full_init_quadrature(Y, num_quadrature, num_minibatch = 100):
        num_minibatch = num_minibatch
        quadrature = NDiagGHQuadrature(1, num_quadrature)
        quadrature_locs, quadrature_weights = quadrature._build_X_W(np.zeros(1), np.ones(1))
        quadrature_weights = quadrature_weights[:,0]
        quadrature_weights = quadrature_weights[None, :]
        quadrature_locs = tf.expand_dims(quadrature_locs, axis=0)
        quadrature_locs = tf.tile(quadrature_locs, multiples=[Y.shape[0],1,1])
        Y = Y[:, None,:]
        Y = tf.tile(Y, multiples=[1, num_quadrature, 1])
        X = quadrature_locs
        return X, Y

    ds_full = full_init_quadrature(y, num_quadrature, N)
    full_elbo = marginal_model.elbo(ds_full)
    print(f"Full Loss: {- full_elbo}")

    loss = - full_elbo

    if plot_fit:
        # Plot the fit to see if everything is ok
        lower, median, upper, samples = marginal_model.predict_credible_layer(
            obs_noise=True,
            sample_size=1000
        )
        textstr = 'like_var=%.2f\nneg_elbo=%.2f\n'%(
            marginal_model.likelihood.variance.numpy(),
            loss.numpy()
        )
        plt.text(-8, 0, textstr, fontsize=8)
        plt.hist(y[:, 0], bins=100)
        plt.hist(median, bins=100, alpha=0.5)
        plt.xlim(-3, 3)

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
    kernel_lengthscale_1: float,
    kernel_lengthscale_2: float,
    likelihood_variance: float,
    work_dir: str,
    save_name: str,
    adam_lr: float,
    nat_lr: float,
    causal: Optional[bool] = None,
    run_number: Optional[int] = None,
    random_restart_number: Optional[int] = None,
    plot_fit: Optional[bool] = False,
):
    """
    Train a conditional model using a partially observed GPLVM.
    """
    # Define kernels
    sq_exp = gpflow.kernels.SquaredExponential(
        lengthscales=[kernel_lengthscale_1, kernel_lengthscale_1 * 1.3]
    )
    sq_exp.variance.assign(kernel_variance)
    matern = gpflow.kernels.Matern32(
        lengthscales=[kernel_lengthscale_2, kernel_lengthscale_2 * 1.3]
    )
    matern.variance.assign(kernel_variance)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel, matern])

    Z = np.concatenate(
            [
                np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
                np.random.randn(num_inducing, 1),
            ],
            axis=1
        )
    conditional_model = GPCDE(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(
            variance=likelihood_variance
        ),
        inducing_variable=Z,
        num_quadrature=100,
        whiten=True
    )
    ds_iter = conditional_model._init_quadrature(X=x, Y=y, num_minibatch=100)

    logf = run_optimizer(
        model=conditional_model,
        ds_iter=ds_iter,
        iterations=50000,
        adam_lr=adam_lr,
        nat_lr=nat_lr,
    )

    N = x.shape[0]
    conditional_model.num_minibatch = N
    full_ds_iter = conditional_model._init_quadrature(x, y, N)
    num_quadrature = 100
    def full_init_quadrature(X, Y, num_minibatch = 100):
        num_data= X.shape[0]
        num_dim = X.shape[1]
        num_minibatch = num_minibatch
        quadrature = NDiagGHQuadrature(1, num_quadrature)
        quadrature_locs, quadrature_weights = quadrature._build_X_W(np.zeros(1), np.ones(1))
        quadrature_weights = quadrature_weights[:,0]
        quadrature_weights = quadrature_weights[None, :]
        quadrature_locs = tf.expand_dims(quadrature_locs, axis=0)
        quadrature_locs = tf.tile(quadrature_locs, multiples=[X.shape[0],1,1])
        X = X[:,None,:]
        Y = Y[:,None,:]
        X = tf.tile(X, multiples=[1, num_quadrature, 1])
        Y = tf.tile(Y, multiples=[1, num_quadrature, 1])
        X = tf.concat([X, quadrature_locs],axis=-1)
        return X, Y

    ds_full = full_init_quadrature(x, y, N)
    full_elbo = conditional_model.elbo(ds_full)
    print(f"Full Loss: {- full_elbo}")

    loss = - full_elbo

    if plot_fit:
        # Plot the fit to see if everything is ok
        obs_new = np.linspace(x.min() - 2, x.max() + 1, 1000)[:, None]
        # Sample from the prior
        lower, median, upper, samples = conditional_model.predict_credible_layer(
            Xnew=obs_new,
            obs_noise=True
        )
        textstr = 'like_var=%.2f\nneg_elbo=%.2f\n'%(
            conditional_model.likelihood.variance.numpy(),
            loss
        )
        plt.text(x.min() - 6, 0, textstr, fontsize=8)
        plt.scatter(x[:, 0], y[:, 0], c='r')
        plt.plot(obs_new, median, c='b', alpha=0.2)
        # plt.scatter(obs_new[:, 0], samples[0, 0], alpha=0.5)
        plt.fill_between(obs_new[:, 0], upper[:, 0], lower[:, 0], alpha=0.5)

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


def causal_score_gplvm_quadrature(args, x, y, run_number, restart_number, causal, save_name):
    num_inducing = args.num_inducing if x.shape[0] > args.num_inducing else x.shape[0]

    # Sample hyperparams
    kernel_variance = 1.0
    loss_x = None
    # Likelihood variance
    kappa = np.random.uniform(
        low=10.0, high=100, size=[1]
    )
    likelihood_variance = 1. / (kappa ** 2)
    # Kernel lengthscale
    lamda = np.random.uniform(
        low=1, high=100, size=[2]
    )
    kernel_lengthscale = 1.0 / lamda
    adam_lr = np.random.choice(adam_learning_rates)
    nat_lr = np.random.choice(natgrad_learning_rates)

    tf.print("X" if causal else "Y")
    loss_x = train_marginal_model(
        y=x,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale_1=kernel_lengthscale[0],
        kernel_lengthscale_2=kernel_lengthscale[1],
        likelihood_variance=likelihood_variance[0],
        work_dir=args.work_dir,
        run_number=run_number,
        random_restart_number=restart_number,
        causal=causal,
        save_name=save_name,
        plot_fit=args.plot_fit,
        adam_lr=adam_lr,
        nat_lr=nat_lr,
    )

    # Sample hyperparams
    kernel_variance = 1.0
    # Likelihood variance
    kappa = np.random.uniform(
        low=10.0, high=100, size=[1]
    )
    likelihood_variance = 1. / (kappa ** 2)
    # Kernel lengthscale
    lamda = np.random.uniform(
        low=1.0, high=100, size=[2]
    )
    kernel_lengthscale = 1.0 / lamda
    adam_lr = np.random.choice(adam_learning_rates)
    nat_lr = np.random.choice(natgrad_learning_rates)

    tf.print("Y|X" if causal else "X|Y")
    loss_y_x = train_conditional_model(
        x=x,
        y=y,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale_1=kernel_lengthscale[0],
        kernel_lengthscale_2=kernel_lengthscale[1],
        likelihood_variance=likelihood_variance[0],
        work_dir=args.work_dir,
        run_number=run_number,
        random_restart_number=restart_number,
        causal=causal,
        save_name=save_name,
        plot_fit=args.plot_fit,
        adam_lr=adam_lr,
        nat_lr=nat_lr,
    )

    return (loss_x, loss_y_x)


def min_causal_score_gplvm_quadrature(args, x, y, weight, target):
    # TODO: The data loading and saving etc should be separate from the model
    # Find data index to start and end the runs on
    data_start_idx = args.data_start
    data_end_idx = args.data_end if args.data_end < len(x) else len(x)
    save_name = f"fullscore-{args.data}-gplvmquadrature-reinit{args.random_restarts}-numind{args.num_inducing}" \
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

            ) = causal_score_gplvm_quadrature(
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

            ) = causal_score_gplvm_quadrature(
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