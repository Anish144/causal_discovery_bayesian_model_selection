from tabnanny import check
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.models import GPModel, SVGP
from gpflow import covariances, kernels, likelihoods
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints, InducingVariables
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import positive, to_default_float
from gpflow.utilities.ops import pca_reduce
from gpflow.models.gpr import GPR
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from functools import partial
from gpflow.conditionals.util import sample_mvn

from ops import cholesky


def batch_kernel_evaluation(X, kernel):
    return kernel(X[0], X[1])


class GeneralisedUnsupGPLVM(SVGP):
    def __init__(
        self,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        likelihood: likelihoods,
        num_mc_samples: int,
        inducing_variable: InducingVariables,
        batch_size: int,
        q_mu=None,
        q_sqrt=None,
        X_prior_mean=None,
        X_prior_var=None,
    ):
        """
        This is the Generalised GPLVM as listed in:
        https://arxiv.org/pdf/2202.12979.pdf

        The key point here are:
        - Uses uncollapsed inducing variables which allows for minibatching
        - Still computes the kernel expectation, but using MC expectation

        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x (L + Q) (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        """
        num_data, num_latent_gps = X_data_mean.shape
        super().__init__(
            kernel,
            likelihood,
            mean_function=None,
            num_latent_gps=num_latent_gps,
            q_diag=False,
            whiten=True,
            inducing_variable=inducing_variable,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
        )

        # Set in data to be a non trainable parameter
        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        self.batch_size = batch_size

        self.num_data = num_data
        self.num_mc_samples = num_mc_samples

        assert np.all(X_data_mean.shape == X_data_var.shape)

        if (inducing_variable is None):
            raise ValueError(
                "BayesianGPLVM needs `inducing_variable`"
            )

        # Make only the non latent part of inducing trainable
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent_gps
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent_gps

    def get_new_mean_vars(self, data_idx) -> MeanAndVariance:
        batch_X_means = tf.gather(self.X_data_mean, data_idx)
        batch_X_vars = tf.gather(self.X_data_var, data_idx)
        new_mean = batch_X_means
        new_variance = batch_X_vars
        return new_mean, new_variance, batch_X_means, batch_X_vars

    def predict_f_samples(
        self,
        Xnew,
        obs_noise=False,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.
        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )

        # check below for shape info
        mean, cov = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        if obs_noise is True and full_cov is False:
            cov += self.likelihood.variance
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(
                mean_for_sample, cov, full_cov, num_samples=num_samples
            )  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            samples = sample_mvn(
                mean, cov, full_output_cov, num_samples=num_samples
            )  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def predict_full_samples_layer(self, sample_size=1000, obs_noise=False, num_latent_samples=50, num_gp_samples=50):
        w = np.random.normal(size=(num_latent_samples, sample_size, 1))
        sampling_func = lambda x: self.predict_f_samples(x, obs_noise=obs_noise, num_samples=num_gp_samples)

        def sample_latent_gp(w_single):
            X = w_single
            samples = sampling_func(X)
            return samples

        samples = tf.map_fn(sample_latent_gp, w)

        return samples

    def predict_credible_layer(
        self,
        sample_size,
        lower_quantile=2.5,
        upper_quantile=97.5,
        num_gp_samples=50,
        num_latent_samples=50,
        obs_noise=False,
    ):

        samples = self.predict_full_samples_layer(
            sample_size=sample_size,
            obs_noise=obs_noise,
            num_gp_samples=num_gp_samples,
            num_latent_samples=num_latent_samples,
        )
        lower = np.percentile(samples, lower_quantile, axis=[0,1])
        median = np.percentile(samples, 50, axis=[0,1])
        upper = np.percentile(samples, upper_quantile, axis=[0,1])

        return lower, median, upper, samples

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        Y_data, data_idx = data
        (
            new_mean,
            new_variance,
            batch_X_means,
            batch_X_vars
        ) = self.get_new_mean_vars(data_idx)

        # We integrate out the latent variable by taking J MC samples
        # [num_mc, num_batch, num_dim]
        X_samples = tfp.distributions.MultivariateNormalDiag(
            loc=new_mean,
            scale_diag=new_variance,
        ).sample(self.num_mc_samples)

        # # KL[q(x) || p(x)]
        batch_prior_means = tf.gather(
            self.X_prior_mean, data_idx
        )
        batch_prior_vars = tf.gather(
            self.X_prior_var, data_idx
        )
        dX_data_var = (
            batch_X_vars
            if batch_X_vars.shape.ndims == 2
            else tf.linalg.diag_part(batch_X_vars)
        )
        NQ = to_default_float(tf.size(batch_X_means))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var + 1e-30))
        KL += 0.5 * tf.reduce_sum(tf.math.log(batch_prior_vars + 1e-30))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(batch_X_means - batch_prior_means) + dX_data_var) / batch_prior_vars
        )

        KL_2 = self.prior_kl()
        f_mean, f_var = self.predict_f(X_samples, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y_data)
        # MC over 1st dim
        var_exp = tf.reduce_mean(var_exp, axis=0)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, KL_2.dtype)
            minibatch_size = tf.cast(tf.shape(Y_data)[0], KL_2.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, KL_2.dtype)
        return (tf.reduce_sum(var_exp) - KL) * scale - KL_2

        # # Evaluate the necessary kernels
        # batch_kern_ready = partial(
        #     batch_kernel_evaluation,
        #     kernel=self.kernel
        # )

        # # [num_mc, num_batch, num_batch]
        # batch_cov_ff = tf.vectorized_map(
        #     batch_kern_ready, [X_samples, X_samples]
        # )

        # psi_0 = tf.reduce_mean(batch_cov_ff, axis=0)
        # # Scalar
        # trace_psi_0 = tf.einsum("ii->", psi_0)

        # inducing_new_axis = self.inducing_variable.Z[None, :, :]
        # inducing_copy = tf.tile(inducing_new_axis, [self.num_mc_samples, 1, 1])
        # # [num_mc, num_batch, num_inducing]
        # batch_cov_fu = tf.vectorized_map(
        #     batch_kern_ready, [X_samples, inducing_copy]
        # )

        # # [num_batch, num_inducing]
        # psi_1 = tf.reduce_mean(batch_cov_fu, axis=0)

        # kern_product = tf.einsum(
        #     "jnm,jnp->jmp", batch_cov_fu, batch_cov_fu
        # )
        # # [num_inducing, num_inducing]
        # psi_2 = tf.reduce_mean(kern_product, axis=0)

        # old_psi0 = expectation(pX, self.kernel)
        # old_psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        # old_psi2 = expectation(
        #         pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
        # )



        # # [num_inducing, num_inducing]
        # cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=1e-6)

        # # Do everything with cholesky of cov_uu
        # L = tf.linalg.cholesky(cov_uu) # [num_inducing, num_inducing]

        # # Find the conditional likelihood term
        # # Gaussian term
        # # Mean
        # A = tf.linalg.triangular_solve(L, tf.linalg.adjoint(psi_1), lower=True)  # [..., M, N]
        # A = tf.linalg.triangular_solve(tf.linalg.adjoint(L), A, lower=False)
        # fmean = tf.linalg.matmul(A, self.q_mu, transpose_a=True)  # [..., N, R]
        # log_prob = self.likelihood.log_prob(fmean, Y_data)
        # term_1 = tf.reduce_sum(log_prob)

        # noise = self.likelihood.variance
        # noise_for_gauss = - 0.5 * (1 / (2 * noise))

        # # 1st Trace term
        # term_2 = noise_for_gauss * trace_psi_0

        # # 2nd Trace term
        # B = tf.linalg.triangular_solve(L, psi_2, lower=True)  # [..., M, N]
        # B = tf.linalg.triangular_solve(tf.linalg.adjoint(L), B, lower=False)
        # term_3 = - noise_for_gauss * tf.einsum(
        #     "ii->", B
        # )

        # # 3rd Trace term
        # C = tf.linalg.triangular_solve(L, self.q_sqrt, lower=True)  # [..., M, N]
        # C = tf.linalg.triangular_solve(
        #     tf.linalg.adjoint(L), self.q_sqrt, lower=False
        # )  # [..., M, N]
        # S_Kmm_inv = tf.linalg.matmul(
        #     self.q_sqrt, C, transpose_b=True
        # )
        # D = tf.linalg.matmul(
        #     S_Kmm_inv,
        #     B,
        #     transpose_b=True
        # )
        # # Sum over output dimension
        # D = tf.reduce_sum(D, axis=0)
        # term_4 = noise_for_gauss * tf.einsum(
        #     "ii->", D
        # )

        # # Combine above to give the final likelihood term
        # final_like_term = term_1
        # final_like_term += term_2
        # final_like_term += term_3
        # final_like_term += term_4

        # # KL[q(x) || p(x)]
        # dX_data_var = (
        #     self.X_data_var
        #     if self.X_data_var.shape.ndims == 2
        #     else tf.linalg.diag_part(self.X_data_var)
        # )
        # NQ = to_default_float(tf.size(self.X_data_mean))
        # D = to_default_float(tf.shape(Y_data)[1])
        # KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var + 1e-30))
        # KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var + 1e-30))
        # KL -= 0.5 * NQ
        # KL += 0.5 * tf.reduce_sum(
        #     (tf.square(self.X_data_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        # )

        # # KL[q(u)) || p(u)]
        # KL_2 = self.prior_kl()

        # bound = final_like_term - KL - KL_2

        # import pdb; pdb.set_trace()


        # num_inducing = self.inducing_variable.num_inducing
        # psi0 = tf.reduce_sum(expectation(pX, self.kernel))
        # # try:
        # psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        # # except:
        # # tf.print(self.kernel.lengthscales, self.kernel.variance, self.likelihood.variance)
        # # tf.print(tf.reduce_min(new_variance[:, 1]))
        # psi2 = tf.reduce_sum(
        #     expectation(
        #         pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
        #     ),
        #     axis=0,
        # )
        # cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=self.jitter)
        # L = tf.linalg.cholesky(cov_uu)
        # # L = cholesky(cov_uu)
        # tf.debugging.assert_all_finite(
        #     L, message="L is not finite!"
        # )
        # sigma2 = self.likelihood.variance
        # # Compute intermediate matrices
        # A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        # tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        # AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        # B = AAT + tf.eye(num_inducing, dtype=default_float())
        # LB = tf.linalg.cholesky(B)
        # # LB = cholesky(B)
        # tf.debugging.assert_all_finite(
        #     LB, message="LB is not finite!"
        # )
        # log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        # c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2

        # # KL[q(x) || p(x)]
        # dX_data_var = (
        #     self.X_data_var
        #     if self.X_data_var.shape.ndims == 2
        #     else tf.linalg.diag_part(self.X_data_var)
        # )
        # NQ = to_default_float(tf.size(self.X_data_mean))
        # D = to_default_float(tf.shape(Y_data)[1])
        # KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var + 1e-30))
        # KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var + 1e-30))
        # KL -= 0.5 * NQ
        # KL += 0.5 * tf.reduce_sum(
        #     (tf.square(self.X_data_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        # )

        # # compute log marginal bound
        # ND = to_default_float(tf.size(Y_data))
        # bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        # bound += -0.5 * D * log_det_B
        # bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        # bound += 0.5 * tf.reduce_sum(tf.square(c))
        # bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        # bound -= KL
        # if np.isnan(LB.numpy()).sum() > 0:
        #     print(f"{self.kernel.variance.numpy()}, {self.likelihood.variance.numpy()}, {self.kernel.lengthscales.numpy()}")
            # import pdb; pdb.set_trace()

        # print(f"{self.kernel.variance.numpy()}, {self.likelihood.variance.numpy()}, {self.kernel.lengthscales.numpy()}")
        # return bound

    # def predict_f(
    #     self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    # ) -> MeanAndVariance:
    #     """
    #     Compute the mean and variance of the latent function at some new points.
    #     Note that this is very similar to the SGPR prediction, for which
    #     there are notes in the SGPR notebook.

    #     Note: This model does not allow full output covariances.

    #     :param Xnew: points at which to predict
    #     """
    #     if full_output_cov:
    #         raise NotImplementedError

    #     new_mean, new_variance = self.get_new_mean_vars()
    #     pX = DiagonalGaussian(new_mean, new_variance)

    #     Y_data = self.data
    #     num_inducing = self.inducing_variable.num_inducing
    #     psi1 = expectation(pX, (self.kernel, self.inducing_variable))
    #     psi2 = tf.reduce_sum(
    #         expectation(
    #             pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
    #         ),
    #         axis=0,
    #     )
    #     jitter = default_jitter()
    #     Kus = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
    #     sigma2 = self.likelihood.variance
    #     L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=self.jitter))

    #     A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
    #     tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
    #     AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
    #     B = AAT + tf.eye(num_inducing, dtype=default_float())
    #     LB = tf.linalg.cholesky(B)
    #     c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2
    #     tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
    #     tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
    #     mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
    #     if full_cov:
    #         var = (
    #             self.kernel(Xnew)
    #             + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
    #             - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
    #         )
    #         shape = tf.stack([1, 1, tf.shape(Y_data)[1]])
    #         var = tf.tile(tf.expand_dims(var, 2), shape)
    #     else:
    #         var = (
    #             self.kernel(Xnew, full_cov=False)
    #             + tf.reduce_sum(tf.square(tmp2), axis=0)
    #             - tf.reduce_sum(tf.square(tmp1), axis=0)
    #         )
    #         shape = tf.stack([1, tf.shape(Y_data)[1]])
    #         var = tf.tile(tf.expand_dims(var, 1), shape)
    #     return mean + self.mean_function(Xnew), var

    # def predict_log_density(self, data: OutputData) -> tf.Tensor:
    #     raise NotImplementedError


def check_condition_number(matrix):
    eig_B = tf.linalg.eigvals(
        matrix, name=None
    )
    eig_real = tf.math.real(eig_B)
    max_eig = tf.math.abs(tf.reduce_max(eig_real))
    min_eig = tf.math.abs(tf.reduce_min(eig_real))
    condition_number = max_eig / min_eig
    return condition_number