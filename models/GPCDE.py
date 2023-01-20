import tensorflow as tf
import numpy as np
from gpflow.models import GPModel, SVGP
from gpflow.base import Parameter
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.mean_functions import MeanFunction
from gpflow import posteriors
from gpflow.kullback_leiblers import gauss_kl
from gpflow.config import default_jitter, default_float
from gpflow.utilities import positive, triangular
from gpflow.models.util import inducingpoint_wrapper
from gpflow.conditionals.util import sample_mvn
from gpflow.quadrature import NDiagGHQuadrature

# from gparkme.posteriors import create_posterior
from typing import Optional


class GPCDE(SVGP):
    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        mean_function: Optional[MeanFunction] = None,
        num_quadrature: int = 10,
        q_diag: bool = False,
        whiten: bool = True,
        q_mu=None,
        q_sqrt=None,
        num_latent_gps=1,
    ):
        super().__init__(
            kernel,
            likelihood,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            q_diag=q_diag,
            whiten=whiten,
            inducing_variable=inducing_variable,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
        )
        self.num_quadrature = num_quadrature

    def _init_quadrature(self, X, Y, num_minibatch=100):
        self.num_data = X.shape[0]
        self.num_dim = X.shape[1]
        self.num_minibatch = num_minibatch
        self.quadrature = NDiagGHQuadrature(1, self.num_quadrature)
        quadrature_locs, quadrature_weights = self.quadrature._build_X_W(
            np.zeros(1), np.ones(1)
        )
        self.quadrature_weights = quadrature_weights[:, 0]
        self.quadrature_weights = self.quadrature_weights[None, :]
        quadrature_locs = tf.expand_dims(quadrature_locs, axis=0)
        quadrature_locs = tf.tile(quadrature_locs, multiples=[X.shape[0], 1, 1])
        X = X[:, None, :]
        Y = Y[:, None, :]
        X = tf.tile(X, multiples=[1, self.num_quadrature, 1])
        Y = tf.tile(Y, multiples=[1, self.num_quadrature, 1])
        X = tf.concat([X, quadrature_locs], axis=-1)
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        ds_iter = iter(ds.repeat().shuffle(self.num_data).batch(num_minibatch))
        return ds_iter

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
        mean, cov = self.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
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

    def predict_full_samples_layer(
        self, Xnew, obs_noise=False, num_latent_samples=50, num_gp_samples=50
    ):
        w = np.random.normal(size=(num_latent_samples, Xnew.shape[0], 1))
        sampling_func = lambda x: self.predict_f_samples(
            x, obs_noise=obs_noise, num_samples=num_gp_samples
        )

        def sample_latent_gp(w_single):
            X = np.concatenate([Xnew, w_single], axis=1)
            samples = sampling_func(X)
            return samples

        samples = tf.map_fn(sample_latent_gp, w)

        return samples

    def predict_credible_layer(
        self,
        Xnew,
        lower_quantile=2.5,
        upper_quantile=97.5,
        num_gp_samples=50,
        num_latent_samples=50,
        obs_noise=False,
    ):

        samples = self.predict_full_samples_layer(
            Xnew,
            obs_noise=obs_noise,
            num_gp_samples=num_gp_samples,
            num_latent_samples=num_latent_samples,
        )
        lower = np.percentile(samples, lower_quantile, axis=[0, 1])
        median = np.percentile(samples, 50, axis=[0, 1])
        upper = np.percentile(samples, upper_quantile, axis=[0, 1])

        return lower, median, upper, samples

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        X = tf.reshape(
            X, (self.num_minibatch * self.num_quadrature, self.num_dim + 1)
        )
        Y = tf.reshape(Y, (self.num_minibatch * self.num_quadrature, 1))
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        # each var_exp is (num_minibatch, num_quadrature, 1)
        var_exp = tf.reshape(var_exp, (self.num_minibatch, self.num_quadrature))
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)

        # var_exp = tf.reduce_sum(var_exp, axis=0) * scale
        quadrature_weights = self.quadrature_weights
        term1 = tf.math.reduce_logsumexp(
            var_exp + tf.math.log(quadrature_weights), axis=1
        )
        return tf.reduce_sum(term1) * scale - kl
