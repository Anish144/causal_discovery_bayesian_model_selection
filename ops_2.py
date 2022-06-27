from gpflow import default_float
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import covariances


@tf.function(autograph=False, jit_compile=True)
def _cholesky(matrix, psi1, psi2, sigma2):
    """Return a Cholesky factor and boolean success."""
    try:
        L = tf.linalg.cholesky(matrix)
        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(matrix.shape[0], dtype=default_float())
        chol = tf.linalg.cholesky(B)
        ok = tf.reduce_all(tf.math.is_finite(chol))
        return chol, ok, A, AAT
    except tf.errors.InvalidArgumentError:
        return matrix, False, tf.transpose(psi1, (1, 0)), matrix


def cholesky(inducing, kernel, psi1, psi2, sigma2, max_attempts: int = 7, jitter: float = 1e-8):
    def update_diag(matrix, jitter):
        diag = tf.linalg.diag_part(matrix)
        diag_add = tf.ones_like(diag) * jitter
        new_diag = diag_add + diag
        new_matrix = tf.linalg.set_diag(matrix, new_diag)
        return new_matrix

    def cond(state):
        return state[0]

    def body(state):
        _, matrix, jitter, _, psi1, psi2, sigma2, _, _ = state
        res, ok, A, AAT = _cholesky(matrix, psi1, psi2, sigma2)
        # tf.print("OK xw= ", ok, "is_not_nan = ", tf.reduce_all(tf.math.is_finite(res)), "jitter = ", jitter)
        new_matrix = tf.cond(ok, lambda: matrix, lambda: update_diag(matrix, jitter))
        break_flag = tf.logical_not(ok)
        if jitter.numpy() > 9e-6:
            tf.print("End", jitter)
        return [(break_flag, new_matrix, jitter * 10, res, psi1, psi2, sigma2, A, AAT)]

    # is_finite = tf.reduce_all(tf.math.is_finite(matrix))
    # tf.print("is_finite = ", is_finite)
    jitter = tf.cast(jitter, default_float())

    cov_uu = covariances.Kuu(inducing, kernel, jitter=0)

    init_state = (True, cov_uu, jitter, cov_uu, psi1, psi2, sigma2, tf.transpose(psi1, (1, 0)), cov_uu)
    result = tf.while_loop(cond, body, [init_state], maximum_iterations=max_attempts)

    return result[-1][3], result[-1][7], result[-1][8]


def cholesky_with_ldl(matrix, jitter: float = 1e-8):
    ldl_fn = tfp.experimental.linalg.no_pivot_ldl

    def correct_diag(matrix, jitter):
        factor, diag = ldl_fn(matrix)
        new_diag = tf.where(diag <= 0.0, jitter, diag)
        factor_times_diag = factor * new_diag
        new_matrix = tf.matmul(factor_times_diag, factor, b_transpose=True)
        result, _ignore = _cholesky(new_matrix)
        return result

    res, ok = _cholesky(matrix)
    final_res = tf.cond(ok, lambda: res, lambda: correct_diag(matrix, jitter))
    return final_res