import tensorflow as tf
import tensorflow_probability as tfp


@tf.function(autograph=False, jit_compile=True)
def _cholesky(matrix):
    """Return a Cholesky factor and boolean success."""
    try:
        chol = tf.linalg.cholesky(matrix)
        ok = tf.reduce_all(tf.math.is_finite(chol))
        return chol, ok
    except tf.errors.InvalidArgumentError:
        return matrix, False


def cholesky(matrix, max_attempts: int = 8, jitter: float = 1e-8):
    def update_diag(matrix, jitter):
        diag = tf.linalg.diag_part(matrix)
        diag_add = tf.ones_like(diag) * jitter
        new_diag = diag_add + diag
        new_matrix = tf.linalg.set_diag(matrix, new_diag)
        return new_matrix

    def cond(state):
        return state[0]

    def body(state):
        _, matrix, jitter, _ = state
        res, ok = _cholesky(matrix)
        # tf.print("OK xw= ", ok, "is_not_nan = ", tf.reduce_all(tf.math.is_finite(res)), "jitter = ", jitter)
        new_matrix = tf.cond(ok, lambda: matrix, lambda: update_diag(matrix, jitter))
        break_flag = tf.logical_not(ok)
        if jitter.numpy() > 9e-8:
            tf.print("End", jitter)
        return [(break_flag, new_matrix, jitter * 10, res)]

    is_finite = tf.reduce_all(tf.math.is_finite(matrix))
    # tf.print("is_finite = ", is_finite)
    jitter = tf.cast(jitter, matrix.dtype)
    init_state = (True, matrix, jitter, matrix)
    result = tf.while_loop(cond, body, [init_state], maximum_iterations=max_attempts)

    return result[-1][-1]


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