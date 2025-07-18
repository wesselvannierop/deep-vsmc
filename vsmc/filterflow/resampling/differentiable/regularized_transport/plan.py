"""
TODO: implement this module and dependencies in jax (only needed for training though)
"""

import tensorflow as tf

from vsmc.filterflow.resampling.differentiable.regularized_transport.sinkhorn import (
    sinkhorn_potentials,
)
from vsmc.filterflow.resampling.differentiable.regularized_transport.utils import (
    cost,
    diameter,
)


@tf.function
def _fillna(tensor):
    return tf.where(tf.math.is_finite(tensor), tensor, tf.zeros_like(tensor))


@tf.function
def transport_from_potentials(x, f, g, eps, logw, n):
    """
    To get the transported particles from the sinkhorn iterates
    :param x: tf.Tensor[B, N, D]
        Input: the state variable
    :param f: tf.Tensor[B, N]
        Potential, output of the sinkhorn iterates
    :param g: tf.Tensor[B, N]
        Potential, output of the sinkhorn iterates
    :param eps: float
    :param logw: tf.Tensor[N]
    :
    :return: the transportation matrix
    :rtype: tf.Tensor[B, N, N]

    """
    float_n = tf.cast(n, float)
    log_n = tf.math.log(float_n)

    cost_matrix = cost(x, x)
    fg = tf.expand_dims(f, 2) + tf.expand_dims(g, 1)  # fg = f + g.T
    temp = fg - cost_matrix
    temp = temp / eps

    temp = temp - tf.reduce_logsumexp(temp, 1, keepdims=True) + log_n
    # We "divide" the transport matrix by its col-wise sum to make sure that weights normalise to logw.
    temp = temp + tf.expand_dims(logw, 1)

    transport_matrix = tf.math.exp(temp)

    return transport_matrix  # , grad


@tf.function
@tf.custom_gradient
def transport(x, logw, eps, scaling, threshold, max_iter, n):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme".

    Args:
        x (Tensor): The input tensor of shape [batch_size, n_particles, latent_dims].
        logw (Tensor): The degenerate logweights tensor of shape [batch_size, n_particles].
        eps (float): Regularization parameter.
        scaling (float): Scaling parameter.
        threshold (float): Convergence threshold.
        max_iter (int): Maximum number of iterations.
        n (int): Number of elements.

    Returns:
        Tensor: Transport matrix of shape [batch_size, n_particles, n_particles].
    """
    float_n = tf.cast(n, float)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    dimension = tf.cast(x.shape[-1], tf.float32)
    centered_x = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    diameter_value = diameter(x, x)
    scale = tf.reshape(diameter_value, [-1, 1, 1]) * tf.sqrt(dimension)
    scaled_x = centered_x / tf.stop_gradient(scale)

    alpha, beta, _, _, _ = sinkhorn_potentials(
        logw, scaled_x, uniform_log_weight, scaled_x, eps, scaling, threshold, max_iter
    )
    transport_matrix = transport_from_potentials(
        scaled_x, alpha, beta, eps, logw, float_n
    )

    def grad(d_transport):
        d_transport = tf.clip_by_value(d_transport, -1.0, 1.0)
        # mask = logw > MIN_RELATIVE_LOG_WEIGHT * tf.math.log(float_n)  # the other particles have died out really.
        with tf.GradientTape() as tape:
            tape.watch([x, logw])
            loss = tf.reduce_sum(transport_matrix * d_transport)
        dx, dlogw = tape.gradient(loss, [x, logw])
        # dlogw = tf.where(mask, dlogw, 0.)
        # dx = tf.where(tf.expand_dims(mask, -1), dx, 0.)  # set all dimensions of the same particle to 0.
        return dx, dlogw, None, None, None, None, None

    return transport_matrix, grad
