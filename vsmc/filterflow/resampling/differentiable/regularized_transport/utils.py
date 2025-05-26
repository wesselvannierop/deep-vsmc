import tensorflow as tf


def _fillna(tensor):
    mask = tf.math.is_finite(tensor)
    return tf.where(mask, tensor, tf.zeros_like(tensor))


@tf.function
def diameter(x, y):
    diameter_x = tf.reduce_max(tf.math.reduce_std(x, 1), -1)
    diameter_y = tf.reduce_max(tf.math.reduce_std(y, 1), -1)
    res = tf.maximum(diameter_x, diameter_y)
    return tf.where(res == 0.0, 1.0, res)


@tf.function
def max_min(x, y):
    max_max = tf.maximum(tf.math.reduce_max(x, [1, 2]), tf.math.reduce_max(y, [1, 2]))
    min_min = tf.minimum(tf.math.reduce_min(x, [1, 2]), tf.math.reduce_min(y, [1, 2]))

    return max_max - min_min


def softmin(epsilon: tf.Tensor, cost_matrix: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
    """Implementation of softmin function

    :param epsilon: float
        regularisation parameter
    :param cost_matrix:
    :param f:
    :return:
    """
    n = cost_matrix.shape[1]
    b = cost_matrix.shape[0]

    f_ = tf.reshape(f, (b, 1, n))
    temp_val = f_ - cost_matrix / tf.reshape(epsilon, (-1, 1, 1))
    log_sum_exp = tf.reduce_logsumexp(temp_val, axis=2)
    res = -tf.reshape(epsilon, (-1, 1)) * log_sum_exp

    return res


@tf.function
def squared_distances(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Computes the square distance matrix on the last dimension between two tensors:

    :param x: tf.Tensor[B, N, D]
    :param y: tf.Tensor[B, M, D]
    :return: tensor of shape [B, N, M]
    :rtype tf.Tensor
    """
    # Reshape x and y for broadcasting
    x_reshaped = tf.expand_dims(x, axis=2)  # [B, N, 1, D]
    y_reshaped = tf.expand_dims(y, axis=1)  # [B, 1, M, D]

    # Compute pairwise distances
    distances = tf.norm(x_reshaped - y_reshaped, axis=-1)

    return tf.square(distances)


@tf.function
def cost(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Computes the square distance matrix on the last dimension between two tensors:

    :param x: tf.Tensor[B, N, D]
    :param y: tf.Tensor[B, M, D]
    :return: tensor of shape [B, N, M]
    :rtype tf.Tensor
    """
    return squared_distances(x, y) / 2.0
