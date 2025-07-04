"""
Importing this module requires the environment variable `KERAS_BACKEND` to be set.

- `ops.pad` does not work when paddings is symbolic tensor, so using tf.pad
- searchsorted from jax should be vmap'ed
- rename `jax.lax.cumlogsumexp` to `cumulative_logsumexp`
"""

import warnings

import keras
import numpy as np
from keras import ops

# import because can be nice to have
from keras.src.optimizers.base_optimizer import clip_by_global_norm, global_norm

from zea.utils import translate

backend = keras.backend.backend()

if backend == "jax":
    import jax
    import jax.numpy as jnp

    _searchsorted = jax.vmap(
        lambda *x: jnp.searchsorted(*x, side="left")
    )  # assumes always batched
    _cumulative_logsumexp = jax.lax.cumlogsumexp
    _pad = ops.pad
elif backend == "tensorflow":
    import tensorflow as tf

    _searchsorted = lambda *x: tf.searchsorted(*x, side="left")
    _cumulative_logsumexp = tf.math.cumulative_logsumexp
    _pad = tf.pad
else:
    warnings.warn(f"Backend {backend} not supported. Some functions may not work.")


def searchsorted(*args, **kwargs):
    return _searchsorted(*args, **kwargs)


def cumulative_logsumexp(*args, **kwargs):
    return _cumulative_logsumexp(*args, **kwargs)


def split_into_sizes(array, sizes: list, axis: int = 0):
    """
    The `array` is split into len(sizes) elements. The shape of the i-th element has the same size
    as the value except along dimension axis where the size is sizes[i].

    Args:
        array (Tensor): The array to split.
        sizes (list): The sizes of the splits.
        axis (int): The axis to split along.

    Returns:
        list: A list of arrays

    Based on: [`tf.split`](https://www.tensorflow.org/api_docs/python/tf/split) which provides
    other functionality compared to `ops.split`.
    """
    assert isinstance(sizes, list), "Sizes must be a list."
    assert sum(sizes) == ops.shape(array)[axis], (
        "Sizes must sum to the length of the array."
    )

    # Calculate split indices
    indices = np.cumsum(sizes[:-1])

    # Use np.split
    return ops.split(array, indices, axis=axis)


def postprocess(data, range_from: tuple = None, range_to: tuple = (0, 1)):
    if range_from is None:
        range_from = [ops.min(image), ops.max(image)]
    data = ops.clip(data, *range_from)
    data = translate(data, range_from, range_to)
    return data


def recursive_map_fn(func, data, num_maps):
    # Base case: when no more maps are needed
    if num_maps == 0:
        return func(data)

    # Apply ops.map and recursively reduce num_maps
    return ops.map(lambda x: recursive_map_fn(func, x, num_maps - 1), data)


def confidence_interval(metric_values, axis=-1, verbose=False):
    """
    Args:
        metric_values (ndarray): The metric values to compute the confidence interval for.
        axis (int, optional): The axis along which to compute the confidence interval.
            The other axes will be considered as batch dimensions. Defaults to -1.

    Sources:
        - https://en.wikipedia.org/wiki/97.5th_percentile_point
        - https://en.wikipedia.org/wiki/Confidence_interval
        - https://en.wikipedia.org/wiki/Standard_error
    """
    metric_values = ops.array(metric_values)
    mean = ops.mean(metric_values, axis=axis)
    std = ops.std(metric_values, axis=axis)
    n = metric_values.shape[axis]
    if verbose:
        print(f"N observations: {n}")
    standard_error = std / ops.sqrt(n)
    upper_ci = standard_error * 1.96
    lower_ci = standard_error * 1.96
    ci = ops.stack([lower_ci, upper_ci])
    return mean, ci


def is_eye_cov(cov_matrix):
    # cov_matrix: [*batch_dims, state_dim, state_dim]
    batch_dims = ops.ndim(cov_matrix) - 2
    mean_cov = ops.mean(cov_matrix, axis=list(range(batch_dims)))
    return ops.all(mean_cov == ops.eye(mean_cov.shape[0]))


def pad_zeros_like(z, axis=-1):
    return ops.concatenate([z, ops.zeros_like(z)], axis=axis)


def pad(*args, **kwargs):
    return _pad(*args, **kwargs)


def pad_to_shape(z, pad_to_shape, pad_value=0.0):
    # Compute the padding required for each dimension
    # TODO: use np.array or ops.array?
    pad_shape = np.array(pad_to_shape) - ops.shape(z)

    # Create the paddings tensor
    paddings = ops.stack([ops.zeros_like(pad_shape), pad_shape], axis=1)

    # Apply the padding
    padded_z = pad(z, paddings, constant_values=pad_value)

    return padded_z


def fit_gaussian(data, weights, eps=1e-6):
    """
    Fit a Gaussian to multiple batches of ND data points with associated weights.

    Args:
      - data: np.ndarray of shape (*batch_dims, num_particles, n_dims)
          The ND coordinates of the particles.
      - weights: np.ndarray of shape (*batch_dims, num_particles)
          The weights associated with each particle.

    Returns:
      - weighted_means: np.ndarray of shape (*batch_dims, n_dims)
          The weighted mean of the coordinates for each batch.
      - weighted_covariances: np.ndarray of shape (*batch_dims, n_dims, n_dims)
          The weighted covariance matrix of the coordinates for each batch.
    """
    # Normalize weights to ensure they sum to 1 for each batch
    weights += eps
    normalized_weights = weights / ops.sum(weights, axis=-1, keepdims=True)

    # Compute the weighted mean for each batch
    weighted_means = ops.einsum("...ij,...i->...j", data, normalized_weights)

    # Compute deviations from the mean
    deviations = data - weighted_means[..., None, :]

    # Compute the weighted covariance matrix for each batch
    weighted_covariances = ops.einsum(
        "...ik,...il,...i->...kl", deviations, deviations, normalized_weights
    )
    weighted_covariances + ops.eye(weighted_covariances.shape[-1]) * eps

    return weighted_means, weighted_covariances
