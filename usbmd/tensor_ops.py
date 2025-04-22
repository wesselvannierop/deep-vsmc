"""Basic tensor operations implemented with the multi-backend `keras.ops`."""

import os

import keras
from keras import ops


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob=None, seed=None):
    """
    Adds salt and pepper noise to the input image.

    Args:
        image (ndarray): The input image, must be of type float32 and normalized between 0 and 1.
        salt_prob (float): The probability of adding salt noise to each pixel.
        pepper_prob (float, optional): The probability of adding pepper noise to each pixel.
            If not provided, it will be set to the same value as `salt_prob`.
        seed: A Python integer or instance of
            `keras.random.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.random.SeedGenerator`.

    Returns:
        ndarray: The noisy image with salt and pepper noise added.
    """
    if pepper_prob is None:
        pepper_prob = salt_prob

    if salt_prob == 0.0 and pepper_prob == 0.0:
        return image

    assert ops.dtype(image) == "float32", "Image should be of type float32."

    noisy_image = ops.copy(image)

    # Add salt noise
    salt_mask = keras.random.uniform(ops.shape(image), seed=seed) < salt_prob
    noisy_image = ops.where(salt_mask, 1.0, noisy_image)

    # Add pepper noise
    pepper_mask = keras.random.uniform(ops.shape(image), seed=seed) < pepper_prob
    noisy_image = ops.where(pepper_mask, 0.0, noisy_image)

    return noisy_image


def extend_n_dims(arr, axis, n_dims):
    """
    Extend the number of dimensions of an array by inserting 'n_dims' ones at the specified axis.

    Args:
        arr: The input array.
        axis: The axis at which to insert the new dimensions.
        n_dims: The number of dimensions to insert.

    Returns:
        The array with the extended number of dimensions.

    Raises:
        AssertionError: If the axis is out of range.
    """
    assert axis <= ops.ndim(
        arr
    ), "Axis must be less than or equal to the number of dimensions in the array"
    assert (
        axis >= -ops.ndim(arr) - 1
    ), "Axis must be greater than or equal to the negative number of dimensions minus 1"
    axis = ops.ndim(arr) + axis + 1 if axis < 0 else axis

    # Get the current shape of the array
    shape = ops.shape(arr)

    # Create the new shape, inserting 'n_dims' ones at the specified axis
    new_shape = shape[:axis] + (1,) * n_dims + shape[axis:]

    # Reshape the array to the new shape
    return ops.reshape(arr, new_shape)


def func_with_one_batch_dim(
    func,
    tensor,
    n_batch_dims: int,
    batch_size: int | None = None,
    func_axis: int | None = None,
    **kwargs,
):
    """
    Applies a function to an input tensor with one or more batch dimensions. The function will
    be executed in parallel on all batch elements.

    Args:
        func (function): The function to apply to the image.
            Will take the `func_axis` output from the function.
        tensor (Tensor): The input tensor.
        n_batch_dims (int): The number of batch dimensions in the input tensor.
            Expects the input to start with n_batch_dims batch dimensions. Defaults to 2.
        batch_size (int, optional): Integer specifying the size of the batch for
            each step to execute in parallel. Defaults to None, in which case the function
            will run everything in parallel.
        func_axis (int, optional): If `func` returns mulitple outputs, this axis will be returned.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The output tensor with the same batch dimensions as the input tensor.

    Raises:
        ValueError: If the number of batch dimensions is greater than the rank of the input tensor.
    """
    # Extract the shape of the batch dimensions from the input tensor
    batch_dims = ops.shape(tensor)[:n_batch_dims]

    # Extract the shape of the remaining (non-batch) dimensions
    other_dims = ops.shape(tensor)[n_batch_dims:]

    # Reshape the input tensor to merge all batch dimensions into one
    reshaped_input = ops.reshape(tensor, [-1, *other_dims])

    # Apply the given function to the reshaped input tensor
    if batch_size is None:
        reshaped_output = func(reshaped_input, **kwargs)
    else:
        reshaped_output = batched_map(func, reshaped_input, batch_size=batch_size)

    # If the function returns multiple outputs, select the one corresponding to `func_axis`
    if isinstance(reshaped_output, (tuple, list)):
        if func_axis is None:
            raise ValueError(
                "func_axis must be specified when the function returns multiple outputs."
            )
        reshaped_output = reshaped_output[func_axis]

    # Extract the shape of the output tensor after applying the function (excluding the batch dim)
    output_other_dims = ops.shape(reshaped_output)[1:]

    # Reshape the output tensor to restore the original batch dimensions
    return ops.reshape(reshaped_output, [*batch_dims, *output_other_dims])


def matrix_power(matrix, power):
    """
    Compute the power of a square matrix.
    Should match the
    [numpy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html)
    implementation.

    Parameters:
        matrix (array-like): A square matrix to be raised to a power.
        power (int): The exponent to which the matrix is to be raised.
                    Must be a non-negative integer.
    Returns:
        array-like: The resulting matrix after raising the input matrix to the specified power.

    """
    if power == 0:
        return ops.eye(matrix.shape[0])
    if power == 1:
        return matrix
    if power % 2 == 0:
        half_power = matrix_power(matrix, power // 2)
        return ops.matmul(half_power, half_power)
    return ops.matmul(matrix, matrix_power(matrix, power - 1))


def boolean_mask(tensor, mask, size=None):
    """
    Apply a boolean mask to a tensor.

    Args:
        tensor (Tensor): The input tensor.
        mask (Tensor): The boolean mask to apply.
        size (int, optional): The size of the output tensor. Only used for Jax backend if you
            want to trace the function. Defaults to None.

    Returns:
        Tensor: The masked tensor.
    """
    backend = os.environ.get("KERAS_BACKEND")
    if backend == "jax" and size is not None:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

        indices = jnp.where(mask, size=size)  # Fixed size allows Jax tracing
        return tensor[indices]
    elif backend == "tensorflow":
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        return tf.boolean_mask(tensor, mask)
    else:
        return tensor[mask]


def flatten(tensor, start_dim=0, end_dim=-1):
    """Should be similar to: https://pytorch.org/docs/stable/generated/torch.flatten.html"""
    # Get the shape of the input tensor
    old_shape = ops.shape(tensor)

    # Adjust end_dim if it's negative
    end_dim = ops.ndim(tensor) + end_dim if end_dim < 0 else end_dim

    # Create a new shape with -1 in the flattened dimensions
    new_shape = [*old_shape[:start_dim], -1, *old_shape[end_dim + 1 :]]

    # Reshape the tensor
    return ops.reshape(tensor, new_shape)


def batched_map(f, xs, batch_size=None, jit=True, **batch_kwargs):
    """
    Map a function over leading array axes.

    Args:
        f (callable): Function to apply element-wise over the first axis.
        xs (Tensor): Values over which to map along the leading axis.
        batch_size (int, optional): Size of the batch for each step.
        jit (bool, optional): If True, use a jitted version of the function for
            faster batched mapping. Else, loop over the data with the original function.
        batch_kwargs (dict, optional): Additional keyword arguments (tensors) to
            batch along with xs. Must have the same first dimension size as xs.

    Returns:
        The mapped tensor(s).

    Idea taken from: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.map.html
    """
    if batch_kwargs is None:
        batch_kwargs = {}

    # Ensure all batch kwargs have the same leading dimension as xs.
    if batch_kwargs:
        assert all(
            ops.shape(xs)[0] == ops.shape(v)[0] for v in batch_kwargs.values()
        ), "All batch kwargs must have the same first dimension size as xs."

    total = ops.shape(xs)[0]
    # TODO: could be rewritten with ops.cond such that it also works for jit=True.
    if not jit and batch_size is not None and total <= batch_size:
        return f(xs, **batch_kwargs)

    ## Non-jitted version: simply iterate over batches.
    if not jit:
        bs = batch_size or 1  # Default batch size to 1 if not specified.
        outputs = []
        for i in range(0, total, bs):
            idx = slice(i, i + bs)
            current_kwargs = {k: v[idx] for k, v in batch_kwargs.items()}
            outputs.append(f(xs[idx], **current_kwargs))
        return ops.concatenate(outputs, axis=0)

    ## Jitted version.

    # Helper to create the batched function for use with ops.map.
    def create_batched_f(kw_keys):
        def batched_f(inputs):
            x, *kw_values = inputs
            kw = dict(zip(kw_keys, kw_values))
            return f(x, **kw)

        return batched_f

    if batch_size is None:
        batched_f = create_batched_f(list(batch_kwargs.keys()))
        return ops.map(batched_f, (xs, *batch_kwargs.values()))

    # Pad and reshape primary tensor.
    xs_padded = pad_array_to_divisible(xs, batch_size, axis=0)
    new_shape = (-1, batch_size) + ops.shape(xs_padded)[1:]
    xs_reshaped = ops.reshape(xs_padded, new_shape)

    # Pad and reshape batch_kwargs similarly.
    reshaped_kwargs = {}
    for k, v in batch_kwargs.items():
        v_padded = pad_array_to_divisible(v, batch_size, axis=0)
        reshaped_kwargs[k] = ops.reshape(
            v_padded, (-1, batch_size) + ops.shape(v_padded)[1:]
        )

    batched_f = create_batched_f(list(reshaped_kwargs.keys()))
    out = ops.map(batched_f, (xs_reshaped, *reshaped_kwargs.values()))
    out_reshaped = ops.reshape(out, (-1,) + ops.shape(out)[2:])
    return out_reshaped[:total]  # Remove any padding added.


if keras.backend.backend() == "jax":
    # For jit purposes
    def _get_padding(N, remainder):
        return N - remainder if remainder != 0 else 0

else:

    def _get_padding(N, remainder):
        return ops.where(remainder != 0, N - remainder, 0)


def pad_array_to_divisible(arr, N, axis=0, mode="constant", pad_value=None):
    """Pad an array to be divisible by N along the specified axis.
    Args:
        arr (Tensor): The input array to pad.
        N (int): The number to which the length of the specified axis should be divisible.
        axis (int, optional): The axis along which to pad the array. Defaults to 0.
        mode (str, optional): The padding mode to use. Defaults to 'constant'.
            One of `"constant"`, `"edge"`, `"linear_ramp"`,
            `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
            `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
            `"circular"`. Defaults to `"constant"`.
        pad_value (float, optional): The value to use for padding when mode='constant'.
            Defaults to None. If mode is not `constant`, this value should be None.
    Returns:
        Tensor: The padded array.
    """
    # Get the length of the specified axis
    length = ops.shape(arr)[axis]

    # Calculate how much padding is needed for the specified axis
    remainder = length % N
    padding = _get_padding(N, remainder)

    # Create a tuple with (before, after) padding for each axis
    pad_width = [(0, 0)] * ops.ndim(arr)  # No padding for other axes
    pad_width[axis] = (0, padding)  # Padding for the specified axis

    # Pad the array
    padded_array = ops.pad(arr, pad_width, mode=mode, constant_values=pad_value)

    return padded_array
