from keras import ops

from dpf.filterflow.constants import (
    MIN_ABSOLUTE_LOG_WEIGHT,
    MIN_ABSOLUTE_WEIGHT,
    MIN_RELATIVE_LOG_WEIGHT,
    MIN_RELATIVE_WEIGHT,
)


def _normalize(weights, axis, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if log:
        normalizer = ops.logsumexp(weights, axis=axis, keepdims=True)
        return weights - normalizer
    normalizer = ops.sum(weights, axis=axis)
    return weights / normalizer


def normalize(weights, axis, n, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    float_n = ops.cast(n, "float32")

    if log:
        normalized_weights = ops.clip(
            _normalize(weights, axis, True), ops.array(-1e3), ops.array(0.0)
        )
        stop_gradient_mask = normalized_weights < ops.maximum(
            MIN_ABSOLUTE_LOG_WEIGHT, MIN_RELATIVE_LOG_WEIGHT * float_n
        )
    else:
        normalized_weights = _normalize(weights, axis, False)
        stop_gradient_mask = normalized_weights < ops.maximum(
            MIN_ABSOLUTE_WEIGHT, MIN_RELATIVE_WEIGHT**float_n
        )
    float_stop_gradient_mask = ops.cast(stop_gradient_mask, "float32")
    return (
        ops.stop_gradient(float_stop_gradient_mask * normalized_weights)
        + (1.0 - float_stop_gradient_mask) * normalized_weights
    )


def _native_mean(weights, particles, keepdims):
    weights = ops.expand_dims(weights, -1)
    res = ops.sum(weights * particles, axis=-2, keepdims=keepdims)
    return res


def _log_mean(log_weights, particles, keepdims):
    max_log_weights = ops.stop_gradient(ops.max(log_weights, axis=-1, keepdims=True))
    weights = ops.exp(log_weights - max_log_weights)
    if keepdims:
        max_log_weights = ops.expand_dims(max_log_weights, -1)

    temp = particles * ops.expand_dims(weights, -1)
    temp = ops.sum(temp, -2, keepdims=keepdims)
    res = max_log_weights + ops.log(temp)

    return res


def mean(state, keepdims=True, is_log=False):
    """Returns the weighted averaged of the state"""
    return mean_raw(state.particles, state.log_weights, keepdims, is_log)


def mean_raw(particles, log_weights, keepdims=True, is_log=False):
    """Returns the weighted averaged of the state"""
    if is_log:
        return _log_mean(log_weights, particles, keepdims)
    else:
        return _native_mean(ops.exp(log_weights), particles, keepdims)


def std_raw(particles, log_weights, avg=None, keepdims=True, is_log=False):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if avg is None:
        avg = mean_raw(particles, log_weights, keepdims=True, is_log=False)
    centered_state = particles - avg
    if is_log:
        var = _log_mean(log_weights, centered_state**2, keepdims=keepdims)
        return 0.5 * var
    else:
        var = _native_mean(ops.exp(log_weights), centered_state**2, keepdims=keepdims)
        return ops.sqrt(var)


def std(state, avg=None, keepdims=True, is_log=False):
    """Normalises weights, either expressed in log terms or in their natural space"""
    return std_raw(state.particles, state.log_weights, avg, keepdims, is_log)
