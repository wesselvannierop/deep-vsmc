import abc

from keras import ops

from zea.tensor_ops import extend_n_dims
from vsmc.filterflow.resampling.base import ResamplerBase, resample
from vsmc.filterflow.state import State
from vsmc.ops import cumulative_logsumexp, searchsorted


def _discrete_percentile_function(
    spacings, n_particles, on_log, weights=None, log_weights=None
):
    """vectorised resampling function, can be used for systematic/stratified/multinomial resampling"""
    if on_log:
        cumlogsumexp = cumulative_logsumexp(log_weights, axis=1)
        log_spacings = ops.log(spacings)
        indices = searchsorted(cumlogsumexp, log_spacings)
    else:
        cum_sum = ops.cumsum(weights, axis=1)
        indices = searchsorted(cum_sum, spacings)

    return ops.clip(indices, 0, n_particles - 1)


class StandardResamplerBase(ResamplerBase, metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""

    DIFFERENTIABLE = False

    def __init__(self, on_log):
        """Constructor

        :param on_log: bool
            Should the resampling use log weights
        :param stop_gradient: bool
            Should the resampling step propagate the stitched gradients or not
        """
        super().__init__()
        self._on_log = on_log

    @staticmethod
    @abc.abstractmethod
    def _get_spacings(n_particles, batch_size, seed):
        """Spacings variates to give for empirical CDF block selection"""

    def apply(self, state: State, flags, seed=None):
        """Resampling method

        :param state State
            Particle filter state
        :param flags: Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """
        batch_size = state.batch_size
        n_particles = state.n_particles

        spacings = self._get_spacings(n_particles, batch_size, seed)
        # TODO: We should be able to get log spacings directly to always stay in log space.
        indices = _discrete_percentile_function(
            spacings, n_particles, self._on_log, state.weights, state.log_weights
        )

        ancestor_indices = ops.where(
            ops.reshape(flags, [-1, 1]),
            indices,
            ops.reshape(ops.arange(n_particles), [1, -1]),
        )

        indices = extend_n_dims(indices, -1, len(state.dimension))
        new_particles = ops.take_along_axis(state.particles, indices, axis=1)

        float_n_particles = ops.cast(n_particles, float)
        uniform_weights = ops.ones_like(state.weights) / float_n_particles
        uniform_log_weights = ops.zeros_like(state.log_weights) - ops.log(
            float_n_particles
        )

        resampled_particles = resample(state.particles, new_particles, flags)
        resampled_weights = resample(state.weights, uniform_weights, flags)
        resampled_log_weights = resample(state.log_weights, uniform_log_weights, flags)

        return state.evolve(
            particles=resampled_particles,
            weights=resampled_weights,
            log_weights=resampled_log_weights,
            ancestor_indices=ancestor_indices,
        )
