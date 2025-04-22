import abc

from keras import ops

from dpf.filterflow.resampling.base import ResamplerBase, resample
from dpf.filterflow.resampling.differentiable.regularized_transport.plan import (
    transport,
)
from dpf.filterflow.state import State


def apply_transport_matrix(state: State, transport_matrix, flags):
    float_n_particles = ops.cast(state.n_particles, "float32")
    transported_particles = ops.matmul(transport_matrix, state.flat_particles)
    uniform_log_weights = -ops.log(float_n_particles) * ops.ones_like(state.log_weights)
    uniform_weights = ops.ones_like(state.weights) / float_n_particles

    resampled_particles = resample(state.flat_particles, transported_particles, flags)
    resampled_weights = resample(state.weights, uniform_weights, flags)
    resampled_log_weights = resample(state.log_weights, uniform_log_weights, flags)

    return state.evolve(
        flat_particles=resampled_particles,
        weights=resampled_weights,
        log_weights=resampled_log_weights,
    )


class RegularisedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Regularised Transform - docstring to come."""

    DIFFERENTIABLE = True

    # TODO: Document this really nicely
    def __init__(
        self, epsilon, scaling=0.75, max_iter=100, convergence_threshold=1e-3, **_kwargs
    ):
        """Constructor

        :param epsilon: float
            Regularizer for Sinkhorn iterates
        :param scaling: float
            Epsilon scaling for sinkhorn iterates
        :param max_iter: int
            max number of iterations in Sinkhorn
        :param convergence_threshold: float
            Fixed point iterates converge when potentials don't move more than this anymore
        """
        self.convergence_threshold = ops.cast(convergence_threshold, "float32")
        self.max_iter = ops.cast(max_iter, "int32")
        self.epsilon = ops.cast(epsilon, "float32")
        self.scaling = ops.cast(scaling, "float32")
        super(RegularisedTransform, self).__init__()

    def apply(self, state: State, flags, seed=None):
        """Resampling method

        :param state State
            Particle filter state
        :param flags: Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """
        transport_matrix = transport(
            state.flat_particles,  # uses the flat_particles attribute of the state object
            state.log_weights,
            self.epsilon,
            self.scaling,
            self.convergence_threshold,
            self.max_iter,
            state.n_particles,
        )

        return apply_transport_matrix(state, transport_matrix, flags)
