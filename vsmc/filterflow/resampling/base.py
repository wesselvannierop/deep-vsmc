import abc

from keras import ops

from vsmc.filterflow.base import Module
from vsmc.filterflow.state import State


def resample(tensor, new_tensor, flags):
    shape = [-1] + [1] * (ops.ndim(tensor) - 1)
    return ops.where(ops.reshape(flags, shape), new_tensor, tensor)


class ResamplerBase(Module, metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""

    DIFFERENTIABLE = False

    @abc.abstractmethod
    def apply(self, state: State, flags, seed=None):
        """Resampling method

        :param state: State
            Particle filter state
        :param flags: Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """


class NoResampling(ResamplerBase):
    DIFFERENTIABLE = True

    def apply(self, state: State, flags, seed=None):
        """Resampling method

        :param state: State
            Particle filter state
        :param flags: Tensor
            Flags for resampling
        :param seed: Tensor
            seed for resampling (if needed)
        :return: resampled state
        :rtype: State
        """
        return state
