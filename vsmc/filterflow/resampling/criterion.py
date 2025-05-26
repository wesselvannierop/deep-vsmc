import abc

import keras
from keras import ops

from vsmc.filterflow.base import Module
from vsmc.filterflow.state import State


class ResamplingCriterionBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, state: State):
        """Flags which batches should be resampled

        :param state: State
            current state
        :return: mask of booleans
        :rtype Tensor
        """


def neff(tensor, assume_normalized: bool, is_log: bool, threshold: float):
    if is_log:
        if assume_normalized:
            log_neff = -ops.logsumexp(2 * tensor, 1)
        else:
            log_neff = 2 * ops.logsumexp(tensor, 1) - ops.logsumexp(2 * tensor, 1)
        flag = log_neff <= ops.log(threshold)
        return flag, ops.exp(log_neff)
    else:
        if assume_normalized:
            neff = 1 / ops.sum(tensor**2, 1)
        else:
            neff = ops.sum(tensor, 1) ** 2 / ops.sum(tensor**2, 1)
        flag = neff <= threshold

        return flag, neff


@keras.saving.register_keras_serializable()
class NeffCriterion(ResamplingCriterionBase):
    """
    Standard Neff criterion for resampling. If the neff of the state tensor falls below a certain threshold
    (either in relative or absolute terms) then the state will be flagged as needing resampling
    """

    def __init__(self, threshold, is_relative, on_log=True, assume_normalized=True):
        super(NeffCriterion, self).__init__()
        self._threshold = threshold
        self._is_relative = is_relative
        self._on_log = on_log
        self._assume_normalized = assume_normalized

    def get_config(self):
        return {
            "threshold": self._threshold,
            "is_relative": self._is_relative,
            "on_log": self._on_log,
            "assume_normalized": self._assume_normalized,
        }

    def apply(self, state: State):
        """Flags which batches should be resampled

        :param state: State
            current state
        :return: mask of booleans, efficient sample size prior resampling
        :rtype Tensor
        """
        threshold = (
            self._threshold
            if not self._is_relative
            else state.n_particles_float * self._threshold
        )
        if self._on_log:
            return neff(
                state.log_weights, self._assume_normalized, self._on_log, threshold
            )
        else:
            return neff(state.weights, self._assume_normalized, self._on_log, threshold)


class AlwaysResample(ResamplingCriterionBase):

    def apply(self, state: State):
        return ops.ones(state.batch_size, "bool"), ops.zeros(
            state.batch_size, "float32"
        )


class NeverResample(ResamplingCriterionBase):

    def apply(self, state: State):
        return ops.zeros(state.batch_size, "bool"), ops.zeros(
            state.batch_size, dtype="float32"
        )
