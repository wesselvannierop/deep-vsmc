import abc

from vsmc.filterflow.base import Module
from vsmc.filterflow.state import State


class TransitionModelBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transition_dist(self, particles):
        """Get the transition distribution for the particles given the previous state and inputs.

        Args:
            particles (Tensor): Previous particle filter state.

        Returns:
            tfp.Distribution: The transition distribution.
        """

    @abc.abstractmethod
    def loglikelihood(self, prior_state: State, proposed_state: State, inputs):
        """Computes the loglikelihood of an observation given proposed particles
        :param prior_state: State
            State at t-1
        :param proposed_state: State
            Some proposed State for which we want the likelihood given previous state
        :param inputs: Tensor
            Input for transition model
        :return: a tensor of loglikelihoods for all particles in proposed state
        :rtype: Tensor
        """

    @abc.abstractmethod
    def sample(self, state: State, inputs, seed=None):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: Tensor
            Input for transition model
        :param seed: Tensor
            Input for distribution
        :return: proposed State
        :rtype: State
        """
