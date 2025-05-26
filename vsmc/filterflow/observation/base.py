import abc

from keras import ops

from vsmc.filterflow.base import Module
from vsmc.filterflow.state import State


class ObservationModelBase(Module, metaclass=abc.ABCMeta):
    def reconstruct(self, sample, masks=None):
        """
        Args:
            sample (Tensor): The sample tensor of shape [..., *state_dims]
            masks (Tensor): The masks tensor - [..., *observation_dims]

        Returns:
            Tensor: The observation tensor of shape [..., *observation_dims]
        """
        # Add n_particles dim
        sample = ops.expand_dims(sample, axis=-self.n_latent_dims - 1)

        # Decode
        images = self.to_observation_domain(sample, masks)

        # Remove n_particles dim
        return ops.squeeze(images, axis=0)

    @abc.abstractmethod
    def to_observation_domain(self, particles, masks=None):
        """Transforms the latent state into an observation"""

    @abc.abstractmethod
    def loglikelihood(self, state: State, observation, inputs=None):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: Tensor
            User/Process given observation
        :param inputs: Tensor
            Control variables (time elapsed, some environment variables, etc). Might be useful if
            the observation model is dependent on some external variables.
        :return: a tensor of loglikelihoods for all particles
        :rtype: Tensor
        """
