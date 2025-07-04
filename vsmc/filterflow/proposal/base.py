import abc

from vsmc.filterflow.base import Module
from vsmc.filterflow.state import State


class ProposalModelBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def proposal_dist(self, state: State, observation):
        """Get the proposal distribution for the particles given the observation.

        Args:
            state (State): Previous particle filter state.
            observation (Tensor): Look ahead observation for adapted particle proposal

        Returns:
            proposal (tfp.Distribution)
        """

    @abc.abstractmethod
    def propose(self, proposal_dist, state: State, inputs, seed=None):
        """Samples from the proposal distribution and updates the state.

        Args:
            proposal_dist (tfp.Distribution): The proposal distribution.
            state (State): Previous particle filter state.
            inputs (Tensor): Control variables (time elapsed, some environment variables, etc).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            State: The proposed state.
        """

    @abc.abstractmethod
    def loglikelihood(self, proposal_dist, proposed_state: State, inputs):
        """Evaluates the log-likelihood of the proposed state.

        Args:
            proposal_dist (tfp.Distribution): The proposal distribution.
            proposed_state (State): Proposed state.
            inputs (Tensor): Control variables (time elapsed, some environment variables, etc).

        Returns:
            Tensor: The log-likelihood of the proposed state.
        """
