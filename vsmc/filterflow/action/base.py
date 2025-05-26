import abc

from vsmc.filterflow.base import Module
from vsmc.filterflow.state import State


class ActionModelBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, state: State, inputs=None, seed=None):
        """
        Sample method for performing an action.

        Args:
            state (State): The current state of the action.
            inputs (Optional): Additional inputs for the action (default: None).
            seed (Optional): Seed for random number generation (default: None).

        Returns:
            The result of the action.
        """

    @abc.abstractmethod
    def apply(self, action, observation):
        """
        Apply the given action to the provided observation.

        Parameters:
            action (object): The action to be applied.
            observation (object): The observation to apply the action to.

        Returns:
            object: The updated observation after applying the action.
        """
