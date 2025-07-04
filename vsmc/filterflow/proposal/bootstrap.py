from vsmc.filterflow.proposal.base import ProposalModelBase
from vsmc.filterflow.state import State


class BootstrapProposalModel(ProposalModelBase):
    """Standard bootstrap proposal: directly uses the transition model as a proposal."""

    def __init__(self, transition_model):
        super(BootstrapProposalModel, self).__init__()
        self._transition_model = transition_model

    def proposal_dist(self, state, observation):
        return self._transition_model.transition_dist(state)

    def propose(self, proposal_dist, state: State, inputs, seed=None):
        """See base class"""
        proposed_particles = proposal_dist.sample(seed=seed)
        return state.evolve(particles=proposed_particles)

    def loglikelihood(self, proposal_dist, proposed_state: State, inputs):
        log_prob = proposal_dist.log_prob(proposed_state.particles)
        return log_prob
