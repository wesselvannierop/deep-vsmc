import keras
from keras import ops

from vsmc.filterflow.proposal.base import ProposalModelBase
from vsmc.filterflow.transition.base import TransitionModelBase
from vsmc.models.preset_loader import from_preset


# TODO: copy paste from vsmc/baselines/lorenz_encoder/train_encoder.py
@keras.saving.register_keras_serializable()
def l2_loss(y_true, y_pred):
    return ops.mean(ops.linalg.norm(y_true - y_pred, axis=-1))


class EncoderProposalModel(ProposalModelBase):
    def __init__(self, model):
        self.model = model

    def proposal_dist(self, particles, observation):
        return self.model(observation)  # just return the proposed particles

    def loglikelihood(self, proposal_dist, proposed_state, inputs):
        return proposed_state.weights  # dummy output

    def propose(self, proposed_particles, state, inputs, seed=None):
        assert state.n_particles == 1, "Only one particle supported"
        proposed_particles = proposed_particles[:, None]  # add n_particles dimension
        return state.evolve(particles=proposed_particles)


class DummyTransitionModel(TransitionModelBase):
    def sample(self, state, inputs, seed=None):
        return state

    def loglikelihood(self, state, proposed_state, inputs):
        return state.weights  # dummy output


def setup_lorenz_encoder(config, coord_dims):
    model = keras.models.load_model(from_preset(config.checkpoint), compile=True)
    proposal_model = EncoderProposalModel(model)
    transition_model = DummyTransitionModel()
    state2coord = keras.layers.Identity()
    state_dims = coord_dims
    # NOTE: test_transition_model = False
    # NOTE: n_particles = 1
    return proposal_model, transition_model, state2coord, state_dims
