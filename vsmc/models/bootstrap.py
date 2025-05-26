import warnings

import keras

import vsmc.tfp_wrapper as tfp
from vsmc.dpf_utils import GaussianTransitionModel, trim_velocity
from vsmc.filterflow.proposal import BootstrapProposalModel

tfd = tfp.distributions


def setup_bootstrap(config, coord_dims):
    evolution_model = config.get("evolution_model")

    if evolution_model == "velocity":
        state_dims = coord_dims * 2
        state2coord = lambda x: trim_velocity(x, coord_dims)
    else:
        warnings.warn("BPF: Not modeling velocity in the state")
        state_dims = coord_dims
        state2coord = keras.layers.Identity()

    # Set transition/proposal model
    transition_model = GaussianTransitionModel(
        state_dims=state_dims,
        evolution_model=config.get("evolution_model"),
        **config.get("transition", {}),
    )
    proposal_model = BootstrapProposalModel(transition_model)

    return proposal_model, transition_model, state2coord, state_dims
