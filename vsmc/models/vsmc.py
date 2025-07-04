import warnings
from math import prod

import keras
from keras import Sequential, layers, ops

import vsmc.ops as dpf_ops
import vsmc.tf_prob  # pylint: disable=unused-import
import vsmc.tfp_wrapper as tfp
from vsmc.dpf_utils import (
    GaussianTransitionModel,
    get_input_size,
    trim_velocity,
    velocity_transition_fn,
)
from vsmc.filterflow.proposal import ProposalModelBase
from vsmc.filterflow.state import State
from vsmc.filterflow.transition import TransitionModelBase
from vsmc.models.helpers import build_image_encoder, build_simple_dense
from vsmc.prob import GaussianMixture

tfd = tfp.distributions


def wrap_equally(equally_weighted_mixture, mixture):
    if mixture == 1 and not equally_weighted_mixture:
        warnings.warn("Mixture set to 1, forcing equally_weighted_mixture=True.")
    if mixture == 1:
        return True
    return equally_weighted_mixture


def gaussian_from_nn(
    nn_output,
    state_dims: list = (2,),
    clip=None,
    equally_weighted_mixture=True,
    mixture=1,
    axis=-1,  # axis to split the output
):
    # Assertions
    if mixture == 1:
        assert equally_weighted_mixture, (
            "GMM with 1 component is obviously equally weighted."
        )

    state_dim = state_dims[axis]
    n_state_dims = len(state_dims)

    # Split NN output
    if equally_weighted_mixture:
        # https://ieeexplore.ieee.org/document/10447783
        mu, logvar = ops.split(nn_output, 2, axis=axis)
        mixture_logits = ops.ones(mixture) / mixture
    else:
        mu, logvar, mixture_logits = dpf_ops.split_into_sizes(
            nn_output,
            [state_dim * mixture, state_dim * mixture, mixture],
            axis=axis,
        )

    # Split mixture and state dimensions
    if mixture > 1:
        mu = ops.reshape(mu, [*ops.shape(mu)[:-n_state_dims], mixture, *state_dims])
        logvar = ops.reshape(
            logvar, [*ops.shape(logvar)[:-n_state_dims], mixture, *state_dims]
        )
    else:
        # If mixture is 1, we set the mixture logits to None
        mixture_logits = None

    # Clip mu (using prior information)
    if clip is not None:
        mu = ops.clip(mu, *clip)

    # Clip logvar and sigma for numerical stability
    logvar = ops.clip(logvar, -20.0, 20.0)
    sigma = ops.exp(0.5 * logvar)
    sigma = ops.clip(sigma, 1e-3, 1e3)

    return mu, sigma, mixture_logits


@keras.saving.register_keras_serializable()
class ProposalModel(layers.Layer, ProposalModelBase):
    def __init__(
        self,
        input_size: tuple = (28, 28),  # 2d means image (and uses conv2d encoder)
        mixture: int = 1,
        state_dims: int = 2,
        clip=None,
        equally_weighted_mixture=True,
        encoded_dim=256,
        encoder_depth=3,
        norm_layer=None,
        proposal_depth=6,
        proposal_hidden_dim=256,
        model_velocity=False,
        verbose=False,
        nfeatbase=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mixture = mixture
        self.equally_weighted_mixture = wrap_equally(equally_weighted_mixture, mixture)
        self.clip = clip
        self.model_velocity = model_velocity

        self.input_size = input_size
        self.encoder_depth = encoder_depth
        self.norm_layer = norm_layer
        self.encoded_dim = encoded_dim
        self.state_dims = state_dims
        self.proposal_depth = proposal_depth
        self.proposal_hidden_dim = proposal_hidden_dim
        self.nfeatbase = nfeatbase

        assert isinstance(self.state_dims, int), (
            "state_dims must be an integer for ProposalModel"
        )

        self.output_dim = 2 * state_dims * mixture
        if not equally_weighted_mixture:
            self.output_dim += mixture

        networks = self.init_networks()
        if verbose:
            for network in networks:
                network.summary()

    @classmethod
    def from_config(cls, config):
        # Backwards compatibility
        if not "input_size" in config:
            config["input_size"] = (config["img_size"], config["img_size"])
            del config["img_size"]
        if "prior" in config:
            assert config["prior"] is None, "Legacy checkpoint..."
            del config["prior"]
        if "rnn" in config:
            del config["rnn"]
        config["verbose"] = False
        return cls(**config)

    def build_proposal(self, nn_output, *args, **kwargs):
        return GaussianMixture(
            *gaussian_from_nn(
                nn_output,
                state_dims=(self.state_dims,),
                axis=-1,
                clip=self.clip,
                mixture=self.mixture,
                equally_weighted_mixture=self.equally_weighted_mixture,
                *args,
                **kwargs,
            )
        )

    def init_networks(self):
        # Build encoder
        if len(self.input_size) == 2:
            self.encoder = build_image_encoder(
                self.input_size,
                encoder_depth=self.encoder_depth,
                norm_layer=self.norm_layer,
                encoded_dim=self.encoded_dim,
                name="proposal_encoder_2d",
                nfeatbase=self.nfeatbase,
            )
        elif len(self.input_size) == 1:
            self.encoder = build_simple_dense(
                self.input_size,
                output_dim=self.encoded_dim,
                depth=self.encoder_depth,
                name="proposal_encoder_1d",
            )
        else:
            raise ValueError("Input size must be 1d or 2d")

        # Build proposal model
        self.nn = build_simple_dense(
            (None, self.state_dims + self.encoded_dim),
            output_dim=self.output_dim,
            hidden_dim=self.proposal_hidden_dim,
            depth=self.proposal_depth,
            name="proposal_nn",
        )

        # self.encoder.build((None, *self.input_size, 1))
        # self.nn.build((None, None, self.state_dims + self.encoded_dim))
        return self.encoder, self.nn

    def proposal_dist(self, particles, observation):
        return self(particles, observation)

    def call(self, particles, observation):
        batch_size, n_particles, _ = ops.shape(particles)

        encoded_observation = self.encoder(observation)
        latent_dim = ops.shape(encoded_observation)[-1]

        # Broadcast to n_particles
        encoded_observation = ops.broadcast_to(
            encoded_observation[:, None, :], (batch_size, n_particles, latent_dim)
        )

        # Evolution model
        if self.model_velocity:
            particles = velocity_transition_fn(particles)

        # Run neural proposal
        combined = ops.concatenate([particles, encoded_observation], axis=-1)
        nn_output = self.nn(combined)
        gmm = self.build_proposal(nn_output)

        return gmm

    def propose(self, proposal_dist, state: State, inputs, seed=None):
        proposed_particles = proposal_dist.sample(seed=seed)
        return state.evolve(particles=proposed_particles)

    def loglikelihood(self, proposal_dist, proposed_state: State, inputs):
        log_prob = proposal_dist.log_prob(proposed_state.particles)
        return log_prob


@keras.saving.register_keras_serializable()
class TransitionModel(layers.Layer, TransitionModelBase):
    def __init__(
        self,
        mixture,
        state_dims: int,
        clip=None,
        equally_weighted_mixture=True,
        depth=0,
        model_velocity=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        output_dim = 2 * state_dims * mixture
        if not equally_weighted_mixture:
            output_dim += mixture
        self.output_dim = output_dim

        self.mixture = mixture
        self.state_dims = state_dims
        self.depth = depth
        self.clip = clip
        self.equally_weighted_mixture = wrap_equally(equally_weighted_mixture, mixture)
        self.model_velocity = model_velocity

        assert isinstance(self.state_dims, int), (
            "state_dims must be an integer for TransitionModel"
        )

        self.init_networks()
        if verbose:
            self.nn.summary()

    @classmethod
    def from_config(cls, config):
        # Backwards compatibility
        if "prior" in config:
            assert config["prior"] is None, "Legacy checkpoint..."
            del config["prior"]
        config["verbose"] = False
        return cls(**config)

    def gaussian_from_nn(self, nn_output, *args, **kwargs):
        return GaussianMixture(
            *gaussian_from_nn(
                nn_output,
                state_dims=(self.state_dims,),
                axis=-1,
                clip=self.clip,
                mixture=self.mixture,
                equally_weighted_mixture=self.equally_weighted_mixture,
                *args,
                **kwargs,
            )
        )

    def init_networks(self):
        nn = [
            layers.InputLayer((None, self.state_dims)),
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="relu"),
            *[layers.Dense(512, activation="relu") for _ in range(self.depth)],
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.output_dim),
        ]
        self.nn = Sequential(nn, name="transition_model")

    def transition_dist(self, state: State):
        return self(state.particles)

    def call(self, particles):
        if self.model_velocity:
            particles = velocity_transition_fn(particles)
        return self.gaussian_from_nn(self.nn(particles))

    def loglikelihood(self, prior_state, proposed_state, inputs):
        dist = self.transition_dist(prior_state)
        log_prob = dist.log_prob(proposed_state.particles)
        return log_prob

    def sample(self, state, inputs, seed=None):
        dist = self.transition_dist(state)
        proposed_particles = dist.sample(seed=seed)
        return state.evolve(particles=proposed_particles)


def setup_vsmc(config, coord_dims, verbose=True):
    assert hasattr(config, "training"), "VSMC needs training config"

    evolution_model = config.get("evolution_model")

    model_velocity = evolution_model == "velocity"

    if model_velocity:
        state_dims = coord_dims * 2

        @keras.saving.register_keras_serializable()
        def trim_to_coord_dims(x):
            return trim_velocity(x, coord_dims)

        state2coord = trim_to_coord_dims
    else:
        state_dims = coord_dims
        state2coord = layers.Identity()

    # Set transition model
    transition_type = config.transition.pop("type")
    if transition_type == "gaussian":
        transition_model = GaussianTransitionModel(
            state_dims=prod(state_dims),
            evolution_model=evolution_model,
            **config.transition.get("kwargs", {}),
        )
    elif transition_type == "dense":
        transition_model = TransitionModel(
            state_dims=prod(state_dims),
            model_velocity=model_velocity,
            verbose=verbose,
            **config.transition.get("kwargs", {}),
        )
    else:
        raise ValueError(f"Unknown transition model: {transition_type}")

    if not transition_type == "gaussian" and evolution_model is not None:
        warnings.warn("Ignoring evolution model as we use a learned transition.")

    # Set proposal model
    proposal_type = config.proposal.pop("type")
    if proposal_type == "dense":
        proposal_model = ProposalModel(
            input_size=get_input_size(config),
            state_dims=prod(state_dims),
            model_velocity=model_velocity,
            verbose=verbose,
            **config.proposal.get("kwargs", {}),
        )
    else:
        raise ValueError(f"Unknown proposal model: {proposal_type}")

    return proposal_model, transition_model, state2coord, state_dims
