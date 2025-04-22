import warnings

import keras
import numpy as np
from keras import Sequential, layers, ops

import dpf.ops as dpf_ops
import dpf.tf_prob  # pylint: disable=unused-import
import tfp_wrapper as tfp
from dpf.data.lorenz_data import LorenzPSF
from dpf.dpf_utils import (
    GaussianTransitionModel,
    get_input_size,
    trim_velocity,
    velocity_transition_fn,
)
from dpf.filterflow.action import ActionModelBase
from dpf.filterflow.proposal import ProposalModelBase
from dpf.filterflow.state import State
from dpf.filterflow.transition import TransitionModelBase
from dpf.models import get_image_encoder_model, get_proposal_model
from dpf.models.helpers import build_image_encoder, build_simple_dense
from dpf.prob import GaussianMixture
from usbmd import tensor_ops

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
        assert (
            equally_weighted_mixture
        ), "GMM with 1 component is obviously equally weighted."

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
class ConvProposalModel(layers.Layer, ProposalModelBase):
    def __init__(
        self,
        input_size: tuple = (256, 256),
        clip=None,
        mixture: int = 1,
        latent_channels: int = 4,
        equally_weighted_mixture=True,
        verbose=False,
        diffusion: dict | None = None,
        nfeatbase=32,
        nfeat_proposal=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.clip = clip
        self.mixture = mixture
        self.latent_channels = latent_channels
        self.equally_weighted_mixture = wrap_equally(equally_weighted_mixture, mixture)
        self.verbose = verbose
        self.diffusion_dict = diffusion
        self.nfeatbase = nfeatbase
        self.nfeat_proposal = nfeat_proposal

        self.observation_encoder = get_image_encoder_model(
            nfeatbase=self.nfeatbase, out_channels=latent_channels
        )
        if verbose:
            self.observation_encoder.summary()
        self.state_dims = (*np.array(input_size) // 8, 4)

        in_channels = self.latent_channels * 2
        out_channels = 2 * self.latent_channels * self.mixture
        if not self.equally_weighted_mixture:
            out_channels += self.mixture

        self.proposal = get_proposal_model(
            in_channels=in_channels,
            out_channels=out_channels,
            nfeat=self.nfeat_proposal,
        )
        if verbose:
            self.proposal.summary()

        if diffusion is not None:
            self.diffusion = DiffusionReconstructor(**diffusion)

    @classmethod
    def from_config(cls, config):
        config["verbose"] = False
        return cls(**config)

    def get_config(self):
        return {
            "input_size": self.input_size,
            "clip": self.clip,
            "mixture": self.mixture,
            "latent_channels": self.latent_channels,
            "equally_weighted_mixture": self.equally_weighted_mixture,
            "diffusion": self.diffusion_dict,
            "nfeatbase": self.nfeatbase,
            "nfeat_proposal": self.nfeat_proposal,
        }

    def build_proposal(self, nn_output, *args, **kwargs):
        return GaussianMixture(
            *gaussian_from_nn(
                nn_output,
                state_dims=self.state_dims,
                clip=self.clip,
                mixture=self.mixture,
                equally_weighted_mixture=self.equally_weighted_mixture,
                *args,
                **kwargs,
            ),
            reinterpreted_batch_ndims=2,  # TODO: hardcoded 2
        )

    def proposal_dist(self, particles, observation, inputs, seed):
        return self(particles, observation, inputs, seed)

    def call(self, particles, observation, masks, seed):
        # batch_size, n_particles, *latent_dims = ops.shape(particles)
        encoded_observation = self.observation_encoder(observation)

        # Broadcast to n_particles
        encoded_observation = ops.broadcast_to(
            encoded_observation[:, None], ops.shape(particles)
        )

        # Run neural proposal
        combined = ops.concatenate([particles, encoded_observation], axis=-1)
        nn_output = tensor_ops.func_with_one_batch_dim(
            self.proposal, combined, n_batch_dims=2
        )
        gmm = self.build_proposal(nn_output)

        if hasattr(self, "diffusion"):
            measurements = ops.repeat(
                observation[:, None], ops.shape(particles)[1], axis=1
            )
            masks = ops.repeat(masks[:, None], ops.shape(particles)[1], axis=1)
            initial_reconstruction = gmm.loc
            gmm.loc = self.diffusion(
                measurements, initial_reconstruction, masks, seed=seed
            )

        return gmm

    def propose(self, proposal_dist, state: State, inputs, seed=None):
        proposed_particles = proposal_dist.sample(seed=seed)
        return state.evolve(particles=proposed_particles)

    def loglikelihood(self, proposal_dist, proposed_state: State, inputs):
        log_prob = proposal_dist.log_prob(proposed_state.particles)
        return log_prob


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
        diffusion: dict | None = None,
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

        assert isinstance(
            self.state_dims, int
        ), "state_dims must be an integer for ProposalModel"

        self.output_dim = 2 * state_dims * mixture
        if not equally_weighted_mixture:
            self.output_dim += mixture

        networks = self.init_networks()
        if diffusion is not None:
            self.diffusion = DiffusionModel(**diffusion)
        else:
            self.diffusion = None
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

        if self.diffusion is not None:
            gmm = self.diffusion(gmm)

        return gmm

    def propose(self, proposal_dist, state: State, inputs, seed=None):
        proposed_particles = proposal_dist.sample(seed=seed)
        return state.evolve(particles=proposed_particles)

    def loglikelihood(self, proposal_dist, proposed_state: State, inputs):
        log_prob = proposal_dist.log_prob(proposed_state.particles)
        return log_prob


@keras.saving.register_keras_serializable()
class ConvTransitionModel(layers.Layer, TransitionModelBase):
    def __init__(
        self,
        state_dims: tuple = (32, 32, 4),
        mixture: int = 1,
        equally_weighted_mixture=True,
        nfeatbase=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_dims = state_dims
        self.mixture = mixture
        self.equally_weighted_mixture = wrap_equally(equally_weighted_mixture, mixture)
        self.nfeatbase = nfeatbase
        self.clip = None  # TODO

        out_channels = 2 * state_dims[-1] * self.mixture
        if not self.equally_weighted_mixture:
            out_channels += self.mixture

        transition = get_proposal_model(
            in_channels=4,
            out_channels=out_channels,
            nfeat=self.nfeatbase,
            name="transition",
            return_layers=True,
        )
        transition = transition[1:]
        transition = [layers.InputLayer(state_dims)] + transition
        self.transition = Sequential(transition, name="transition")

    @classmethod
    def from_config(cls, config):
        if not "nfeatbase" in config:
            config["nfeatbase"] = config.pop("nfeat")  # Backwards compatibility
        return cls(**config)

    def get_config(self):
        return {
            "state_dims": self.state_dims,
            "mixture": self.mixture,
            "equally_weighted_mixture": self.equally_weighted_mixture,
            "nfeatbase": self.nfeatbase,
        }

    def build_transition(self, nn_output, *args, **kwargs):
        return GaussianMixture(
            *gaussian_from_nn(
                nn_output,
                state_dims=self.state_dims,
                clip=self.clip,
                mixture=self.mixture,
                equally_weighted_mixture=self.equally_weighted_mixture,
                *args,
                **kwargs,
            ),
            reinterpreted_batch_ndims=2,  # TODO: hardcoded 2
        )

    def transition_dist(self, state: State):
        return self(state.particles)

    def call(self, particles):
        nn_output = tensor_ops.func_with_one_batch_dim(
            self.transition, particles, n_batch_dims=2
        )
        return self.build_transition(nn_output)

    def loglikelihood(self, prior_state, proposed_state, inputs):
        dist = self.transition_dist(prior_state)
        log_prob = dist.log_prob(proposed_state.particles)
        return log_prob

    def sample(self, state, inputs, seed=None):
        dist = self.transition_dist(state)
        proposed_particles = dist.sample(seed=seed)
        return state.evolve(particles=proposed_particles)


@keras.saving.register_keras_serializable()
class TransitionModel(layers.Layer, TransitionModelBase):
    def __init__(
        self,
        mixture: int = 1,
        state_dims: int = 2,
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

        assert isinstance(
            self.state_dims, int
        ), "state_dims must be an integer for TransitionModel"

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
            state_dims=state_dims,
            evolution_model=evolution_model,
            **config.transition.get("kwargs", {}),
        )
    elif transition_type == "dense":
        transition_model = TransitionModel(
            state_dims=state_dims,
            model_velocity=model_velocity,
            verbose=verbose,
            **config.transition.get("kwargs", {}),
        )
    elif transition_type == "conv":
        transition_model = ConvTransitionModel(
            state_dims=state_dims,
            **config.transition.get("kwargs", {}),
        )
        transition_model.transition.summary()
    else:
        raise ValueError(f"Unknown transition model: {transition_type}")

    if not transition_type == "gaussian" and evolution_model is not None:
        warnings.warn("Ignoring evolution model as we use a learned transition.")

    # Set proposal model
    proposal_type = config.proposal.pop("type")
    if proposal_type == "dense":
        proposal_model = ProposalModel(
            input_size=get_input_size(config),
            state_dims=state_dims,
            model_velocity=model_velocity,
            verbose=verbose,
            **config.proposal.get("kwargs", {}),
        )
    elif proposal_type == "conv":
        proposal_model = ConvProposalModel(
            input_size=get_input_size(config),
            latent_channels=4,  # TODO: hardcoded
            verbose=verbose,
            **config.proposal.get("kwargs", {}),
        )
    else:
        raise ValueError(f"Unknown proposal model: {proposal_type}")

    return proposal_model, transition_model, state2coord, state_dims
