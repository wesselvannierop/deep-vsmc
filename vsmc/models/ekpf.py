import vsmc.runnable

vsmc.runnable.runnable()


import warnings

import numpy as np
import tensorflow as tf
from keras import ops
from zea import tensor_ops

import vsmc.tfp_wrapper as tfp
from vsmc.data.lorenz_data import LorenzPSF
from vsmc.dpf_utils import get_evolution_model, trim_velocity
from vsmc.experiments import setup_experiment
from vsmc.filterflow.proposal import ProposalModelBase
from vsmc.filterflow.state import State
from vsmc.filterflow.transition import TransitionModelBase

tfd = tfp.distributions

LOCATION_STATE_DIMS = 3


def _observation_fn(z):
    # TODO: add LorenzPSF **kwargs
    z = z[..., :LOCATION_STATE_DIMS]
    imgs = LorenzPSF()(z)
    flat_imgs = tensor_ops.flatten(imgs, 1)
    return flat_imgs


def observation_fn(z, scale_diag):
    x = _observation_fn(z)
    return tfd.MultivariateNormalDiag(x, scale_diag=scale_diag)


def observation_jacobian_fn_batch(z):
    with tf.GradientTape() as tape:
        tape.watch(z)
        y = _observation_fn(z)

    j = tape.batch_jacobian(y, z)
    return j


class ProposalModelEKF(ProposalModelBase):
    def __init__(
        self,
        state_scale_diag,
        observation_scale_diag,
        transition_fn,
        transition_jacobian_fn_batch,
        sample_mean=False,
    ):
        super().__init__()
        self.state_scale_diag = state_scale_diag
        self.observation_scale_diag = observation_scale_diag
        self.sample_mean = sample_mean
        self.transition_fn = transition_fn
        self.transition_jacobian_fn_batch = transition_jacobian_fn_batch

        self.is_ekf = True  # Mark this model as an EKF proposal model

    def proposal_dist(self, state, observation):
        particles = state.particles
        particles_cov = state.particles_cov

        # Stack batch dimensions
        particles = ops.reshape(particles, [-1, *particles.shape[2:]])
        particles_cov = ops.reshape(particles_cov, [-1, *particles_cov.shape[2:]])

        initial_state_prior = tfd.MultivariateNormalFullCovariance(
            particles, particles_cov
        )
        # TODO: maybe use: extended_kalman_filter_one_step
        flat_observation = tensor_ops.flatten(observation, 1)
        flat_observation = ops.repeat(
            flat_observation, state.particles.shape[1], axis=0
        )
        results = tfp.experimental.sequential.extended_kalman_filter(
            flat_observation[None],
            initial_state_prior,
            lambda z: self.transition_fn(z, self.state_scale_diag),
            lambda z: observation_fn(z, self.observation_scale_diag),
            self.transition_jacobian_fn_batch,
            observation_jacobian_fn_batch,
        )
        filtered_mean, filtered_cov = results[:2]

        # Unstack batch dimensions
        filtered_means = ops.reshape(filtered_mean, state.particles.shape)
        filtered_covs = ops.reshape(filtered_cov, state.particles_cov.shape)
        return tfd.MultivariateNormalFullCovariance(filtered_means, filtered_covs)

    def propose(self, proposal_dist, state: State, inputs, seed=None):
        if self.sample_mean:
            proposed_particles = proposal_dist.mean()
        else:
            proposed_particles = proposal_dist.sample(seed=seed)
        particles_cov = proposal_dist.covariance()
        return state.evolve(particles=proposed_particles, particles_cov=particles_cov)

    def loglikelihood(self, proposal_dist, proposed_state: State, inputs):
        return proposal_dist.log_prob(proposed_state.particles)


class TransitionModelEKF(TransitionModelBase):
    def __init__(
        self,
        state_scale_diag,
        transition_fn,
        transition_jacobian_fn_batch,
        sample_mean=False,
    ):
        super().__init__()
        self.state_scale_diag = state_scale_diag
        self.sample_mean = sample_mean
        self.transition_fn = transition_fn
        self.transition_jacobian_fn_batch = transition_jacobian_fn_batch

        self.is_ekf = True  # Mark this model as an EKF transition model

    def transition_dist(self, prior_state):
        # taken from tfp -> extended_kalman_filter_one_step
        current_state = prior_state.particles
        current_covariance = prior_state.particles_cov

        # Stack batch dimensions
        current_state = ops.reshape(current_state, [-1, *current_state.shape[2:]])
        current_covariance = ops.reshape(
            current_covariance, [-1, *current_covariance.shape[2:]]
        )

        current_jacobian = self.transition_jacobian_fn_batch(current_state)
        state_prior = self.transition_fn(current_state, self.state_scale_diag)

        predicted_cov = (
            ops.matmul(
                current_jacobian,
                tf.matmul(current_covariance, current_jacobian, transpose_b=True),
            )
            + state_prior.covariance()
        )
        predicted_mean = state_prior.mean()

        # Unstack batch dimensions
        predicted_mean = ops.reshape(predicted_mean, prior_state.particles.shape)
        predicted_cov = ops.reshape(predicted_cov, prior_state.particles_cov.shape)

        dist = tfd.MultivariateNormalFullCovariance(predicted_mean, predicted_cov)
        return dist

    def loglikelihood(self, prior_state, proposed_state, inputs):
        dist = self.transition_dist(prior_state)
        return dist.log_prob(proposed_state.particles)

    def sample(self, state, inputs, seed=None):
        dist = self.transition_dist(state)
        if self.sample_mean:
            proposed_particles = dist.mean()
        else:
            proposed_particles = dist.sample(seed=seed)
        predicted_cov = dist.covariance()
        return state.evolve(particles=proposed_particles, particles_cov=predicted_cov)


def setup_ekpf(config, coord_dims):
    evolution_model_name = config.get("evolution_model")
    evolution_model = get_evolution_model(evolution_model_name)

    def transition_fn(z1, scale_diag):
        z2 = evolution_model(z1)
        return tfd.MultivariateNormalDiag(z2, scale_diag=scale_diag)

    def transition_jacobian_fn_batch(z1):
        with tf.GradientTape() as tape:
            tape.watch(z1)
            z2 = evolution_model(z1)

        j = tape.batch_jacobian(z2, z1)
        return j

    if evolution_model_name != "velocity":
        state_dims = coord_dims
    else:
        state_dims = (2 * np.array(coord_dims)).tolist()

    state_scale_diag = config.transition.sigma * ops.ones(state_dims)
    transition_model = TransitionModelEKF(
        state_scale_diag,
        sample_mean=config.sample_mean,
        transition_fn=transition_fn,
        transition_jacobian_fn_batch=transition_jacobian_fn_batch,
    )

    observation_scale_diag = config.data.awgn_std * ops.ones(
        ops.prod(config.data.image_shape), "float32"
    )
    if config.sample_mean and config.dpf.n_particles > 1:
        warnings.warn("sample_mean only makes sense with one particle")
    proposal_model = ProposalModelEKF(
        state_scale_diag=state_scale_diag,
        observation_scale_diag=observation_scale_diag,
        sample_mean=config.sample_mean,
        transition_fn=transition_fn,
        transition_jacobian_fn_batch=transition_jacobian_fn_batch,
    )

    state2coord = lambda x: trim_velocity(x, coord_dims)

    return proposal_model, transition_model, state2coord, state_dims
