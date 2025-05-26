"""
The Sequential Monte Carlo module.

NOTE: the Filter class should NOT be modified after initialization!
See: https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static

TODO: only grayscale images (only h, w) are supported for now.
"""

import time
import warnings

import keras
from keras import ops

import vsmc.tfp_wrapper as tfp
from usbmd.backend import jit
from vsmc.prob import Distribution

from .action.base import ActionModelBase
from .base import Module
from .observation.base import ObservationModelBase
from .proposal.base import ProposalModelBase
from .resampling.base import ResamplerBase
from .resampling.criterion import ResamplingCriterionBase
from .state import State, StateSeries, write_state_array
from .transition.base import TransitionModelBase
from .utils import normalize


class Filter(Module):
    def __init__(
        self,
        transition_model,
        proposal_model,
        store_proposal=False,
        store_all=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transition_model = transition_model
        self._proposal_model = proposal_model
        self.store_proposal = store_proposal
        self.store_all = store_all
        self.body = self._body

    def update(self, state: State, observation, inputs, seed=None):
        """
        :param state: State
            current state of the filter
        :param observation: Tensor
            observation to compare the state against
        :param inputs: Tensor
            inputs for the observation_model
        :return: Updated weights
        """
        raise NotImplementedError

    def predict(self, state: State, inputs, seed=None):
        """Predict step of the filter

        :param state: State
            prior state of the filter
        :param inputs: Tensor
            Inputs used for prediction
        :return: Predicted State
        :rtype: State
        """
        return self._transition_model.sample(state, inputs, seed=seed)

    def _body(self, i, loop_state):
        state, proposal_dists, state_array, observation_series, input_series, seed = (
            loop_state
        )
        observation = observation_series[i]
        inputs = input_series[i] if input_series is not None else None
        seed, seed2 = tfp.random.split_seed(seed, n=2, salt=f"_return_body_{i}")
        state, proposal_dists = self.update(
            state, observation, inputs, proposal_dists, seed2
        )
        if self.store_all:
            # NOTE: for i=0 will overwrite initial state
            state_array = write_state_array(state, None, state_array, i)
        return (
            state,
            proposal_dists,
            state_array,
            observation_series,
            input_series,
            seed,
        )

    def jit(self, state_dims, image_shape, n_particles, *args, **kwargs):
        print("[filterflow] SMC compiling!")
        if keras.backend.backend() == "jax":
            self.body = jit(self._body, *args, **kwargs)
        elif keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            # TensorArray must be constructed in tf.function
            if not isinstance(state_dims, (list, tuple)):
                state_dims = [state_dims]
            self.call = jit(
                self.call,
                *args,
                **kwargs,
                input_signature=[
                    tf.TensorSpec(
                        shape=[None, n_particles, *state_dims], dtype=tf.float32
                    ),
                    tf.TensorSpec(shape=[None, None, *image_shape], dtype=tf.float32),
                    tf.TensorSpec(shape=[], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, None, *image_shape], dtype=tf.bool),
                    tf.TensorSpec(shape=[2], dtype=tf.int32),
                ],
                jit_compile=False,
            )

    def call(
        self,
        initial_particles,
        observation_series,
        n_observations: int = None,
        input_series=None,
        seed=None,
    ):
        if keras.backend.backend() == "tensorflow":
            print("[filterflow] Tracing!")  # An eager-only side effect.

        batch_size = ops.shape(observation_series)[1]

        if self._action_model is not None:
            h, w = self._action_model.img_height, self._action_model.img_width
            action = ops.zeros([batch_size, h, w])
        else:
            action = None

        initial_state = State(initial_particles, action=action)

        # init series
        state_array = write_state_array(initial_state, n_observations)

        state, proposal_dists, state_array, _, _, _ = ops.fori_loop(
            0,
            n_observations,
            self.body,
            (initial_state, [], state_array, observation_series, input_series, seed),
        )

        if self.store_all:
            return state, proposal_dists, StateSeries(state_array)
        else:
            return state, proposal_dists, None

    def __call__(
        self,
        observation_series,
        initial_particles,
        n_observations: int = None,
        input_series=None,
        seed=None,
        verbose=0,
    ):
        """
        Args:
            observation_series (Tensor): Observations to compare against.
                shape: [n_observations, batch_size, ...]
            initial_particles: Initial particles of the filter
            n_observations (int): Number of observations.
            input_series (Tensor): Inputs for the observation model.
            seed (int): Seed for the random number generator.
            verbose (int): Verbosity level.
        """
        if seed is None and keras.backend.backend() == "jax":
            import jax

            seed = jax.random.key(0)

        if input_series is None and keras.backend.backend() == "tensorflow":
            # TODO: this is a hack to make it work with tf.function
            input_series = ops.zeros_like(observation_series, dtype="bool")

        if n_observations is None:
            n_observations = ops.shape(observation_series)[0]

        # sanitize_seed maps any seed flavor to a "stateless-compatible" seed
        seed = tfp.random.sanitize_seed(seed, salt="_return")

        if verbose > 0:
            print("Running filter with n_observations:", n_observations)
            print("This may take a while...")
            start_time = time.perf_counter()

        out = self.call(
            initial_particles, observation_series, n_observations, input_series, seed
        )

        if verbose > 0:
            end_time = time.perf_counter()
            step_time = (end_time - start_time) * 1e3 / n_observations
            print(f"Filter finished ran with {step_time:.4f}ms / step")

        return out


class SMC(Filter):
    def __init__(
        self,
        observation_model: ObservationModelBase,
        transition_model: TransitionModelBase,
        proposal_model: ProposalModelBase,
        resampling_criterion: ResamplingCriterionBase,
        resampling_method: ResamplerBase,
        action_model: ActionModelBase = None,
        compute_entropy=False,
        enhance_gradient_steps=1,
        prior: Distribution = None,
        **kwargs,
    ):
        super().__init__(
            transition_model=transition_model, proposal_model=proposal_model, **kwargs
        )
        self._observation_model = observation_model
        self._resampling_criterion = resampling_criterion
        self._resampling_method = resampling_method
        self._compute_entropy = compute_entropy
        self._action_model = action_model
        self.enhance_gradient_steps = enhance_gradient_steps
        self.prior = prior

        if action_model is not None:
            warnings.warn("Input series is overridden by action model.")

    def update(self, state: State, observation, inputs, proposal_dists, seed):
        """
        :param state: State
            current state of the filter
        :param observation: Tensor
            observation to compare the state against
        :param inputs: Tensor
            inputs for the observation_model
        :return: Updated weights
        """
        t = state.t
        float_t = ops.cast(t, "float32")
        float_t_1 = float_t + 1.0
        seed1, seed2, seed3 = tfp.random.split_seed(seed, n=3, salt="update")
        resampling_flag, ess = self._resampling_criterion.apply(state)
        # update running average efficient sample size
        state = state.evolve(ess=ess / float_t_1 + state.ess * (float_t / float_t_1))
        # perform resampling
        resampled_state = self._resampling_method.apply(state, resampling_flag, seed1)
        # perform sequential IS step
        new_state, proposal_dists = self.propose_and_weight(
            resampled_state, observation, inputs, proposal_dists, seed2
        )
        new_state = self._resampling_correction_term(
            resampling_flag, new_state, state, observation, inputs, seed3
        )
        # increment t
        return new_state.evolve(t=t + 1), proposal_dists

    def _resampling_correction_term(
        self,
        resampling_flag,
        new_state: State,
        prior_state: State,
        observation,
        inputs,
        seed=None,
    ):
        b, n = prior_state.batch_size, prior_state.n_particles
        uniform_log_weights = ops.zeros([b, n]) - ops.log(ops.cast(n, "float32"))
        baseline_state, _ = self.propose_and_weight(
            state=prior_state.evolve(
                log_weights=uniform_log_weights,
                weights=ops.exp(uniform_log_weights),
            ),
            observation=observation,
            inputs=inputs,
            seed=seed,
        )
        float_flag = ops.cast(resampling_flag, "float32")
        centered_reward = ops.reshape(
            float_flag * (new_state.log_likelihoods - baseline_state.log_likelihoods),
            [-1, 1],
        )
        resampling_correction = prior_state.resampling_correction + ops.mean(
            ops.stop_gradient(centered_reward) * prior_state.log_weights, 1
        )
        return new_state.evolve(resampling_correction=resampling_correction)

    def entropy(
        self,
        transition_fn,
        proposed_particles,
        observation_log_likelihoods,
        weights,
        prev_log_weights,
    ):
        """
        Source: https://ieeexplore.ieee.org/document/5712013
        """
        entropy = ops.logsumexp(observation_log_likelihoods + prev_log_weights, axis=1)

        something = []
        n_particles = proposed_particles.shape[1]
        for j in range(n_particles):
            particle = State(
                ops.repeat(
                    proposed_particles[:, j : j + 1],
                    n_particles,
                    axis=1,
                )
            )
            something.append(transition_fn(particle) + prev_log_weights[:, j])
        something = ops.stack(something, axis=-1)
        something = ops.logsumexp(something, axis=-1)

        entropy -= ops.sum(
            (observation_log_likelihoods + something) * weights,
            axis=1,
        )
        return entropy

    def propose_and_weight(
        self, state: State, observation, inputs, proposal_dists=None, seed=None
    ):
        """
        :param state: State
            current state of the filter
        :param observation: Tensor
            observation to compare the state against
        :param inputs: Tensor
            inputs for the observation_model
        :return: Updated weights
        """
        log_likelihood_increment = ops.zeros_like(state.log_likelihoods)

        seed1, seed2, seed3, seed4 = tfp.random.split_seed(
            seed, n=4, salt="propose_and_weight"
        )

        # Action model can be used to change the observations in an online manner
        if self._action_model is not None:
            pred_state = self._transition_model.sample(state, inputs, seed1)
            action = self._action_model.sample(pred_state, inputs, seed=seed2)
            observation = self._action_model.apply(action, observation)
            # TODO: add a flag to decide whether to use action as input
            inputs = action
        else:
            action = None

        for i in range(self.enhance_gradient_steps):
            proposal_dist = self._proposal_model.proposal_dist(
                state.particles, observation, inputs, seed4
            )
            if self.store_proposal and proposal_dists is not None:
                proposal_dists.append(proposal_dist)
            proposed_state = self._proposal_model.propose(
                proposal_dist, state, inputs, seed3
            )
            observation_log_likelihoods = self._observation_model.loglikelihood(
                proposed_state, observation, inputs
            )

            log_weights = self._transition_model.loglikelihood(
                state, proposed_state, inputs
            )
            log_weights += observation_log_likelihoods
            log_weights -= self._proposal_model.loglikelihood(
                proposal_dist, proposed_state, inputs
            )
            log_weights += state.log_weights
            if self.prior is not None:
                log_weights += self.prior.log_prob(proposed_state.particles)

            log_likelihood_increment += ops.logsumexp(log_weights, axis=1)
        log_likelihoods = (
            state.log_likelihoods
            + log_likelihood_increment / self.enhance_gradient_steps
        )

        # log_weights, proposed_state are just the last sampled set of particles
        normalized_log_weights = normalize(log_weights, 1, state.n_particles, True)
        weights = ops.exp(normalized_log_weights)

        # Optionally compute entropy
        if self._compute_entropy:
            transition_fn = lambda x: self._transition_model.loglikelihood(
                state, x, inputs
            )
            entropy = self.entropy(
                transition_fn,
                proposed_state.particles,
                observation_log_likelihoods,
                weights,
                state.log_weights,
            )
        else:
            entropy = None

        return (
            proposed_state.evolve(
                weights=weights,
                log_weights=normalized_log_weights,
                log_likelihoods=log_likelihoods,
                entropy=entropy,
                action=action,
            ),
            proposal_dists,
        )
