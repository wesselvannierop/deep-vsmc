import pickle
import time
import warnings
from functools import partial
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import ops
from zea import log, tensor_ops

import vsmc.ops as dpf_ops
import vsmc.tfp_wrapper as tfp
from vsmc.data.lorenz import LorenzObservationModel
from vsmc.data.lorenz_data import lorenz_kde_prior
from vsmc.filterflow import SMC
from vsmc.filterflow.resampling import (
    NeffCriterion,
    RegularisedTransform,
    SystematicResampler,
)
from vsmc.filterflow.resampling.criterion import ResamplingCriterionBase
from vsmc.filterflow.state import State, StateSeries
from vsmc.keras_helpers import (
    CSVLoggerEval,
    EpochCounterCallback,
    MyProgbarLogger,
    MyWandbMetricsLogger,
    deserialize_config,
)
from vsmc.load_model import load_model
from vsmc.models import setup_bootstrap, setup_ekpf, setup_vsmc
from vsmc.models.base import BaseTrainer, update_instance_with_attributes
from vsmc.models.preset_loader import from_preset
from vsmc.prob import Distribution
from vsmc.tf_prob import entropy_mc, mc_kld
from vsmc.utils import del_keys, rename_key

tfd = tfp.distributions

LOAD_OWN_VARIABLES = True


def check_config(config):
    # When training, the model should be learned
    if config.get("train"):
        assert config.model == "learned", "Only learned model can be trained"
    # When evaluating, a checkpoint should be provided
    elif config.model == "learned":
        assert config.training.checkpoint is not None, "Need a checkpoint to evaluate"
    if config.training.get("checkpoint") is not None and config.get("train"):
        assert hasattr(config, "epoch_checkpoint"), (
            "Need epoch_checkpoint for checkpoint"
        )

    assert not "validation" in config, "Using old config -> refactor!"


def backward_compatible(config):
    del_keys(
        config,
        [
            "val_interval",
            "plot_actions",
            "plot_config",
            "gradient_clipping",
            "change_seed",
            "initial_seed",
            "method_name",
            "visualize_batch",
            "visualize_idx",
        ],
    )
    if not isinstance(config["state_dims"], (tuple, list)):
        config["state_dims"] = (config["state_dims"],)
    rename_key(config, "test_transition_model", "enable_test_transition_model", True)
    rename_key(config, "criterion_instance", "criterion")
    return config


@keras.saving.register_keras_serializable()
class LearnedPF(BaseTrainer):
    def __init__(
        self,
        observation_model,
        transition_model,
        proposal_model,
        initial_state: tuple,
        state_dims: tuple,
        action_model=None,
        prior: Distribution = None,
        initial_state_dist: str = "normal",
        n_particles: int = 28,
        criterion: ResamplingCriterionBase | None = None,
        save_folder: str = "results",
        enhance_gradient_steps: int = 1,
        supervised: bool = False,
        pre_smc_operation=None,
        state2coord=None,
        all_observations_at_epoch: int = 30,
        predict_step_avg=5,
        dump_results=False,
        enable_compute_elbo=False,
        enable_test_transition_model=False,
        elbo_likelihood_sigma=0.1,
        store_proposal=False,
        skip_grad=None,
        normalization_range: tuple = [0, 1],
        **kwargs,
    ):
        # Add all attributes to the instance
        self.observation_model = observation_model
        self.transition_model = transition_model
        self.proposal_model = proposal_model
        self.initial_state = initial_state
        self.state_dims = state_dims
        self.action_model = action_model
        self.prior = prior
        self.initial_state_dist = initial_state_dist
        self.n_particles = n_particles
        if criterion is None:
            criterion = NeffCriterion(0.5, is_relative=True)
        self.criterion = criterion
        self.save_folder = Path(save_folder)
        self.enhance_gradient_steps = enhance_gradient_steps
        self.supervised = supervised
        if pre_smc_operation is None:
            pre_smc_operation = keras.layers.Identity()
        self.pre_smc_operation = pre_smc_operation
        if state2coord is None:
            state2coord = keras.layers.Identity()
        self.state2coord = state2coord
        self.all_observations_at_epoch = all_observations_at_epoch
        self.predict_step_avg = predict_step_avg
        self.dump_results = dump_results
        self.enable_compute_elbo = enable_compute_elbo
        self.enable_test_transition_model = enable_test_transition_model
        self.elbo_likelihood_sigma = elbo_likelihood_sigma
        self.store_proposal = store_proposal
        self.skip_grad = skip_grad
        self.normalization_range = normalization_range
        super().__init__(**kwargs)

        self.seed = keras.random.SeedGenerator(42)  # only used for initial state

        self.range_to = (0, 1)
        self._decode_modes = ["wm"]

        # Initialize metrics
        for metric in self.mean_metrics:
            setattr(self, metric + "_tracker", keras.metrics.Mean(name=metric))
        self.my_metrics = [
            getattr(self, metric + "_tracker") for metric in self.mean_metrics
        ]
        self.skip_metrics = ["grad_norm", "n_observations"]

    def postprocess(self, data):
        return dpf_ops.postprocess(data, self.normalization_range, self.range_to)

    @property
    def mean_metrics(self):
        x = ["loss"]
        for mode in self._decode_modes:
            x += [f"l2norm_{mode}"]
        if self.enable_compute_elbo:
            x += ["elbo", "kl", "log_likelihood", "entropy", "joint"]
        if self.enable_test_transition_model:
            x += ["l2norm_1pred", "l2norm_2pred"]
        return x

    @property
    def metrics(self):
        return self.my_metrics

    @property
    def dpf_attrs(self):
        """All attributes of LearnedPF that are necessary for the SMC class."""
        return [
            "observation_model",
            "transition_model",
            "proposal_model",
            "criterion",
            "action_model",
            "enhance_gradient_steps",
            "store_proposal",
            "supervised",
            "prior",
            "state_dims",
            "image_shape",
            "n_particles",
        ]

    @property
    def dpf_ready(self):
        """Check if all necessary attributes are present."""
        return all(hasattr(self, attr) for attr in self.dpf_attrs)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self.dpf_attrs and self.dpf_ready:
            print("(Re)building SMC instances")
            self.val_smc = self.get_val_dpf()
            self.train_smc = self.get_trainable_dpf()

    def get_dpf(self, resampler, store_all=None, jit=False):
        # NOTE: when the init signature changes, add to: self.dpf_attrs!
        smc = SMC(
            observation_model=self.observation_model,
            transition_model=self.transition_model,
            proposal_model=self.proposal_model,
            resampling_criterion=self.criterion,
            resampling_method=resampler,
            action_model=self.action_model,
            enhance_gradient_steps=self.enhance_gradient_steps,
            store_proposal=self.store_proposal,
            store_all=self.supervised if store_all is None else store_all,
            prior=self.prior,
        )
        if jit:
            smc.jit(self.state_dims, self.image_shape, self.n_particles)
        return smc

    def get_trainable_dpf(self):
        resampler = RegularisedTransform(
            0.1, scaling=0.75, max_iter=100, convergence_threshold=1e-3
        )
        # NOTE: store_all is enabled to calculate l2norm_wm
        return self.get_dpf(resampler, store_all=True)

    def get_val_dpf(self):
        return self.get_dpf(SystematicResampler(), store_all=True)

    def call(
        self,
        observations,
        initial_particles=None,
        input_series=None,
        n_observations: int = None,
        training: bool = None,
        seed=None,
    ):
        # If no initial_particles are provided, try to generate them
        if initial_particles is None:
            batch_size = ops.shape(observations)[1]
            initial_particles = self.get_initial_state(batch_size)

        if training:
            return self.train_smc(
                observations, initial_particles, n_observations, input_series, seed=seed
            )
        else:
            return self.val_smc(
                observations, initial_particles, n_observations, input_series, seed=seed
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proposal_model": self.proposal_model,
                "transition_model": self.transition_model,
                "observation_model": self.observation_model,
                "state2coord": self.state2coord,
                "action_model": self.action_model,
                "initial_state": self.initial_state,
                "prior": self.prior,
                "initial_state_dist": self.initial_state_dist,
                "n_particles": self.n_particles,
                "state_dims": self.state_dims,
                "criterion": self.criterion,
                "save_folder": str(self.save_folder),
                "enhance_gradient_steps": self.enhance_gradient_steps,
                "supervised": self.supervised,
                "pre_smc_operation": self.pre_smc_operation,
                "all_observations_at_epoch": self.all_observations_at_epoch,
                "predict_step_avg": self.predict_step_avg,
                "dump_results": self.dump_results,
                "enable_compute_elbo": self.enable_compute_elbo,
                "enable_test_transition_model": self.enable_test_transition_model,
                "elbo_likelihood_sigma": self.elbo_likelihood_sigma,
                "skip_grad": self.skip_grad,
            }
        )
        return config

    @property
    def shapes_dict(self):
        # 1 represents the batch_size, or n_observations
        return {
            "observations": [1, 1, *self.image_shape],
            "initial_particles": [1, self.n_particles, *self.state_dims],
            "n_observations": None,
            "input_series": None,
        }

    @property
    def image_shape(self):
        return self.observation_model.image_shape

    def call_on_shapes_dict(self):
        inputs = {}
        for key, shape in self.shapes_dict.items():
            if shape is None:
                inputs[key] = None
            else:
                inputs[key] = ops.ones(shape)
        self(**inputs)

    def build_from_config(self, config):
        if "shapes_dict" in config:
            super().build_from_config(config)
        else:
            print("Building from old config")
            self.call_on_shapes_dict()

    @classmethod
    def from_config(cls, config):
        config = backward_compatible(config)
        config = deserialize_config(
            config,
            [
                "proposal_model",
                "transition_model",
                "observation_model",
                "action_model",
                "state2coord",
                "pre_smc_operation",
                "criterion",
                "prior",
            ],
        )
        return cls(**config)

    def get_initial_state(self, batch_size, real_initial_position=None):
        state_shape = [batch_size, self.n_particles, *self.state_dims]
        initial_state = self.initial_state.copy()

        if initial_state[0] == "true" and real_initial_position is None:
            raise ValueError(
                "Could not set initial state to real position. Provide `real_initial_position`."
            )

        if initial_state[0] == "true":
            # Add n_particles dimension
            initial_state[0] = ops.repeat(
                real_initial_position[:, None], self.n_particles, axis=1
            )
            # If the state is velocity-based, we need to add zeros
            initial_state[0] = dpf_ops.pad_to_shape(initial_state[0], state_shape)

        # Set the initial state
        if self.initial_state_dist == "normal":
            particles = keras.random.normal(state_shape, *initial_state, seed=self.seed)
        elif self.initial_state_dist == "uniform":
            particles = keras.random.uniform(
                state_shape, *initial_state, seed=self.seed
            )
        else:
            raise ValueError(
                f"Unknown initial state distribution: {self.initial_state_dist}"
            )

        return particles

    @staticmethod
    def _n_observations(seq_len, epoch, all_observations_at_epoch):
        """How many observations to use for training at this step."""
        unclipped = (epoch + 1) * seq_len // (all_observations_at_epoch + 1)
        return np.clip(unclipped, 1, seq_len).astype(np.int32)

    def n_observations(self, seq_len):
        return self._n_observations(seq_len, self.epoch, self.all_observations_at_epoch)

    def recompile_at_epochs(self, seq_len):
        at_epoch = [None]
        for epoch in range(self.all_observations_at_epoch + 1):
            x = self._n_observations(seq_len, epoch, self.all_observations_at_epoch)
            if x != at_epoch[-1]:
                at_epoch.append(x)
        return at_epoch[1:]

    def recompile_now(self, epoch, seq_len):
        return epoch in self.recompile_at_epochs(seq_len)

    def posterior_sample(self, states, mode="ml"):
        particles = states.get_particle(mode)
        return self.state2coord(particles)

    @property
    def n_latent_dims(self):
        return len(self.state_dims)

    def particle_l2_error(self, coords, real_positions):
        """
        Args:
            coords: [*batch_dims, *state_dims]
            real_positions: [*batch_dims, *state_dims]
        Returns:
            l2_error: [*batch_dims]
        """
        flat_coords = tensor_ops.flatten(coords, start_dim=-self.n_latent_dims)
        flat_real_positions = tensor_ops.flatten(
            real_positions, start_dim=-self.n_latent_dims
        )
        return ops.norm(flat_coords - flat_real_positions, axis=-1, ord=2)

    def supervised_loss(self, states, real_positions, wrt="wm", type="norm"):
        coords = self.posterior_sample(states, mode=wrt)
        if type == "norm":
            loss = ops.mean(self.particle_l2_error(coords, real_positions))
        elif type == "mse":
            loss = ops.mean(
                ops.square(coords - real_positions),
                axis=list(range(-self.n_latent_dims, 0)),
            )
        else:
            raise ValueError(f"Unknown loss type: {type}")
        return loss

    def supervised_transition_loss(self, states, real_positions):
        # stocastically sample a particle
        particle = states.get_particle("sample")
        # remove last timestep if there is no 'real_positions' for it
        gt_n_observations = ops.shape(real_positions)[0]
        particle = particle[: gt_n_observations - 1]
        # transition
        transitioned_distribution = self.transition_model(particle)
        # sample from the transitioned distribution
        transitioned_particles = transitioned_distribution.sample()
        # get the corresponding range of real_positions
        n_observations = ops.shape(transitioned_particles)[0]
        real_positions = real_positions[1 : n_observations + 1]
        return ops.mean(self.particle_l2_error(transitioned_particles, real_positions))

    def vsmc_loss(self, state, n_observations):
        if isinstance(state, State):
            # state.log_likelihoods: [batch_size]
            loss = -ops.mean(state.log_likelihoods)
        elif isinstance(state, StateSeries):
            # states.log_likelihoods: [n_observations, batch_size]
            loss = -ops.mean(state.log_likelihoods[-1])
        else:
            raise ValueError(f"Unknown state type: {type(state)}")
        return loss / ops.cast(n_observations, "float32")

    def train_step(self, data):
        """
        Args:
            data: (real_positions, observations, masks)
                where real_positions.shape = (n_observations, batch_size, *state_dims)
                where observations.shape = (n_observations, batch_size, *obs_dims)
                where masks.shape = (n_observations, batch_size, *obs_dims)
        """
        # Prepare data and get initial state
        real_positions, observations, masks = self.pre_smc_operation(data)
        seq_len = ops.shape(observations)[0]
        batch_size = ops.shape(observations)[1]
        n_observations = self.n_observations(seq_len=seq_len)
        real_initial_position = real_positions[0]
        initial_state = self.get_initial_state(batch_size, real_initial_position)

        # Compute the loss
        with tf.GradientTape() as tape:
            state, proposal_dists, states = self(
                observations,
                initial_state,
                masks,
                n_observations=n_observations,
                training=True,
            )

            if self.supervised:
                loss = self.supervised_loss(states, real_positions[:n_observations])
                loss += self.supervised_transition_loss(states, real_positions)
            else:
                loss = self.vsmc_loss(state, n_observations)

        # Compute the gradients
        grads = tape.gradient(loss, self.trainable_variables)
        grad_norm = dpf_ops.global_norm(grads)

        # Skip gradient update: only apply gradients if grad_norm is below the skip threshold
        # TODO: maybe reflect this in the grad_norm metric
        self.apply_gradients(grads, self.trainable_variables, self.skip_grad, grad_norm)

        # Compute the mean L2 norm of the weighted mean
        l2norm_wm = self.supervised_loss(
            states, real_positions[:n_observations], wrt="wm", type="norm"
        )

        results = {
            "loss": loss,
            "grad_norm": grad_norm,
            "l2norm_wm": ops.mean(l2norm_wm),
            "n_observations": n_observations,
        }

        self.update_metrics(results)
        return results  # instantaneous metrics!

    def test_transition_model(self, states, real_positions, seq_len, batch_size):
        # TODO: this causes retracing (function pfor.<locals>.f)
        step = ops.cast(ops.ceil(seq_len / self.predict_step_avg), "int32")
        t_range = ops.arange(0, seq_len - 2, step)

        def test_without_observations(i, state):
            # NOTE that it takes the (not weighted) mean of the particles!
            t = t_range[i]
            # 1 step into the future
            pred_state = self.val_smc.predict(states.read(t), None)
            l2norm_1pred = self.supervised_loss(
                pred_state, real_positions[t + 1], wrt="mean", type="norm"
            )
            # 2 steps into the future
            pred_state_2 = self.val_smc.predict(pred_state, None)
            l2norm_2pred = self.supervised_loss(
                pred_state_2, real_positions[t + 2], wrt="mean", type="norm"
            )
            l2norm_1pred_sum, l2norm_2pred_sum = state
            # Sum up the l2norm (mean over the batches) for this timestep
            l2norm_1pred_sum += ops.sum(l2norm_1pred)
            l2norm_2pred_sum += ops.sum(l2norm_2pred)
            return l2norm_1pred_sum, l2norm_2pred_sum

        l2norm_1pred_sum, l2norm_2pred_sum = ops.fori_loop(
            0,
            ops.shape(t_range)[0],
            test_without_observations,
            (0.0, 0.0),
        )
        avg_factor = ops.cast((ops.shape(t_range)[0] * batch_size), "float32")
        return {
            "l2norm_1pred": l2norm_1pred_sum / avg_factor,
            "l2norm_2pred": l2norm_2pred_sum / avg_factor,
        }

    def active_sampling(self, targets, seed=None) -> StateSeries:
        """
        Small wrapper around self.call which resizes and reorders the targets,
            and returns the state series.
        Args:
            targets: [batch_size, h, w, seq_len]
        """

        # Resize for dpf
        targets = ops.image.resize(targets, self.image_shape)

        # Move seq_len to the front
        targets = ops.moveaxis(targets, -1, 0)

        _, _, state_series = self.call(targets, training=False, seed=seed)
        return state_series

    def test_step(self, data):
        """
        Args:
            data: (real_positions, observations, masks)
                where real_positions.shape = (n_observations, batch_size, *state_dims)
                where observations.shape = (n_observations, batch_size, *obs_dims)
                where masks.shape = (n_observations, batch_size, *obs_dims)
        """
        # Unpack and prepare data
        real_positions, observations, masks = self.pre_smc_operation(data)
        seq_len = ops.shape(observations)[0]
        batch_size = ops.shape(observations)[1]
        real_initial_position = real_positions[0]
        initial_state = self.get_initial_state(batch_size, real_initial_position)

        # Run PF
        state, proposal_dists, states = self(
            observations,
            initial_state,
            masks,
            training=False,
        )

        if self.supervised:
            loss = self.supervised_loss(states, real_positions)
        else:
            loss = self.vsmc_loss(state, seq_len)
        results = {"loss": loss}

        # Override the masks if an action model is present
        if self.action_model is not None:
            masks = states.action

        # ELBO
        if self.enable_compute_elbo and not self.is_training:
            elbo, kl, log_likelihood, entropy, joint, _ = self.compute_elbo(
                states=states,
                observations=observations,
                masks=masks,
                dump=self.dump_results,
                likelihood_sigma=self.elbo_likelihood_sigma,
            )
            results |= {
                "elbo": ops.mean(elbo),
                "kl": ops.mean(kl),
                "log_likelihood": ops.mean(log_likelihood),
                "entropy": ops.mean(entropy),
                "joint": ops.mean(joint),
            }

        # Test without observations
        if self.enable_test_transition_model:
            results |= self.test_transition_model(
                states, real_positions, seq_len, batch_size
            )

        # Compute metrics and log
        for mode in self._decode_modes:
            # one_particle: [n_observations, batch_size, *state_dims]
            one_particle = self.posterior_sample(states, mode=mode)

            # Compute the L2 norm between the particles and the real positions
            l2norm_particles = self.particle_l2_error(one_particle, real_positions)
            results[f"l2norm_{mode}"] = ops.mean(l2norm_particles)

        # Dump states and batch
        if self.dump_results and not self.is_training:
            pickle.dump(results, open(self.save_folder / "results.pkl", "wb"))
            pickle.dump(data, open(self.save_folder / "batch.pkl", "wb"))
            states.dump(self.save_folder / "states.pkl")

        self.update_metrics(results)

        return results  # instantaneous metrics!

    def compute_elbo(
        self,
        states,
        observations,
        masks,
        verbose=True,
        dump=False,
        use_kde_cache=True,
        likelihood_sigma=0.1,  # TODO: sweep over sigma
        mc_samples=1000,
        mc_batch_size=30,
    ):
        # TODO: elbo computation is lorenz only! (adapt for other state dimensions etc)
        # TODO: exclude initial state from evaluation, not sure if it is already done
        if verbose:
            print(f"ELBO likelihood_sigma: {likelihood_sigma}")
            print("Computing ELBO")

        # Trim the velocity from the state
        particles = self.state2coord(states.particles)

        if hasattr(states, "particles_cov"):
            # This means that we are probably evaluating the EKF
            print("ELBO for EKF")
            particles_cov = states.particles_cov[..., :3, :3]
            components_distribution = tfd.MultivariateNormalFullCovariance(
                particles,
                particles_cov,
            )
            particle_posterior = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=states.weights),
                components_distribution=components_distribution,
            )
        else:
            # This means that we are probably evaluating the particle filter
            mean, cov = dpf_ops.fit_gaussian(particles, states.weights)
            particle_posterior = tfd.MultivariateNormalFullCovariance(mean, cov)

        # KL divergence between the particle posterior and the prior
        prior_dist = lorenz_kde_prior(use_cache=use_kde_cache)
        start_time = time.time()
        kl = mc_kld(
            particle_posterior,
            prior_dist,
            n_samples=mc_samples,
            batch_size=mc_batch_size,
        )
        if verbose:
            print("KL computation time: ", time.time() - start_time)

        def log_likelihood_fn(proposed_state):
            return ops.sum(
                self.observation_model._loglikelihood(
                    proposed_state,
                    observations,
                    masks,
                    sigma=likelihood_sigma,
                    normalize=True,
                ),
                axis=-1,
            )

        def prior_log_prob_fn(proposed_state):
            return ops.sum(prior_dist.log_prob(proposed_state), axis=-1)

        # Compute the log likelihood of the observed data
        log_likelihood = ops.zeros(particle_posterior.batch_shape)
        log_prob_prior = ops.zeros(particle_posterior.batch_shape)
        for _ in range(mc_samples // mc_batch_size):
            proposed_particles = particle_posterior.sample(mc_batch_size)
            # move mc_samples from 0 to -2. This is necessary for the observation model
            proposed_particles = ops.moveaxis(proposed_particles, 0, -2)
            log_likelihood += log_likelihood_fn(proposed_particles)
            log_prob_prior += prior_log_prob_fn(proposed_particles)
        log_likelihood /= mc_samples
        log_prob_prior /= mc_samples
        # log_likelihood = log_likelihood / (self.observation_model.img_size**2)

        # Compute the ELBO
        elbo = log_likelihood - kl

        # Compute the entropy
        try:
            # Use tensorflow probability entropy if possible
            entropy = particle_posterior.entropy()
        except NotImplementedError:
            # Otherwise, use the Monte Carlo estimate of entropy
            entropy = entropy_mc(
                particle_posterior,
                n_samples=mc_samples,
                batch_size=mc_batch_size,
            )

        joint = log_likelihood + log_prob_prior
        elbo2 = joint + entropy

        if dump:
            dump_path = self.save_folder / "elbo"
            dump_path.mkdir(exist_ok=True)
            elbo.numpy().dump(str(dump_path / "elbo.npy"))
            kl.numpy().dump(str(dump_path / "kl.npy"))
            ops.convert_to_numpy(log_likelihood).dump(
                str(dump_path / "log_likelihood.npy")
            )
            ops.convert_to_numpy(entropy).dump(str(dump_path / "entropy.npy"))
            ops.convert_to_numpy(joint).dump(str(dump_path / "joint.npy"))

        return elbo, kl, log_likelihood, entropy, joint, elbo2


def datasets(config, only_val=False, augmentations=None):
    """
    Should return a tuple of two datasets: training and validation
    Batches should be a tuple of:
        - observations: [batch_size, n_observations, *obs_dims]
        - masks (optional): [batch_size, n_observations, *obs_dims]
        - real_positions: [batch_size, n_observations, *state_dims]
    """
    if config.experiment == "lorenz":
        from vsmc.data.lorenz import lorenz_experiment_data

        datasets = lorenz_experiment_data(config)
    else:
        raise ValueError(f"Unknown experiment: {config.experiment}")

    # Apply augmentations
    aug_datasets = []
    for dataset in datasets:
        if augmentations is not None and dataset is not None:
            aug_datasets.append(dataset.map(augmentations))
        else:
            aug_datasets.append(dataset)

    return aug_datasets


def get_coord_dims(config):
    """Returns a tuple with the dimensions of the coordinate/state space."""
    if config.experiment == "lorenz":
        return (3,)
    else:
        raise ValueError(f"Unknown experiment: {config.experiment}")


def make_init_dict(config):
    # Start with config.dpf
    init_dict = config.as_dict().get("dpf", {})
    if "dpf" in config:
        config.dpf._mark_accessed_recursive()

    # Add the rest of the config
    if "save_path" in config:
        init_dict["save_folder"] = str(config.save_path)
    init_dict["normalization_range"] = config.data.get("normalization_range", [0, 1])
    return init_dict


def dpf_prep(config, verbose=True):
    check_config(config)
    init_dict = make_init_dict(config)

    if config.get("training", {}).get("checkpoint") is not None:
        print(
            f"Loading model from checkpoint: {log.yellow(config.training.checkpoint)}. "
            "So the rest of the config is ignored."
        )
        checkpoint = from_preset(config.training.checkpoint)

        init_dict["observation_model"] = config.get("observation_model", {})
        pf = load_model(
            checkpoint,
            **init_dict,
            custom_objects={"LorenzObservationModel": LorenzObservationModel},
        )

        # Maybe update optimizer params
        update_instance_with_attributes(
            pf.optimizer, config.get("training", {}).get("optimizer", {})
        )

        return pf
        # return build_and_compile(pf, config)

    # Defaults (may be overwritten by config)
    pre_smc_operation = None
    action_model = None
    prior = None

    # Coord / state dimensions
    coord_dims = get_coord_dims(config)

    # Init models
    if config.model == "bootstrap":
        proposal_model, transition_model, state2coord, state_dims = setup_bootstrap(
            config, coord_dims
        )
    elif config.model == "ekpf":
        proposal_model, transition_model, state2coord, state_dims = setup_ekpf(
            config, coord_dims
        )
    elif config.model == "learned":
        proposal_model, transition_model, state2coord, state_dims = setup_vsmc(
            config, coord_dims, verbose=verbose
        )
    elif config.model == "encoder":
        from vsmc.baselines.lorenz_encoder.dpf_integration import setup_lorenz_encoder

        proposal_model, transition_model, state2coord, state_dims = (
            setup_lorenz_encoder(config, coord_dims)
        )
    else:
        raise ValueError(f"Unknown model: {config.model}")

    # Data
    if config.experiment == "lorenz":
        from vsmc.data.lorenz import lorenz_experiment

        observation_model = lorenz_experiment(config, state2coord)
    else:
        raise ValueError(f"Unknown experiment: {config.experiment}")

    init_dict |= dict(
        observation_model=observation_model,
        transition_model=transition_model,
        proposal_model=proposal_model,
        action_model=action_model,
        prior=prior,
        state_dims=state_dims,
        pre_smc_operation=pre_smc_operation,
        state2coord=state2coord,
    )

    pf = LearnedPF(**init_dict)
    return build_and_compile(pf, config)


def build_and_compile(pf, config):
    # Init optimizer
    optimizer_kwargs = config.get("training", {}).get("optimizer", {})
    optimizer = keras.optimizers.AdamW(**optimizer_kwargs)

    # Build
    pf.call_on_shapes_dict()

    unbuilt_layers = [l for l in pf.layers if not l.built]
    if len(unbuilt_layers) > 0:
        warnings.warn(f"Unbuilt layers: {unbuilt_layers}")

    # Compile
    run_eagerly = config.get("run_eagerly", False)
    pf.compile(optimizer=optimizer, jit_compile=False, run_eagerly=run_eagerly)
    return pf


def dpf_evaluate(pf, config, val_dataset, n_val_epochs=20, verbose="auto"):
    run_eagerly = config.get("run_eagerly", False) or config.dpf.get(
        "enable_compute_elbo", False
    )
    pf.compile(run_eagerly=run_eagerly, jit_compile=False)
    pf.is_training = False
    for _ in range(n_val_epochs):
        pf.evaluate(
            val_dataset,
            verbose=verbose,
            callbacks=get_val_callbacks(config, val_dataset),
        )

    # Load metrics and print mean performance
    metrics = pd.read_csv(config.save_path / "metrics-val.csv")
    mean, std = metrics["l2norm_wm"].mean(), metrics["l2norm_wm"].std()
    print(f"Mean l2norm WM: {mean}")
    print(f"Std l2norm WM: {std}")


def get_fit_callbacks(config, dataset, val_dataset, recompile_now_callable):
    callbacks = [
        EpochCounterCallback(len(dataset), recompile_now_callable),
        keras.callbacks.ModelCheckpoint(
            config.save_path / "pf_{epoch}.keras",
            monitor="val_l2norm_wm",
            verbose=1,
            save_best_only=config.get("save_best_only", False),
            save_weights_only=False,
            mode="min",
            save_freq="epoch",
        ),
        keras.callbacks.CSVLogger(config.save_path / "metrics.csv", append=False),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_l2norm_wm",
            factor=0.1,
            patience=10,
            verbose=1,
            mode="min",
        ),
        MyWandbMetricsLogger(log_freq="batch"),
        *get_val_callbacks(config, val_dataset),
    ]
    return callbacks


def get_val_callbacks(config, val_dataset):
    callbacks = [
        CSVLoggerEval(config.save_path / "metrics-val.csv", append=True),
        MyProgbarLogger(),
    ]
    return callbacks


def update_save_path(config, save_path: str):
    config.save_path = save_path
    config.save_path.mkdir()
    return config


def dpf_run(
    config, pf=None, dataset=None, val_dataset=None, verbose=True, keras_verbose="auto"
):
    # Load settings
    if config.get("train", False):
        n_retrain = config.get("training", {}).get("n_retrain", 1)
    else:
        n_retrain = 0
    base_save_path = config.save_path
    n_val_epochs = config.get("n_val_epochs", 20)
    only_val = n_retrain == 0

    # Create the pf instance if not provided
    if pf is not None:
        warnings.warn(
            "Provided a pf instance. Proceed carefully as there may be a mismatch with the config!"
        )
        if n_retrain > 1:
            raise ValueError(
                "Provided a pf instance, but n_retrain > 1. This is not allowed."
            )
    else:
        pf = dpf_prep(config, verbose=verbose)

    if pf.supervised:
        warnings.warn("Supervised mode is enabled.")

    # Load datasets if not provided
    if dataset is None or val_dataset is None:
        dataset, val_dataset = datasets(config, only_val=only_val)

    # Train (multiple times if n_retrain > 1)
    for i in range(n_retrain):
        # Change save path for multiple runs (must be done before dpf_prep)
        if n_retrain > 1:
            config = update_save_path(config, base_save_path / f"run{i}")

        # Reset the model
        if i > 0:
            pf = dpf_prep(config, verbose=verbose)

        print(f"start optimization (run {i + 1}/{n_retrain})")
        if i == 0:
            config.training._mark_accessed("epochs")
            config.training._mark_accessed("validation_freq")
            config._log_all_unaccessed()
        # limit steps per epoch to avoid long epoch time
        steps_per_epoch = int(1e4) if len(dataset) > 1e4 else None
        recompile_now_callable = partial(pf.recompile_now, seq_len=config.data.n_frames)
        pf.fit(
            dataset,
            epochs=config.training.epochs,
            validation_data=val_dataset,
            validation_freq=config.training.validation_freq,
            verbose=keras_verbose,
            callbacks=get_fit_callbacks(
                config, dataset, val_dataset, recompile_now_callable
            ),
            steps_per_epoch=steps_per_epoch,
            initial_epoch=config.get("epoch_checkpoint", 0),
        )
        pf.save(config.save_path / "pf-last.keras")

    if only_val:
        dpf_evaluate(pf, config, val_dataset, n_val_epochs=n_val_epochs)

    config._log_all_unaccessed()
    return pf
