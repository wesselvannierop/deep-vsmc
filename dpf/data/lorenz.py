if __name__ == "__main__":
    import dpf.runnable  # isort: skip

    dpf.runnable.runnable(device="auto:1", hide_first_for_tf=False)

import math
import warnings
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import ops
from tqdm import tqdm

import dpf.filterflow as filterflow
import tfp_wrapper as tfp
from dpf.data.lorenz_data import Lorenz, LorenzPSF, get_lorenz_kwargs
from dpf.data.utils import backwards_compatibility
from dpf.dpf_utils import Masker
from usbmd import tensor_ops
from usbmd.utils import save_to_gif, to_8bit

tfd = tfp.distributions

# TEMP_DIR = "temp"
TEMP_DIR = "temp"


def fast_debug_config(config, debugging=True):
    if debugging:
        warnings.warn("Using fast debug config!")
        fast_config = dict(
            n_particles=4,
            sequence_length=3,
            batch_size=2,
            nr_of_sequences=2,
            val_sequence_length=3,
            val_batch_size=2,
            val_nr_of_sequences=2,
        )
        config.update(fast_config)


def evolution_process_taylor(
    x, coefficients=2, delta_t=0.02, sigma=10, rho=28, beta=8 / 3
):
    # Create the A_ matrix by leveraging numpy broadcasting
    batch_shape = x.shape[:-1]
    row1 = ops.stack(
        [
            -sigma * ops.ones(batch_shape),
            sigma * ops.ones(batch_shape),
            ops.zeros(batch_shape),
        ],
        axis=-1,
    )
    row2 = ops.stack(
        [rho - x[..., 2], -ops.ones(batch_shape), ops.zeros(batch_shape)], axis=-1
    )
    row3 = ops.stack(
        [x[..., 1], ops.zeros(batch_shape), -beta * ops.ones(batch_shape)], axis=-1
    )
    A_ = ops.stack([row1, row2, row3], axis=-2)

    # Initialize A as an identity matrix, broadcasted to match batch dimensions
    A = ops.eye(3, dtype="float32")
    A = ops.copy(ops.broadcast_to(A, batch_shape + (3, 3)))

    # Perform the Taylor series expansion
    for i in range(1, coefficients + 1):
        A += tensor_ops.matrix_power(A_ * delta_t, i) / math.factorial(i)

    # Perform the batched matrix-vector multiplication
    return ops.einsum("...ij,...j->...i", A, x)


def lorenz_experiment(config, state2coord):
    observation_model = LorenzObservationModel(
        sigma=config.observation_model.likelihood_sigma,
        state2coord=state2coord,
        image_shape=config.data.image_shape[:2],
    )
    return observation_model


def get_observation_fn(data_config, action=False):
    if action:
        return None

    if data_config.get("observation_fn", None) is None:
        if hasattr(data_config, "observation_fn_kwargs"):
            warnings.warn(
                "Config has observation_fn_kwargs but does not provide a observation_fn"
                " -> Ignoring observation_fn_kwargs"
            )
        observation_fn = None
    elif data_config.observation_fn == "mask":
        observation_fn = Masker(
            data_config.image_shape[:2],
            **data_config.get("observation_fn_kwargs", {}),
        )
    else:
        raise ValueError(f"Unknown observation_fn: {data_config.observation_fn}")
    return observation_fn


def lorenz_experiment_data(config, example=True, example_name="lorenz.gif"):
    lorenz_kwargs = get_lorenz_kwargs(config)
    lorenz_train = Lorenz(partition="train", **lorenz_kwargs)
    lorenz_val = Lorenz(partition="val", **lorenz_kwargs)

    action = config.get("action", None) is not None
    train_kwargs = dict(
        salt_prob=config.data.get("salt_prob", 0.0),
        pepper_prob=config.data.get("pepper_prob", 0.0),
        awgn_std=config.data.get("awgn_std", 0.0),
        observation_fn=get_observation_fn(config.data, action),
        normalization_range=config.data.get("normalization_range", [0, 1]),
    )
    if hasattr(config, "val_data"):
        val_kwargs = dict(
            salt_prob=config.val_data.get("salt_prob", 0.0),
            pepper_prob=config.val_data.get("pepper_prob", 0.0),
            awgn_std=config.val_data.get("awgn_std", 0.0),
            observation_fn=get_observation_fn(config.val_data, action),
            normalization_range=config.val_data.get("normalization_range", [0, 1]),
        )
    else:
        val_kwargs = train_kwargs

    # Train
    if config.get("training", {}).get("train", False):
        dataset = lorenz_train.tf_dataset(
            batch_size=config.batch_size,
            seq_length=config.sequence_length,
            shuffle=True,
            **train_kwargs,
        )
    else:
        dataset = None

    val_dataset = lorenz_val.tf_dataset(
        batch_size=config.val_batch_size,
        seq_length=config.val_sequence_length,
        shuffle=False,
        repeat=1,
        **val_kwargs,
    )

    if example:
        _, observations, masks = next(iter(val_dataset))
        imgs = observations[:, 0]  # first batch item (full sequence)
        masks = masks[:, 0]  # first batch item (full sequence)
        imgs = to_8bit(imgs, dynamic_range=[0, 1], pillow=False)
        masks = ops.logical_not(masks)
        assert imgs[masks].sum() == 0, "Masks are not correct!"
        masks = ops.cast(masks, "float32") * 0.5
        masks = to_8bit(masks, dynamic_range=[0, 1], pillow=False)
        imgs += masks
        save_to_gif(imgs, config.save_path / example_name)

    return dataset, val_dataset


@keras.saving.register_keras_serializable()
class LorenzObservationModel(filterflow.ObservationModelBase):
    def __init__(
        self, state2coord=None, sigma=0.1, image_shape=(28, 28), x_max=35, y_max=35
    ):
        if state2coord is None:
            state2coord = keras.layers.Identity()
        self.state2coord = state2coord
        self.sigma = sigma
        self.image_shape = image_shape
        self.x_max = x_max
        self.y_max = y_max

        self.lorenz_psf = LorenzPSF(image_shape, x_max, y_max)

    def decoder(self, x):
        return self.lorenz_psf(self.state2coord(x))

    def get_config(self):
        return {
            "state2coord": self.state2coord,
            "sigma": self.sigma,
            "image_shape": self.image_shape,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @classmethod
    def from_config(cls, config):
        config["state2coord"] = keras.layers.deserialize(config["state2coord"])
        config = backwards_compatibility(config)
        return cls(**config)

    def to_observation_domain(self, particles, masks=None):
        """
        Args:
            particles (Tensor): The particle tensor of shape [*batch_dims, n_particles, state_dims]
            masks (Tensor): The masks tensor - [*batch_dims, *observation_dims]

        Returns:
            Tensor: The observation tensor of shape [n_particles, *batch_dims, *observation_dims]
        """
        # Move n_particles to the front
        particles = ops.moveaxis(particles, -2, 0)

        images = self.decoder(particles)
        if masks is not None:
            images = images * ops.cast(masks, ops.dtype(images))
        return images

    def _loglikelihood(
        self, particles, observation, masks=None, sigma=None, normalize=False
    ):
        """
        Normalize makes sure that the loglikelihood makes sense for number of pixels observed.
        If very little pixels are observed (without normalization), the loglikelihood will
            be very high, because 0-0=0.
        """
        images = self.to_observation_domain(particles, masks)

        if sigma is None:
            sigma = self.sigma

        if normalize:
            log_likelihood = tfd.Normal(loc=images, scale=sigma).log_prob(observation)
            if masks is not None:
                log_likelihood = ops.where(masks, log_likelihood, 0)
                pixels_observed = ops.cast(ops.sum(masks, axis=[-1, -2]), "float32")
            log_likelihood = ops.sum(log_likelihood, axis=[-1, -2])
            if masks is not None:
                log_likelihood /= pixels_observed
                log_likelihood *= ops.cast(ops.prod(ops.shape(images)[-2:]), "float32")
        else:
            log_likelihood = tfd.Independent(
                tfd.Normal(loc=images, scale=sigma),
                reinterpreted_batch_ndims=2,
            ).log_prob(observation)

        return ops.moveaxis(log_likelihood, 0, -1)

    def loglikelihood(self, state, observation, inputs=None):
        """
        Calculate the log-likelihood of the state given the observation.

        Args:
            state (object): The state object containing particles.
                - particles (Tensor): The particle tensor of shape [*batch_dims, n_particles, state_dims].
            observation (Tensor): The observation tensor - [*batch_dims, *observation_dims]

        Returns:
            Tensor: The log-likelihood tensor of shape [*batch_dims, n_particles].
        """
        return self._loglikelihood(state.particles, observation, inputs)
