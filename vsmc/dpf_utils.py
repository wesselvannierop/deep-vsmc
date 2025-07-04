import time
import warnings
from pathlib import Path

import keras
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import wandb
from keras import ops
from zea import log, tensor_ops

import vsmc.ops as dpf_ops
from vsmc.filterflow.transition import TransitionModelBase
from vsmc.prob import GaussianMixture

DEFAULT_MARKER_SIZE = mpl.rcParams["lines.markersize"] ** 2


def scale_marker_size(
    weight, scale_factor=DEFAULT_MARKER_SIZE, offset=DEFAULT_MARKER_SIZE // 6
):
    return weight * scale_factor + offset


def get_evolution_model(name):
    if name is None:
        return keras.layers.Identity()
    elif name == "velocity":
        return velocity_transition_fn
    elif name == "lorenz":
        from vsmc.data.lorenz import evolution_process_taylor

        return evolution_process_taylor
    else:
        raise ValueError(f"Invalid evolution model: {name}")


def mean_without_outliers(data, threshold=10, axis=None):
    # data.shape: (batch_size, n_observations)
    below_threshold = data < threshold
    above_threshold = data >= threshold
    accepted = tensor_ops.boolean_mask(data, below_threshold)
    mean = ops.mean(accepted, axis=axis)
    n_rejected = ops.sum(ops.cast(above_threshold, "float32"), axis=axis)
    p_rejected = n_rejected / ops.cast(ops.prod(data.shape), "float32")
    return mean, p_rejected


def velocity_transition_fn(z1):
    velocity = z1[..., 3:]
    z2 = z1 + dpf_ops.pad_zeros_like(velocity)
    return z2


class Masker:
    def __init__(self, image_shape=(28, 28), block_size=4, p=0.5, mask_fn=1) -> None:
        # p indicates the probability of a block being masked
        self.image_shape = np.array(image_shape)
        self.block_size = block_size
        self.block_img_size = self.image_shape // self.block_size
        self.p = p  # may also be tuple
        self.cached_masks = None
        if mask_fn == 0:
            self.generate_masks = self.generate_random_block_masks
        else:
            self.generate_masks = self.generate_random_block_masks2

    def get_p(self):
        # If p is a tuple, sample from uniform distribution with bounds p[0] and p[1]
        if isinstance(self.p, (tuple, list)):
            if self.p[0] == self.p[1]:
                return self.p[0]
            else:
                return keras.random.uniform((1,), self.p[0], self.p[1])
        else:
            return self.p

    def get_N(self):
        # N: Number of blocks to be masked
        N = int(np.round(np.prod(self.block_img_size) * self.get_p()))
        if N == 0:
            warnings.warn("No blocks will be masked")
        return N

    def generate_random_block_masks2(self, n_masks=(1,)):
        """Uses keras.random.uniform to generate random masks. Much faster for large n_masks."""
        batch_dims = ops.array(n_masks)
        if batch_dims.ndim == 0:
            batch_dims = batch_dims[None]

        mini_mask_shape = ops.concatenate((batch_dims, self.block_img_size))
        pepper_mask = keras.random.uniform(mini_mask_shape) < self.get_p()

        # Repeat the mask to match the original image size
        pepper_mask = ops.repeat(pepper_mask, self.block_size, axis=-1)
        pepper_mask = ops.repeat(pepper_mask, self.block_size, axis=-2)
        masks = ops.logical_not(pepper_mask)
        return masks

    # TODO: Add seed for deterministic behavior
    def generate_random_block_masks(self, n_masks=(1,)):
        """Will always generate the same number of occlusions for every mask"""
        n_masks = ops.convert_to_numpy(n_masks)

        batch_dims = np.array(n_masks)
        if batch_dims.ndim == 0:
            batch_dims = batch_dims[None]
        final_dims = np.concatenate((batch_dims, self.image_shape))

        # Total number of masks to generate
        total_masks = np.prod(batch_dims)

        # Randomly choose indices for masks
        inds = []
        for _ in range(total_masks):
            ind = np.random.choice(
                range(np.prod(self.block_img_size)),
                self.get_N(),
                replace=False,
            )
            inds.append(ind)
        ind = np.stack(inds)

        # Create the base mask with all zeros
        mask = np.zeros((total_masks, np.prod(self.block_img_size)), dtype=bool)

        # Set the chosen indices to True
        np.put_along_axis(mask, ind, True, axis=1)

        # Reshape to match block_img_size and expand back to original image size
        mask = mask.reshape((total_masks, *self.block_img_size))
        mask = np.repeat(mask, self.block_size, axis=-1)
        mask = np.repeat(mask, self.block_size, axis=-2)

        # Reshape to the final dimensions and negate the mask
        masks = np.logical_not(mask).reshape(final_dims)
        return ops.convert_to_tensor(masks)

    @staticmethod
    def mask_image_given_masks(images, masks):
        return images * ops.cast(masks, images.dtype)

    def __call__(self, images):
        self.cached_masks = self.generate_masks(images.shape[:-2])
        return self.mask_image_given_masks(images, self.cached_masks), self.cached_masks


def trim_velocity(particles, dims=3):
    # particles: (..., dx)
    return particles[..., :dims]


def _mask_image(images, masks):
    return images * ops.cast(masks, images.dtype)


def visualize_trajectory(
    real_positions,
    ml_particles,
    weighted_mean_particles,
    save_path,
    name="default",
    xlim=(-35, 35),
    ylim=(-35, 35),
):
    # This part is still in 2D
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(real_positions[:, 0], real_positions[:, 1])
    axs[0].plot(ml_particles[:, 0], ml_particles[:, 1])
    axs[0].set_title("ML particles")
    axs[1].plot(real_positions[:, 0], real_positions[:, 1])
    axs[1].plot(weighted_mean_particles[:, 0], weighted_mean_particles[:, 1])
    axs[1].set_title("WM particles")
    for ax in axs:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    plt.savefig(f"{save_path}/trajectory_{name}.png")
    log.success(f"Succesfully saved trajectory to -> {save_path}/trajectory_{name}.png")
    return fig


def clip_by_coords(data, lims):
    assert data.shape[-1] == len(lims), "Data and lims must have same dimensions"

    def clip(index, state):
        state[index] = ops.clip(state[index], *lims[index])

    return ops.fori_loop(0, len(lims), clip, data)


@keras.saving.register_keras_serializable()
class GaussianTransitionModel(TransitionModelBase):
    def __init__(
        self, sigma: list, state_dims, clip_range=None, evolution_model: str = None
    ):
        self.sigma = sigma
        self.state_dims = state_dims
        self.clip_range = clip_range
        self.evolution_model_name = evolution_model

        self.state_scale_diag = sigma * ops.ones(self.state_dims, "float32")
        self.evolution_model = get_evolution_model(evolution_model)
        self.model_velocity = evolution_model == "velocity"

    def get_config(self):
        return {
            "sigma": self.sigma,
            "state_dims": self.state_dims,
            "clip_range": self.clip_range,
            "evolution_model": self.evolution_model_name,
        }

    def transition_dist(self, particles):
        # Apply evolution model (could be identity)
        particles = self.evolution_model(particles)

        return GaussianMixture(
            particles, self.state_scale_diag, reinterpreted_batch_ndims=2
        )

    def loglikelihood(self, prior_state, proposed_state, inputs):
        dist = self.transition_dist(prior_state.particles)
        return dist.log_prob(proposed_state.particles)

    def sample(self, state, inputs, seed=None):
        # Sample from transition model
        dist = self.transition_dist(state.particles)
        proposed_particles = dist.sample(seed=seed)

        if self.clip_range is not None:
            proposed_particles = ops.clip(proposed_particles, *self.clip_range)
        return state.evolve(particles=proposed_particles)


def get_input_size(config):
    input_size = config.data.get("input_size")
    if input_size is None:
        assert hasattr(config.data, "image_shape"), (
            "config needs data.input_size or data.image_shape"
        )
        input_size = config.data.image_shape[:2]
    return input_size


def mean_from_npz(path, pattern="batch_*.npz"):
    # Find files
    filepaths = Path(path).glob(pattern)

    means = {}
    stds = {}
    for filepath in filepaths:
        # Load file
        data = np.load(filepath)

        # Loop over keys
        for key in data.files:
            if key not in means:
                means[key] = []
                stds[key] = []
            # Mean over batch and sequence
            means[key].append(np.mean(data[key]))
            stds[key].append(np.std(data[key]))

    print(f"Found {len(means[key])} files")

    # Mean over batches
    for key in means:
        means[key] = np.mean(means[key])
        stds[key] = np.mean(stds[key])
    return means, stds
