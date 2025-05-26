import math
import os
import pickle
import warnings
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from jax.scipy.stats import gaussian_kde as jax_kde
from keras import ops
from scipy.integrate import odeint

import tfp_wrapper as tfp
from usbmd import log, tensor_ops
from usbmd.utils import translate
from vsmc.data.utils import backwards_compatibility

tfd = tfp.distributions

DEFAULT_TR_TT = 1024 * 8
DEFAULT_VAL_TT = 128 * 32

TEMP_DIR = "temp"


class Lorenz:
    """Taken from: https://github.com/vgsatorras/hybrid-inference/blob/master/datasets/lorenz.py"""

    def __init__(
        self,
        partition="train",
        max_len=DEFAULT_TR_TT + 2 * DEFAULT_VAL_TT,
        tr_tt=DEFAULT_TR_TT,
        val_tt=DEFAULT_VAL_TT,
        test_tt=DEFAULT_VAL_TT,
        gnn_format=False,
        sparse=True,
        sample_dt=0.02,
        no_pickle=False,
        pickle_path=TEMP_DIR,
        pickle_tt=16384,
        pickle_sample_dt=0.2,
        seed=0,
        image_shape=(28, 28),
        x_max=35,
        y_max=35,
        solve_equations_dt=1e-5,
        awgn_std_state_transition=0.5,
        rho=28.0,
        sigma=10.0,
        beta=8.0 / 3.0,
        x0=[1.0, 1.0, 1.0],
    ):
        # Only used for generating images
        self.image_shape = image_shape
        self.x_max = x_max
        self.y_max = y_max
        self.img_maker = LorenzPSF(image_shape=image_shape, x_max=x_max, y_max=y_max)

        self.partition = partition  # training set or test set
        self.max_len = max_len
        self.gnn_format = gnn_format
        self.sparse = sparse
        self.x0 = x0
        self.sample_dt = sample_dt
        self.dt = solve_equations_dt

        H = np.eye(3)
        R = np.eye(3) * awgn_std_state_transition**2
        self.meas_model = MeasurementModel(H, R)

        self.rho = rho
        self.sigma = sigma
        self.beta = beta

        self.pickle_tt = pickle_tt
        self.pickle_sample_dt = pickle_sample_dt
        self.seed = seed

        if pickle_tt < tr_tt + val_tt + test_tt:
            warnings.warn(
                "Pickle length is smaller than the sum of the train, val and test lengths."
                "Will generate new data."
            )
            no_pickle = True
        if pickle_sample_dt < sample_dt:
            warnings.warn(
                "Pickle sample_dt is smaller than the sample_dt. Will generate new data."
            )
            no_pickle = True
        self.no_pickle = no_pickle
        self.pickle_path = Path(pickle_path)
        self.pickle_path.mkdir(parents=True, exist_ok=True)
        self._pickle_system(max_sample_dt=pickle_sample_dt)

        self.data = self._generate_sample(seed=seed, tt=tr_tt + val_tt + test_tt)

        if self.partition == "test":
            self.data = [self.data[0][0:test_tt], self.data[1][0:test_tt]]
        elif self.partition == "val":
            self.data = [
                self.data[0][test_tt : (test_tt + val_tt)],
                self.data[1][test_tt : (test_tt + val_tt)],
            ]
        elif self.partition == "train":
            self.data = [
                self.data[0][(test_tt + val_tt) : (test_tt + val_tt + tr_tt)],
                self.data[1][(test_tt + val_tt) : (test_tt + val_tt + tr_tt)],
            ]
        else:
            raise Exception("Wrong partition")
        self._split_data()

        """
        tr_samples = int(tr_tt/max_len)
        test_samples = int(test_tt / max_len)
        val_samples = int(val_tt / max_len)
        if self.partition == 'train':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples, test_samples + tr_samples)]
        elif self.partition == 'val':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples + tr_samples, test_samples + tr_samples + val_samples)]
        elif self.partition == 'test':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples)]
        else:
            raise Exception('Wrong partition')
        """

        print(
            "%s partition created, \t num_samples %d \t num_timesteps: %d"
            % (self.partition, len(self.data), self.total_len())
        )

    def __len__(self):
        return len(self.data)

    def dump(self, path, object):
        with open(path, "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            return pickle.load(f)

    def _split_data(self):
        num_splits = math.ceil(float(self.data[0].shape[0]) / self.max_len)
        data = []
        for i in range(int(num_splits)):
            i_start = i * self.max_len
            i_end = (i + 1) * self.max_len
            data.append([self.data[0][i_start:i_end], self.data[1][i_start:i_end]])
        self.data = data

    def _generate_sample(self, seed, tt):
        np.random.seed(seed)
        sample = self._simulate_system(tt=tt, x0=self.x0)

        # returns state, measurement
        return list(sample)

    def f(self, state, t):
        x, y, z = state  # unpack the state vector
        return (
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        )  # derivatives

    def _pickle_system(self, max_sample_dt=0.2):
        if not self.no_pickle and not (self.pickle_path / "lorenz_states.pkl").exists():
            print("Dumping Lorenz data to pickle")
            print("Simulating Lorenz system, this might take a while...")
            t = np.arange(0.0, self.pickle_tt * max_sample_dt, self.dt)
            states = odeint(self.f, self.x0, t)
            self.dump(self.pickle_path / "lorenz_states.pkl", states)

    def _simulate_system(self, tt, x0):
        """
        Simulates the Lorenz system for a given time period and initial conditions.
        Will solve odeint with self.dt and then downsample to self.sample_dt.

        Parameters:
            tt (float): Total time period for simulation.
            x0 (array-like): Initial conditions for the system.

        Returns:
            tuple: A tuple containing two arrays - `states` and `meas`.
                - `states` (ndarray): Array of system states at different time steps.
                - `meas` (ndarray): Array of measurements corresponding to the system states.
        """
        # Load states
        if self.no_pickle:
            print("Simulating Lorenz system, this might take a while...")
            t = np.arange(0.0, tt * self.sample_dt, self.dt)
            states = odeint(self.f, x0, t)
        else:
            print("Loading Lorenz data from pickle")
            states = self.load(self.pickle_path / "lorenz_states.pkl")

        # Downsample states
        states_ds = np.zeros((tt, 3))
        for i in range(states_ds.shape[0]):
            states_ds[i] = states[i * int(self.sample_dt / self.dt)]
        states = states_ds

        # Measurement
        meas = np.zeros(states.shape)
        for i in range(len(states)):
            meas[i] = self.meas_model(states[i])
        return states, meas

    def total_len(self):
        total = 0
        for state, meas in self.data:
            total += meas.shape[0]
        return total

    def get_data(self, noise_trajectory=True, seq_length=None):
        data = self.data[0][noise_trajectory]
        data = data.astype(np.float32)
        if seq_length is None:
            seq_length = data.shape[0]
        return data.reshape(-1, seq_length, 3)

    def images(self, noise_trajectory=True, seq_length=None):
        # (n_sequences, n_frames, h, w)
        return self.img_maker(self.get_data(noise_trajectory, seq_length))

    def tf_dataset(
        self,
        seq_length,
        batch_size,
        repeat=1,
        shuffle=False,
        drop_remainder=True,
        noise_trajectory=True,
        normalization_range=(0, 1),
        salt_prob=0.0,
        pepper_prob=None,
        awgn_std=0.0,
        observation_fn=None,
    ):
        # Get the data
        real_positions = self.get_data(noise_trajectory=False, seq_length=seq_length)
        imgs = self.images(noise_trajectory, seq_length)

        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices((imgs, real_positions))
        dataset = dataset.repeat(repeat)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(dataset) if len(dataset) < 1000 else 1000
            )

        # Noise images (before batching such that each item can get differently noised)
        dataset = dataset.map(
            lambda imgs, real_positions: (
                real_positions,
                *LorenzPSF.noise_images(
                    imgs, salt_prob, pepper_prob, awgn_std, observation_fn
                ),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # batch item is now: (real_positions, observations, masks)

        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        # Normalize images
        dataset = dataset_normalization(dataset, normalization_range)

        # Reshape (move seq_len to the front)
        dataset = dataset.map(
            lambda real_positions, observations, masks: (
                ops.moveaxis(real_positions, 1, 0),
                ops.moveaxis(observations, 1, 0),
                ops.moveaxis(masks, 1, 0),
            ),
        )

        return dataset


def anneal(value_range, temperature, max_temperature):
    assert temperature >= 0
    assert max_temperature > 0
    _value_range = np.array(value_range).copy()
    process = temperature / max_temperature
    _value_range[1] = ops.clip(
        _value_range[1] * process + _value_range[0], _value_range[0], _value_range[1]
    )
    return _value_range


def dataset_noise_annealing(
    dataset,
    epochs,
    all_noise_at_epoch=30,
    salt_prob: tuple = [0.0, 0.0],
    pepper_prob=None,  # TODO
    awgn_std: tuple = [0.0, 0.6],
    observation_fn=None,
):
    if all_noise_at_epoch > 0:
        # Anneal salt noise
        _salt_prob = anneal(salt_prob, epochs, all_noise_at_epoch)

        # Anneal gaussian noise
        _awgn_std = anneal(awgn_std, epochs, all_noise_at_epoch)

        # Anneal masker
        _p = observation_fn.p
        _p = anneal(_p, epochs, all_noise_at_epoch)
        observation_fn.p = _p
    else:
        _salt_prob = salt_prob
        _awgn_std = awgn_std

    # Noise images
    dataset = dataset.map(
        lambda imgs, *args: (
            *LorenzPSF.noise_images(
                imgs, _salt_prob, pepper_prob, _awgn_std, observation_fn
            ),
            *args,
        ),
    )
    # batch is now: (img, mask, real_positions)
    return dataset


def dataset_normalization(dataset, normalization_range=(0, 1)):
    dataset = dataset.map(
        lambda real_positions, observations, masks: (
            real_positions,
            translate(observations, (0, 1), normalization_range),
            masks,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset


@keras.saving.register_keras_serializable()
class LorenzPSF:
    """
    Sources:
        - https://github.com/KalmanNet/Latent_KalmanNet_TSP/blob/main/model_Lorenz.py
            uses y_max=40 and x_max=30 creating a squashed blob
        - https://arxiv.org/abs/2304.07827
    """

    def __init__(self, image_shape=(28, 28), x_max=35, y_max=35):
        self.grid = self.create_grid(image_shape, x_max, y_max)
        # grid.shape [float32]: (2, h, w)

    @classmethod
    def from_config(cls, config):
        config = backwards_compatibility(config)
        return cls(**config)

    def get_config(self):
        return {
            "image_shape": self.image_shape,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @staticmethod
    def create_grid(image_shape=(28, 28), x_max=35, y_max=35):
        x_range = ops.linspace(-x_max, x_max, image_shape[0])
        y_range = ops.linspace(-y_max, y_max, image_shape[1])
        x_grid, y_grid = ops.meshgrid(x_range, y_range, indexing="xy")
        grid = ops.stack([x_grid, y_grid], axis=0)
        grid = ops.cast(grid, "float32")
        return grid

    @staticmethod
    def noise_images(
        imgs, salt_prob=0.01, pepper_prob=None, awgn_std=0.0, observation_fn=None
    ):
        # If awgn_std is a tuple, take a uniform sample from the range
        if isinstance(awgn_std, (tuple, list)):
            awgn_std = keras.random.uniform((1,), awgn_std[0], awgn_std[1])

        imgs = tensor_ops.add_salt_and_pepper_noise(imgs, salt_prob, pepper_prob)
        imgs += keras.random.normal(imgs.shape, mean=0.0, stddev=awgn_std)
        if observation_fn is not None:
            imgs, masks = observation_fn(imgs)
        else:
            masks = ops.ones(imgs.shape, "float32")
        return imgs, masks

    def __call__(self, coordinate, eps=1e-6):
        # coordinate.shape [float32]: (*batch_dims, 3)
        assert (
            coordinate.shape[-1] == 3
        ), f"Last dimension of coordinate must be 3, got {coordinate.shape[-1]}"

        coordinate = coordinate[..., None, None] * ops.ones(
            self.grid[0].shape, "float32"
        )
        z = ops.abs(coordinate[..., 2, :, :])
        xy = coordinate[..., 0:2, :, :]
        distances = ops.linalg.norm(self.grid - xy, ord=2, axis=-3)
        exp_input = -(distances**2) / ((2 * z) + eps)
        exp_input = ops.clip(exp_input, -30.0, 30.0)  # prevent overflow
        images = ops.exp(exp_input)
        return ops.clip(images, 0.0, 1.0)


def __plot_trajectory(states):
    fig = plt.figure(linewidth=0.0)
    ax = fig.add_subplot(projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.5)
    plt.axis("off")
    return fig


class MeasurementModel:
    def __init__(self, H, R):
        self.H = H
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = self.H @ x + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement


def find_max():
    dataset = Lorenz(
        partition="train",
        no_pickle=True,
        max_len=10000,
        test_tt=0,
        val_tt=0,
        tr_tt=10000,
        sample_dt=0.02,
    )
    data = dataset.get_data(False)[0]
    print("Max x: ", np.max(data[:, 0]), "Min x: ", np.min(data[:, 0]))
    print("Max y: ", np.max(data[:, 1]), "Min y: ", np.min(data[:, 1]))
    print("Max z: ", np.max(data[:, 2]), "Min z: ", np.min(data[:, 2]))
    # Max x:  19.488646 Min x:  -18.04019
    # Max y:  27.183012 Min y:  -24.419317
    # Max z:  47.832016 Min z:  0.9617391


def compute_prior():
    dataset = Lorenz(
        partition="train",
        no_pickle=True,
        max_len=10000,
        test_tt=0,
        val_tt=0,
        tr_tt=10000,
        sample_dt=0.02,
    )
    prior = dataset.get_data(False)[0]
    quantized_prior = np.floor(prior).astype(int)
    len_prior = prior.shape[0]

    from collections import defaultdict

    # Initialize a default dictionary to count occurrences
    heatmap = defaultdict(float)

    # Count occurrences
    max_coord = 0
    max_norm = 0
    for coord in quantized_prior:
        key = tuple(coord)

        # Check for outliers
        if coord[0] > 35:
            print(coord)
        if coord[1] > 35:
            print(coord)
        if coord[2] > 50:
            print(coord)

        # Check max norm
        if np.linalg.norm(coord) > max_norm:
            max_coord = coord
            max_norm = np.linalg.norm(coord)

        # Increment the count
        heatmap[key] += 1 / len_prior

    print(f"Max norm: {max_norm}")
    print(f"Max coord: {max_coord}")

    # Convert to a regular dictionary for easier manipulation
    heatmap = dict(heatmap)
    import pickle

    pickle.dump(heatmap, open("prior.pkl", "wb"))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Extract the coordinates and counts
    x, y, z, c = [], [], [], []
    for coord, count in heatmap.items():
        x.append(coord[0])
        y.append(coord[1])
        z.append(coord[2])
        c.append(count)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=c, cmap="viridis", alpha=0.5)

    # Add color bar which maps values to colors
    plt.colorbar(sc)

    # Labeling the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig("prior.png")


def compute_kde_prior(len_prior=10000, plot=False):
    dataset = Lorenz(
        partition="train",
        no_pickle=True,
        max_len=len_prior,
        test_tt=0,
        val_tt=0,
        tr_tt=len_prior,
        sample_dt=0.02,
    )
    prior = dataset.get_data(False)[0]
    x, y, z = prior[:, 0], prior[:, 1], prior[:, 2]

    # # Perform Kernel Density Estimation
    kde = jax_kde(np.vstack([x, y, z]))
    pickle.dump(kde, open(f"{TEMP_DIR}/kde_jax.pkl", "wb"))

    if plot:
        plot_kde_prior(
            [np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)], kde
        )
    return kde


def plot_kde_prior(
    full_extent, kde=None, cmap="magma", colorbar=True, cache=True, no_labels=False
):
    min_x, max_x, min_y, max_y, min_z, max_z = full_extent
    if Path(f"{TEMP_DIR}/density.npy").exists() and cache:
        print("Loading cached density")
        density = np.load(f"{TEMP_DIR}/density.npy", allow_pickle=True)
    else:
        print("Computing density")
        xx, yy, zz = np.mgrid[
            min_x:max_x:100j,
            min_y:max_y:100j,
            min_z:max_z:100j,
        ]
        positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
        density = np.reshape(kde(positions).T, xx.shape)
        density.dump(f"{TEMP_DIR}/density.npy")

    summed_density = []
    for axis in [0, 1, 2]:
        summed_density.append(np.sum(density, axis=axis))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for axis in [0, 1, 2]:
        extent = (
            full_extent[2:]
            if axis == 0
            else (full_extent[0:2] + full_extent[4:6] if axis == 1 else full_extent[:4])
        )
        labels = ["X", "Y", "Z"]
        labels.pop(axis)

        axs[axis].imshow(
            summed_density[axis],
            cmap=cmap,
            extent=extent,
            origin="lower",
            vmin=0,
        )
        if colorbar and axis == 2:
            plt.colorbar(label="Probability Density")
        if not no_labels:
            axs[axis].set_xlabel(labels[0])
            axs[axis].set_ylabel(labels[1])
        else:
            axs[axis].axis("off")
    if not no_labels:
        fig.suptitle("Probability Map using KDE")
    path = f"{TEMP_DIR}/prob_map.png"
    plt.savefig(path)
    print(f"Saved KDE plot to {log.green(path)}")


def get_kde_prior(use_cache=True):
    if not os.path.exists(f"{TEMP_DIR}/kde_jax.pkl") or not use_cache:
        compute_kde_prior()
    else:
        kde = pickle.load(open(f"{TEMP_DIR}/kde_jax.pkl", "rb"))
        # kde = jax.device_put(kde, jax.devices()[-1]) # TODO TEMP Move to last GPU
    return kde


def lorenz_kde_prior(use_cache=True):
    kde = get_kde_prior(use_cache)

    def kde_2bd(s, kde):
        """Support multiple batch dimensions for the KDE. Expects non batch dim to be the last."""
        s = s.numpy()
        shape = s.shape
        s = np.moveaxis(s, -1, 0)
        s = s.reshape(3, -1)
        prob = kde(s)  # needs (3, n) shape
        return prob.transpose().reshape(*shape[:-1])

    class KDE:
        def log_prob(self, x):
            return kde_2bd(x, kde.logpdf)

        def prob(self, x):
            return kde_2bd(x, kde)

        def sample(self, key, shape):
            return kde.resample(key, shape)

    return KDE()


def lorenz_prior(sigma=1.0):
    """Approximates the prior distribution of the Lorenz system using a Gaussian Mixture Model."""
    prior = pickle.load(open(f"{TEMP_DIR}/prior.pkl", "rb"))
    probs = np.array(list(prior.values()))
    probs = ops.convert_to_tensor(probs, "float32")
    locs = np.array(list(prior.keys()))
    locs = ops.convert_to_tensor(locs, "float32")
    prior_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=probs),
        components_distribution=tfd.MultivariateNormalDiag(
            locs, ops.ones_like(locs) * sigma
        ),
    )
    return prior_dist


def get_lorenz_kwargs(config):
    tr_tt = config.nr_of_sequences * config.sequence_length
    val_tt = config.val_sequence_length * config.val_nr_of_sequences
    return dict(
        no_pickle=False,
        max_len=tr_tt + 2 * val_tt,
        test_tt=val_tt,
        val_tt=val_tt,
        tr_tt=tr_tt,
        sample_dt=config.get("sample_dt", 0.02),
        image_shape=config.data.image_shape[:2],
    )


def compute_snr(noise_std=0.1):
    dataset = Lorenz().tf_dataset(seq_length=None, batch_size=1)

    # Step 2: Compute Noise Power (same for all images)
    P_noise = noise_std**2

    # Initialize a list to store SNR for each image
    snr_list = []

    for data in dataset:
        # Step 1: Compute Signal Power for each image
        P_signal = np.mean(data[1] ** 2)

        # Step 3: Compute SNR for each image
        SNR = P_signal / P_noise
        snr_list.append(SNR)

    # Step 4: Aggregate SNR (compute mean SNR across the dataset)
    mean_snr = np.mean(snr_list)

    # Step 5: (Optional) Convert mean SNR to dB
    mean_snr_dB = 10 * np.log10(mean_snr)

    return mean_snr, mean_snr_dB
