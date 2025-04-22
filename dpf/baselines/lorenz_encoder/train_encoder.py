from dpf.runnable import runnable

runnable("jax", "auto:1", hide_first_for_tf=False)

import sys

debugging = True if sys.gettrace() else False
if debugging:
    debug = dict(jit_compile=False, run_eagerly=True)
else:
    debug = dict()

import jax
import keras
from keras import ops
from wandb.integration.keras import WandbMetricsLogger

from dpf.data.lorenz_data import LorenzPSF, lorenz_kde_prior
from dpf.dpf_utils import Masker
from dpf.models.vsmc import build_image_encoder
from experiments import setup_experiment


class Dataset(keras.utils.PyDataset):
    def __init__(self, n_samples, batch_size, p_range=(0.1, 0.9), awgn_std=0.1):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.x_range = (-35, 35)
        self.y_range = (-35, 35)
        self.z_range = (1, 50)
        self.image_shape = (28, 28)
        self.p_range = p_range
        self.awgn_std = awgn_std
        self.masker = Masker(image_shape=self.image_shape, block_size=4, p=p_range)
        self.lorenz_psf = LorenzPSF(
            image_shape=self.image_shape, x_max=self.x_range[1], y_max=self.y_range[1]
        )
        self.key = jax.random.key(0)
        self.prior = lorenz_kde_prior()

    def __len__(self):
        # return the number of batches
        return self.n_samples // self.batch_size

    def __getitem__(self, idx):
        # Generate random coords
        # x = keras.random.uniform((self.batch_size,), *self.x_range)
        # y = keras.random.uniform((self.batch_size,), *self.y_range)
        # z = keras.random.uniform((self.batch_size,), *self.z_range)
        # coords = ops.stack([x, y, z], axis=-1)
        self.key, subkey = jax.random.split(self.key, 2)
        coords = self.prior.sample(subkey, (self.batch_size,))  # [3, batch_size]
        coords = ops.transpose(coords)  # [batch_size, 3]

        # Use Lorenz PSF to generate images
        images = self.lorenz_psf(coords)
        images, _ = self.lorenz_psf.noise_images(
            images, salt_prob=0.0, awgn_std=self.awgn_std, observation_fn=self.masker
        )

        return images, coords  # input, target


def l2_loss(y_true, y_pred):
    return ops.mean(ops.linalg.norm(y_true - y_pred, axis=-1))


config_partial_observations = dict(
    awgn_std=0.1,
    val_awgn_std=0.1,
    p_range=(0.0, 0.8),
    val_p_range=(0.8, 0.8),
    save_path="{output_dir}/dpf/lorenz_encoder/{timestamp}-p",
    wandb=dict(project="lorenz_encoder", entity="wessel"),
)
config_awgn_noise = dict(
    awgn_std=(0.1, 0.6),
    val_awgn_std=0.6,
    p_range=(0.0, 0.0),
    val_p_range=(0.0, 0.0),
    save_path="{output_dir}/dpf/lorenz_encoder/{timestamp}-awgn",
    wandb=dict(project="lorenz_encoder", entity="wessel"),
)
# root_config = config_partial_observations
root_config = config_awgn_noise


if __name__ == "__main__":
    # Setup experiment
    config, run = setup_experiment(config=root_config)
    wandb_enabled = hasattr(config, "wandb")

    # Create dataset
    # epochs and n_samples are arbitrary because it is random data
    dataset = Dataset(
        n_samples=3200, batch_size=32, p_range=config.p_range, awgn_std=config.awgn_std
    )
    val_dataset = Dataset(
        n_samples=320,
        batch_size=32,
        p_range=config.val_p_range,
        awgn_std=config.val_awgn_std,
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.save_path / "lorenz_encoder.keras",
            monitor="loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            save_freq="epoch",
            initial_value_threshold=None,
        ),
        keras.callbacks.CSVLogger(config.save_path / "metrics.csv", append=False),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=20, verbose=1, mode="min"
        ),
    ]
    if wandb_enabled:
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))

    # Model
    model = build_image_encoder(encoder_depth=4, encoded_dim=3, nfeatbase=32)
    model.summary()

    # Fit
    optimizer = keras.optimizers.AdamW(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=l2_loss, **debug)
    model.fit(
        dataset,
        epochs=200,
        callbacks=callbacks,
        validation_data=val_dataset,
        validation_freq=10,
    )
