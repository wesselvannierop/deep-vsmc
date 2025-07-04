import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras

from vsmc.baselines.lorenz_encoder.train_encoder import Dataset, l2_loss
from vsmc.experiments import setup_experiment
from vsmc.models.preset_loader import from_preset

if __name__ == "__main__":
    # Setup experiment
    config, run = setup_experiment(
        config=dict(
            save_path="experiments/lorenz_encoder",
            checkpoint="models/lorenz_encoder.keras",
        )
    )

    # Create dataset
    dataset = Dataset(n_samples=3200, batch_size=32, p_range=(0.1, 0.1), awgn_std=0.0)

    # Model
    # model = build_image_encoder(encoder_depth=4, encoded_dim=3, nfeatbase=16)
    # model.summary()
    # if checkpoint := config.get("checkpoint"):
    #     model.load_weights(checkpoint)
    model = keras.models.load_model(from_preset(config.checkpoint), compile=False)

    # Fit
    model.compile(optimizer="adam", loss=l2_loss)
    output = model.evaluate(
        dataset, callbacks=[keras.callbacks.CSVLogger(config.save_path / "val.csv")]
    )
    print(model.metrics_names, output)
