import os
import sys
import warnings
from pathlib import Path

import keras
import wandb

from zea import Config, log
from zea.utils import get_date_string

debugging = True if sys.gettrace() else False
print("Debugging mode is on!" if debugging else "Debugging mode is off!")


def dict_bool_to_key(**kwargs):
    """
    When a boolean is True, replace it with the key.
    When a boolean is False, replace it with "not_" + key.
    """
    for key in kwargs:
        if isinstance(kwargs[key], bool):
            if kwargs[key]:
                kwargs[key] = key
            else:
                kwargs[key] = "not_" + key
        if isinstance(kwargs[key], dict):
            kwargs[key] = dict_bool_to_key(**kwargs[key])
    return kwargs


class Experiment:
    def __init__(
        self,
        config: str | Path | dict | Config,
        path: str | Path = None,
        backend: str | None = None,
        seed: int = 21,
        debug_run=False,
    ):
        self.path = path
        self.backend = backend
        self.seed = seed
        self.debug_run = debug_run or debugging

        # Load config
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            assert config_path.exists(), f"Config file does not exist: {config_path}"
            config = Config.load_from_yaml(config_path)
            config.config_path = str(config_path)  # Save the config path
            print(f"Setting up experiment for: {log.yellow(config.config_path)}")
        elif isinstance(config, dict):
            print("Config is given as an argument.")
            config = Config(config)  # Convert to Config object, if not already
        else:
            raise ValueError("config should be a path or a dictionary.")
        self.config = config

        if self.path is None:
            assert (
                "save_path" in self.config
            ), "path not given and save_path not found in config."
            self.path = self.config.save_path

        # Add data key if not present
        if not "data" in self.config:
            self.config.data = {}

        # Settings
        self.backend = backend or os.environ.get("KERAS_BACKEND")

        # Backend specific setup
        if self.backend == "tensorflow":
            import tensorflow as tf

            tf.config.run_functions_eagerly(self.run_eagerly)
            # tf.debugging.enable_check_numerics()
        elif self.backend == "jax":
            import jax

            jax.disable_jit(disable=self.run_eagerly)
        else:
            warnings.warn(f"[Experiment] does not recognize: {self.backend}")

        # Seed random number generators
        keras.utils.set_random_seed(self.seed)

        self.path = self.format_path(self.path, self.config)
        config.save_path = self.path

        # Copy config file to experiment path
        self.config.save_to_yaml(self.path / "config.yaml")

        # Freeze such that no new attributes can be added
        self.config.freeze()

    @property
    def run_eagerly(self):
        return self.debug_run

    def setup_wandb(self, **kwargs):
        mode = kwargs.pop("mode", "online")
        mode = "offline" if (self.debug_run and mode == "online") else mode
        self.run = wandb.init(mode=mode, config=self.config, dir=self.path, **kwargs)
        print(f"wandb: {self.run.job_type} run {self.run.name}\n")
        return self.run

    def format_path(self, path, config: Config):
        # Use config & timestamp to format the path
        format_kwargs = Config(dict_bool_to_key(**config.deep_copy()))
        timestamp = get_date_string()
        format_kwargs.timestamp = timestamp
        path = Path(path.format(**format_kwargs))

        # If the experiment path already exists, add the timestamp to the path
        if path.exists():
            config_path = config.get("config_path", None)
            config_str = "_" + Path(config_path).stem if config_path else ""
            path /= timestamp + config_str

        # If in debugging mode, add _debug to the experiment path
        if self.debug_run:
            path = path.parent / (path.name + "_debug")

        # Make the experiment path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def make_subdirs(self, *subdirs):
        for subdir in subdirs:
            (self.path / subdir).mkdir()


def setup_experiment(config, path=None, backend=None, seed=21, debug_run=False):
    experiment = Experiment(config, path, backend, seed, debug_run)
    config = experiment.config
    if "wandb" in config:
        run = experiment.setup_wandb(config.wandb)
    else:
        run = None
    return config, run
