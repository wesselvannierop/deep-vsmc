import os
import shutil
import sys
import warnings
from pathlib import Path

import git
import keras
import pathspec
import wandb

from usbmd import Config, log
from usbmd.utils import get_date_string

debugging = True if sys.gettrace() else False
print("Debugging mode is on!" if debugging else "Debugging mode is off!")

gitignore_path = Path(".gitignore")
if gitignore_path.exists():
    with gitignore_path.open("r") as f:
        gitignore_patterns = f.read().splitlines()
    spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_patterns)
else:
    spec = pathspec.PathSpec([])


def get_git_root(path: Path | str) -> Path:
    git_repo = git.Repo(str(path), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)


REPO_ROOT = get_git_root(__file__)


def copy_repo_py_files(to_dir: Path, verbose=False, zip=True):
    """
    Copy all the .py files in the repository to the `to_dir`.
    Uses your .gitignore to exclude files and folders.
    """

    # Assert that the destination directory does not exist
    assert not to_dir.exists(), "Destination directory already exists."

    # Assert that the destination directory is outside the repository or is ignored by .gitignore
    if to_dir.resolve().is_relative_to(REPO_ROOT):
        assert spec.match_file(
            to_dir.resolve().relative_to(REPO_ROOT)
        ), "Destination directory lives inside this repository is not ignored by .gitignore. \
            This means that the now copied files will be copied the next experiment too, \
            leading to an ever growing number of files being copied."

    to_dir.mkdir()

    py_files = []
    for root, dirs, files in os.walk(REPO_ROOT, topdown=True):
        files = [f for f in files if not f[0] == "."]
        dirs[:] = [d for d in dirs if not d[0] == "."]
        dirs[:] = [d for d in dirs if not spec.match_file(d)]
        py_files.extend(Path(root) / f for f in files if f.endswith(".py"))

    for p in py_files:
        to_p = to_dir / p.relative_to(REPO_ROOT)
        if verbose:
            print(f"Copying {p} to {to_p}")

        to_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, to_p)

    if zip:
        shutil.make_archive(to_dir, "zip", to_dir)
        shutil.rmtree(to_dir)


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

        # Copy config file to experiment path
        self.config.save_to_yaml(self.path / "config.yaml")

        # Copy repo code to experiment path
        copy_repo_py_files(self.path / "code")

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
        # Use config, timestamp, and output_dir to format the path
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
