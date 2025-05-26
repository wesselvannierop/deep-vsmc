import decimal
import itertools
from pathlib import Path

import numpy as np

from usbmd import Config, log

DECIMAL_PRECISION = 10  # max 15?
decimal.getcontext().prec = DECIMAL_PRECISION


def update_nested_dict(nested_dict: dict, keys: list, value):
    """
    Updates a nested dictionary inplace with the given value at the position specified by the keys.

    :param nested_dict: The dictionary to update
    :param keys: A list of keys specifying the position in the nested dictionary
    :param value: The value to assign to the specified position
    """
    d = nested_dict
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
    return nested_dict


class Sweeper:
    """
    Sweeps over every possible combination of the given parameters.
    """

    def __init__(
        self,
        config: Config,
        sweep_param_dict: dict,
        experiment_path: str | Path,
        name: str | Path,
    ):
        self.base_config = config
        self.sweep_param_dict = self._fix_decimal_issues(sweep_param_dict)
        self.combinations = list(self._generate_combinations())

        self.name = name
        self.experiment_path = Path(experiment_path) / self.name
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        log.info(f"Setting up Sweeper on: {log.bold(log.yellow(self.experiment_path))}")

        # Set save path in base config
        self.base_config.unfreeze()
        self.base_config.save_path = str(
            self.experiment_path / self._generate_save_path()
        )
        self.base_config.freeze()

    @staticmethod
    def _fix_decimal_issues(sweep_param_dict):
        for key, values in sweep_param_dict.items():
            if isinstance(values[0], float):
                x = np.array(values).astype(np.float64)
                sweep_param_dict[key] = [
                    float(decimal.Decimal(y).normalize()) for y in x
                ]
        return sweep_param_dict

    def _generate_combinations(self):
        # Extract keys and values
        keys = list(self.sweep_param_dict.keys())
        values = list(self.sweep_param_dict.values())
        for value in values:
            assert isinstance(
                value, (list, tuple)
            ), "Values in the sweep dict must be a list or tuple"

        # Generate combinations
        combinations = list(itertools.product(*values))

        # Combine keys with their respective values
        result = [dict(zip(keys, combination)) for combination in combinations]

        # Display the result
        for combo in result:
            yield combo

    def __len__(self):
        return len(self.combinations)

    def run(self, fn):
        """Runs `fn` with argument `config` for each configuration in the sweep."""
        for config in iter(self):
            fn(config)

    def __iter__(self):
        # Loop over all combinations
        for i, combination in enumerate(self.combinations):
            # Create a deep copy of the base config
            config = self.base_config.deep_copy()
            # Loop over all parameters in the combination and update the config
            for param_name, param_value in combination.items():
                _param_name = param_name.split(".")
                update_nested_dict(config, _param_name, param_value)

            log.info(f"SWEEPER ({i + 1}/{len(self)}) -- Running with config: {config}")
            yield config

    def _generate_save_path(self):
        # Example: "{timestamp}-sample_dt={sample_dt:.2f}-mixture={mixture}"
        path = "{timestamp}"
        for key in self.sweep_param_dict.keys():
            path += "-" + key + "={" + key + "}"
        return path
