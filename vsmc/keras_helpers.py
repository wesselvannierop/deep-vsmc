import collections
import csv
import gc
import warnings
from typing import Any, Dict, Optional

import keras
import numpy as np
import wandb
from keras.src.utils import file_utils
from wandb.integration.keras import WandbMetricsLogger


class AlwaysContains:
    """Fake dict that will always returns True for `item in self`"""

    def __contains__(self, item):
        return True


class MyProgbar(keras.utils.Progbar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "stateful_metrics"), (
            "Progbar must have stateful_metrics attribute in order to work properly. "
            "Maybe keras has been updated"
        )

        # override stateful_metrics to always return True
        self.stateful_metrics = AlwaysContains()


class MyProgbarLogger(keras.callbacks.ProgbarLogger):
    """
    Wrapper around keras.callbacks.ProgbarLogger that does not do averaging on the metrics
    and allows for customizing the unit name and other parameters. Will replace the original
    ProgbarLogger if provided as a callback.

    Will also show the validation progress bar when called from `fit`.
    """

    def __init__(self, unit_name="step", **kwargs):
        super().__init__()
        self.unit_name = unit_name
        self.progbar_kwargs = kwargs

    def on_train_begin(self, logs=None):
        # When this logger is called inside `fit`, normally validation is silent, but we want to show it
        self._called_in_fit = False

    def _maybe_init_progbar(self):
        if self.progbar is None:
            self.progbar = MyProgbar(
                target=self.target, verbose=self.verbose, **self.progbar_kwargs
            )


def deserialize(item):
    """alias for keras.utils.deserialize_keras_object"""
    return keras.utils.deserialize_keras_object(item)


def deserialize_config(config, keys):
    for key in keys:
        if key in config:
            config[key] = deserialize(config[key])
        else:
            warnings.warn(f"Key {key} not found in config during deserialization.")
    return config


def get_rescaling_layer(a, b, x_min=0, x_max=255):
    """Rescaling layer.

    Args:
        a (float): minimum value of range to map to.
        b (float): maximum value of range to map to.
        x_min (float, optional): min value of input image. Defaults to 0.
        x_max (float, optional): max value of input image. Defaults to 255.

    Returns:
        tf layer: Rescaling layer.

    """
    scale = (b - a) / (x_max - x_min)
    offset = a
    return keras.layers.Rescaling(scale=scale, offset=offset)


class MyWandbMetricsLogger(WandbMetricsLogger):
    """
    Changes to original: groups validation metrics under 'val/'.
    """

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        logs = dict() if logs is None else {f"epoch/{k}": v for k, v in logs.items()}

        # custom (group validation metrics under val/):
        keys_to_modify = [key for key in logs.keys() if key.startswith("epoch/val_")]
        for key in keys_to_modify:
            new_key = "val/" + key.removeprefix("epoch/val_")
            logs[new_key] = logs.pop(key)

        logs["epoch/epoch"] = epoch

        lr = self._get_lr()
        if lr is not None:
            logs["epoch/learning_rate"] = lr

        wandb.log(logs)


class ClearMemory(keras.callbacks.Callback):
    """Keras callback to clear memory after each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()


class CSVLoggerEval(keras.callbacks.Callback):
    """Callback that streams validation results to a CSV file.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.

    Args:
        filename: Filename of the CSV file, e.g. `'run/log.csv'`.
        separator: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing
            training). False: overwrite existing file.
    """

    def __init__(self, filename, separator=",", append=False):
        super().__init__()
        self.sep = separator
        self.filename = file_utils.path_to_string(filename)
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True

    def on_test_begin(self, logs=None):
        if self.append:
            if file_utils.exists(self.filename):
                with file_utils.File(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = file_utils.File(self.filename, mode)

    def on_test_end(self, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return f'"[{", ".join(map(str, k))}]"'
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": "val"})
        row_dict.update((key, handle_value(logs.get(key, "NA"))) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

        # Close file
        self.csv_file.close()
        self.writer = None


class EpochCounterCallback(keras.callbacks.Callback):
    """Makes the epoch and step available in the model."""

    def __init__(self, len_training_dataset: int, recompile_now_callable=None):
        super().__init__()
        self.len_training_dataset = len_training_dataset
        self._recompile_now_callable = recompile_now_callable

    def recompile_now(self, epoch):
        if self._recompile_now_callable is not None:
            return self._recompile_now_callable(epoch)
        else:
            return getattr(self.model, "recompile_now", False)

    def on_train_begin(self, logs=None):
        assert hasattr(self.model, "step"), "Model must have an step attribute"
        self.model.is_training = True

    def on_epoch_begin(self, epoch, logs=None):
        self.model.epoch = epoch
        self.model.step.assign(self.len_training_dataset * epoch)
        print(f"Step at epoch={epoch + 1} begin:", self.model.step)

        # Recompile when discriminator is getting activated
        if self.recompile_now(epoch):
            self.model.make_train_function(force=True)  # recompile train_step!

    def on_train_batch_end(self, batch, logs=None):
        self.model.step.assign_add(1)
