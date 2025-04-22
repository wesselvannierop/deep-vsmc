from pathlib import Path

import keras
from keras import ops

from usbmd import log


def update_instance_with_attributes(
    instance, attributes: dict, ignore_keys: list | None = None, verbose: bool = True
):
    """
    Updates a attributes on an instance with a attributes object.
    """
    name = instance.__class__.__name__
    if ignore_keys is None:
        ignore_keys = []
    if verbose:
        print(f"------ Updating {name} attibutes ------")
    for key, value in attributes.items():
        if key in ignore_keys:
            if verbose:
                print(f"Skipping {key}, this cannot be changed after {name} creation.")
            continue
        if hasattr(instance, key) and getattr(instance, key) != value:
            if verbose:
                print(f"Updating {key} from {getattr(instance, key)} to {value}")
            setattr(instance, key, value)
        elif not hasattr(instance, key):
            log.warning(f"Adding new attribute {key} with value {value}")
            setattr(instance, key, value)
    if verbose:
        print(f"------ Done updating {name} attributes ------")


class BaseModel(keras.Model):
    """Base model class for keras models."""

    def load_own_variables(self, store):
        """Basically skip_mismatch=True for loading variables."""
        try:
            super().load_own_variables(store)
        except Exception as e:
            log.warning(f"Could not load model variables: {e}")

    def save_model_json(self, directory):
        """Save model as JSON file."""
        json_model = self.to_json(indent=4)
        json_model_path = str(Path(directory) / "model.json")
        with open(json_model_path, "w", encoding="utf-8") as file:
            file.write(json_model)
        log.success(
            f"Succesfully saved model architecture to {log.yellow(json_model_path)}"
        )

    def update_metrics(self, results: dict):
        """Update metric states with the instantaneous results from a batch."""
        for metric in self.metrics:
            if metric.name in results:
                metric.update_state(results[metric.name])

    def update_metrics_stateless(self, metrics_variables, results: dict):
        # Update metrics.
        new_metrics_vars = []
        for metric in self.metrics:
            if metric.name not in results:
                continue

            this_metric_vars = metrics_variables[
                len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)
            ]
            this_metric_vars = metric.stateless_update_state(
                this_metric_vars, results[metric.name]
            )
            new_metrics_vars += this_metric_vars
        return new_metrics_vars

    def _get_metrics_result_or_logs(self, logs):
        """This method will replace the logs with the values of the metrics when available.
        Normally, keras does this only when logs.keys == self.metrics.keys.

        My paradigm is to return the instantaneous values of the metrics in the logs such that wandb
        and the progress bar will always show the instantaneous values. The mean is only calculated
        at the end of the epoch by taking self.get_metrics_result().

        NOTE: this is automatically called in the train_step and test_step methods, at the end of
        each epoch.

        Args:
            logs: A `dict` of metrics returned by train / test step function.

        Returns:
            A `dict` containing values of the metrics listed in `self.metrics` and `logs` when
            the corresponding metrics are not available.
        """
        # If no skip_metrics attribute, create it
        if not hasattr(self, "skip_metrics"):
            self.skip_metrics = []

        # Get mean values
        metric_logs = self.get_metrics_result()
        all_zero = all(v == 0 for v in metric_logs.values())
        if all_zero:
            log.warning(
                "All metrics are zero. This is likely a bug. "
                + "Did you forget to update them?"
            )

        # Log missing metrics
        missing_logs = (set(logs.keys()) - set(self.skip_metrics)) - set(
            metric_logs.keys()
        )
        if missing_logs:
            log.warning(
                f"Some metrics could not be averaged: {list(missing_logs)}. "
                + "Logs will be taken instead. This may happen when train and test "
                + "metrics are not the same."
            )

        # Update logs with mean values
        return logs | metric_logs

    @property
    def backend(self):
        return keras.backend.backend()

    def update_model_from_config(self, config, ignore_keys=None, verbose=True):
        """
        Updates a attributes on a model instance with a config object.
        """
        update_instance_with_attributes(self, config, ignore_keys, verbose)


class BaseTrainer(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "step"):
            self.step = keras.Variable(0, trainable=False, dtype="int", name="step")
        if not hasattr(self, "epoch"):
            self.epoch = 0

    def _apply_gradients(self, gradients, variables):
        """Apply gradients to variables."""
        self.optimizer.apply_gradients(zip(gradients, variables))

    def apply_gradients(self, gradients, variables, skip_grad=None, grad_norm=None):
        """
        Apply gradients to variables if the gradient norm is less than skip_grad.
        If skip_grad is None, the gradients are always applied.
        """
        pred = True if skip_grad is None else ops.less_equal(grad_norm, skip_grad)
        ops.cond(
            pred,
            lambda: self._apply_gradients(gradients, variables),
            lambda: None,
        )
