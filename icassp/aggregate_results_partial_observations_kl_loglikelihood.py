import os

os.environ["KERAS_BACKEND"] = "numpy"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zea import Config, log
from zea.utils import get_date_string

DARKMODE = True

timestamp = get_date_string()
TEMP_DIR = "temp"
RESULTS_DIR = "results"
SAVE_DIR = Path(TEMP_DIR) / timestamp
SAVE_DIR.mkdir()

PRESENTATION_VERSION = True
if PRESENTATION_VERSION:
    fig_kwargs = {
        "figsize": (4.49, 2.6),
    }
    legend_kwargs = {
        "loc": "outside center right",
    }
else:
    fig_kwargs = {}
    legend_kwargs = {
        "loc": "center right",
    }


plt.style.use("pyutils/styles/icassp.mplstyle")
if DARKMODE:
    plt.style.use("pyutils/styles/darkmode.mplstyle")

sweep_folder = Path(
    f"{RESULTS_DIR}/lorenz/icassp/partial-observations-tracking-performance-240909"
)

# metric = "elbo"
metric1 = "kl"
metric2 = "log_likelihood"
# metric = "entropy"
timestamps = []


zorder = {
    "dpf": 5,
    "bpf": 4,
    "ekpf": 3,
    "encoder": 2,
}
key2model = {
    "dpf": "DPF (ours)",
    "bpf": "BPF",
    "ekpf": "EKF",
    "bpf10x": "BPF (10$\\times$ particles)",
    "encoder": "Encoder (supervised)",
}
key2linestyle = {
    "bpf10x": {"color": "tab:purple", "linestyle": (5, (10, 3)), "marker": "d"},
}
metric2label = {
    "elbo": "ELBO [-]",
    "l2norm_wm": "Euclidean distance [-]",
    "kl": "KL to prior [-]",
    "log_likelihood": "Log-likelihood [-]",
}

# Load metrics from models
metrics_files = sweep_folder.rglob("metrics-val-final.csv")
results = {
    "dpf": {},
    "bpf": {},
    "ekpf": {},
}

for metrics_file in metrics_files:
    config_path = metrics_file.parent / "config.yaml"
    splits = str(metrics_file.parent.parent.name).split("-")
    model = splits[1]
    if model == "encoder":
        continue
    if model not in results.keys():
        results[model] = {}
    _timestamp = int(splits[-1])
    if timestamps != [] and _timestamp not in timestamps:
        continue
    config = Config.load_from_yaml(config_path)
    metrics = pd.read_csv(metrics_file)
    for metric in [metric1, metric2]:
        if metric not in results[model].keys():
            results[model][metric] = {}
        if metric in metrics.columns:
            p_occlusion = (
                config.data.observation_fn_kwargs.p
                if hasattr(config.data, "observation_fn_kwargs")
                else config.observation_fn_kwargs.p
            )
            p_observation = 1 - p_occlusion
            results[model][metric][p_observation] = metrics[metric].tolist()
        else:
            print(f"Metric {metric} not found in {metrics_file}")


def confidence_interval(metric_values, axis=-1, verbose=False):
    """
    Args:
        metric_values (ndarray): The metric values to compute the confidence interval for.
        axis (int, optional): The axis along which to compute the confidence interval.
            The other axes will be considered as batch dimensions. Defaults to -1.

    Sources:
        - https://en.wikipedia.org/wiki/97.5th_percentile_point
        - https://en.wikipedia.org/wiki/Confidence_interval
        - https://en.wikipedia.org/wiki/Standard_error
    """
    metric_values = np.array(metric_values)
    mean = np.nanmean(metric_values, axis=axis)
    std = np.nanstd(metric_values, axis=axis)
    n = metric_values.shape[axis]
    if verbose:
        print(f"N observations: {n}")
    standard_error = std / np.sqrt(n)
    upper_ci = standard_error * 1.96
    lower_ci = standard_error * 1.96
    ci = np.stack([lower_ci, upper_ci])
    return mean, ci


# Plotting
errorbar = False
fill_between = True
assert errorbar != fill_between, "Choose only one plotting method"
fig, axs = plt.subplots(2, 1, sharex=True, **fig_kwargs)
for model in results.keys():
    if results[model] == {}:
        continue
    for idx, metric in enumerate([metric1, metric2]):
        p_observation = np.sort(list(results[model][metric].keys()))
        axs[idx].set_ylabel(metric2label.get(metric, metric))
        metric_values = [results[model][metric][item] for item in p_observation]  # sort
        mean, ci = confidence_interval(metric_values, verbose=True)
        if idx == 1:
            label = key2model.get(model, model)
        else:
            label = None
        if errorbar:
            axs[idx].errorbar(p_observation, mean, yerr=ci, label=label)
        if fill_between:
            axs[idx].plot(
                p_observation,
                mean,
                label=label,
                zorder=zorder.get(model, 1) + 1,
                **key2linestyle.get(model, {}),
            )
            color = key2linestyle.get(model, {}).get("color")
            extra_dict = dict(facecolor=color) if color is not None else {}
            axs[idx].fill_between(
                p_observation,
                mean - ci[0],
                mean + ci[1],
                alpha=0.5,
                **extra_dict,
            )
# legend center right
fig.legend(fontsize=7, **legend_kwargs)
axs[-1].set_xlabel("Proportion of observations [-]")

for ax in axs:
    ax.grid(True)
axs[0].set_ylim([5, 22])
axs[1].set_ylim([640, 690])
# if metric == "elbo":
#     plt.ylim([-10, 0])
# elif metric == "l2norm_wm":
#     plt.ylim([0, 10])
# elif metric == "kl":
#     plt.ylim([0, 15])

# Save
save_path = SAVE_DIR / "aggregate_results_partial_observations_kl_ll.png"
plt.savefig(save_path.with_suffix(".png"), transparent=True)
plt.savefig(save_path.with_suffix(".pdf"))
plt.close()

log.info(f"Saved to {log.green(save_path)}")
