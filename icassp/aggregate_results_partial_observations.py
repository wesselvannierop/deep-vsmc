import os

os.environ["KERAS_BACKEND"] = "numpy"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vsmc.ops as dpf_ops
from usbmd import Config, log
from usbmd.utils import get_date_string

DARKMODE = True

timestamp = get_date_string()
TEMP_DIR = "temp"
RESULTS_DIR = "results"
SAVE_DIR = Path(TEMP_DIR) / timestamp
SAVE_DIR.mkdir()


plt.style.use("pyutils/styles/icassp.mplstyle")
if DARKMODE:
    plt.style.use("pyutils/styles/darkmode.mplstyle")

sweep_folder = Path(
    f"{RESULTS_DIR}/lorenz/icassp/partial-observations-tracking-performance-240909"
)
# metric = "elbo"
# metric = "kl"
metric = "l2norm_wm"
# metric = "entropy"
timestamps = []
metrics_files = sweep_folder.rglob("metrics-val-final.csv")

results = {
    "dpf": {},
    "bpf": {},
    "ekpf": {},
    "encoder": {},
}
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
    "encoder": "Encoder (supervised)",
    "bpf10x": "BPF (10$\\times$ particles)",
}
metric2label = {
    "elbo": "ELBO [-]",
    "l2norm_wm": "Euclidean distance [-]",
}

# Load metrics from models
for metrics_file in metrics_files:
    config_path = metrics_file.parent / "config.yaml"
    splits = str(metrics_file.parent.parent.name).split("-")
    model = splits[1]
    if model not in results.keys():
        results[model] = {}
    _timestamp = int(splits[-1])
    if timestamps != [] and _timestamp not in timestamps:
        continue
    config = Config.load_from_yaml(config_path)
    metrics = pd.read_csv(metrics_file)
    if metric in metrics.columns:
        p_occlusion = (
            config.data.observation_fn_kwargs.p
            if hasattr(config.data, "observation_fn_kwargs")
            else config.observation_fn_kwargs.p
        )
        p_observation = 1 - p_occlusion
        # if p_observation < 0.6:
        #     continue  # TODO: run models for p_observation < 0.6
        results[model][p_observation] = metrics[metric].tolist()


# Plotting
errorbar = False
fill_between = True
assert errorbar != fill_between, "Choose only one plotting method"
fig = plt.figure()
for model in results.keys():
    if results[model] == {}:
        continue
    p_observation = np.sort(list(results[model].keys()))
    metric_values = [results[model][item] for item in p_observation]  # sort
    mean, ci = dpf_ops.confidence_interval(metric_values, verbose=True)
    if errorbar:
        plt.errorbar(p_observation, mean, yerr=ci, label=model)
    if fill_between:
        plt.plot(
            p_observation,
            mean,
            label=key2model.get(model, model),
            zorder=zorder.get(model, 1) + 1,
        )
        plt.fill_between(p_observation, mean - ci[0], mean + ci[1], alpha=0.5)
plt.legend(fontsize=7)
plt.xlabel("Proportion of observations [-]")
plt.ylabel(metric2label.get(metric, metric))
plt.grid(visible=True)
if metric == "elbo":
    plt.ylim([-10, 0])
elif metric == "l2norm_wm":
    plt.ylim([0.5, 10])
elif metric == "kl":
    plt.ylim([0, 15])

# Save
save_path = SAVE_DIR / f"aggregate_results_partial_observations_{metric}.png"
plt.savefig(save_path.with_suffix(".png"), transparent=True)
plt.savefig(save_path.with_suffix(".pdf"))
plt.close()

log.info(f"Saved to {log.green(save_path)}")
