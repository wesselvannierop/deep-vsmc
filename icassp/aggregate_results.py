import os

os.environ["KERAS_BACKEND"] = "numpy"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from usbmd import Config, log
from usbmd.utils import get_date_string

import vsmc.ops as dpf_ops

DARKMODE = True

timestamp = get_date_string()
TEMP_DIR = "temp"
RESULTS_DIR = "results"
SAVE_DIR = Path(TEMP_DIR) / timestamp
SAVE_DIR.mkdir()



plt.style.use("pyutils/styles/icassp.mplstyle")
if DARKMODE:
    plt.style.use("pyutils/styles/darkmode.mplstyle")

sweep_folder = Path(f"{RESULTS_DIR}/lorenz/icassp/tracking-performance-240910")
metric = "l2norm_wm"
timestamps = []
metrics_files = sweep_folder.rglob("metrics-val-final.csv")

results = {
    "dpf": {},
    "bpf": {},
    "ekpf": {},
    "encoder": {},
    "bpf10x": {},
}
key2model = {
    "dpf": "DPF (ours)",
    "bpf": "BPF",
    "ekpf": "EKF",
    "encoder": "Encoder (supervised)",
    "bpf10x": "BPF (10$\\times$ particles)",
}
metric2label = {
    "l2norm_wm": "Euclidean distance [-]",
}

for metrics_file in metrics_files:
    config_path = metrics_file.parent / "config.yaml"
    splits = str(metrics_file.parent.parent.name).split("-")
    model = splits[1]
    _timestamp = int(splits[-1])
    if _timestamp not in timestamps and timestamps != []:
        continue
    config = Config.load_from_yaml(config_path)
    metrics = pd.read_csv(metrics_file)
    if model not in results.keys():
        results[model] = {}
    results[model][config.data.awgn_std] = metrics[metric].tolist()


errorbar = False
fill_between = True
assert errorbar != fill_between, "Choose only one plotting method"
plt.figure()
for model in results.keys():
    if results[model] == {}:
        continue
    awgn_stds = np.sort(list(results[model].keys()))
    awgn_stds = awgn_stds[awgn_stds != 0.0]
    l2norm_wms = [
        results[model][awgn_std]
        for awgn_std in awgn_stds
        # if len(results[model][awgn_std]) == 20
    ]
    awgn_stds = awgn_stds[: len(l2norm_wms)]
    mean, ci = dpf_ops.confidence_interval(l2norm_wms, verbose=True)
    snr = 10 * np.log10(0.02 / awgn_stds**2)
    x = awgn_stds
    if errorbar:
        plt.errorbar(x, mean, yerr=ci, label=model)
    if fill_between:
        plt.plot(x, mean, label=key2model[model])
        plt.fill_between(x, mean - ci[0], mean + ci[1], alpha=0.5)
plt.legend(fontsize=7)
# plt.xlabel("Signal-to-noise ratio [dB]")
plt.xlabel("Std. of additive white Gaussian noise [-]")
plt.ylabel(metric2label[metric])
plt.grid(visible=True)
plt.ylim(0.5, 9)
plt.savefig(SAVE_DIR / "tracking_performance.png", transparent=True)
plt.savefig(SAVE_DIR / "tracking_performance.pdf")
plt.close()

log.info(f"Saved to {log.green(SAVE_DIR / "tracking_performance.png")}")
