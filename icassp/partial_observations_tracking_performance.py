import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DPF")
    parser.add_argument(
        "--run_sweep",
        action="store_true",
        help="Actually run the sweep",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use remote data paths",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tensorflow",  # jax needs stateless test_step
    )
    return parser.parse_args()


import vsmc.runnable  # isort: skip

args = parse_args()
vsmc.runnable.runnable(
    args.backend, "auto:2" if args.backend == "tensorflow" else "auto:1"
)

import time
from pathlib import Path

import numpy as np

from zea import Config
from vsmc.experiments import setup_experiment
from vsmc.learned_pf import dpf_run
from vsmc.sweeper import Sweeper

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    experiment_path = Path(
        f"results/lorenz/icassp/partial-observations-tracking-performance-250103"
    )
    config_paths = {
        "dpf": "dpf/experiments/icassp/dpf_lorenz_partial_observations.yaml",
        # "dpf_evolution": "dpf/experiments/icassp/transition-known/dpf_lorenz_evolution.yaml",
        "bpf": "dpf/experiments/icassp/bootstrap_lorenz.yaml",
        # "bpf_evolution": "dpf/experiments/icassp/transition-known/bootstrap_lorenz.yaml",
        "bpf10x": "dpf/experiments/icassp/bootstrap_lorenz_10x.yaml",
        # "ekpf": "dpf/experiments/icassp/ekpf_lorenz.yaml",
        # "ekpf_evolution": "dpf/experiments/icassp/transition-known/ekpf_lorenz.yaml",
        # "encoder": "dpf/experiments/icassp/encoder_lorenz_partial_observations.yaml",
    }

    for model, config_path in config_paths.items():
        base_config = Config.load_from_yaml(config_path)
        base_config.freeze()
        sweep_name = f"icassp-{model}-{timestamp}"

        if model != "ekpf":
            p = np.arange(0.0, 0.9, 0.1)
        else:
            p = np.arange(0.0, 0.12, 0.02)

        sweeper = Sweeper(
            base_config,
            {
                "data.observation_fn": ["mask"],
                "data.observation_fn_kwargs.p": p,
            },
            experiment_path,
            sweep_name,
        )

        for job_index, config in enumerate(sweeper):
            config.unfreeze()
            config.save_to_yaml(
                str(sweeper.experiment_path / f"config_{job_index}.yaml")
            )

            if args.run_sweep or True:
                config, run = setup_experiment(config=config)
                config.freeze()
                dpf_run(config, verbose=False)
                if run is not None:
                    run.finish()

        print(f"jobs: {len(sweeper)}")
        print(f"sweep_name: {sweep_name}")
        print(f"results saved to: {sweeper.experiment_path}")
