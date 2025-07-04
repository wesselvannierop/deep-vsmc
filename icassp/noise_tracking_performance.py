import vsmc.runnable  # isort: skip

vsmc.runnable.runnable()

import argparse
import time
from pathlib import Path

import numpy as np
from zea import Config

from vsmc.experiments import setup_experiment
from vsmc.learned_pf import dpf_run
from vsmc.sweeper import Sweeper


def parse_args():
    parser = argparse.ArgumentParser(description="DPF")
    parser.add_argument(
        "--run_sweep",
        action="store_true",
        help="Actually run the sweep",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    experiment_path = Path(
        f"{args.output_dir}/dpf/lorenz/icassp/tracking-performance"
    )
    config_paths = {
        "dpf": "configs/icassp/dpf_lorenz.yaml",
        "bpf10x": "configs/icassp/bootstrap_lorenz_10x.yaml",
        "bpf": "configs/icassp/bootstrap_lorenz.yaml",
        "ekpf": "configs/icassp/ekpf_lorenz.yaml",
        "encoder": "configs/icassp/encoder_lorenz.yaml",
    }

    for key, config_path in config_paths.items():
        base_config = Config.from_yaml(config_path)
        sweep_name = f"icassp-{key}-{timestamp}"
        sweeper = Sweeper(
            base_config,
            {
                "data.awgn_std": np.arange(0.1, 0.65, 0.05),
            },
            experiment_path,
            sweep_name,
        )

        for job_index, config in enumerate(sweeper):
            config.unfreeze()
            config.save_to_yaml(
                str(sweeper.experiment_path / f"config_{job_index}.yaml")
            )
            if key != "dpf":
                config.likelihood_sigma = config.data.awgn_std
            config.validation.elbo_likelihood_sigma = config.data.awgn_std
            if hasattr(config, "val_data"):
                config.val_data.awgn_std = config.data.awgn_std

            if args.run_sweep:
                config, run = setup_experiment(config=config)
                config.freeze()
                dpf_run(config)
                if run is not None:
                    run.finish()

        print(f"jobs: {len(sweeper)}")
        print(f"sweep_name: {sweep_name}")
        print(f"results saved to: {sweeper.experiment_path}")
